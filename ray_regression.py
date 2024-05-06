import argparse
import json
import os
import imageio

# from inference.predict import predict_cameras
import mitsuba as mi
import numpy as np
import torch
from tqdm import tqdm, trange

from generate_rays import readRaysFromFile
from model.diffuser import RayDiffuser
from model.scheduler import NoiseScheduler
from utils.rays_conversion import pluckerRays2Point

# from vit_pytorch import SimpleViT
# from vit_pytorch.vit_for_small_dataset import ViT


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="data/RayDiffusionData/main_xml/scene0001_01"
    )
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--n_iteration", type=int, default=500)
    parser.add_argument("--model_dir", type=str, default="output/model.pth")
    return parser


def compute_image_uv(num_patches_x, num_patches_y):
    """
    Compute the UV coordinates of the origins of the rays in the image space.  Bottom-left corner of the image is (-1, -1) and top-right corner is (1, 1)
    """
    cell_w = 2 / num_patches_x
    cell_h = 2 / num_patches_y
    x = torch.linspace(-1 + cell_w / 2, 1 - cell_w / 2, num_patches_x)
    y = torch.linspace(-1 + cell_h / 2, 1 - cell_h / 2, num_patches_y)
    x, y = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([x, y], dim=-1)


def visualizeRays(rays, num_patches_x, num_patches_y):
    rays = np.array(rays)  # (N, 6, H * W), H = num_patches_y, W = num_patches_x
    rays = rays.reshape(-1, 6, num_patches_y, num_patches_x).transpose(0, 2, 3, 1)
    direction_imgs = rays[..., :3]
    moment_imgs = rays[..., 3:]
    images = np.concatenate([direction_imgs, moment_imgs], axis=2)
    images = (images + 1) * 255 // 2
    images = images.astype(np.uint8)

    return images  # (N, H, W, 6)


def main(data_dir, output_dir, n_iteration, model_dir=None):
    with open(os.path.join(data_dir, "params.json"), "r") as f:
        json_data = json.load(f)
        num_patches_x = json_data["num_patches_x"]
        num_patches_y = json_data["num_patches_y"]
        num_images = json_data["num_images"]

    device = torch.device("cuda:0")

    all_images = []
    all_rays = []
    for i in range(num_images):
        image_file = os.path.join(data_dir, "images", f"image{i}.exr")
        image = mi.Bitmap(image_file)
        image = np.array(image, dtype=np.float32)
        image = image.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        all_images.append(image)

        rays_file = os.path.join(data_dir, "rays", f"rays{i}.txt")
        rays = readRaysFromFile(rays_file)
        rays = np.array(rays, dtype=np.float32)
        rays = rays.transpose(1, 0).reshape(
            6, num_patches_y, num_patches_x
        )  # (num_patches_x * num_patches_y, 6) -> (6, num_patches_y, num_patches_x)
        all_rays.append(rays)

    all_images = np.stack(all_images, axis=0)
    all_images = torch.tensor(all_images).unsqueeze(0).to(device)
    all_rays = np.stack(all_rays, axis=0)
    all_rays = torch.tensor(all_rays).unsqueeze(0).to(device)

    noise_scheduler = NoiseScheduler(
        type="linear",
        max_timesteps=100,
        beta_start=0.0001,
        beta_end=0.2,
    )

    model = RayDiffuser(
        depth=num_patches_x,
        width=num_patches_y,
        P=1,
        max_num_images=num_images,
        noise_scheduler=noise_scheduler,
        feature_extractor="dino",
        append_ndc=True,
    ).to(device)

    if model_dir is not None:
        model.load_state_dict(torch.load(model_dir))

    batch_size = all_images.shape[0]
    num_images = all_images.shape[1]
    t = model.noise_scheduler.max_timesteps
    x_t = torch.randn(batch_size, num_images, 6, num_patches_x, num_patches_y, device=device)
    image_features = model.feature_extractor(all_images, autoresize=True)
    print(f"image_features.shape: {image_features.shape}")
    # (H, W, 2)
    uv = compute_image_uv(num_patches_x=num_patches_x, num_patches_y=num_patches_y).to(device)
    # (B, N, 2, H, W)
    uv = (
        uv.permute(2, 0, 1)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, num_images, 2, num_patches_x, num_patches_y)
    )
    print(f"uv.shape: {uv.shape}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if model_dir is None:
        progress_bar = trange(n_iteration)
        losses = []
        for _ in progress_bar:
            optimizer.zero_grad()
            eps_pred, noise_sample = model(
                features=image_features,
                rays_noisy=x_t,
                t=t,
                ndc_coordinates=uv,
            )

            loss = torch.nn.functional.mse_loss(eps_pred, all_rays)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            # print loss on progress bar
            progress_bar.set_description(f"Loss: {loss.item()}")

        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
        print(f"Model saved to {os.path.join(output_dir, 'model.pth')}")

        # save losses as a plot
        import matplotlib.pyplot as plt

        plt.plot(losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(output_dir, "loss.png"))
        plt.close()

        pred_rays_vis = visualizeRays(eps_pred.cpu().detach().numpy(), num_patches_x, num_patches_y)
        gt_rays_vis = visualizeRays(all_rays.cpu().numpy(), num_patches_x, num_patches_y)
        rays_vis = np.concatenate([pred_rays_vis, gt_rays_vis], axis=1)
        os.makedirs(os.path.join(output_dir, "visualization"), exist_ok=True)
        for i in range(rays_vis.shape[0]):
            imageio.imwrite(os.path.join(output_dir, "visualization", f"vis{i}.png"), rays_vis[i])
    else:
        with torch.no_grad():
            eps_pred, noise_sample = model(
                features=image_features,
                rays_noisy=x_t,
                t=t,
                ndc_coordinates=uv,
            )
        # compute light source position and loss w.r.t. ground truth
        gt_light_pos = np.array([-1, 1, 1])
        pred_rays = (
            eps_pred.cpu()
            .detach()
            .numpy()
            .reshape(-1, 6, num_patches_y * num_patches_x)
            .transpose(0, 2, 1)
        )
        pred_positions = []
        position_losses = []
        for i in range(pred_rays.shape[0]):
            pred_light_pos = pluckerRays2Point(pred_rays[i])
            position_loss = np.mean(np.linalg.norm(pred_light_pos - gt_light_pos, axis=1))
            pred_positions.append(pred_light_pos)
            position_losses.append(position_loss)

        with open(os.path.join(output_dir, "position_losses.txt"), "w") as f:
            for i in range(len(position_losses)):
                f.write(f"{i}\n")
                f.write(f"{pred_positions[i]}\n")
                f.write(f"{position_losses[i]}\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
