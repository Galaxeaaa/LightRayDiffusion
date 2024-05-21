import argparse
import glob
import json
import os
from datetime import datetime

import imageio

# from inference.predict import predict_cameras
import mitsuba as mi
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataloader.raydiffusion import RayDiffusionData
from generate_rays import readRaysFromFile
from model.diffuser import RayDiffuser
from model.scheduler import NoiseScheduler
from utils.rays_conversion import pluckerRays2Point
from utils.visualization import visualizeRays


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/RayDiffusionData/scenes_on_cluster")
    # parser.add_argument("--data_dir", type=str, default="test_output")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--max_n_iteration", type=int, default=500)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--max_num_images", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--lr_scheduler", type=str, default="ExponentialLR")
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.99)
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


def train(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    max_n_iteration = args.max_n_iteration
    max_num_images = args.max_num_images
    num_images = args.num_images
    learning_rate = args.learning_rate
    lr_scheduler = args.lr_scheduler
    lr_scheduler_gamma = args.lr_scheduler_gamma

    # load data
    train_data = RayDiffusionData(data_dir, split="train")
    dataloader = DataLoader(train_data, batch_size=num_images, shuffle=True)
    num_patches_x = train_data.num_patches_x
    num_patches_y = train_data.num_patches_y

    device = torch.device("cuda:0")

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
        max_num_images=max_num_images,
        noise_scheduler=noise_scheduler,
        feature_extractor="dino",
        append_ndc=True,
    ).to(device)

    t = model.noise_scheduler.max_timesteps

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = eval(lr_scheduler)(optimizer, gamma=lr_scheduler_gamma)

    progress_bar = trange(max_n_iteration)
    n_iteration = max_n_iteration
    losses = []
    for i_iter in progress_bar:
        for all_images, all_rays, all_light_centers, all_cam_mats, all_origins in iter(dataloader):
            all_images = all_images.unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)
            all_rays = all_rays.unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)

            batch_size = all_images.shape[0]
            num_images = all_images.shape[1]
            # (H, W, 2)
            uv = compute_image_uv(num_patches_x=num_patches_x, num_patches_y=num_patches_y).to(device)
            # (B, N, 2, H, W)
            uv = (
                uv.permute(2, 0, 1)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, num_images, 2, num_patches_y, num_patches_x)
            )
            x_t = torch.randn(
                batch_size, num_images, 6, num_patches_y, num_patches_x, device=device
            )
            image_features = model.feature_extractor(all_images, autoresize=True)
            mask = all_rays[..., :3] != 0

            optimizer.zero_grad()
            eps_pred, noise_sample = model(
                features=image_features,
                rays_noisy=x_t,
                t=t,
                ndc_coordinates=uv,
                mask=mask,
            )

            loss = torch.nn.functional.mse_loss(eps_pred, all_rays)
            progress_bar.set_description(f"Loss: {loss.item()}")
            losses.append(loss.item())
            if loss < 0.001:
                n_iteration = i_iter
                break
            loss.backward()
            optimizer.step()
        # progress_bar.set_description(f"lr: {scheduler.get_last_lr()}")
        scheduler.step()

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    print(f"Model saved to {os.path.join(output_dir, 'model.pth')}")

    # plot losses in logaritmic scale
    import matplotlib.pyplot as plt

    with open(os.path.join(output_dir, "losses.txt"), "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")

    losses = np.array(losses)
    losses = losses[losses > 0]
    losses = np.log(losses)

    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Log Loss")
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

    # pred_rays_vis = visualizeRays(eps_pred.cpu().detach().numpy().reshape(-1, 6, num_patches_y, num_patches_x).transpose(0, 2, 3, 1))
    # gt_rays_vis = visualizeRays(all_rays.cpu().numpy().reshape(-1, 6, num_patches_y, num_patches_x).transpose(0, 2, 3, 1))
    # rays_vis = np.concatenate([pred_rays_vis, gt_rays_vis], axis=1)
    # os.makedirs(os.path.join(output_dir, "visualization"), exist_ok=True)
    # for i in range(rays_vis.shape[0]):
    #     imageio.imwrite(os.path.join(output_dir, "visualization", f"vis{i}.png"), rays_vis[i])

    config_file = os.path.join(output_dir, "config.json")
    config = {
        "learning_rate": learning_rate,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_gamma": lr_scheduler_gamma,
        "data_dir": data_dir,
        "n_loaded_images": len(train_data),
        "n_iteration": n_iteration,
        "max_n_iteration": max_n_iteration,
        "num_patches_x": num_patches_x,
        "num_patches_y": num_patches_y,
        "model": {
            "max_num_images": max_num_images,
        },
    }
    json.dump(config, open(config_file, "w"), indent=4, sort_keys=True)


def validate(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    model_dir = args.model_dir
    split = args.split
    with open(os.path.join(os.path.dirname(model_dir), "config.json"), "r") as f:
        config = json.load(f)
        num_patches_x = config["num_patches_x"]
        num_patches_y = config["num_patches_y"]
        max_num_images = config["model"]["max_num_images"]
    num_images = 1

    # load data
    val_data = RayDiffusionData(data_dir, split=split)
    dataloader = DataLoader(val_data, batch_size=num_images, shuffle=False)
    num_patches_x = val_data.num_patches_x
    num_patches_y = val_data.num_patches_y

    device = torch.device("cuda:0")

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
        max_num_images=max_num_images,
        noise_scheduler=noise_scheduler,
        feature_extractor="dino",
        append_ndc=True,
    ).to(device)

    model.load_state_dict(torch.load(model_dir))

    t = model.noise_scheduler.max_timesteps
    # (H, W, 2)
    uv = compute_image_uv(num_patches_x=num_patches_x, num_patches_y=num_patches_y).to(device)
    # (B, N, 2, H, W)
    uv = (
        uv.permute(2, 0, 1)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(1, num_images, 2, num_patches_y, num_patches_x)
    )

    torch.no_grad()
    gt_positions = []
    pred_positions = []
    position_losses = []
    progress_bar = trange(len(val_data))
    dataloader_iter = iter(dataloader)
    for i_iter in progress_bar:
        all_images, all_rays, all_light_centers, all_cam_mats, all_origins = next(dataloader_iter)
        all_images = all_images.unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)
        all_rays = all_rays.unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)

        batch_size = all_images.shape[0]
        # num_images = all_images.shape[1]
        x_t = torch.randn(batch_size, num_images, 6, num_patches_y, num_patches_x, device=device)
        image_features = model.feature_extractor(all_images, autoresize=True)

        eps_pred, noise_sample = model(
            features=image_features,
            rays_noisy=x_t,
            t=t,
            ndc_coordinates=uv,
        )

        # visualize rays
        pred_rays_vis = visualizeRays(
            eps_pred.cpu()
            .detach()
            .numpy()
            .reshape(-1, 6, num_patches_y, num_patches_x)
            .transpose(0, 2, 3, 1)
            .squeeze()
        )  # (B=1, 6, H * W) -> (H, W, 6)
        gt_rays_vis = visualizeRays(
            all_rays.cpu()
            .numpy()
            .reshape(-1, 6, num_patches_y, num_patches_x)
            .transpose(0, 2, 3, 1)
            .squeeze()
        )  # (B=1, 6, H * W) -> (H, W, 6)
        rays_vis = np.concatenate([pred_rays_vis, gt_rays_vis], axis=-3)
        visualization_dir = os.path.join(output_dir, f"visualization_{split}")
        os.makedirs(visualization_dir, exist_ok=True)
        imageio.imwrite(os.path.join(visualization_dir, f"vis{i_iter}.png"), rays_vis)

        gt_light_pos = all_light_centers.cpu().numpy()
        pred_rays = (
            eps_pred.cpu()
            .detach()
            .numpy()
            .reshape(-1, 6, num_patches_y * num_patches_x)
            .transpose(0, 2, 1)
            .squeeze()
        )  # (B=1, H * W, 6) -> (H * W, 6)
        pred_light_pos = pluckerRays2Point(pred_rays)
        cam_pos = all_cam_mats[0][0].cpu().numpy().astype(np.float32)
        origin = all_origins[0].cpu().numpy().astype(np.float32)
        camera_extrinsics = mi.ScalarTransform4f.look_at(
            origin=all_cam_mats[0][0].tolist(),
            target=all_cam_mats[0][1].tolist(),
            up=all_cam_mats[0][2].tolist(),
        )
        # # [DEBUG] compute gt light position from rays and convert to world space
        # gt_rays = (
        #     all_rays.cpu()
        #     .numpy()
        #     .reshape(-1, 6, num_patches_y * num_patches_x)
        #     .transpose(0, 2, 1)
        #     .squeeze()
        # )
        # computed_gt_light_pos = pluckerRays2Point(gt_rays)
        # computed_gt_light_pos = camera_extrinsics @ np.array(computed_gt_light_pos)
        # computed_gt_light_pos = np.array(
        #     [computed_gt_light_pos[0], computed_gt_light_pos[1], computed_gt_light_pos[2]],
        #     dtype=np.float32,
        # )
        # computed_gt_light_pos = computed_gt_light_pos - cam_pos + origin
        # convert pred light position to world space
        pred_light_pos = camera_extrinsics @ np.array(pred_light_pos)
        pred_light_pos = np.array([pred_light_pos[0], pred_light_pos[1], pred_light_pos[2]])
        pred_light_pos = pred_light_pos - cam_pos + origin

        # compute position loss
        position_loss = np.linalg.norm(pred_light_pos - gt_light_pos[0])
        gt_positions.append(gt_light_pos[0])
        pred_positions.append(pred_light_pos)

        position_losses.append(position_loss)

    with open(os.path.join(output_dir, f"position_losses_{split}.txt"), "w") as f:
        f.write(f"Average position loss: {np.mean(position_losses)}\n")
        for i in range(len(position_losses)):
            f.write(f"{i}\n")
            f.write(f"gt:\t{gt_positions[i]}\n")
            f.write(f"pred:\t{pred_positions[i]}\n")
            f.write(f"{position_losses[i]}\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    # args.model_dir = "output/20240519_142029/model.pth"
    # args.split = "train"
    if args.model_dir is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(args.output_dir, current_time)
        train(args)
        args.model_dir = os.path.join(args.output_dir, "model.pth")
        validate(args)
    else:
        folder_name = os.path.dirname(args.model_dir)
        folder_name = os.path.basename(folder_name)
        args.output_dir = os.path.join(args.output_dir, folder_name)
        validate(args)
