import torch
from vit_pytorch.vit_for_small_dataset import ViT
from vit_pytorch import SimpleViT
import os
from generate_rays import readRaysFromFile
import argparse
from model.diffuser import RayDiffuser
from model.scheduler import NoiseScheduler

# from inference.predict import predict_cameras
import mitsuba as mi
import numpy as np
from tqdm import tqdm, trange


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="examples/robot/images")
    parser.add_argument("--model_dir", type=str, default="models/co3d_diffusion")
    parser.add_argument("--mask_dir", type=str, default="")
    parser.add_argument("--bbox_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="output_cameras.html")
    return parser


def compute_image_uv(num_patches_x, num_patches_y):
    """
    Compute the UV coordinates of the origins of the rays in the image space.  Bottom-left corner of the image is (-1, -1) and top-right corner is (1, 1)
    """
    cell_w = 2 / num_patches_x
    cell_h = 2 / num_patches_y
    x = torch.linspace(-1 + cell_w / 2, 1 - cell_w / 2, num_patches_x)
    y = torch.linspace(-1 + cell_h / 2, 1 - cell_h / 2, num_patches_y)
    x, y = torch.meshgrid(x, y)
    return torch.stack([x, y], dim=-1)


def main(image_dir, model_dir, mask_dir, bbox_path, output_path):
    n_iteration = 100

    num_patches_x = 10
    num_patches_y = 10
    device = torch.device("cuda:0")

    image = mi.Bitmap("test_gt^^.exr")
    images = np.array(image)
    images = torch.tensor(images)
    images = images.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (H, W, 3) -> (1, 1, 3, H, W)
    images = images.to(device)
    print(f"images.shape: {images.shape}")

    # read ray information from file
    ray_file = "rays_test.txt"
    rays = readRaysFromFile(ray_file)
    rays = np.array(rays, dtype=np.float32)
    # (num_patches_x * num_patches_y, 6) -> (1, 1, 6, num_patches_x, num_patches_y)
    rays = (
        torch.tensor(rays)
        .unsqueeze(0)
        .unsqueeze(0)
        .reshape(1, 1, 6, num_patches_x, num_patches_y)
        .to(device)
    )
    print(f"rays.shape: {rays.shape}")

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
        max_num_images=1,
        noise_scheduler=noise_scheduler,
        feature_extractor="dino",
        append_ndc=True,
    ).to(device)

    batch_size = images.shape[0]
    num_images = images.shape[1]
    t = model.noise_scheduler.max_timesteps
    x_t = torch.randn(batch_size, num_images, 6, num_patches_x, num_patches_y, device=device)
    image_features = model.feature_extractor(images, autoresize=True)
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

    progress_bar = trange(n_iteration)
    for _ in progress_bar:
        optimizer.zero_grad()
        eps_pred, noise_sample = model(
            features=image_features,
            rays_noisy=x_t,
            t=t,
            ndc_coordinates=uv,
        )

        loss = torch.nn.functional.mse_loss(eps_pred, rays)
        loss.backward()
        optimizer.step()
        # print loss on progress bar
        progress_bar.set_description(f"Loss: {loss.item()}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
