import wandb
import configargparse
import glob
import json
import os
from datetime import datetime
from utils.matrix import lookat_matrix, invert_camera_matrix

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import imageio

# from inference.predict import predict_cameras
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


def getParser():
    def none_or_str(value):
        if value == "None":
            return None
        return str(value)

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", "-c", is_config_file=True, help="config file path")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--data_dir", type=none_or_str, default="data/RayDiffusionData/scenes_on_cluster"
    )
    # parser.add_argument("--data_dir", type=none_or_str, default="test_output")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.99)
    parser.add_argument("--lr_scheduler", type=none_or_str, default="ExponentialLR")
    parser.add_argument("--max_n_epochs", type=int, default=500)
    parser.add_argument("--max_num_images", type=int, default=8)
    parser.add_argument("--model_dir", type=none_or_str, default=None)
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--num_scenes", type=int, default=None)
    parser.add_argument("--num_lights", type=int, default=None)
    parser.add_argument("--output_dir", type=none_or_str, default="output")
    parser.add_argument("--split", type=none_or_str, default="val")
    parser.add_argument("--split_by", type=none_or_str, default="image")
    parser.add_argument("--eval_interval", type=int, default=50)
    return parser


def computeImageUV(num_patches_x, num_patches_y):
    """
    Compute the UV coordinates of the origins of the rays in the image space.  Bottom-left corner of the image is (-1, -1) and top-right corner is (1, 1)
    """
    cell_w = 2 / num_patches_x
    cell_h = 2 / num_patches_y
    x = torch.linspace(-1 + cell_w / 2, 1 - cell_w / 2, num_patches_x)
    y = torch.linspace(-1 + cell_h / 2, 1 - cell_h / 2, num_patches_y)
    x, y = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([x, y], dim=-1)


def transformToWorldSpace(p, cam_mat, origin):
    """
    Args
    ----
    p:
        3D point in camera space
    cam_mat: (4, 4)
        Camera matrix
    origin:
        Origin of the camera
    """
    p = np.array(p)
    cam_mat = np.array(cam_mat)
    origin = np.array(origin)

    cam_pos = cam_mat[0]
    camera_extrinsics = lookat_matrix(
        cam_mat[0], cam_mat[1], cam_mat[2]
    )  # from camera space to world space
    p = np.array(p)
    p = np.concatenate([p, np.ones((p.shape[0], 1))], axis=-1).transpose()
    p = camera_extrinsics @ p
    p = p[:3].transpose()
    p = p - cam_pos + origin
    # (N, 3)

    return p


def train(args):
    wandb.init(project="RayDiffusion", config=vars(args))
    wandb.config = vars(args)

    # load data
    train_data = RayDiffusionData(
        args.data_dir, split="train", split_by=args.split_by, num_scenes=args.num_scenes, num_lights=args.num_lights
    )
    dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    num_patches_x = train_data.num_patches_x
    num_patches_y = train_data.num_patches_y

    # detect if cuda is available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        raise ValueError("CUDA is not available")

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
        max_num_images=args.max_num_images,
        noise_scheduler=noise_scheduler,
        feature_extractor="dino",
        append_ndc=True,
    ).to(device)

    t = model.noise_scheduler.max_timesteps

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = eval(args.lr_scheduler)(optimizer, gamma=args.lr_scheduler_gamma)

    progress_bar = trange(args.max_n_epochs)
    epoch = args.max_n_epochs
    losses = []
    for i in progress_bar:
        # for ( all_images, all_rays, all_light_centers, all_cam_mats, all_origins, all_scale,) in dataloader:
        losses = []
        for all_images, all_rays, _, _, _, _ in dataloader:
            all_images = (
                all_images.unsqueeze(1).permute(0, 1, 4, 2, 3).to(device)
            )  # (B, H, W, 3) -> (B, N, 3, H, W)
            all_rays = (
                all_rays.unsqueeze(1).permute(0, 1, 4, 2, 3).to(device)
            )  # (B, H, W, 6) -> (B, N, 6, H, W)
            B = all_images.shape[0]
            N = all_images.shape[1]
            uv = (
                computeImageUV(num_patches_x=num_patches_x, num_patches_y=num_patches_y)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(B, N, 2, num_patches_y, num_patches_x)
                .to(device)
            )  # (H, W, 2) -> (B, N, 2, H, W)
            x_t = torch.randn(
                B, N, 6, num_patches_y, num_patches_x, device=device
            )  # (B, N, 6, H, W)
            image_features = model.feature_extractor(
                all_images, autoresize=True
            )  # (B, N, C_feat, H, W)
            mask = all_rays[..., :3] != 0

            optimizer.zero_grad()
            eps_pred, noise_sample = model(
                features=image_features,
                rays_noisy=x_t,
                t=t,
                ndc_coordinates=uv,
                mask=mask,
            )
            # eps_pred: (B, N, 6, H, W)

            loss = torch.nn.functional.mse_loss(eps_pred, all_rays)
            progress_bar.set_description(f"Loss: {loss.item()}")
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # Evaluate model every eval_interval iterations

        if i != 0 and i % args.eval_interval == 0:
            validate(args, model=model)

        wandb.log({"train_loss": np.mean(losses)})
        scheduler.step()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pth"))
    print(f"Model saved to {os.path.join(args.output_dir, 'model.pth')}")

    with open(os.path.join(args.output_dir, "losses.txt"), "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")

    losses = np.array(losses)
    losses = losses[losses > 0]
    losses = np.log(losses)

    # plt.plot(losses)
    # plt.xlabel("Iteration")
    # plt.ylabel("Log Loss")
    # plt.savefig(os.path.join(output_dir, "loss.png"))
    # plt.close()

    # # Visualize rays
    # pred_rays_vis = visualizeRays(eps_pred.cpu().detach().numpy().reshape(-1, 6, num_patches_y, num_patches_x).transpose(0, 2, 3, 1))
    # gt_rays_vis = visualizeRays(all_rays.cpu().numpy().reshape(-1, 6, num_patches_y, num_patches_x).transpose(0, 2, 3, 1))
    # rays_vis = np.concatenate([pred_rays_vis, gt_rays_vis], axis=1)
    # os.makedirs(os.path.join(output_dir, "visualization"), exist_ok=True)
    # for i in range(rays_vis.shape[0]):
    #     imageio.imwrite(os.path.join(output_dir, "visualization", f"vis{i}.png"), rays_vis[i])

    # config_file = os.path.join(output_dir, "config.json")
    # config = {
    #     "learning_rate": learning_rate,
    #     "lr_scheduler": lr_scheduler,
    #     "lr_scheduler_gamma": lr_scheduler_gamma,
    #     "data_dir": data_dir,
    #     "n_loaded_images": len(train_data),
    #     "n_iteration": epoch,
    #     "max_n_epochs": max_n_epochs,
    #     "num_patches_x": num_patches_x,
    #     "num_patches_y": num_patches_y,
    #     "model": {
    #         "max_num_images": max_num_images,
    #     },
    # }
    # json.dump(config, open(config_file, "w"), indent=4, sort_keys=True)


def validate(args, model=None):
    # Load data
    num_images = 1
    val_data = RayDiffusionData(
        args.data_dir, split=args.split, split_by=args.split_by, num_scenes=args.num_scenes, num_lights=args.num_lights
    )
    dataloader = DataLoader(val_data, batch_size=num_images, shuffle=False)
    num_patches_x = val_data.num_patches_x
    num_patches_y = val_data.num_patches_y

    device = torch.device("cuda:0")

    if model == None:
        with open(os.path.join(os.path.dirname(args.model_dir), "config.json"), "r") as f:
            config = json.load(f)
            num_patches_x = config["num_patches_x"]
            num_patches_y = config["num_patches_y"]
            max_num_images = config["model"]["max_num_images"]

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

        model.load_state_dict(torch.load(args.model_dir))
    else:
        model = model.to(device)

    t = model.noise_scheduler.max_timesteps
    # (H, W, 2)
    uv = computeImageUV(num_patches_x=num_patches_x, num_patches_y=num_patches_y).to(device)
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
    position_loss_percentages = []
    progress_bar = tqdm(enumerate(dataloader))

    for i, (
        all_images,
        all_rays,
        all_light_centers,
        all_cam_mats,
        all_origins,
        all_scale,
    ) in progress_bar:
        all_images = all_images.unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)
        all_rays = all_rays.unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)

        batch_size = all_images.shape[0]
        # num_images = all_images.shape[1]
        x_t = torch.randn(
            batch_size, num_images, 6, num_patches_y, num_patches_x, device=device
        )  # (B, N, 6, H, W)
        image_features = model.feature_extractor(
            all_images, autoresize=True
        )  # (B, N, C_feat, H, W)

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
        visualization_dir = os.path.join(args.output_dir, f"visualization_{args.split}")
        os.makedirs(visualization_dir, exist_ok=True)
        imageio.imwrite(os.path.join(visualization_dir, f"vis{i}.png"), rays_vis)

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
        pred_light_pos = transformToWorldSpace(pred_light_pos, all_cam_mats[0], all_origins[0])

        # compute position loss
        position_loss = np.linalg.norm(pred_light_pos - gt_light_pos[0])
        gt_positions.append(gt_light_pos[0])
        pred_positions.append(pred_light_pos)

        position_losses.append(position_loss)
        position_loss_percentages.append(position_loss / all_scale[0])

    wandb.log({"val_mean_position_loss": np.mean(position_losses)})
    wandb.log({"val_mean_position_loss_percentage": np.mean(position_loss_percentages)})

    with open(os.path.join(args.output_dir, f"position_losses_{args.split}.txt"), "w") as f:
        f.write(f"Average position loss: {np.mean(position_losses)}\n")
        f.write(f"Average position loss percentage: {np.mean(position_loss_percentages)}\n")
        f.write(f"Medium position loss percentage: {np.median(position_loss_percentages)}\n")
        for i in range(len(position_losses)):
            f.write(f"{i}\n")
            f.write(f"gt:\t{gt_positions[i]}\n")
            f.write(f"pred:\t{pred_positions[i]}\n")
            f.write(f"{position_losses[i]}\n")
            f.write(f"{position_loss_percentages[i]}\n")


if __name__ == "__main__":
    parser = getParser()
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
