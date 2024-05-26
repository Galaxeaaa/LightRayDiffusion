import json
import os
from glob import glob
import mitsuba as mi

import imageio
import numpy as np
from torch.utils.data import Dataset
from generate_rays import readRaysFromFile


class RayDiffusionData(Dataset):
    def __init__(self, data_dir: str, split: str = "all"):
        self.data_dir = data_dir
        self.image_file_list = []
        num_patches_x, num_patches_y = 0, 0
        scene_dirs = glob(os.path.join(data_dir, "scene*"))
        scene_dirs = sorted(scene_dirs)
        # scene_dirs = scene_dirs[:1]
        for scene_dir in scene_dirs:
            light_dirs = glob(os.path.join(scene_dir, "light*"))
            light_dirs = sorted(light_dirs)
            # light_dirs = (
            #     light_dirs[: int(len(light_dirs) * 0.8)]
            #     if split == "train"
            #     else light_dirs[int(len(light_dirs) * 0.8) :]
            # )
            for light_dir in light_dirs:
                params_file = os.path.join(light_dir, "params.json")
                with open(params_file, "r") as f:
                    params = json.load(f)
                    num_images = params["num_images"]
                    num_patches_x = params["num_patches_x"]
                    num_patches_y = params["num_patches_y"]

                if split == "train":
                    idx_range = range(int(num_images * 0.8))
                elif split == "val":
                    idx_range = range(int(num_images * 0.8), num_images)
                elif split == "all":
                    idx_range = range(num_images)
                else:
                    raise ValueError(f"Split \"{split}\" not recognized.")

                for image_file in [
                    os.path.join(light_dir, f"image{i}.exr") for i in idx_range
                ]:
                    self.image_file_list.append(image_file)
        if num_patches_x == 0 or num_patches_y == 0:
            raise ValueError("num_patches_x and num_patches_y must be greater than 0")
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y

        print(f"Loaded RayDiffusion {split} data from {data_dir}. Totally {len(self)} images from {len(scene_dirs)} scenes.")

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        image_file = self.image_file_list[idx]
        image_idx = int(os.path.splitext(os.path.basename(image_file))[0].split("image")[1])
        parent_dir = os.path.dirname(image_file)
        params_file = os.path.join(parent_dir, "params.json")
        rays_file = os.path.join(parent_dir, f"rays{image_idx}.txt")
        with open(params_file, "r") as f:
            params = json.load(f)
        num_patches_x = params["num_patches_x"]
        num_patches_y = params["num_patches_y"]
        num_images = params["num_images"]
        light_center = np.array(params["light_center"], dtype=np.float32)
        try:
            scale = params["scale"]
        except KeyError:
            raise ValueError(f"scale not found in params.json for scene {parent_dir}")

        image = imageio.imread(image_file)
        image = np.array(image, dtype=np.float32)
        rays, origin = readRaysFromFile(rays_file)
        rays = np.array(rays, dtype=np.float32)

        scene_name = os.path.basename(os.path.dirname(parent_dir))

        with open(os.path.join("data/scenes_on_cluster/xml", scene_name, "cam.txt"), "r") as f:
            n_cams = int(f.readline())
            lines = f.readlines()
            camera_lookat_mat = [
                list(map(float, line.split(" ")))
                for line in lines[image_idx * 3 : image_idx * 3 + 3]
            ]
            camera_lookat_mat = np.array(camera_lookat_mat)

        return image, rays, light_center, camera_lookat_mat, origin, scale
