import json

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm, trange

from dataloader.convert_xml import convertXML2Dict
from dataloader.openrooms import OpenRoomsDemoSceneData, OpenRoomsSceneData
from utils.visualization import display

if __name__ == "__main__":
    idx_view = 0
    cwd = "."
    scene_name = "scene0001_01"
    res_h = 420
    res_w = 560
    light_idx = 5

    data_dir = os.path.join(cwd, "data/scenes_on_cluster/xml", scene_name)
    xml_filename = os.path.join(data_dir, "main.xml")
    gt_filename = os.path.join(cwd, "gt.exr")
    dataset = OpenRoomsSceneData(data_dir=data_dir)
    # Set up sensor and integrator
    scene_dict = convertXML2Dict(xml_filename)
    scene_dict["sensor"]["film"]["height"] = res_h
    scene_dict["sensor"]["film"]["width"] = res_w
    scene_dict["integrator"]["type"] = "prb"
    # Remove all emitters
    scene_dict.pop("env_emitter")
    for obj in scene_dict.values():
        if "emitter" in obj:
            obj.pop("emitter")

    gt_light_position = [0, 0, 0]
    light_color = [1, 3, 3]
    # Create a point light placeholder. The actual center and intensity will be set later in parameters
    scene_dict["emitter_opt"] = {
        "type": "point",
        "position": gt_light_position,
        "intensity": {
            "type": "rgb",
            "value": light_color,
        },
    }

    cam_lookat_mat = dataset[idx_view]["camera_lookat_mat"]
    scene_dict["sensor"]["to_world"] = mi.ScalarTransform4f.look_at(
        origin=cam_lookat_mat[0], target=cam_lookat_mat[1], up=cam_lookat_mat[2]
    )

    scene = mi.load_dict(scene_dict)

    gt_image = mi.Bitmap(gt_filename)
    # gt_image = mi.render(scene, spp=528)
    # mi.util.write_bitmap("gt.exr", gt_image, "rgb")
    # exit(0)

    params = mi.traverse(scene)

    vis_res = 6
    losses = []
    progress_bar = tqdm(total=vis_res ** 2)
    for y in np.linspace(1, -1, vis_res):
        for z in np.linspace(1, -1, vis_res):
            params["emitter_opt.position"] = [2, y, z]
            image = mi.render(scene, params, spp=8)
            # mi.util.write_bitmap(f"tmp_{y}_{z}.exr", image, "rgb")
            loss = np.mean(np.abs(image - gt_image))
            losses.append(loss)
            progress_bar.update(1)
    
    print(losses)
    
    losses = np.array(losses, dtype=np.float32).reshape(vis_res, vis_res)
    plt.imshow(losses, cmap="viridis")
    plt.show()