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
    scene_name = "scene0013_00"
    res_h = 420
    res_w = 560

    data_dir = os.path.join(cwd, "data/scenes_on_cluster/xml", scene_name)
    xml_filename = os.path.join(data_dir, "main.xml")
    gt_filename = os.path.join(cwd, "gt_0013.exr")
    dataset = OpenRoomsSceneData(data_dir=data_dir)
    # Set up sensor and integrator
    scene_dict = convertXML2Dict(xml_filename)
    scene_dict["sensor"]["film"]["height"] = res_h
    scene_dict["sensor"]["film"]["width"] = res_w
    scene_dict["integrator"]["type"] = "prb_projective"
    # Remove all emitters
    scene_dict.pop("env_emitter")
    for obj in scene_dict.values():
        if "emitter" in obj:
            obj.pop("emitter")

    gt_light_position = [0, 0, 0]
    light_color = [1, 1, 1]
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

    # scene_dict["x_axis"] = {
    #     "type": "cylinder",
    #     "p0": [0, 0, 0],
    #     "p1": [1, 0, 0],
    #     "radius": 0.01,
    #     "bsdf": {
    #         "type": "diffuse",
    #         "reflectance": {
    #             "type": "rgb",
    #             "value": [1, 0, 0],
    #         },
    #     }
    # }

    # scene_dict["y_axis"] = {
    #     "type": "cylinder",
    #     "p0": [0, 0, 0],
    #     "p1": [0, 1, 0],
    #     "radius": 0.01,
    #     "bsdf": {
    #         "type": "diffuse",
    #         "reflectance": {
    #             "type": "rgb",
    #             "value": [0, 1, 0],
    #         },
    #     }
    # }
    
    # scene_dict["z_axis"] = {
    #     "type": "cylinder",
    #     "p0": [0, 0, 0],
    #     "p1": [0, 0, 1],
    #     "radius": 0.01,
    #     "bsdf": {
    #         "type": "diffuse",
    #         "reflectance": {
    #             "type": "rgb",
    #             "value": [0, 0, 1],
    #         },
    #     }
    # }

    # origin = [-1.2, -0.3, 1.4]
    origin = [0, 0, 0]
    dir1 = np.array([-1, 0, -1], dtype=np.float32)
    dir1 = dir1 / np.linalg.norm(dir1)
    dir2 = np.array([0, 1, 0], dtype=np.float32)
    dir2 = dir2 / np.linalg.norm(dir2)

    # scene_dict["axis1"] = {
    #     "type": "cylinder",
    #     "p0": origin,
    #     "p1": origin + dir1,
    #     "radius": 0.01,
    #     "emitter": {
    #         "type": "area",
    #         "radiance": {
    #             "type": "rgb",
    #             "value": [1, 0, 0],
    #         },
    #     }
    # }

    # scene_dict["axis2"] = {
    #     "type": "cylinder",
    #     "p0": origin,
    #     "p1": origin + dir2,
    #     "radius": 0.01,
    #     "emitter": {
    #         "type": "area",
    #         "radiance": {
    #             "type": "rgb",
    #             "value": [0, 1, 0],
    #         },
    #     }
    # }


    scene = mi.load_dict(scene_dict)

    gt_image = mi.Bitmap(gt_filename)
    # gt_image = mi.render(scene, spp=8)
    # display(mi.util.convert_to_bitmap(gt_image))
    # mi.util.write_bitmap("gt_0013.exr", gt_image, "rgb")
    # exit(0)

    params = mi.traverse(scene)

    # vis_image = mi.render(scene, params, spp=8)
    # display(mi.util.convert_to_bitmap(vis_image))
    # exit(0)

    vis_res = 6
    losses = []
    progress_bar = tqdm(total=vis_res ** 2)
    u, v = np.meshgrid(np.linspace(0, 1, vis_res), np.linspace(0, 1, vis_res))
    u = u[..., None]
    v = v[..., None]
    points = origin + u * dir1 + v * dir2

    for points in points.reshape(-1, 3):
        params["emitter_opt.position"] = points
        image = mi.render(scene, params, spp=8)
        # mi.util.write_bitmap(f"tmp_{x}.exr", image, "rgb")
        loss = np.mean(np.abs(image - gt_image))
        losses.append(loss)
        progress_bar.update(1)
    
    print(losses)
    
    losses = np.array(losses, dtype=np.float32).reshape(vis_res, vis_res)
    # Visualize loss in 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 1, vis_res)
    v = np.linspace(0, 1, vis_res)
    u, v = np.meshgrid(u, v)
    ax.plot_surface(u, v, losses, cmap='viridis')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('Loss')
    plt.show()