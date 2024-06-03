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
    light_idx = 5

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

    init_light_position = [0, 2, 0]
    light_color = [1, 1, 1]
    # Create a point light placeholder. The actual center and intensity will be set later in parameters
    scene_dict["emitter_opt"] = {
        "type": "point",
        "position": init_light_position,
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
    display(mi.util.convert_to_bitmap(gt_image))


    params = mi.traverse(scene)

    init_image = mi.render(scene, params, spp=128)
    display(mi.util.convert_to_bitmap(init_image))

    opt = mi.ad.Adam(lr=0.05)
    opt["light_translate"] = mi.Point3f(0, 0, 0)
    # params.update(opt)

    images = []
    loss_hist = []

    pbar = tqdm(range(100))
    for i in pbar:
        image = mi.render(scene, params, spp=4)
        images.append(image)
        loss = dr.mean(dr.abs(image - gt_image))

        dr.backward(loss)
        opt.step()
        # params.update(opt)
        params["emitter_opt.position"] += opt["light_translate"]

        pbar.set_description(f"Loss: {loss[0]:.5f}.")
        loss_hist.append(loss[0])

    # generate video using imageio
    images = [mi.util.convert_to_bitmap(img) for img in images]
    imageio.mimsave("test.mp4", images, duration=100)

    # display loss history
    plt.plot(loss_hist)
    plt.show()
    plt.savefig("loss.png")