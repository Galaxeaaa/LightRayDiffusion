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
    num_patches_y = 10
    num_patches_x = 10
    num_lights_per_scene = 3
    light_idx = 5
    cell_h = res_h // num_patches_y
    cell_w = res_w // num_patches_x

    data_dir = os.path.join(cwd, "data/scenes_on_cluster/xml", scene_name)
    xml_filename = os.path.join(data_dir, "main.xml")
    gt_filename = os.path.join(
        "data/RayDiffusionData/scenes_on_cluster",
        scene_name,
        f"light{light_idx}/image{idx_view}.exr",
    )
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
    # Read GT position from json
    with open(
        os.path.join(
            "data/RayDiffusionData/scenes_on_cluster", scene_name, f"light{light_idx}/params.json"
        ),
        "r",
    ) as f:
        params = json.load(f)
        position = params["light_center"]
    # Create a point light placeholder. The actual center and intensity will be set later in parameters
    scene_dict["emitter_opt"] = {
        "type": "point",
        "position": position,
        "intensity": {
            "type": "rgb",
            "value": np.random.uniform(1, 5, 3).tolist(),
        },
    }
    print(position)

    cam_lookat_mat = dataset[idx_view]["camera_lookat_mat"]
    scene_dict["sensor"]["to_world"] = mi.ScalarTransform4f.look_at(
        origin=cam_lookat_mat[0], target=cam_lookat_mat[1], up=cam_lookat_mat[2]
    )

    scene = mi.load_dict(scene_dict)

    gt_image = mi.Bitmap(gt_filename)
    display(mi.util.convert_to_bitmap(gt_image))
    # mask = (
    #     np.array(gt_image[..., 0] < 1.0)
    #     * np.array(gt_image[..., 1] < 1.0)
    #     * np.array(gt_image[..., 2] < 1.0)
    # )
    # mask = mask[..., None]
    # mask = mi.TensorXf(mask)

    params = mi.traverse(scene)
    params.keep(r"emitter_opt\.intensity\.value")
    params.update()

    init_image = mi.render(scene, params, spp=128)
    display(mi.util.convert_to_bitmap(init_image))

    opt = mi.ad.Adam(lr=0.05, params=params)
    params.update(opt)

    images = []
    loss_hist = []

    pbar = tqdm(range(100))
    for i in pbar:
        image = mi.render(scene, params, spp=4, seed=i)
        images.append(image)
        loss = dr.mean(dr.sqr(image - gt_image))

        dr.backward(loss)
        opt.step()
        # opt[params["emitter_opt.intensity.value"]] = dr.clamp(params["emitter_opt.intensity.value"], 0, 10)
        params.update(opt)

        pbar.set_description(f"Loss: {loss[0]:.5f}.")
        loss_hist.append(loss[0])
        # print(params["emitter_opt.intensity.value"])

    # generate video using imageio
    # image_end = mi.render(scene, params, spp=128)
    # mi.util.write_bitmap("test_end.exr", image_end, "rgb")
    images = [mi.util.convert_to_bitmap(img) for img in images]
    imageio.mimsave("test.gif", images, duration=100)

    # display loss history
    plt.plot(loss_hist)
    plt.show()

    final_image = mi.render(scene, params, spp=512)
    display(mi.util.convert_to_bitmap(final_image))
