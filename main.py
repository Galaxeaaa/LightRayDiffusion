import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")
from functools import partial
import os
import argparse
import yaml
import cv2
from pprint import pprint
from tqdm import tqdm, trange

import torch
import matplotlib.pyplot as plt

from dataloader.openrooms import get_dataloader, OpenRoomsDemoSceneData

# from util.img_utils import clear_color, mask_generator,to_output,imwrite
# from util.logger import get_logger
import numpy as np

import imageio
from dataloader.convert_xml import convertXML2Dict


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def loadSensor(origin, lookat, up):
    sensor = mi.sensors.PerspectiveSensor(origin=origin, target=lookat, up=up, fov=45)
    return sensor


def display(image):
    plt.axis("off")
    plt.imshow(image)
    plt.show()


def lossFn(img, gt):
    return dr.mean(dr.sqr(img - gt))


def totalVariation(img):
    return dr.sum(dr.abs(img[..., :-1] - img[..., 1:])) + dr.sum(dr.abs(img[:-1] - img[1:]))


if __name__ == "__main__":
    data_dir = "data/Demo/main_xml/scene0001_01"
    xml_filename = "test.xml"
    gt_filename = "test_gt_wall_noenv.exr"
    idx_view = 0
    voxel_res = 5

    dataset = OpenRoomsDemoSceneData(data_dir="data/Demo/main_xml/scene0001_01")
    cam_lookat_mat = dataset[idx_view]["camera_lookat_mat"]
    scene_dict = convertXML2Dict(xml_filename)

    scene_dict["sensor"]["to_world"] = mi.ScalarTransform4f.look_at(
        origin=cam_lookat_mat[0], target=cam_lookat_mat[1], up=cam_lookat_mat[2]
    )
    scene_dict.pop("env_emitter")
    scene_dict["integrator"]["type"] = "prb"

    # for obj in scene_dict.values():
    #     if "emitter" in obj:
    #         obj.pop("emitter")
    center = [-1, 0, 0]
    radius = 0.5
    scene_dict["emitter_gt"] = {
        "type": "sphere",
        "center": center,
        "radius": radius,
        # "bsdf": {
        #     "type": "diffuse",
        #     "reflectance": {"type": "rgb", "value": [0.3, 0.3, 0.75]},
        # },
        # "to_world": mi.ScalarTransform4f.translate(init_center).scale(init_radius),
        "emitter": {
            "type": "area",
            "radiance": {
                "type": "rgb",
                "value": [3, 3, 7.5],
            },
        },
    }
    scene = mi.load_dict(scene_dict)
    # gt_image = mi.Bitmap(gt_filename)
    print("Rendering ground truth image.")
    gt_image = mi.render(scene, spp=512, seed=0)
    display(mi.util.convert_to_bitmap(gt_image))
    mask = (
        np.array(gt_image[..., 0] < 1.0)
        * np.array(gt_image[..., 1] < 1.0)
        * np.array(gt_image[..., 2] < 1.0)
    )
    mask = mask[..., None]
    mask = mi.TensorXf(mask)
    display(mask)
    display(mi.util.convert_to_bitmap(mask * gt_image))

    scene_dict.pop("emitter_gt")
    x, y, z = dr.meshgrid(
        dr.linspace(mi.Float, center[0] - radius / 2, center[0] + radius / 2, voxel_res),
        dr.linspace(mi.Float, center[1] - radius / 2, center[1] + radius / 2, voxel_res),
        dr.linspace(mi.Float, center[2] - radius / 2, center[2] + radius / 2, voxel_res),
    )
    point_light_positions = mi.Point3f(x, y, z)
    for i in range(voxel_res):
        for j in range(voxel_res):
            for k in range(voxel_res):
                index = i * voxel_res * voxel_res + j * voxel_res + k
                scene_dict[f"point_light_opt_{i}_{j}_{k}"] = {
                    "type": "point",
                    "position": [
                        point_light_positions[0][index],
                        point_light_positions[1][index],
                        point_light_positions[2][index],
                    ],
                    "intensity": {"type": "rgb", "value": [np.random.rand() for _ in range(3)]},
                }

    scene = mi.load_dict(scene_dict)
    params = mi.traverse(scene)

    params.keep(r"point_light_opt_\d+_\d+_\d+\.intensity\.value")

    # params.keep(r"emitter_opt\.(to_world|emitter\.radiance\.value)")
    # params.keep(r"emitter_opt\.(emitter\.radiance\.value)")
    # print(f"Optimizing parameters {list(params.keys())}.")
    # params.update()

    # init_image = mi.render(scene, params, spp=128)
    # display(init_image)
    # exit(0)

    # init_to_world = mi.Transform4f(params["emitter_opt.to_world"])
    # init_vertex_positions = dr.unravel(mi.Point3f, params["emitter_opt.vertex_positions"])

    opt = mi.ad.Adam(lr=0.05, params=params)
    params.update(opt)
    # opt["trans"] = mi.Point3f(0, 0.1, 0)
    # dr.enable_grad(opt["trans"])
    # opt["trans"] = mi.Float(0)
    # opt["scale"] = mi.Float(2.0)

    def applyTransform(params, opt):
        # opt["trans"] = dr.clamp(opt["trans"], -0.5, 0.5)
        trans = mi.Transform4f.translate(opt["trans"])
        # params["emitter_opt.to_world"] = trans @ init_to_world
        # params["emitter_opt.vertex_positions"] = dr.ravel(trans @ init_vertex_positions)
        params.update()

    # applyTransform(params, opt)

    images = []
    loss_hist = []

    pbar = tqdm(range(100))
    for i in pbar:
        # applyTransform(params, opt)
        image = mi.render(scene, params, spp=16, seed=i)
        images.append(image)
        loss = lossFn(image * mask, gt_image * mask)

        def totalVariation(params):
            tv_loss = 0
            for i in range(voxel_res):
                for j in range(voxel_res):
                    for k in range(voxel_res):
                        if i > 0:
                            tv_loss += dr.sum(
                                dr.abs(
                                    params[f"point_light_opt_{i}_{j}_{k}.intensity.value"]
                                    - params[f"point_light_opt_{i-1}_{j}_{k}.intensity.value"]
                                )
                            )
                        if j > 0:
                            tv_loss += dr.sum(
                                dr.abs(
                                    params[f"point_light_opt_{i}_{j}_{k}.intensity.value"]
                                    - params[f"point_light_opt_{i}_{j-1}_{k}.intensity.value"]
                                )
                            )
                        if k > 0:
                            tv_loss += dr.sum(
                                dr.abs(
                                    params[f"point_light_opt_{i}_{j}_{k}.intensity.value"]
                                    - params[f"point_light_opt_{i}_{j}_{k-1}.intensity.value"]
                                )
                            )
            return tv_loss

        loss += totalVariation(params) * 0.01

        dr.backward(loss)
        opt.step()
        # clamp
        for i in range(voxel_res):
            for j in range(voxel_res):
                for k in range(voxel_res):
                    params[f"point_light_opt_{i}_{j}_{k}.intensity.value"] = dr.clamp(
                        params[f"point_light_opt_{i}_{j}_{k}.intensity.value"], 0, 10
                    )
        params.update(opt)

        pbar.set_description(f"Loss: {loss[0]:.5f}.")
        loss_hist.append(loss[0])

    # generate video using imageio
    # image_end = mi.render(scene, params, spp=128)
    # mi.util.write_bitmap("test_end.exr", image_end, "rgb")
    images = [mi.util.convert_to_bitmap(img) for img in images]
    imageio.mimsave("test.mp4", images)

    # display loss history
    plt.plot(loss_hist)
    plt.show()

    # display(original_image)
    final_image = mi.render(scene, params, spp=512)
    mi.util.write_bitmap("test_result.exr", final_image, "rgb")

    # visualize point light colors by replacing the point light with a sphere
    for i in range(voxel_res):
        for j in range(voxel_res):
            for k in range(voxel_res):
                intensity = params[f"point_light_opt_{i}_{j}_{k}.intensity.value"]
                intensity = list(intensity)
                intensity = [x[0] for x in intensity]
                scene_dict.pop(f"point_light_opt_{i}_{j}_{k}")
                index = i * voxel_res * voxel_res + j * voxel_res + k
                scene_dict[f"point_light_opt_{i}_{j}_{k}"] = {
                    "type": "sphere",
                    "center": [
                        point_light_positions[0][index],
                        point_light_positions[1][index],
                        point_light_positions[2][index],
                    ],
                    "radius": 0.01,
                    "emitter": {
                        "type": "area",
                        "radiance": {"type": "rgb", "value": intensity},
                    },
                }
    scene = mi.load_dict(scene_dict)
    vis_image = mi.render(scene, spp=512)
    display(mi.util.convert_to_bitmap(vis_image))
