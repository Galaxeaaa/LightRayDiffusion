import gc
import json
import os
import sys

import drjit as dr
import imageio
import mitsuba as mi
import numpy as np
from tqdm import trange

from utils.visualization import display, visualizeRays

mi.set_variant("cuda_ad_rgb")

from dataloader.convert_xml import convertXML2Dict
from dataloader.openrooms import (OpenRoomsDemoSceneData, OpenRoomsSceneData,
                                  get_dataloader)
from utils.PluckerRay import PluckerRay
from utils.rays_conversion import pluckerRays2Point


def clean_up():
    """
    Ensure no leftover instances from other tests are still in registry, wait
    for all running kernels to finish and reset the JitFlag to default. Also
    periodically frees the malloc cache to prevent the testcases from hogging
    all system memory.
    """
    dr.kernel_history_clear()
    dr.flush_malloc_cache()
    dr.malloc_clear_statistics()
    dr.flush_kernel_cache()

    if hasattr(dr, "sync_thread"):
        dr.sync_thread()
        dr.registry_clear()
        dr.set_flags(dr.JitFlag.Default)


def intersectCameraRayWithScene(u, v, fov, aspect_ratio, camera_extrinsics, scene):
    """
    u, v: pixel coordinates starting from top-left corner. u is the vertical coordinate and v is the horizontal coordinate.
    """
    # compute ray from camera to pixel in camera space
    cam_ray_dir = np.array(
        [
            np.tan(np.deg2rad(fov) / 2) - v * 2 * np.tan(np.deg2rad(fov) / 2),
            aspect_ratio * np.tan(np.deg2rad(fov) / 2)
            - u * 2 * aspect_ratio * np.tan(np.deg2rad(fov) / 2),
            1,
        ]
    )
    # convert camera_ray_dir to world space
    # !!! Somehow camera_extrinsics actually convert from camera space to world space
    cam_pos = camera_extrinsics.translation()
    cam_ray_dir = (
        camera_extrinsics @ cam_ray_dir - cam_pos
    )  # mi.Transform4f can only transform positions, so we have to subtract the camera position
    cam_ray_dir = cam_ray_dir / np.linalg.norm(cam_ray_dir)
    ray = mi.Ray3f(o=cam_pos, d=cam_ray_dir)
    # find nearest intersection with the scene
    si = scene.ray_intersect(ray)
    if dr.none(si.is_valid()):
        return np.array([np.nan, np.nan, np.nan])
    pixel_pos = np.array(si.p).flatten()

    # convert positions to camera space
    # light_pos = camera_extrinsics.inverse() @ light_pos
    # pixel_pos = camera_extrinsics.inverse() @ pixel_pos

    # d = light_pos - pixel_pos
    # d = d / np.linalg.norm(d)
    # d = np.array(d).flatten()
    # m = np.cross(pixel_pos, d).flatten()

    # return PluckerRay(direction=d, moment=m), pixel_pos
    return pixel_pos


def writeRaysToFile(rays, origin, filename, num_patches_x, num_patches_y):
    with open(filename, "w") as f:
        f.write(f"{num_patches_x} {num_patches_y}\n")
        f.write(f"{origin[0]} {origin[1]} {origin[2]}\n")
        for ray in rays:
            f.write(
                f"{ray.direction[0]} {ray.direction[1]} {ray.direction[2]} {ray.moment[0]} {ray.moment[1]} {ray.moment[2]}\n"
            )


def readRaysFromFile(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        num_patches_x, num_patches_y = map(int, lines[0].split())
        origin = list(map(float, lines[1].split()))
        origin = np.array(origin)
        rays = []
        for line in lines[2:]:
            ray = list(map(float, line.split()))
            rays.append(ray)
        rays = np.array(rays)
        assert (
            rays.shape[0] == num_patches_x * num_patches_y
        ), f"Number of rays read from file is not equal to num_patches_x * num_patches_y recorded in the file. {rays.shape[0]} != {num_patches_x * num_patches_y}"
        rays = rays.reshape(num_patches_y, num_patches_x, 6)
    return rays, origin


def newSceneWithRandomLightCenter(scene_dict):
    light_center = [
        np.random.uniform(-5, 5),
        np.random.uniform(-2, 2),
        np.random.uniform(-5, 5),
    ]
    scene_dict["emitter_opt"]["position"] = light_center
    scene = mi.load_dict(scene_dict)
    return scene


if __name__ == "__main__":
    scene_idx = int(sys.argv[1])
    light_idx = int(sys.argv[2])

    cwd = "."
    res_h = 420
    res_w = 560
    num_patches_y = 10
    num_patches_x = 10
    num_lights_per_scene = 3
    cell_h = res_h // num_patches_y
    cell_w = res_w // num_patches_x

    with open("usable_scenes_xml.txt", "r") as f:
        scene_names = f.readlines()
        scene_names = [scene_name.strip() for scene_name in scene_names]

    for scene_name in scene_names[scene_idx:scene_idx + 1]:
        print(f"Generating data for scene {scene_name}...")
        data_dir = os.path.join(cwd, "data/scenes_on_cluster/xml", scene_name)
        xml_filename = os.path.join(data_dir, "main.xml")
        data_group_name = data_dir.split("/")[-2]
        # scene_name = data_dir.split("/")[-1]
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
        # Create a point light placeholder. The actual center and intensity will be set later in parameters
        scene_dict["emitter_opt"] = {
            "type": "point",
            "position": [0, 0, 0],
            "intensity": {
                "type": "rgb",
                # random intensity between 1 and 10, always be white and yellow-ish
                "value": [0, 0, 0],
            },
        }

        # scene_dict["origin"] = {
        #     "type": "sphere",
        #     "center": [0, 0, 0],
        #     "radius": 0.1,
        #     "bsdf": {
        #         "type": "diffuse",
        #         "reflectance": {"type": "rgb", "value": [0, 0, 1]},
        #     },
        # }

        scene = mi.load_dict(scene_dict)

        # for i_light in range(num_lights_per_scene):
        for i_light in [light_idx]:
            params = mi.traverse(scene)
            light_RG = np.random.uniform(2, 10)
            light_B = np.random.uniform(light_RG * 0.5, light_RG * 1.1)
            params["emitter_opt.intensity.value"] = [light_RG, light_RG, light_B]

            # light_RG = 5
            # light_B = 5
            # light_center = np.array([0.0, 0.0, 0.0])

            # Load light center from file
            params_dir = os.path.join(
                cwd,
                "data",
                "RayDiffusionData",
                data_group_name,
                scene_name,
                f"light{i_light}",
                "params.json",
            )
            with open(params_dir, "r") as f:
                jsonfile = json.load(f)
                light_center = jsonfile["light_center"]
                num_patches_x = jsonfile["num_patches_x"]
                num_patches_y = jsonfile["num_patches_y"]
                num_images = jsonfile["num_images"]

            camera_lookat_mat = dataset[0]["camera_lookat_mat"]
            params["sensor.to_world"] = mi.ScalarTransform4f.look_at(
                origin=camera_lookat_mat[0], target=camera_lookat_mat[1], up=camera_lookat_mat[2]
            )
            params.update()
            min_cam_pos = np.array([0, 0, 0])
            max_cam_pos = np.array([0, 0, 0])
            for i_view in range(len(dataset)):
                cam_lookat_mat = dataset[i_view]["camera_lookat_mat"]
                cam_pos = cam_lookat_mat[0]
                min_cam_pos = np.minimum(min_cam_pos, cam_pos)
                max_cam_pos = np.maximum(max_cam_pos, cam_pos)
            
            scale = np.max(max_cam_pos - min_cam_pos)

            # print(f"Trying to render an image...")
            # centers = []
            # while True:
            #     light_center = [
            #         np.random.uniform(-5, 5),
            #         np.random.uniform(-2, 2),
            #         np.random.uniform(-5, 5),
            #     ]
            #     params["emitter_opt.position"] = light_center
            #     params.update()
            #     try_image = mi.render(scene, spp=35)
            #     try_image = np.array(try_image, dtype=np.float32)
            #     avg_color = np.mean(try_image, axis=(0, 1))
            #     if np.all(avg_color > 0.01):
            #         centers.append(light_center)
            #         break

            # # [TEMP] find bounding box of light centers
            # centers = np.array(centers)
            # min_center = np.min(centers, axis=0)
            # max_center = np.max(centers, axis=0)
            # print(f"min_center: {min_center}")
            # print(f"max_center: {max_center}")
            # exit(0)

            # print(f"Light center: {light_center}")

            # Generate gt rays for light ray diffusion
            print(f"Generating data...")

            # Output directory
            output_dir = os.path.join(
                cwd, "data", "RayDiffusionData", data_group_name, scene_name, f"light{i_light}"
            )
            # output_dir = os.path.join("test_output_2", scene_name, f"light{i_light}")
            os.makedirs(output_dir, exist_ok=True)

            origins = []
            for i_view in range(len(dataset)):
                break
                # for i_view in range(10, len(dataset)):
                # Set up camera
                cam_lookat_mat = dataset[i_view]["camera_lookat_mat"]
                camera_extrinsics = mi.ScalarTransform4f.look_at(
                    origin=cam_lookat_mat[0], target=cam_lookat_mat[1], up=cam_lookat_mat[2]
                )
                params["sensor.to_world"] = camera_extrinsics
                params.update()

                # Generate rays
                rays = []
                pixel_positions = []
                for x in range(num_patches_y):
                    for y in range(num_patches_x):
                        x_ = x * cell_h + cell_h // 2
                        y_ = y * cell_w + cell_w // 2
                        u = x_ / res_h
                        v = y_ / res_w
                        pixel_pos = intersectCameraRayWithScene(
                            u=u,
                            v=v,
                            fov=scene_dict["sensor"]["fov"],
                            aspect_ratio=res_h / res_w,
                            camera_extrinsics=camera_extrinsics,
                            scene=scene,
                        )
                        pixel_positions.append(pixel_pos)
                        # if ray is None or pixel_pos is None:
                        #     ray = PluckerRay(
                        #         direction=np.array([0, 0, 0]), moment=np.array([0, 0, 0])
                        #     )
                        #     rays.append(ray)
                        #     continue
                        # rays.append(ray)

                        # # Validate the ray
                        # m_hat = np.cross(camera_extrinsics.inverse() @ light_center, ray.direction)
                        # loss = np.linalg.norm(m_hat - ray.moment)
                        # if loss > 1e-3:
                        #     print(scene_name)
                        #     print(i_light)
                        #     print(x, y)
                        #     print(f"m_hat: {m_hat}")
                        #     print(f"ray.moment: {ray.moment}")
                        #     print(f"loss: {loss}")
                        #     print()

                        # scene_dict[f"pixel_{x}_{y}"] = {
                        #     "type": "sphere",
                        #     "center": np.array(pixel_pos).flatten(),
                        #     "radius": 0.03,
                        #     "bsdf": {
                        #         "type": "diffuse",
                        #         "reflectance": {"type": "rgb", "value": [1, 0, 0]},
                        #     },
                        # }

                # Compute mean pixel position
                pixel_positions = np.array(pixel_positions).reshape(num_patches_y, num_patches_x, 3)
                mean_pixel_pos = np.mean(
                    pixel_positions, axis=(0, 1), where=~np.isnan(pixel_positions)
                )
                origins.append(mean_pixel_pos.tolist())
                for x in range(num_patches_y):
                    for y in range(num_patches_x):
                        # compute ray from pixel to light in camera space with origin at mean_pixel_pos
                        if np.isnan(pixel_positions[x, y]).any():
                            ray = PluckerRay(
                                direction=np.array([0, 0, 0]), moment=np.array([0, 0, 0])
                            )
                            rays.append(ray)
                            continue
                        p = camera_extrinsics.inverse() @ (
                            pixel_positions[x, y] - mean_pixel_pos + cam_lookat_mat[0]
                        )
                        l = camera_extrinsics.inverse() @ (
                            light_center - mean_pixel_pos + cam_lookat_mat[0]
                        )
                        # p = camera_extrinsics.inverse() @ pixel_positions[x, y]
                        # l = camera_extrinsics.inverse() @ light_center
                        d = l - p
                        d = d / np.linalg.norm(d)
                        d = np.array(d).flatten()
                        m = np.cross(p, d).flatten()
                        ray = PluckerRay(direction=d, moment=m)
                        rays.append(ray)

                # Write ground truth ray information
                ray_file = os.path.join(output_dir, f"rays{i_view}.txt")
                writeRaysToFile(rays, mean_pixel_pos, ray_file, num_patches_x, num_patches_y)
                # # Visualize rays
                # rays = list(map(lambda ray: ray.data(), rays))
                # rays = np.array(rays)
                # rays = rays.reshape(num_patches_y, num_patches_x, 6)
                # rays_vis = visualizeRays(rays)
                # imageio.imwrite(os.path.join(output_dir, f"rays{i_view}.png"), rays_vis)

                # # Validate light center
                # rays = list(map(lambda ray: ray.data(), rays))
                # rays = np.array(rays)
                # center = pluckerRays2Point(rays)
                # center = camera_extrinsics @ np.array(center) - cam_lookat_mat[0] + mean_pixel_pos
                # print(f"center: {center}")
                # print("loss: ", np.linalg.norm(center - light_center))
                # print()

                # Render ground truth image
                gt_image = mi.render(scene, spp=528, seed=0)
                image_file = os.path.join(output_dir, f"image{i_view}.exr")
                mi.util.write_bitmap(image_file, gt_image, "rgb")

            # write parameters to json file
            parameters = {
                "num_patches_x": num_patches_x,
                "num_patches_y": num_patches_y,
                "num_images": len(dataset),
                "light_center": list(light_center),
                "scale": scale,
            }
            json_file = os.path.join(output_dir, f"params.json")
            with open(json_file, "w") as f:
                json.dump(parameters, f, sort_keys=True, indent=4)
