import gc
import json
import os

import drjit as dr
import imageio
import mitsuba as mi
import numpy as np

from utils.visualization import display, visualizeRays

mi.set_variant("cuda_ad_rgb")

from dataloader.convert_xml import convertXML2Dict
from dataloader.openrooms import OpenRoomsDemoSceneData, OpenRoomsSceneData, get_dataloader
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


def generateRay2Light(u, v, fov, aspect_ratio, light_pos, camera_extrinsics, scene):
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
        return None, None
    pixel_pos = np.array(si.p).flatten()

    # convert positions to camera space
    # light_pos = camera_extrinsics.inverse() @ light_pos
    # pixel_pos = camera_extrinsics.inverse() @ pixel_pos

    d = light_pos - pixel_pos
    d = d / np.linalg.norm(d)
    d = np.array(d).flatten()
    m = np.cross(pixel_pos, d).flatten()

    return PluckerRay(direction=d, moment=m), pixel_pos


def writeRaysToFile(rays, filename, num_patches_x, num_patches_y):
    with open(filename, "w") as f:
        f.write(f"{num_patches_x} {num_patches_y}\n")
        for ray in rays:
            f.write(
                f"{ray.direction[0]} {ray.direction[1]} {ray.direction[2]} {ray.moment[0]} {ray.moment[1]} {ray.moment[2]}\n"
            )


def readRaysFromFile(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        num_patches_x, num_patches_y = map(int, lines[0].split())
        rays = []
        for line in lines[1:]:
            ray = list(map(float, line.split()))
            rays.append(ray)
        rays = np.array(rays)
        assert (
            rays.shape[0] == num_patches_x * num_patches_y
        ), f"Number of rays read from file is not equal to num_patches_x * num_patches_y recorded in the file. {rays.shape[0]} != {num_patches_x * num_patches_y}"
        rays = rays.reshape(num_patches_y, num_patches_x, 6)
    return rays


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
    cwd = "."
    res_h = 420
    res_w = 560
    num_patches_y = 10
    num_patches_x = 10
    num_lights_per_scene = 1
    cell_h = res_h // num_patches_y
    cell_w = res_w // num_patches_x

    with open("usable_scenes_xml.txt", "r") as f:
        scene_names = f.readlines()
        scene_names = [scene_name.strip() for scene_name in scene_names]

    for scene_name in scene_names[:1]:
        print(f"Generating rays for scene {scene_name}...")
        data_dir = os.path.join(cwd, "data/scenes_on_cluster/xml", scene_name)
        xml_filename = os.path.join(data_dir, "main.xml")
        data_group_name = data_dir.split("/")[-2]
        # scene_name = data_dir.split("/")[-1]
        dataset = OpenRoomsSceneData(data_dir=data_dir)

        scene_dict = convertXML2Dict(xml_filename)
        scene_dict["sensor"]["film"]["height"] = res_h
        scene_dict["sensor"]["film"]["width"] = res_w
        scene_dict["integrator"]["type"] = "prb"
        # remove all emitters
        scene_dict.pop("env_emitter")
        for obj in scene_dict.values():
            if "emitter" in obj:
                obj.pop("emitter")

        # scene_dict["origin"] = {
        #     "type": "sphere",
        #     "center": [0, 0, 0],
        #     "radius": 0.1,
        #     "bsdf": {
        #         "type": "diffuse",
        #         "reflectance": {"type": "rgb", "value": [0, 0, 1]},
        #     },
        # }

        for i_light in range(num_lights_per_scene):
            light_RG = np.random.uniform(2, 10)
            light_B = np.random.uniform(1, light_RG * 1.1)

            light_center = [
                np.random.uniform(-5, 5),
                np.random.uniform(-2, 2),
                np.random.uniform(-5, 5),
            ]

            light_center = np.array([1.0, 0.0, 0.0])

            # params_dir = os.path.join(
            #     cwd, "data", "RayDiffusionData", data_group_name, scene_name, f"light{i_light}", "params.json"
            # )
            # with open(params_dir, "r") as f:
            #     params = json.load(f)
            #     light_center = params["light_center"]

            scene_dict["emitter_opt"] = {
                "type": "point",
                "position": light_center,
                "intensity": {
                    "type": "rgb",
                    # random intensity between 1 and 10, always be white and yellow-ish
                    "value": [light_RG, light_RG, light_B],
                },
            }

            camera_lookat_mat = dataset[0]["camera_lookat_mat"]
            scene_dict["sensor"]["to_world"] = mi.ScalarTransform4f.look_at(
                origin=camera_lookat_mat[0], target=camera_lookat_mat[1], up=camera_lookat_mat[2]
            )
            # scene = mi.load_dict(scene_dict)

            # # try to render an image to check if the light at valid position
            # print(f"Trying to render an image...")
            # while True:
            #     try_image = mi.render(scene, spp=64)
            #     try_image = np.array(try_image, dtype=np.float32)
            #     avg_color = np.mean(try_image, axis=(0, 1))
            #     if np.all(avg_color > 0.01):
            #         break
            #     scene = newSceneWithRandomLightCenter(scene_dict)

            light_center = np.array([1.0, 0.0, 0.0])
            print(f"Light center: {light_center}")

            # generate gt rays for light ray diffusion
            print(f"Generating data...")
            for i_view in range(len(dataset)):
            # for i_view in range(10):
                cam_lookat_mat = dataset[i_view]["camera_lookat_mat"]
                camera_extrinsics = mi.ScalarTransform4f.look_at(
                    origin=cam_lookat_mat[0], target=cam_lookat_mat[1], up=cam_lookat_mat[2]
                )
                scene_dict["sensor"]["to_world"] = camera_extrinsics
                scene = mi.load_dict(scene_dict)
                rays = []
                for x in range(num_patches_y):
                    for y in range(num_patches_x):
                        x_ = x * cell_h + cell_h // 2
                        y_ = y * cell_w + cell_w // 2
                        u = x_ / res_h
                        v = y_ / res_w
                        ray, pixel_pos = generateRay2Light(
                            u=u,
                            v=v,
                            fov=scene_dict["sensor"]["fov"],
                            aspect_ratio=res_h / res_w,
                            camera_extrinsics=camera_extrinsics,
                            light_pos=light_center,
                            scene=scene,
                        )
                        if ray is None or pixel_pos is None:
                            ray = PluckerRay(
                                direction=np.array([0, 0, 0]), moment=np.array([0, 0, 0])
                            )
                            rays.append(ray)
                            continue
                        rays.append(ray)

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

                # write data to file
                output_dir = os.path.join(
                    cwd, "data", "RayDiffusionData", data_group_name, scene_name, f"light{i_light}"
                )
                output_dir = os.path.join("test_output", scene_name, f"light{i_light}")
                os.makedirs(output_dir, exist_ok=True)

                # write parameters to json file
                parameters = {
                    "num_patches_x": num_patches_x,
                    "num_patches_y": num_patches_y,
                    "num_images": len(dataset),
                    "light_center": list(light_center),
                }
                json_file = os.path.join(output_dir, f"params.json")
                with open(json_file, "w") as f:
                    json.dump(parameters, f, sort_keys=True, indent=4)

                # write ground truth ray information
                ray_file = os.path.join(output_dir, f"rays{i_view}.txt")
                writeRaysToFile(rays, ray_file, num_patches_x, num_patches_y)
                rays = list(map(lambda ray: ray.data(), rays))
                rays = np.array(rays)
                rays = rays.reshape(num_patches_y, num_patches_x, 6)
                rays_vis = visualizeRays(rays)
                imageio.imwrite(os.path.join(output_dir, f"rays{i_view}.png"), rays_vis)

                # # validate light center
                # rays = list(map(lambda ray: ray.data(), rays))
                # rays = np.array(rays)
                # center = pluckerRays2Point(rays)
                # center = camera_extrinsics @ np.array(center)
                # print(f"center: {center}")
                # print("loss: ", np.linalg.norm(center - light_center))
                # print()

                # write ground truth rendering image
                scene = mi.load_dict(scene_dict)
                gt_image = mi.render(scene, spp=528, seed=0)
                image_file = os.path.join(output_dir, f"image{i_view}.exr")
                mi.util.write_bitmap(image_file, gt_image, "rgb")
