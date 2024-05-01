import drjit as dr
import mitsuba as mi
import os
from utils.visualization import display
import numpy as np

mi.set_variant("cuda_ad_rgb")

from dataloader.convert_xml import convertXML2Dict
from dataloader.openrooms import get_dataloader, OpenRoomDemoSceneData
from PluckerRay import PluckerRay


def generateRay2Light(u, v, fov, aspect_ratio, cam_pos, light_pos, camera_extrinsics, scene):
    """
    u, v: pixel coordinates starting from top-left corner
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

    d = light_pos - pixel_pos
    d = d / np.linalg.norm(d)
    d = np.array(d).flatten()
    m = np.cross(pixel_pos, d).flatten()

    return PluckerRay(direction=d, moment=m), pixel_pos


def writeRaysToFile(rays, filename):
    with open(filename, "w") as f:
        for ray in rays:
            f.write(
                f"{ray.direction[0]} {ray.direction[1]} {ray.direction[2]} {ray.moment[0]} {ray.moment[1]} {ray.moment[2]}\n"
            )


def readRaysFromFile(filename):
    rays = []
    with open(filename, "r") as f:
        for line in f:
            ray = line.strip().split(" ")
            rays.append(
                [float(ray[0]), float(ray[1]), float(ray[2]), float(ray[3]), float(ray[4]), float(ray[5])]
            )

    return rays


if __name__ == "__main__":
    cwd = "."
    data_dir = os.path.join(cwd, "data/Demo/main_xml/scene0001_01")
    xml_filename = os.path.join(cwd, "test.xml")
    gt_filename = os.path.join(cwd, "test_gt_wall_noenv.exr")
    idx_view = 0
    res_h = 480
    res_w = 640
    patch_res_h = 10
    patch_res_w = 10
    cell_h = res_h // patch_res_h
    cell_w = res_w // patch_res_w

    dataset = OpenRoomDemoSceneData(data_dir=data_dir)
    cam_lookat_mat = dataset[idx_view]["camera_lookat_mat"]

    scene_dict = convertXML2Dict(xml_filename)
    scene_dict["sensor"]["to_world"] = mi.ScalarTransform4f.look_at(
        origin=cam_lookat_mat[0], target=cam_lookat_mat[1], up=cam_lookat_mat[2]
    )
    scene_dict["sensor"]["film"]["height"] = res_h
    scene_dict["sensor"]["film"]["width"] = res_w
    scene_dict.pop("env_emitter")
    scene_dict["integrator"]["type"] = "prb"
    camera_extrinsics = mi.Transform4f.look_at(
        origin=cam_lookat_mat[0], target=cam_lookat_mat[1], up=cam_lookat_mat[2]
    )
    camera_projection = mi.Transform4f.perspective(
        fov=scene_dict["sensor"]["fov"],
        near=1e-2,  # default
        far=1e4,  # default
    )

    for obj in scene_dict.values():
        if "emitter" in obj:
            obj.pop("emitter")
    light_center = [-1, 1, 1]
    scene_dict["emitter_gt"] = {
        "type": "point",
        "position": light_center,
        "intensity": {
            "type": "rgb",
            "value": [5, 5, 5],
        },
    }

    # generate gt rays for light ray diffusion
    depth = dataset[idx_view]["depth"]

    scene = mi.load_dict(scene_dict)
    rays = []
    for x in range(patch_res_h):
        for y in range(patch_res_w):
            x_ = x * cell_h + cell_h // 2
            y_ = y * cell_w + cell_w // 2
            u = x_ / res_h
            v = y_ / res_w
            ray, pixel_pos = generateRay2Light(
                u=u,
                v=v,
                fov=scene_dict["sensor"]["fov"],
                aspect_ratio=res_h / res_w,
                cam_pos=cam_lookat_mat[0],
                camera_extrinsics=mi.Transform4f.look_at(
                    origin=cam_lookat_mat[0], target=cam_lookat_mat[1], up=cam_lookat_mat[2]
                ),
                light_pos=light_center,
                scene=scene,
            )
            if ray is None or pixel_pos is None:
                ray = PluckerRay(direction=np.array([0, 0, 0]), moment=np.array([0, 0, 0]))
            rays.append(ray)
            # # validate ray
            # light_pos = np.array(light_center)
            # ray_dir = light_pos - pixel_pos
            # ray_dir = ray_dir / np.linalg.norm(ray_dir)
            # print(np.dot(ray.direction, ray_dir))
            # m_ = np.cross(pixel_pos, ray_dir)
            # print(f"m: {ray.moment}, m_: {m_}")

    # write ray information to file
    ray_file = os.path.join(cwd, "rays_test.txt")
    with open(ray_file, "w") as f:
        for ray in rays:
            f.write(
                f"{ray.direction[0]} {ray.direction[1]} {ray.direction[2]} {ray.moment[0]} {ray.moment[1]} {ray.moment[2]}\n"
            )
    exit(0)

    scene_dict["x_axis"] = {
        "type": "cylinder",
        "p0": [0, 0, 0],
        "p1": [1, 0, 0],
        "radius": 0.01,
        "material": {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": [1, 0, 0]},
        },
    }
    scene_dict["y_axis"] = {
        "type": "cylinder",
        "p0": [0, 0, 0],
        "p1": [0, 1, 0],
        "radius": 0.01,
        "material": {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": [0, 1, 0]},
        },
    }
    scene_dict["z_axis"] = {
        "type": "cylinder",
        "p0": [0, 0, 0],
        "p1": [0, 0, 1],
        "radius": 0.01,
        "material": {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": [0, 0, 1]},
        },
    }

    # load scene
    scene = mi.load_dict(scene_dict)
    # gt_image = mi.Bitmap(gt_filename)
    print("Rendering ground truth image.")
    gt_image = mi.render(scene, spp=512, seed=0)
    display(mi.util.convert_to_bitmap(gt_image))
