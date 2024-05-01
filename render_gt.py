import mitsuba as mi
import drjit as dr
import numpy as np
from main import display
# from emitter.invisible_sphere import InvisibleSphereEmitter

mi.set_variant("cuda_ad_rgb")
from dataloader.openrooms import OpenRoomDemoSceneData
from dataloader.convert_xml import convertXML2Dict

def loadSceneData(data_dir):
    scene_data = OpenRoomDemoSceneData(data_dir)
    return scene_data

if __name__ == "__main__":
    # mi.register_emitter("invisible_sphere", InvisibleSphereEmitter)
    data_dir = "data/Demo/main_xml/scene0001_01"
    xml_filename = "test.xml"
    out_filename = "test_gt^^.exr"
    idx_view = 0

    dataset = loadSceneData(data_dir)
    cam_lookat_mat = dataset[idx_view]["camera_lookat_mat"]

    scene_dict = convertXML2Dict(xml_filename)
    # for obj in scene_dict.values():
    #     if "emitter" in obj:
    #         obj.pop("emitter")
    scene_dict.pop("env_emitter")
    scene_dict["integrator"]["hide_emitters"] = True
    init_center = [-1, 0, 0]
    init_radius = 0.5
    scene_dict["emitter_opt"] = {
        "type": "point",
        # "center": init_center,
        # "radius": init_radius,
        # "emitter": {
        #     "type": "area",
        #     "radiance": {
        #         "type": "rgb",
        #         "value": [3, 3, 7.5],
        #     },
        # }
        "position": init_center,
        "intensity": {
            "type": "rgb",
            "value": [3, 3, 7.5],
        },
    }
    scene = mi.load_dict(scene_dict)

    params = mi.traverse(scene)
    params["sensor.to_world"] = mi.Transform4f.look_at(
        origin=cam_lookat_mat[0], target=cam_lookat_mat[1], up=cam_lookat_mat[2]
    )
    params.update()

    gt_image = mi.render(scene, params, spp=512)
    # display(mi.util.convert_to_bitmap(gt_image))
    mi.util.write_bitmap(out_filename, gt_image, "rgb")