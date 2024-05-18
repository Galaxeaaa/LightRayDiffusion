import mitsuba as mi
import drjit as dr
import numpy as np
from main import display
# from emitter.invisible_sphere import InvisibleSphereEmitter

mi.set_variant("cuda_ad_rgb")
from dataloader.openrooms import OpenRoomsDemoSceneData, OpenRoomsSceneData
from dataloader.convert_xml import convertXML2Dict

def loadSceneData(data_dir):
    # scene_data = OpenRoomsDemoSceneData(data_dir)
    scene_data = OpenRoomsSceneData(data_dir)
    return scene_data

if __name__ == "__main__":
    # mi.register_emitter("invisible_sphere", InvisibleSphereEmitter)
    data_dir = "data/scenes_on_cluster/xml/scene0001_01"
    xml_filename = "test.xml"
    out_filename = "test_gt^^.exr"
    idx_view = 0

    scene_data = loadSceneData(data_dir)
    cam_lookat_mat = scene_data.camera_lookat_mats[idx_view]

    scene_dict = convertXML2Dict(scene_data.scene_xml)
    # remove all emitters
    for obj in scene_dict.values():
        if "emitter" in obj:
            obj.pop("emitter")
    scene_dict.pop("env_emitter")
    # add a point light
    init_center = [0, 1.5, 0]
    RG = np.random.uniform(2, 10)
    B = np.random.uniform(1, RG / 2)
    scene_dict["emitter_opt"] = {
        "type": "point",
        "position": init_center,
        "intensity": {
            "type": "rgb",
            # random intensity between 1 and 10, always be white and yellow-ish
            "value": [RG, RG, B],
        },
    }
    scene = mi.load_dict(scene_dict)

    params = mi.traverse(scene)
    params["sensor.to_world"] = mi.Transform4f.look_at(
        origin=cam_lookat_mat[0], target=cam_lookat_mat[1], up=cam_lookat_mat[2]
    )
    params.update()

    gt_image = mi.render(scene, params, spp=128)
    display(mi.util.convert_to_bitmap(gt_image))
    # mi.util.write_bitmap(out_filename, gt_image, "rgb")