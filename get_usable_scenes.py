import glob
import os
from tqdm import tqdm

import mitsuba as mi

mi.set_variant("cuda_ad_rgb")
from dataloader.convert_xml import convertXML2Dict
from dataloader.openrooms import OpenRoomsSceneData

cwd = "."
root_dir = os.path.join(cwd, "data/scenes/xml")
scene_dirs = glob.glob(os.path.join(root_dir, "scene*"))
failures = 0
successes = 0
successes_list = []
bar = tqdm(scene_dirs)
for scene_dir in bar:
    xml_filename = os.path.join(scene_dir, "main.xml")
    scene_data = OpenRoomsSceneData(scene_dir)
    cam_lookat_mat = scene_data.camera_lookat_mats[0]
    try:
        scene_dict = convertXML2Dict(scene_data.scene_xml)
    except Exception as e:
        failures += 1
        continue
    try:
        scene = mi.load_dict(scene_dict)
    except Exception as e:
        failures += 1
        continue
    successes += 1
    successes_list.append(scene_dir.split("/")[-1])

    bar.set_description(f"Successes: {successes}, Fails: {failures}")    

with open("usable_scenes_xml_.txt", "w") as f:
    f.write("\n".join(successes_list))
