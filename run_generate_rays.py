import os
from tqdm import tqdm

with open("usable_scenes_xml.txt", "r") as f:
    scene_names = f.readlines()
    scene_names = [scene_name.strip() for scene_name in scene_names]

i_start = scene_names.index("scene0024_01")

for scene_idx in tqdm(range(i_start, len(scene_names))):
    for light_idx in range(5):
        os.system(f"python generate_rays.py {scene_idx} {light_idx}")