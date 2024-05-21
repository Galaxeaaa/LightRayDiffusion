from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import imageio
import struct
import pickle


def get_dataloader(dataset: Dataset, batch_size: int, num_workers: int, train: bool):
    dataloader = DataLoader(
        dataset, batch_size, shuffle=train, num_workers=num_workers, drop_last=train
    )
    return dataloader


class OpenRoomsDemoSceneData(Dataset):
    def __init__(self, data_dir: str):
        self.images = [os.path.normpath(i) for i in glob(data_dir + "/im_*.hdr")]
        self.n_views = len(self.images)
        self.camera_lookat_mats = self.loadCameras(data_dir + "/cam.txt")
        self.base_colors = glob(data_dir + "/imbaseColor_*.png")
        self.normals = glob(data_dir + "/imnormal_*.png")
        self.depth_data = glob(data_dir + "/imdepth_*.dat")
        self.all_light_masks = glob(
            data_dir + "/immask_*.png"
        )  # a single light mask for all lights
        # self.svlightings = glob(data_dir + "/imenv_*.hdr")
        self.light_boxes = []  # light box for each light
        self.light_masks = []  # light mask for each light
        self.direct_shadings = []  # direct shading for each light
        self.direct_shadings_wo_occlusion = []  # direct shading without occlusion for each light
        self.shadows = []  # shadow map for each light
        for i in range(self.n_views):
            self.light_boxes.append(glob(data_dir + f"/light_{i + 1}/box*.dat"))
            self.light_masks.append(glob(data_dir + f"/light_{i + 1}/mask*.png"))
            self.direct_shadings.append(glob(data_dir + f"/light_{i + 1}/imDS*.rgbe"))
            self.direct_shadings_wo_occlusion.append(
                glob(data_dir + f"/light_{i + 1}/imDSNoOcclu*.rgbe")
            )
            self.shadows.append(glob(data_dir + f"/light_{i + 1}/imShadow*.png"))
            self.direct_shadings[-1] = [x for x in self.direct_shadings[-1] if "NoOcclu" not in x]
            assert (
                len(self.light_boxes[-1])
                == len(self.light_masks[-1])
                == len(self.direct_shadings[-1])
                == len(self.direct_shadings_wo_occlusion[-1])
                == len(self.shadows[-1])
            ), f"The numbers of light properties are not consistent for view {i + 1}. Check the dataset."

        assert (
            len(self.images)
            == len(self.base_colors)
            == len(self.normals)
            == len(self.depth_data)
            == len(self.light_masks)
            == len(self.direct_shadings)
            == len(self.direct_shadings_wo_occlusion)
            == len(self.shadows)
        ), "The view numbers of different properties are not consistent. Check the dataset."

        print(f"Loaded scene {data_dir} with {self.n_views} views.")

    def __len__(self):
        return self.n_views

    def __getitem__(self, idx):
        image = imageio.imread(self.images[idx])

        base_color = imageio.imread(self.base_colors[idx])
        base_color = (base_color.astype(np.float32) / 255.0) ** (2.2)

        normal = imageio.imread(self.normals[idx])

        with open(self.depth_data[idx], "rb") as fIn:
            # Read the height and width of depth
            hBuffer = fIn.read(4)
            height = struct.unpack("i", hBuffer)[0]
            wBuffer = fIn.read(4)
            width = struct.unpack("i", wBuffer)[0]
            # Read depth
            dBuffer = fIn.read(4 * width * height)
            depth = np.array(struct.unpack("f" * height * width, dBuffer), dtype=np.float32)
            depth = depth.reshape(height, width)

        all_light_mask = imageio.imread(self.all_light_masks[idx])

        light_boxes = []
        light_masks = []
        direct_shadings = []
        direct_shadings_wo_occlusion = []
        shadows = []
        n_lights = len(self.light_boxes[idx])
        for i in range(n_lights):
            with open(self.light_boxes[idx][i], "rb") as fIn:
                info = pickle.load(fIn)
                light_boxes.append(info)
            light_masks.append(imageio.imread(self.light_masks[idx][i]))
            direct_shadings.append(imageio.imread(self.direct_shadings[idx][i]))
            direct_shadings_wo_occlusion.append(
                imageio.imread(self.direct_shadings_wo_occlusion[idx][i])
            )
            shadows.append(imageio.imread(self.shadows[idx][i]))

        light_masks = np.array(light_masks, dtype=np.float32)
        direct_shadings = np.array(direct_shadings, dtype=np.float32)
        direct_shadings_wo_occlusion = np.array(direct_shadings_wo_occlusion, dtype=np.float32)
        shadows = np.array(shadows, dtype=np.float32)

        camera_lookat_mat = self.camera_lookat_mats[idx]

        return {
            "image": image,  # (480, 640, 3)
            "base_color": base_color,  # (480, 640, 3)
            "normal": normal,  # (480, 640, 3)
            "depth": depth,  # (480, 640)
            "all_light_mask": all_light_mask,  # (3, 480, 640, 3)
            "light_boxes": light_boxes,  # List[{"isWindow": bool, "box3D": {"center", "xAxis",
            # "yAxis", "zAxis", "xLen", "yLen", "zLen"}, "box2D": {"x1", "x2", "y1", "y2"}}, ...]
            "light_masks": light_masks,  # (n_light, 120, 160, 3)
            "direct_shadings": direct_shadings,  # (n_light, 120, 160, 3)
            "direct_shadings_wo_occlusion": direct_shadings_wo_occlusion,  # (n_light, 120, 160, 3)
            "shadows": shadows,  # (n_light, 120, 160, 3)
            "camera_lookat_mat": camera_lookat_mat,  # (3, 3)
        }

    def loadCameras(self, data_dir: str):
        camera_lookat_mats = []
        with open(data_dir, "r") as fIn:
            n_cams = int(fIn.readline())
            assert (
                n_cams == self.n_views
            ), "The number of cameras in cam.txt is not consistent with the number of views."
            for i_cam in range(n_cams):
                look_at_mat = np.zeros((3, 3))
                for i in range(3):
                    look_at_mat[i] = list(map(float, fIn.readline().split()))
                camera_lookat_mats.append(look_at_mat)

        return camera_lookat_mats  # [origin, lookat, up]: 3x3


class OpenRoomsSceneData(Dataset):
    def __init__(self, data_dir: str):
        self.scene_xml = data_dir + "/main.xml"
        self.camera_lookat_mats = self.loadCameras(data_dir + "/cam.txt")
        self.n_views = len(self.camera_lookat_mats)

    def __len__(self):
        return self.n_views

    def __getitem__(self, idx):
        return {
            "scene_xml": self.scene_xml,
            "camera_lookat_mat": self.camera_lookat_mats[idx],
        }

    def loadCameras(self, data_dir: str):
        camera_lookat_mats = []
        with open(data_dir, "r") as fIn:
            n_cams = int(fIn.readline())
            for _ in range(n_cams):
                look_at_mat = np.zeros((3, 3))
                for i in range(3):
                    look_at_mat[i] = list(map(float, fIn.readline().split()))
                camera_lookat_mats.append(look_at_mat)

        return camera_lookat_mats  # [origin, lookat, up]: 3x3
