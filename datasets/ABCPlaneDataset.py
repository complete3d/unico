import json
import os
import random
import sys

import h5py
import numpy as np
import torch.utils.data as data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import data_transforms
from .build import DATASETS


@DATASETS.register_module()
class ABCPlane(data.Dataset):
    def __init__(self, config, logger=None):
        self.category_file_path = config.category_file_path
        self.complete_points_path = config.complete_points_path
        self.complete_planes_path = config.complete_planes_path
        self.input_points_path = config.input_points_path
        self.subset = config.subset
        self.large_file_path = config.large_file_path

        self.num_points = config.N_POINTS
        self.num_planes = config.NUM_PLANES

        with open(self.category_file_path, "r") as f:
            self.dataset_categories = json.loads(f.read())

        with open(self.large_file_path, "r") as f:
            large_file_list = json.loads(f.read())

        if self.num_planes == 40:
            filter_file_list = (
                large_file_list["40-50"]
                + large_file_list["50-60"]
                + large_file_list["60+"]
            )
        elif self.num_planes == 50:
            filter_file_list = large_file_list["50-60"] + large_file_list["60+"]
        elif self.num_planes == 60:
            filter_file_list = large_file_list["60+"]
        else:
            filter_file_list = []

        self.file_list = []
        for dc in self.dataset_categories:
            samples = dc[self.subset]
            for sample in samples:
                if sample in filter_file_list:
                    continue
                self.file_list.append({"model_id": sample, "file_path": sample + ".ply"})

        self.num_renderings = config.num_renderings if self.subset == "train" else 1
        self.transforms = self._get_transforms()

    def _get_transforms(self):
        return data_transforms.Compose(
            [
                {
                    "callback": "UpSamplePlanes",
                    "parameters": {"n_planes": self.num_planes},
                    "objects": ["planes_gt"],
                },
                {
                    "callback": "UpSamplePoints",
                    "parameters": {"n_points": 2048},
                    "objects": ["points_pc"],
                },
                {
                    "callback": "UpSamplePoints",
                    "parameters": {"n_points": self.num_points},
                    "objects": ["points_gt"],
                },
                {
                    "callback": "ToTensor",
                    "objects": ["points_gt", "planes_gt", "points_pc"],
                },
            ]
        )

    def pc_norm_with_centroid_and_scale(self, pc, centroid, m):
        pc[:, :3] = pc[:, :3] - centroid
        pc[:, :3] = pc[:, :3] / m
        return pc

    def plane_norm_with_centroid_and_scale(self, plane, centroid, m):
        plane[:, 3] = plane[:, 3] + np.dot(plane[:, :3], centroid)
        plane[:, 3] = plane[:, 3] / m
        return plane

    def pc_norm(self, pc):
        centroid = np.mean(pc[:, :3], axis=0)
        pc[:, :3] = pc[:, :3] - centroid
        m = np.max(np.sqrt(np.sum(pc[:, :3] ** 2, axis=1)))
        pc[:, :3] = pc[:, :3] / m
        assert m != 0
        return pc, centroid, m

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        render_idx = random.randint(0, self.num_renderings - 1) if self.subset == "train" else 0

        points_gt = h5py.File(self.complete_points_path, "r")
        planes_gt = h5py.File(self.complete_planes_path, "r")
        points_pc = h5py.File(self.input_points_path, "r")

        data["points_gt"] = points_gt[sample["model_id"]][:].astype(np.float32)
        data["planes_gt"] = planes_gt[sample["model_id"]][:].astype(np.float32)
        data["points_pc"] = points_pc[sample["model_id"]][f"{render_idx:02d}"][:].astype(np.float32)[:, :3]

        for key in ["points_gt", "points_pc", "planes_gt"]:
            if key == "points_gt":
                data[key], gt_centroid, gt_scale = self.pc_norm(data[key])
            elif key == "points_pc":
                data[key] = self.pc_norm_with_centroid_and_scale(data[key], gt_centroid, gt_scale)
            else:
                data[key] = self.plane_norm_with_centroid_and_scale(data[key], gt_centroid, gt_scale)

        points_gt.close()
        planes_gt.close()
        points_pc.close()

        if self.transforms is not None:
            data = self.transforms(data)

        assert data["points_gt"].shape[0] == self.num_points

        return sample["model_id"], (
            data["points_gt"][..., :3],
            data["points_gt"][..., -1],
            data["planes_gt"][..., :4],
            data["planes_gt"][..., -1],
            data["points_pc"],
        )

    def __len__(self):
        return len(self.file_list)
