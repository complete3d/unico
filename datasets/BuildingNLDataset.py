import os
import sys
import random
import json

import numpy as np
import torch.utils.data as data
import h5py

# Set up the base directory and add it to the system path for local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import data_transforms
from .io import IO
from .build import DATASETS


@DATASETS.register_module()
class BuildingNL(data.Dataset):
    """
    Dataset class for BuildingNL dataset.

    This dataset loads complete and partial point clouds along with plane data.
    It supports training and testing subsets and applies a sequence of transformations
    to the data.
    """

    def __init__(self, config, logger=None):
        """
        Initialize the BuildingNL dataset.

        Args:
            config: Configuration object containing dataset paths, number of points,
                    number of planes, subset type, and other parameters.
        """
        
        self.category_file_path = config.category_file_path
        self.complete_points_path = config.complete_points_path
        self.complete_planes_path = config.complete_planes_path
        self.input_points_path = config.input_points_path
        self.subset = config.subset
        self.large_file_path = config.large_file_path

        self.num_points = config.N_POINTS
        self.num_planes = config.NUM_PLANES

        
        with open(self.category_file_path, 'r') as f:
            self.dataset_categories = json.loads(f.read())
            
        filter_file_list = []
        with open(self.large_file_path, 'r') as f:
            large_file_list = json.loads(f.read())
            
        if self.num_planes == 40:
            filter_file_list = large_file_list['40-50'] + large_file_list['50-60'] + large_file_list['60+'] 
        elif self.num_planes == 50:
            filter_file_list = large_file_list['50-60'] + large_file_list['60+']
        elif self.num_planes == 60:
            filter_file_list = large_file_list['60+']
            
        self.file_list = []
        for dc in self.dataset_categories:
            if self.subset == "train":
                samples = dc[self.subset]
                for s in samples:
                    if s in filter_file_list:
                        continue
                    self.file_list.append({
                        'model_id':
                        s,
                        'file_path': s + '.ply'
                    })
            else:
                samples = dc[self.subset]
                for s in samples:
                    if s in filter_file_list:
                        continue
                    self.file_list.append({
                        'model_id':
                        s,
                        'file_path': s + '.ply'
                    })
        # self.file_list = self.file_list[:100]
        # Use 2 renderings for training and 1 for testing
        self.num_renderings = config.num_renderings if self.subset == 'train' else 1

        # Initialize the data transformations
        self.transforms = self._get_transforms()

    
    def _get_transforms(self):
        """
        Define the sequence of data transformations to be applied on each sample.

        Returns:
            A composed transformation function.
        """
        return data_transforms.Compose([
            {
                'callback': 'UpSamplePlanes',
                'parameters': {
                    'n_planes': self.num_planes
                },
                'objects': ['planes_gt']
            },
            {
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['points_pc']
            },
            {
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': 16384
                },
                'objects': ['points_gt']
            },
            {
                'callback': 'ToTensor',
                'objects': ['points_gt', 'planes_gt', 'points_pc']
            }
        ])
    
    def pc_norm_with_centroid_and_scale(self, pc, centroid, m):
        """ pc: NxC, return NxC """
        pc[:, :3] = pc[:, :3] - centroid
        pc[:, :3] = pc[:, :3] / m
        return pc
    
    def plane_norm_with_centroid_and_scale(self, plane, centroid, m):
        """ pc: NxC, return NxC """
        plane[:, 3] = plane[:, 3] + np.dot(plane[:, :3], centroid)
        plane[:, 3] = plane[:, 3] / m
        return plane

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc[:, :3], axis=0)
        pc[:, :3] = pc[:, :3] - centroid
        m = np.max(np.sqrt(np.sum(pc[:, :3]**2, axis=1)))
        pc[:, :3] = pc[:, :3] / m
        assert m != 0
        return pc, centroid, m

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple containing:
              - model_id: Identifier of the model.
              - A tuple with:
                  - points_gt (first 3 coordinates)
                  - points_gt (last coordinate)
                  - planes_gt (first 3 coordinates)
                  - planes_gt (last coordinate)
                  - points_pc: Partial points data.
        """
        sample = self.file_list[idx]
        data = {}

        # Select a random rendering index for training; use 0 for testing
        render_idx = random.randint(0, self.num_renderings - 1) if self.subset == 'train' else 0
        
        points_gt = h5py.File(self.complete_points_path, 'r')
        planes_gt = h5py.File(self.complete_planes_path, 'r')
        points_pc = h5py.File(self.input_points_path, 'r')

        # Load complete point cloud data
        data['points_gt'] = points_gt[sample['model_id']][:].astype(np.float32)
        # Load complete plane data
        data['planes_gt'] = planes_gt[sample['model_id']][:].astype(np.float32)
        

        # Load partial point cloud data for the selected rendering index
        data['points_pc'] = points_pc[sample['model_id']][f"{render_idx:02d}"][:].astype(np.float32)[:, :3]
        

        for ri in ['points_gt', 'points_pc', 'planes_gt']:

            if ri == 'points_gt':
                data[ri], gt_centroid, gt_scale = self.pc_norm(data[ri])
            elif ri == 'points_pc':
                data[ri] = self.pc_norm_with_centroid_and_scale(data[ri], gt_centroid, gt_scale)
            else:
                data[ri] = self.plane_norm_with_centroid_and_scale(data[ri], gt_centroid, gt_scale)

        # Close the h5py files
        points_gt.close()
        planes_gt.close()
        points_pc.close()

        # Apply data transformations if they are defined
        if self.transforms is not None:
            data = self.transforms(data)

        # Ensure the complete points data has the expected number of points
        assert data['points_gt'].shape[0] == self.num_points

        return sample['model_id'], (
            data['points_gt'][..., :3],
            data['points_gt'][..., -1],
            data['planes_gt'][..., :4],
            data['planes_gt'][..., -1],
            data['points_pc']
        )

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.file_list)
