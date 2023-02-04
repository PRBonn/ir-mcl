"""The functions for building the IPB2DMapping dataset class for
IPBLab Indoor Datasets and Pre-2014 Robotics 2D-Laser Datasets
Code partially borrowed from
https://github.com/kwea123/nerf_pl/blob/master/datasets/blender.py.
MIT License
Copyright (c) 2020 Quei-An Chen
"""

import os
import json

import numpy as np

import torch
import torch.utils.data as data

from .ray_utils import get_ray_directions, get_rays


class IPB2DMapping(data.Dataset):
    """
    IPB 2D Mapping Dataset Class.

    :param root_dir: the root directory of the dataset.
    :param split: the type of the current split. ({'train', 'val', 'test'}, default: 'train)
    """
    def __init__(self, root_dir, split='train'):
        super(IPB2DMapping, self).__init__()
        self.root_dir = root_dir

        assert split in ['train', 'val', 'test'],\
            "Not supported split type \"{}\"".format(split)
        self.split = split

        self.load_data()

    def load_data(self):
        file_path = os.path.join(self.root_dir, '{}.json'.format(self.split))

        with open(file_path, 'r') as f:
            self.meta = json.load(f)

        self.num_beams = self.meta['num_beams']
        self.angle_min = self.meta['angle_min']
        self.angle_max = self.meta['angle_max']
        self.angle_res = self.meta['angle_res']
        self.max_range = self.meta['max_range']
        self.meta['scans'] = self.meta['scans']

        self.near = 0.02
        self.far = np.floor(self.max_range)
        self.bound = np.array([self.near, self.far])

        # ray directions for all beams in the lidar coordinate, shape: (N, 2)
        self.directions = get_ray_directions(self.angle_min, self.angle_max, self.angle_res)

        if self.split == 'train':
            self.all_rays = []
            self.all_ranges = []

            for scan in self.meta['scans']:
                pose = np.array(scan['transform_matrix'])[:2, :3]
                T_w2l = torch.FloatTensor(pose)

                rays_o, rays_d = get_rays(self.directions, T_w2l)
                range_readings, valid_mask_gt = self.load_scan(scan['range_readings'])

                # remove invalid range reading (exceed the limitation of lidar)
                # invalid data will be not used for training
                rays_o = rays_o[valid_mask_gt]
                rays_d = rays_d[valid_mask_gt]
                range_readings = range_readings[valid_mask_gt]

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             self.near * torch.ones_like(rays_d[..., :1]),
                                             self.far * torch.ones_like(rays_d[..., :1])],
                                            dim=1)]
                self.all_ranges += [range_readings]

            self.all_rays = torch.cat(self.all_rays, dim=0)  # shape: (len(frames) * (N-invalid), 6)
            self.all_ranges = torch.cat(self.all_ranges, dim=0)  # shape: (len(frames) * (N-invalid), )

    def load_scan(self, range_readings):
        range_readings = np.array(range_readings)

        # valid mask ground truth (< max_range)
        valid_mask_gt = range_readings.copy()
        valid_mask_gt[np.logical_and(valid_mask_gt > 0, valid_mask_gt < self.max_range)] = 1
        valid_mask_gt[valid_mask_gt >= self.max_range] = 0

        # set invalid value (no return) to 0
        range_readings[range_readings >= self.max_range] = 0

        range_readings = torch.Tensor(range_readings)
        valid_mask_gt = torch.BoolTensor(valid_mask_gt)

        return range_readings, valid_mask_gt

    def __getitem__(self, index):
        if self.split == 'train':
            sample = {'rays': self.all_rays[index],
                      'ranges': self.all_ranges[index]}

        else:
            scan = self.meta['scans'][index]
            odom = torch.FloatTensor(scan['odom'])
            T_w2l = torch.FloatTensor(scan['transform_matrix'])[:2, :3]

            # load range reading
            range_readings, valid_mask_gt = self.load_scan(scan['range_readings'])

            rays_o, rays_d = get_rays(self.directions, T_w2l)

            near, far = self.near * torch.ones_like(rays_d[..., :1]), self.far * torch.ones_like(rays_d[..., :1])
            rays = torch.cat([rays_o, rays_d, near, far], -1)

            sample = {'rays': rays,
                      'ranges': range_readings,
                      'odom': odom,
                      'valid_mask_gt': valid_mask_gt}

        return sample

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        else:
            return len(self.meta['scans'])
