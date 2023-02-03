"""The class for the implicit representation-based observation model
Code partially borrowed from
https://github.com/PRBonn/range-mcl/blob/main/src/sensor_model.py.
MIT License
Copyright (c) 2021 Xieyuanli Chen, Ignacio Vizzo, Thomas Läbe, Jens Behley,
Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.
"""

import numpy as np
import torch

from .rendering import NOFRendering, NOGRendering


class SensorModel:
    def __init__(self, scans, params, map_size):
        # load the map module.
        self.scans = scans

        use_cuda: bool = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        print('Using: ', device)

        self.sensor_model_type = params['sensor_model']

        if self.sensor_model_type == 'nof':
            self.model = NOFRendering(
                directions=params['directions'], near=params['near'], far=params['far'],
                ckpt_path=params['ckpt_path'], L_pos=params['L_pos'], feature_size=params['feature_size'],
                use_skip=params['use_skip'], N_samples=params['N_samples'], chunk=params['chunk'],
                use_disp=params['use_disp'], device=device
            )

        elif self.sensor_model_type == 'nog':
            self.model = NOGRendering(
                directions=params['directions'], near=params['near'], far=params['far'],
                ckpt_path=params['ckpt_path'], L_pos=params['L_pos'], feature_size=params['feature_size'],
                use_skip=params['use_skip'], N_samples=params['N_samples'], map_size=params['nog_size'],
                grid_res=params['nog_res'], device=device)

        self.map_min_x = map_size[0]
        self.map_max_x = map_size[1]
        self.map_min_y = map_size[2]
        self.map_max_y = map_size[3]

    def update_weights(self, particles, frame_idx, T_b2l=None):
        current_scan = self.scans[frame_idx]

        # TODO: skip the particles outside the map

        particle_poses = particles[..., :-1]
        # 1. particles_poses (N, 3) to particles_mat (N, 4, 4)
        xs, ys, yaws = particle_poses[:, 0], particle_poses[:, 1], particle_poses[:, 2]
        particles_mat = [[np.cos(yaws), -np.sin(yaws), np.zeros_like(xs), xs],
                         [np.sin(yaws), np.cos(yaws), np.zeros_like(xs), ys],
                         [np.zeros_like(xs), np.zeros_like(xs), np.ones_like(xs), np.zeros_like(xs)],
                         [np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.ones_like(xs)]]
        particles_mat = np.array(particles_mat)

        # 2. transfer robot poses to lidar poses: T_w2b @ T_b2l -> T_w2l
        if T_b2l is not None:
            particles_mat = np.einsum('ijn, jk->ikn', particles_mat, T_b2l)
        particles_mat = particles_mat[:2, [0, 1, 3], :]

        # generate synthetic laser scan
        particle_scans = self.model.render(particles_mat)

        # calculate similarity
        # L1 Distance
        diffs = np.abs(particle_scans - current_scan)
        # Gaussian Kernel
        scores = np.exp(-0.5 * np.mean(diffs[:, current_scan > 0], axis=1) ** 2 / (0.5 ** 2))

        # normalize the particles' weight
        particles[:, 3] = particles[:, 3] * scores
        particles[:, 3] /= np.sum(particles[:, 3])

        return particles
