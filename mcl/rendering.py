"""The classes for rendering 2D LiDAR scan of particles
"""

import time

import numpy as np

import torch

from nof.dataset.ray_utils import get_rays
from nof.render import render_rays, render_rays_grid
from nof.networks.models import NOF, Embedding
from nof.nof_utils import load_ckpt


class NOFRendering:
    """
    Rendering novel views with Neural Occupancy Field
    """

    def __init__(self, directions, near, far,
                 ckpt_path, L_pos=10, feature_size=256, use_skip=True,
                 N_samples=256, chunk=32 * 1024, use_disp=False,
                 device=torch.device('cpu')):
        # lidar params
        self.directions = directions
        self.near = near
        self.far = far

        # models
        self.device = device
        self.embedding_position = Embedding(in_channels=2, N_freq=L_pos)
        self.nof_model = NOF(feature_size=feature_size,
                             in_channels_xy=2 + 2 * L_pos * 2,
                             use_skip=use_skip)
        # loading pretrained weights
        load_ckpt(self.nof_model, ckpt_path, model_name='nof')
        self.nof_model.to(self.device).eval()

        # rendering params
        self.N_samples = N_samples
        self.chunk = chunk
        self.use_disp = use_disp

    def get_laser_rays(self, pose):
        # load range reading
        rays_o, rays_d = get_rays(self.directions, pose)
        rays_o = torch.FloatTensor(rays_o.copy())
        rays_d = torch.FloatTensor(rays_d.copy())

        near, far = self.near * torch.ones_like(rays_d[..., :1]), \
                    self.far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)

        return rays

    def render(self, poses):
        preds = []
        for pose in poses:
            # data processing
            rays = self.get_laser_rays(pose).to(self.device)

            with torch.no_grad():
                rendered_rays = render_rays(
                    model=self.nof_model, embedding_xy=self.embedding_position, rays=rays,
                    N_samples=self.N_samples, use_disp=self.use_disp, chunk=self.chunk
                )

            preds.append(rendered_rays['depth'].cpu().numpy())

        preds = np.array(preds)

        return preds


class NOGRendering:
    """
    Rendering novel views with Neural Occupancy Grid
    """

    def __init__(self, directions, near, far, ckpt_path,
                 L_pos=10, feature_size=256, use_skip=True, N_samples=256,
                 map_size=[-50, 50, -50, 50], grid_res=0.05, device=torch.device('cpu')):
        # lidar params
        self.directions = directions
        self.near = near
        self.far = far

        # models
        print("\nInitialize the Neural Occupancy Field model......")
        t = time.time()
        self.device = device
        self.embedding_position = Embedding(in_channels=2, N_freq=L_pos)
        self.nof_model = NOF(feature_size=feature_size,
                             in_channels_xy=2 + 2 * L_pos * 2, use_skip=use_skip)
        # loading pretrained weights
        load_ckpt(self.nof_model, ckpt_path, model_name='nof')
        self.embedding_position.to(self.device).eval()
        self.nof_model.to(self.device).eval()
        print("Models are ready! Time consume: {:.2f}s".format(time.time() - t))

        # generate grid
        self.grid = None
        self.map_size = map_size
        self.grid_res = grid_res
        self.get_grid()

        # rendering params
        self.N_samples = N_samples

    def get_grid(self):
        map_size = np.array(self.map_size)
        x_steps = int((map_size[1] - map_size[0]) / self.grid_res)
        y_steps = int((map_size[3] - map_size[2]) / self.grid_res)

        # generate grids of coordinates
        x = torch.linspace(map_size[0], map_size[1] - self.grid_res, steps=x_steps)
        y = torch.linspace(map_size[2], map_size[3] - self.grid_res, steps=y_steps)

        grid_x, grid_y = torch.meshgrid(x, y)
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).to(self.device)

        print('\nPredicting the occupancy probability for each cells of grid......')
        t = time.time()

        # inference
        with torch.no_grad():
            N_cells = grid.shape[0]
            chunk = 2500000
            out_grid = []
            for i in range(0, N_cells, chunk):
                out_grid.append(self.nof_model(self.embedding_position(grid[i:i + chunk])))
            self.grid = torch.cat(out_grid, dim=0).reshape(x_steps, y_steps)
            # # create grid will use amount of GPU memory, it should be released from the cache.
            torch.cuda.empty_cache()
        print("Inference Finished! Time consume: {:.2f}s".format(time.time() - t))

    def _get_rays(self, Ts_w2l):
        # rays_ds: shape (N_rays, 2, N_particles)
        rays_ds = np.einsum('ij, jkn->ikn',
                            self.directions, Ts_w2l[:, :2, :].transpose(1, 0, 2))

        # normalize direction vector
        rays_ds = rays_ds / np.linalg.norm(rays_ds, axis=1, keepdims=True)

        # The origin of all rays is the camera origin in world coordinate
        rays_os = np.broadcast_to(Ts_w2l[:, 2, :], rays_ds.shape)

        rays_ds = rays_ds.transpose((2, 0, 1))
        rays_os = rays_os.transpose((2, 0, 1))

        return rays_os, rays_ds

    def get_laser_rays(self, Ts_w2l):
        # load range reading
        rays_os, rays_ds = self._get_rays(Ts_w2l)  # shape (N_particles, N_rays, 2)

        rays_os = torch.FloatTensor(rays_os.copy())
        rays_ds = torch.FloatTensor(rays_ds.copy())

        near, far = self.near * torch.ones_like(rays_ds[..., :1]), \
                    self.far * torch.ones_like(rays_ds[..., :1])

        rays = torch.cat([rays_os, rays_ds, near, far], -1)

        return rays

    def render(self, Ts_w2l):
        # data processing
        rays = self.get_laser_rays(Ts_w2l).to(self.device)

        preds = render_rays_grid(self.grid, self.map_size, self.grid_res,
                                 rays=rays, N_samples=self.N_samples, chunk=500)

        preds = preds.cpu().numpy()
        return preds
