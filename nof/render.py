"""The functions for rendering a 2D laser scan from NOF's output
Code partially borrowed from https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py.
MIT License
Copyright (c) 2020 Quei-An Chen
"""

import torch
from .networks import NOF, Embedding

__all__ = ['render_rays']


def inference(model: NOF, embedding_xy: Embedding, samples_xy: torch.Tensor,
              z_vals: torch.Tensor, chunk=1024 * 32, noise_std=1, epsilon=1e-10):
    """
    Helper function that performs model inference.

    :param model: NOF model
    :param embedding_xy: position embedding module
    :param samples_xy: sampled position (shape: (N_rays, N_samples, 2))
    :param z_vals: depths of the sampled positions (shape: N_rays, N_samples)
    :param chunk: the chunk size in batched inference (default: 1024*32)
    :param noise_std: factor to perturb the model's prediction of sigma (default: 1)
    :param epsilon: a small number to avoid the 0 of weights_sum (default: 1e-10)

    :return:
        depth_final: rendered range value for each Lidar beams (shape: (N_rays,))
        weights: weights of each sample (shape: (N_rays, N_samples))
        opacity: the cross entropy of the predicted occupancy values
    """
    N_rays = samples_xy.shape[0]
    N_samples = samples_xy.shape[1]

    # Embed directions
    samples_xy = samples_xy.view(-1, 2)  # shape: (N_rays * N_samples, 2)

    # prediction, to get rangepred and raw sigma
    B = samples_xy.shape[0]  # N_rays * N_samples
    out_chunks = []
    for i in range(0, B, chunk):
        # embed position by chunk
        embed_xy = embedding_xy(samples_xy[i:i + chunk])
        # embed_xy = samples_xy[i:i + chunk]
        out_chunks += [model(embed_xy)]

    out = torch.cat(out_chunks, dim=0)
    prob_occ = out.view(N_rays, N_samples)  # shape: (N_rays, N_samples)

    # Volume Rendering: synthesis the 2d lidar scan
    prob_free = 1 - prob_occ  # (1-p)
    prob_free_shift = torch.cat(
        [torch.ones_like(prob_free[:, :1]), prob_free], dim=-1
    )
    prob_free_cum = torch.cumprod(prob_free_shift, dim=-1)[:, :-1]
    weights = prob_free_cum * prob_occ

    # add noise
    noise = torch.randn(weights.shape, device=weights.device) * noise_std
    weights = weights + noise

    # normalize
    weights = weights / (torch.sum(weights, dim=-1).reshape(-1, 1) + epsilon)

    depth_final = torch.sum(weights * z_vals, dim=-1)  # shape: (N_rays,)

    # opacity regularization
    opacity = torch.mean(torch.log(0.1 + prob_occ) + torch.log(0.1 + prob_free) + 2.20727)

    return depth_final, weights, opacity


def render_rays(model: NOF, embedding_xy: Embedding, rays: torch.Tensor,
                N_samples=64, use_disp=False, perturb=0, noise_std=1, chunk=1024 * 3):
    """
    Render rays by computing the output of @model applied on @rays

    :param model: NOF model, defined by models.NOF()
    :param embedding_xy: embedding model for position, defined by models.Embedding()

    :param rays: the input data, include: ray original, directions, near and far depth bounds
                 (shape: (N_rays, 2+2+2))
    :param N_samples: number of samples pre ray (default: 64)
    :param use_disp: whether to sample in disparity space (inverse depth) (default: False)
    :param perturb: factor to perturb the sampling position on the ray
                    (0 for default, for coarse model only)
    :param noise_std: factor to perturb the model's prediction of sigma (1 for default)
    :param chunk: the chunk size in batched inference

    :return result: dictionary containing final range value, weights and opacity
    """
    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, :2], rays[:, 2:4]  # shape: (N_rays, 2)
    near, far = rays[:, 4].view(-1, 1), rays[:, 5].view(-1, 1)  # shape: (N_rays, 1)

    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # shape: (N_samples,)
    z_steps = z_steps.expand(N_rays, N_samples)

    if use_disp:
        # linear sampling in disparity space
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)
    else:
        # linear sampling in depth space
        z_vals = near * (1 - z_steps) + far * z_steps

    # perturb sampling depths (z_vals)
    if perturb > 0:
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # shape: (N_rays, N_samples-1)
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], dim=-1)

        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    samples_xy = rays_o.unsqueeze(1) + \
                 rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # shape: (N_rays, N_samples, 2)

    depth, weights, opacity = \
        inference(model, embedding_xy, samples_xy, z_vals, chunk, noise_std)

    results = {'depth': depth,
               'weights': weights.sum(1),
               'opacity': opacity
               }

    return results


def render_rays_grid(grid: torch.Tensor, map_size, grid_res: float,
                     rays: torch.Tensor, N_samples=64, chunk=1000, epsilon=1e-10):
    """
    Render rays from neural occupancy grid, only used for accelerating MCL

    :param grid: occupancy grid from NOF
    :param map_size: the size of the grid
    :param grid_res: the cell's resolutions of grid
    :param rays: the input data, include: ray original, directions, near and far depth bounds
                 (shape: (N_rays, 2+2+2))
    :param N_samples: number of samples pre ray (default: 64)
    :param chunk: the chunk size in batched inference
    :param epsilon: a small number to avoid the 0 of weights_sum (default: 1e-10)

    :return result: dictionary containing final range value, weights and opacity
    """
    # Decompose the inputs
    N_particles, N_rays = rays.shape[:2]

    rays_os, rays_ds = rays[..., :2], rays[..., 2:4]  # shape: (N_particles, N_rays, 2)

    # shape: (N_particles, N_rays, 1)
    near, far = rays[..., 4].view(-1, N_rays, 1), rays[..., 5].view(-1, N_rays, 1)

    # Sampling depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # shape: (N_samples,)

    render_scans = []
    for i in range(0, N_particles, chunk):
        # linear sampling in depth space
        z_vals = near[i:i + chunk] * (1 - z_steps) + far[i:i + chunk] * z_steps

        # shape: (chunk, N_rays, N_samples, 2)
        samples_xy = rays_os[i:i + chunk].unsqueeze(2) + \
                     rays_ds[i:i + chunk].unsqueeze(2) * z_vals.unsqueeze(3)

        N_chunk = samples_xy.shape[0]

        # Embed directions
        samples_xy = samples_xy.view(-1, 2)  # shape: (N_chunk * N_rays * N_samples, 2)

        # prediction, to get rangepred and raw sigma
        samples_xy[:, 0] -= map_size[0]
        samples_xy[:, 1] -= map_size[2]
        samples_grid = torch.round(samples_xy / grid_res).type(torch.long)

        out = grid[samples_grid[:, 0], samples_grid[:, 1]]
        prob_occ = out.view(N_chunk, N_rays, N_samples)  # shape: (N_chunk, N_rays, N_samples)

        # Volume Rendering: synthesis the 2d lidar scan
        prob_free = 1 - prob_occ  # (1-p)
        prob_free_shift = torch.cat(
            [torch.ones_like(prob_free[..., :1]), prob_free], dim=-1
        )
        prob_free_cum = torch.cumprod(prob_free_shift, dim=-1)[..., :-1]
        weights = prob_free_cum * prob_occ

        # normalize
        weights = weights / (torch.sum(weights, dim=-1).reshape(N_chunk, N_rays, 1) + epsilon)
        depth = torch.sum(weights * z_vals, dim=-1)  # shape: (N_chunk, N_rays)

        render_scans.append(depth)

    render_scans = torch.cat(render_scans, dim=0)

    return render_scans
