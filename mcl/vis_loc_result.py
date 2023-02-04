"""The functions for visualizing the localization result
Code partially borrowed from
https://github.com/PRBonn/range-mcl/blob/main/src/vis_loc_result.py.
MIT License
Copyright (c) 2021 Xieyuanli Chen, Ignacio Vizzo, Thomas Läbe, Jens Behley,
Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.
"""

import os
import matplotlib.animation as animation

import utils
import numpy as np
import matplotlib.pyplot as plt

from .visualizer import Visualizer


def plot_traj_result(results, poses, odoms=None, occ_map=None, numParticles=1000, grid_res=0.2, start_idx=0,
                     ratio=0.8, converge_thres=5, eva_thres=100):
    """ Plot the final localization trajectory.
    Args:
      results: localization results including particles in every timestamp.
      poses: ground truth poses.
      odoms: odometry
      occ_map: occupancy grid map background
      numParticles: number of particles.
      grid_res: the resolution of the grids.
      start_idx: the start index.
      ratio: the ratio of particles used to estimate the poes.
      converge_thres: a threshold used to tell whether the localization converged or not.
      eva_thres: a threshold to check the estimation results.
    """
    # get ground truth xy and yaw separately
    gt_location = poses[start_idx:, :-1]
    gt_heading = poses[start_idx:, -1]

    if occ_map is not None:
        map_res = 0.05
        ox, oy = occ_map.shape[0] // 2, occ_map.shape[1] // 2

        occ_set = []
        unknow_set = []
        X, Y = occ_map.shape
        for x in range(X):
            for y in range(Y):
                if occ_map[x, y] > 0.9:
                    occ_set.append([(x - ox) * map_res, (y - oy) * map_res])
                elif occ_map[x, y] == 0.5:
                    unknow_set.append([(x - ox) * map_res, (y - oy) * map_res])

        occ_set = np.array(occ_set)
        unknow_set = np.array(unknow_set)

    estimated_traj = []

    for frame_idx in range(start_idx, len(poses)):
        particles = results[frame_idx]
        # collect top 80% of particles to estimate pose
        idxes = np.argsort(particles[:, 3])[::-1]
        idxes = idxes[:int(ratio * numParticles)]

        partial_particles = particles[idxes]
        if np.sum(partial_particles[:, 3]) == 0:
            continue

        estimated_pose = utils.particles2pose(partial_particles)
        estimated_traj.append(estimated_pose)

    estimated_traj = np.array(estimated_traj)

    # generate statistics for location (x, y)
    diffs_xy = np.array(estimated_traj[:, :2] * grid_res - gt_location)
    diffs_dist = np.linalg.norm(diffs_xy, axis=1)  # diff in euclidean

    # generate statistics for yaw
    diffs_heading = np.minimum(abs(estimated_traj[:, 2] - gt_heading),
                               2. * np.pi - abs(estimated_traj[:, 2] - gt_heading)) * 180. / np.pi

    # check if every eva_thres success converged
    if len(diffs_dist) > eva_thres and np.all(diffs_dist[eva_thres::eva_thres] < converge_thres):
        # calculate location error
        diffs_location = diffs_dist[eva_thres:]
        mean_location = np.mean(diffs_location)
        mean_square_error = np.mean(diffs_location * diffs_location)
        rmse_location = np.sqrt(mean_square_error)

        mean_heading = np.mean(diffs_heading)
        mean_square_error_heading = np.mean(diffs_heading * diffs_heading)
        rmse_heading = np.sqrt(mean_square_error_heading)

        # print('rmse_location: ', rmse_location)
        # print('rmse_heading: ', rmse_heading)

    # plot results
    plt.close('all')
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)

    if occ_map is not None:
        ax.plot(occ_set[:, 0], occ_set[:, 1], '.', alpha=0.5, c='black')
        # ax.plot(unknow_set[:, 0], unknow_set[:, 1], '.', alpha=0.5, c='gray')

    ax.plot(poses[:, 0], poses[:, 1], c='r', label='ground_truth')

    if odoms is not None:
        ax.plot(odoms[:, 0], odoms[:, 1], c='c', label='odometry')

    ax.plot(estimated_traj[:, 0] * grid_res, estimated_traj[:, 1] * grid_res, label='estimated trajectory')
    plt.legend()
    plt.show()


def save_loc_result(frame_idx, map_size, poses, particles, est_poses, results_folder):
    """ Save the intermediate plots of localization results.
    Args:
      frame_idx: index of the current frame.
      map_size: size of the map.
      poses: ground truth poses.
      particles: current particles.
      est_poses: pose estimates.
      results_folder: folder to store the plots
    """
    particles[:, -1] /= np.max(particles[:, -1])
    # collect top 80% of particles to estimate pose
    idxes = np.argsort(particles[:, 3])[::-1]
    idxes = idxes[:int(0.8 * len(particles))]

    partial_particles = particles[idxes]

    normalized_weight = partial_particles[:, 3] / np.sum(partial_particles[:, 3])
    estimated_xy = partial_particles[:, :2].T.dot(normalized_weight.T)
    est_poses.append(estimated_xy)
    est_traj = np.array(est_poses)
    # plot results
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)

    ax.scatter(particles[:, 0], particles[:, 1], c=particles[:, 3], cmap='Blues', s=10)
    ax.plot(poses[:frame_idx + 1, 0], poses[:frame_idx + 1, 1], c='r', label='ground_truth')
    ax.plot(est_traj[:, 0], est_traj[:, 1], label='weighted_mean_80%')

    ax.axis('square')
    ax.set(xlim=map_size[:2], ylim=map_size[2:])

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(results_folder, str(frame_idx).zfill(6)))
    plt.close()


def vis_offline(results, poses, map_poses, mapsize, odoms=None, occ_map=None,
                numParticles=1000, grid_res=0.2, start_idx=0):
    """ Visualize localization results offline.
    Args:
      results: localization results including particles in every timestamp.
      poses: ground truth poses.
      map_poses: poses used to generate the map.
      odoms: odometry.
      occ_map: occupancy grid map background
      mapsize: size of the map.
      numParticles: number of particles.
      grid_res: the resolution of the grids.
      start_idx: the start index.
    """
    visualizer = Visualizer(mapsize, poses, map_poses, odoms=odoms, occ_map=occ_map,
                            numParticles=numParticles, grid_res=grid_res, start_idx=start_idx)

    # for animation
    anim = animation.FuncAnimation(visualizer.fig, visualizer.update_offline,
                                   frames=len(poses), fargs=[results])

    return anim


if __name__ == '__main__':
    pass
