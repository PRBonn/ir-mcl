"""The class for visualizing the localization result
Code partially borrowed from
https://github.com/PRBonn/range-mcl/blob/main/src/visualizer.py.
MIT License
Copyright (c) 2021 Xieyuanli Chen, Ignacio Vizzo, Thomas Läbe, Jens Behley,
Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.
"""
import copy

import utils
import numpy as np
import matplotlib.pyplot as plt


class Visualizer(object):
    """ This class is a visualizer for localization results.
    """

    def __init__(self, map_size, poses, map_poses, odoms=None, occ_map=None,
                 numParticles=1000, grid_res=0.2, start_idx=0, converge_thres=5):
        """ Initialization:
        mapsize: the size of the given map
        poses: ground truth poses.
        map_poses: poses used to generate the map.
        odoms: odometry.
        occ_map: occupancy grid map background.
        numParticles: number of particles.
        grid_res: the resolution of the grids.
        start_idx: the start index.
        converge_thres: a threshold used to tell whether the localization converged or not.
        """
        self.numpoints = numParticles
        self.grid_res = grid_res

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(3, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [5, 1, 1]})
        self.ax0 = self.ax[0]
        self.ax1 = self.ax[1]
        self.ax2 = self.ax[2]

        # set size
        self.point_size = 1
        self.map_size = map_size
        self.plot_size = np.array(map_size) * grid_res
        self.start_idx = start_idx
        self.converge_idx = copy.copy(start_idx)

        # set ground truth
        self.location_gt = poses[:, :2]
        gt_heading = []
        for pose in poses:
            gt_heading.append(pose[-1])
        self.heading_gt = np.array(gt_heading)

        # set odometry
        self.plot_odom = False
        if odoms is not None:
            self.plot_odom = True
            self.location_odom = odoms[:, :2]
            odom_heading = []
            for odom in odoms:
                odom_heading.append(odom[-1])
            self.heading_odom = np.array(odom_heading)

        # init estimates and errors
        self.location_estimates = np.zeros((len(poses), 2))
        self.location_err = np.zeros(len(poses))
        self.heading_err = np.zeros(len(poses))

        # Then setup FuncAnimation.
        self.err_thres = converge_thres
        self.setup_plot()

        # set the map
        self.ax0.plot(map_poses[:, 0], map_poses[:, 1], '--', alpha=0.5, c='black', label='map')
        if occ_map is not None:
            map_res = 0.05
            ox, oy = occ_map.shape[0] // 2, occ_map.shape[1] // 2

            occ_set = []
            unknow_set = []
            # X, Y = occ_map.shape
            X_min, X_max = int(self.map_size[0] / map_res + ox), int(self.map_size[1] / map_res + ox)
            Y_min, Y_max = int(self.map_size[2] / map_res + oy), int(self.map_size[3] / map_res + oy)
            for x in range(X_min, X_max):
                for y in range(Y_min, Y_max):
                    if occ_map[x, y] > 0.9:
                        occ_set.append([(x - ox) * map_res, (y - oy) * map_res])
                    elif occ_map[x, y] == 0.5:
                        unknow_set.append([(x - ox) * map_res, (y - oy) * map_res])

            occ_set = np.array(occ_set)
            unknow_set = np.array(unknow_set)
            self.ax0.plot(occ_set[:, 0], occ_set[:, 1], '.', alpha=0.5, c='black')
            # self.ax0.plot(unknow_set[:, 0], unknow_set[:, 1], '.', alpha=0.5, c='gray')

        # for zoom animation
        self.x_min_offset = 0
        self.x_max_offset = 0
        self.y_min_offset = 0
        self.y_max_offset = 0

    def setup_plot(self):
        """ Initial drawing of the scatter plot.
        """
        # setup ax0
        self.ax_gt, = self.ax0.plot([], [], c='r', label='reference pose')
        self.ax_est, = self.ax0.plot([], [], c='b', label='estimated pose')

        if self.plot_odom:
            self.ax_odom, = self.ax0.plot([], [], c='c', label='odometry')

        self.scat = self.ax0.scatter([], [], c=[], s=self.point_size, cmap="Blues", vmin=0, vmax=1)

        self.ax0.axis('square')
        self.ax0.set(xlim=self.plot_size[:2], ylim=self.plot_size[2:])
        self.ax0.set_xlabel('X [m]')
        self.ax0.set_ylabel('Y [m]')
        self.ax0.legend()

        # setup ax1
        self.ax_location_err, = self.ax1.plot([], [], c='g')

        self.ax1.set(xlim=[0, len(self.location_gt)], ylim=[0, self.err_thres])
        self.ax1.set_xlabel('Frame id')
        self.ax1.set_ylabel('Location err [m]')

        # setup ax2
        self.ax_heading_err, = self.ax2.plot([], [], c='g')

        self.ax2.set(xlim=[0, len(self.location_gt)], ylim=[0, self.err_thres])
        self.ax2.set_xlabel('Frame id')
        self.ax2.set_ylabel('Heading err [deg]')

        # combine all artists
        self.patches = [self.ax_gt, self.ax_est, self.scat, self.ax_location_err, self.ax_heading_err]

        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.patches

    def get_estimates(self, sorted_data, selection_rate=0.8):
        """ calculate the estimated poses.
        """
        # only use the top selection_rate particles to estimate the position
        selected_particles = sorted_data[-int(selection_rate * self.numpoints):]
        estimated_pose = utils.particles2pose(selected_particles)
        estimated_location = estimated_pose[:2]
        estimated_heading = estimated_pose[2]

        return estimated_location, estimated_heading

    def compute_errs(self, frame_idx, particles):
        """ Calculate the errors.
        """
        sorted_data = particles[particles[:, 3].argsort()]
        new_location_estimate, new_heading_estimate = self.get_estimates(sorted_data)
        self.location_estimates[frame_idx] = new_location_estimate
        self.location_err[frame_idx] = np.linalg.norm(new_location_estimate - self.location_gt[frame_idx])
        self.heading_err[frame_idx] = np.minimum(
            abs(new_heading_estimate - self.heading_gt[frame_idx]),
            2. * np.pi - abs(new_heading_estimate - self.heading_gt[frame_idx])) * 180. / np.pi

        return sorted_data

    def update(self, frame_idx, particles):
        """ Update the scatter plot.
        """
        # particles[:, -1] /= np.max(particles[:, -1])
        particle_xyc = self.compute_errs(frame_idx, particles)[-10000:]

        # Only show the estimated trajectory when localization successfully converges
        if self.location_err[frame_idx] < self.err_thres:
            # set ground truth
            self.ax_gt.set_data(self.location_gt[self.start_idx:frame_idx + 1, 0],
                                self.location_gt[self.start_idx:frame_idx + 1, 1])

            # set odometry
            if self.plot_odom:
                self.ax_odom.set_data(self.location_odom[self.start_idx:frame_idx + 1, 0],
                                      self.location_odom[self.start_idx:frame_idx + 1, 1])

            # set estimated pose
            self.ax_est.set_data(self.location_estimates[self.converge_idx:frame_idx + 1, 0],
                                 self.location_estimates[self.converge_idx:frame_idx + 1, 1])

            # Set x and y data
            self.scat.set_offsets(particle_xyc[:, :2] * self.grid_res)
            # Set colors
            self.scat.set_array(particle_xyc[:, 3])

            # set err
            self.ax_location_err.set_data(np.arange(self.start_idx, frame_idx),
                                          self.location_err[self.start_idx:frame_idx])
            self.ax_heading_err.set_data(np.arange(self.start_idx, frame_idx),
                                         self.heading_err[self.start_idx:frame_idx])

        else:
            # set ground truth
            self.ax_gt.set_data(self.location_gt[self.start_idx:frame_idx + 1, 0],
                                self.location_gt[self.start_idx:frame_idx + 1, 1])

            # set odometry
            if self.plot_odom:
                self.ax_odom.set_data(self.location_odom[self.start_idx:frame_idx + 1, 0],
                                      self.location_odom[self.start_idx:frame_idx + 1, 1])

            # Set x and y data
            self.scat.set_offsets(particle_xyc[:, :2] * self.grid_res)
            # Set colors according to weights
            self.scat.set_array(particle_xyc[:, 3])

            # set err
            self.ax_location_err.set_data(np.arange(self.start_idx, frame_idx),
                                          self.location_err[self.start_idx:frame_idx])
            self.ax_heading_err.set_data(np.arange(self.start_idx, frame_idx),
                                         self.heading_err[self.start_idx:frame_idx])

            self.converge_idx += 1

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.patches

    def update_offline(self, frame_idx, results):
        """ Update the scatter plot.
        """
        particles = results[frame_idx]
        # particles[:, -1] /= np.max(particles[:, -1])
        particle_xyc = self.compute_errs(frame_idx, particles)[-10000:]

        # Only show the estimated trajectory when localization successfully converges
        if self.location_err[frame_idx] < self.err_thres:
            # set ground truth
            self.ax_gt.set_data(self.location_gt[self.start_idx:frame_idx + 1, 0],
                                self.location_gt[self.start_idx:frame_idx + 1, 1])

            # set odometry
            if self.plot_odom:
                self.ax_odom.set_data(self.location_odom[self.start_idx:frame_idx + 1, 0],
                                      self.location_odom[self.start_idx:frame_idx + 1, 1])

            # set estimated pose
            self.ax_est.set_data(self.location_estimates[self.converge_idx:frame_idx + 1, 0],
                                 self.location_estimates[self.converge_idx:frame_idx + 1, 1])

            # Set x and y data
            self.scat.set_offsets(particle_xyc[:, :2] * self.grid_res)
            # Set colors
            self.scat.set_array(particle_xyc[:, 3])

            # set err
            self.ax_location_err.set_data(np.arange(self.start_idx, frame_idx),
                                          self.location_err[self.start_idx:frame_idx])
            self.ax_heading_err.set_data(np.arange(self.start_idx, frame_idx),
                                         self.heading_err[self.start_idx:frame_idx])

        else:
            # set ground truth
            self.ax_gt.set_data(self.location_gt[self.start_idx:frame_idx + 1, 0],
                                self.location_gt[self.start_idx:frame_idx + 1, 1])

            # Set x and y data
            self.scat.set_offsets(particle_xyc[:, :2] * self.grid_res)
            # Set colors according to weights
            self.scat.set_array(particle_xyc[:, 3])

            # set err
            self.ax_location_err.set_data(np.arange(self.start_idx, frame_idx),
                                          self.location_err[self.start_idx:frame_idx])
            self.ax_heading_err.set_data(np.arange(self.start_idx, frame_idx),
                                         self.heading_err[self.start_idx:frame_idx])

            self.converge_idx += 1

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.patches


if __name__ == '__main__':
    pass
