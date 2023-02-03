"""The main function for global localization experiments
Code partially borrowed from
https://github.com/PRBonn/range-mcl/blob/main/src/main_range_mcl.py.
MIT License
Copyright (c) 2021 Xieyuanli Chen, Ignacio Vizzo, Thomas Läbe, Jens Behley,
Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.
"""

import argparse
import os
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt

from utils import load_data, summary_loc
from mcl.initialization import init_particles_pose_tracking, init_particles_uniform
from mcl.motion_model import gen_commands_srrg
from mcl.sensor_model import SensorModel
from mcl.srrg_utils.pf_library.pf_utils import PfUtils
from mcl.vis_loc_result import plot_traj_result
from mcl.visualizer import Visualizer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str,
                        default='~/ir-mcl/config/loc_config_test1.yml',
                        help='the path for the configuration file.')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # load config file
    config_filename = args.config_file
    config = yaml.safe_load(open(config_filename))

    # load parameters
    start_idx = config['start_idx']
    grid_res = config['grid_res']
    numParticles = config['numParticles']
    reduced_num = config['num_reduced']
    visualize = config['visualize']
    result_path = config['result_path']

    # load input data
    map_pose_file = config['map_pose_file']
    data_file = config['data_file']
    mapsize = config['map_size']

    print('\nLoading data......')
    t = time.time()
    # load poses
    timestamps_mapping, map_poses, _, _, _ = load_data(map_pose_file)
    timestamps_gt, poses_gt, odoms, scans, params = \
        load_data(data_file, max_beams=config['max_beams'])

    if config['T_b2l']:
        T_b2l = np.loadtxt(config['T_b2l'])
    else:
        T_b2l = None

    # loading the occupancy grid map for visualization
    occmap = np.load(config['map_file'])

    print('All data are loaded! Time consume: {:.2f}s'.format(time.time() - t))

    # load parameters of sensor model
    params.update({'map_file': config['map_file']})
    params.update({'map_res': config['map_res']})

    params.update({'sensor_model': config['sensor_model']})
    params.update({'L_pos': config['L_pos']})
    params.update({'feature_size': config['feature_size']})
    params.update({'use_skip': config['use_skip']})
    params.update({'N_samples': config['N_samples']})
    params.update({'chunk': config['chunk']})
    params.update({'use_disp': config['use_disp']})
    params.update({'ckpt_path': config['ckpt_path']})
    params.update({'nog_size': config['nog_size']})
    params.update({'nog_res': config['nog_res']})

    # initialize particles
    print('\nMonte Carlo localization initializing...')
    # map_size, road_coords = gen_coords_given_poses(map_poses)
    if config['pose_tracking']:
        init_noise = config['init_noise']
        particles = init_particles_pose_tracking(
            numParticles, poses_gt[start_idx], noises=init_noise)
    else:
        particles = init_particles_uniform(mapsize, numParticles)

    # initialize sensor model
    # load parameters of sensor model
    print("\nSetting up the Sensor Model......")
    sensor_model = SensorModel(scans, params, mapsize)
    update_weights = sensor_model.update_weights
    print("Sensor Model Setup Successfully!\n")

    # generate odom commands
    commands = gen_commands_srrg(odoms)
    srrg_utils = PfUtils()

    # initialize a visualizer
    if visualize:
        plt.ion()
        visualizer = Visualizer(mapsize, poses=poses_gt, map_poses=map_poses, occ_map=occmap,
                                odoms=odoms, grid_res=grid_res, start_idx=start_idx)

    # Starts irmcl
    results = np.full((len(poses_gt), numParticles, 4), 0, np.float32)
    # add a small offset to avoid showing the meaning less estimations before convergence
    offset = 0
    is_converged = False

    for frame_idx in range(start_idx, len(poses_gt)):
        curr_timestamp = timestamps_gt[frame_idx]
        start = time.time()

        # only update while the robot moves
        if np.linalg.norm(commands[frame_idx]) > 1e-8:
            # motion model
            particles = srrg_utils.motion_model(particles, commands[frame_idx])

            # IR-based sensor model
            particles = update_weights(particles, frame_idx, T_b2l=T_b2l)

            if curr_timestamp > 20 and not is_converged:
                is_converged = True
                offset = frame_idx
                print('Initialization is finished!')
                # cutoff redundant particles and leave only num of particles
                idxes = np.argsort(particles[:, 3])[::-1]
                particles = particles[idxes[:reduced_num]]

            # resampling
            particles = srrg_utils.resample(particles)

        cost_time = np.round(time.time() - start, 10)
        print('finished frame {} at timestamp {:.2f}s with time cost: {}s'.format(
            frame_idx, timestamps_gt[frame_idx], cost_time))

        curr_numParticles = particles.shape[0]
        results[frame_idx, :curr_numParticles] = particles

        if visualize:
            visualizer.update(frame_idx, particles)
            visualizer.fig.canvas.draw()
            visualizer.fig.canvas.flush_events()


    # evaluate localization results (through evo)
    if not os.path.exists(os.path.dirname(result_path)):
        os.makedirs(os.path.dirname(result_path))

    result_dir = os.path.dirname(result_path)
    summary_loc(results, start_idx, numParticles,
                timestamps_gt, result_dir, config['gt_file'])

    np.savez_compressed(result_path, timestamps=timestamps_gt, odoms=odoms, poses_gt=poses_gt,
                        particles=results, start_idx=start_idx, numParticles=numParticles)

    print('save the localization results at:', result_path)

    if config['plot_loc_results']:
        plot_traj_result(results, poses_gt, grid_res=grid_res, occ_map=occmap,
                         numParticles=numParticles, start_idx=start_idx + offset)

