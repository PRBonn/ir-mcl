import argparse

import matplotlib.pyplot as plt
import numpy as np

from utils import load_data
from vis_loc_result import plot_traj_result, vis_offline


def get_args():
    parser = argparse.ArgumentParser()

    # results
    parser.add_argument('--loc_results', type=str,
                        default='results/fr079/loc_results.npz',
                        help='the file path of localization results')

    # map
    parser.add_argument('--occ_map', type=str, default=None,
                        help='the file path of the occupancy grid map for visualization.')
    parser.add_argument('--map_size', nargs='+', type=float, default=[-25, 20, -10, 10],
                        help='the size of the map.')

    # output GIF
    parser.add_argument('--output_gif', type=str,
                        default=None,
                        help='the GIF path for saving the localization process.')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # loading localization results
    loc_results = np.load(args.loc_results)

    poses_gt = loc_results['poses_gt']
    particles = loc_results['particles']
    start_idx = loc_results['start_idx']
    numParticles = loc_results['numParticles']

    if args.occ_map:
        occ_map = np.load(args.occ_map)
    else:
        occ_map = None

    grid_res = 1
    offset = 284

    if args.output_gif:
        mapsize = args.map_size
        anim = vis_offline(particles, poses_gt, poses_gt, mapsize, occ_map=occ_map,
                           numParticles=numParticles, grid_res=grid_res, start_idx=start_idx)
        anim.save(args.output_gif, fps=10)
        plt.show()
