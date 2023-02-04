"""The functions for particles initialization
Code partially borrowed from
https://github.com/PRBonn/range-mcl/blob/main/src/initialization.py.
MIT License
Copyright (c) 2021 Xieyuanli Chen, Ignacio Vizzo, Thomas Läbe, Jens Behley,
Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.
"""

import numpy as np

np.random.seed(0)


def init_particles_uniform(map_size, numParticles):
    """ Initialize particles uniformly.
      Args:
        map_size: size of the map.
        numParticles: number of particles.
      Return:
        particles.
    """
    [x_min, x_max, y_min, y_max] = map_size
    particles = []
    rand = np.random.rand
    for i in range(numParticles):
        x = (x_max - x_min) * rand(1).item() + x_min
        y = (y_max - y_min) * rand(1).item() + y_min
        theta = -np.pi + 2 * np.pi * rand(1).item()
        weight = 1
        particles.append([x, y, theta, weight])

    return np.array(particles)


def init_particles_pose_tracking(numParticles, init_pose, noises=[2, 2, np.pi / 6.0], init_weight=1.0):
    """ Initialize particles with a noisy initial pose.
    Here, we use ground truth pose with noises defaulted as [±5 meters, ±5 meters, ±π/6 rad]
    to mimic a non-accurate GPS information as a coarse initial guess of the global pose.
    Args:
      numParticles: number of particles.
      init_pose: initial pose.
      noises: range of noises.
      init_weight: initialization weight.
    Return:
      particles.
    """
    mu = np.array(init_pose)
    cov = np.diag(noises)
    particles = np.random.multivariate_normal(mean=mu, cov=cov, size=numParticles)
    init_weights = np.ones((numParticles, 1)) * init_weight
    particles = np.hstack((particles, init_weights))

    return np.array(particles, dtype=float)
