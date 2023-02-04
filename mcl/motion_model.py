"""The functions for the motion model of mobile robot
Code partially borrowed from
https://github.com/PRBonn/range-mcl/blob/main/src/motion_model.py.
MIT License
Copyright (c) 2021 Xieyuanli Chen, Ignacio Vizzo, Thomas Läbe, Jens Behley,
Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def sample(b):
    tot = 0
    for i in range(12):
        tot += np.random.uniform(-b, b)

    tot = tot / 2
    return tot


def motion_model(particles, u, alpha=[0.0, 0.0, 0.0, 0.0], real_command=False, duration=0.1):
    """ MOTION performs the sampling from the proposal.
    distribution, here the rotation-translation-rotation motion model

    input:
       particles: the particles as in the main script
       u: the command in the form [rot1 trasl rot2] or real odometry [v, w]
       noise: the variances for producing the Gaussian noise for
       perturbating the motion,  noise = [noiseR1 noiseTrasl noiseR2]

    output:
       the same particles, with updated poses.

    The position of the i-th particle is given by the 3D vector
    particles(i).pose which represents (x, y, theta).

    Assume Gaussian noise in each of the three parameters of the motion model.
    These three parameters may be used as standard deviations for sampling.
    """
    num_particles = len(particles)
    if not real_command:
        # Sample-based Odometry Model
        alpha_1, alpha_2, alpha_3, alpha_4 = alpha
        rot1, trans, rot2 = u

        rot1_hat = rot1 + sample(np.repeat(alpha_1 * np.abs(rot1) + alpha_2 * trans, num_particles))
        trans_hat = trans + sample(np.repeat(alpha_3 * trans + alpha_4 * (np.abs(rot1) + np.abs(rot2)), num_particles))
        rot2_hat = rot2 + sample(np.repeat(alpha_1 * np.abs(rot2) + alpha_2 * trans, num_particles))

        # update pose using motion model
        particles[:, 0] += trans_hat * np.cos(particles[:, 2] + rot1_hat)
        particles[:, 1] += trans_hat * np.sin(particles[:, 2] + rot1_hat)
        particles[:, 2] += rot1_hat + rot2_hat

    else:  # use real commands with duration
        # noise in the [v, w] commands when moving the particles
        MOTION_NOISE = [0.05, 0.05]
        vNoise = MOTION_NOISE[0]
        wNoise = MOTION_NOISE[1]

        # use the Gaussian noise to simulate the noise in the motion model
        v = u[0] + vNoise * np.random.randn(num_particles)
        w = u[1] + wNoise * np.random.randn(num_particles)
        gamma = wNoise * np.random.randn(num_particles)

        # update pose using motion models
        particles[:, 0] += - v / w * np.sin(particles[:, 2]) + v / w * np.sin(particles[:, 2] + w * duration)
        particles[:, 1] += v / w * np.cos(particles[:, 2]) - v / w * np.cos(particles[:, 2] + w * duration)
        particles[:, 2] += w * duration + gamma * duration

    return particles


def gen_commands(poses, r1_noisy, d_noisy, r2_noisy):
    """ Create commands out of the ground truth with noise.
    input:
      ground truth poses and noisy coefficients

    output:
      commands for each frame.
    """
    # compute noisy-free commands
    # set the default command = [0,0,0]'
    commands = np.zeros((len(poses), 3))

    # compute relative poses
    headings = poses[:, 2]

    dx = (poses[1:, 0] - poses[:-1, 0])
    dy = (poses[1:, 1] - poses[:-1, 1])

    direct = np.arctan2(dy, dx)  # atan2(dy, dx), 1X(S-1) direction of the movement

    r1 = []
    r2 = []
    distance = []

    for idx in range(len(poses) - 1):
        r1.append(direct[idx] - headings[idx])
        r2.append(headings[idx + 1] - direct[idx])
        distance.append(np.sqrt(dx[idx] * dx[idx] + dy[idx] * dy[idx]))

    r1 = np.array(r1)
    r2 = np.array(r2)
    distance = np.array(distance)

    # add noise to commands
    commands_ = np.c_[r1, distance, r2]
    commands[1:] = commands_ + np.array([r1_noisy * np.random.randn(len(commands_)),
                                         d_noisy * np.random.randn(len(commands_)),
                                         r2_noisy * np.random.randn(len(commands_))]).T

    return commands


def odom2matrix(pose):
    """
    Generate 2D transformation from pose, the rotation counterclockwise about the origin
    :param pose: shape: (3,)
    :return:
    """
    odom_x, odom_y, odom_theta = pose
    matrix = [[np.cos(odom_theta), -np.sin(odom_theta), 0.0, odom_x],
              [np.sin(odom_theta), np.cos(odom_theta), 0.0, odom_y],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]]

    return np.array(matrix)


def matrix2odom(T):
    x, y = T[:2, -1]
    _, _, yaw = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=False)

    return np.array([x, y, yaw])


def gen_commands_srrg(poses):
    """ Create commands from the Odometer.
    input:
      odometer reading

    output:
      commands for each frame.
    """
    # compute noisy-free commands
    # set the default command = [0,0,0]'
    commands = np.zeros((len(poses), 3))

    for idx in range(1, len(poses)):
        T_last = odom2matrix(poses[idx - 1])
        T_current = odom2matrix(poses[idx])
        T_rel = np.linalg.inv(T_last) @ T_current
        commands[idx] = matrix2odom(T_rel)

    return commands


def gen_motion_reckon(commands):
    """ Generate motion reckon only for comparison.
    """

    particle = [0, 0, 0, 1]
    motion_reckon = []
    for cmmand in commands:
        # use the Gaussian noise to simulate the noise in the motion model
        rot1 = cmmand[0]
        tras1 = cmmand[1]
        rot2 = cmmand[2]

        # update pose using motion model
        particle[0] = particle[0] + tras1 * np.cos(particle[2] + rot1)
        particle[1] = particle[1] + tras1 * np.sin(particle[2] + rot1)
        particle[2] = particle[2] + rot1 + rot2

        motion_reckon.append([particle[0], particle[1]])

    return np.array(motion_reckon)


if __name__ == '__main__':
    pass
