import numpy as np
from pf_library.pybind import pf_library_pybind


class PfUtils:
    def __init__(self):
        self.utils_provider = pf_library_pybind._MotionModelAndResampling()

    def motion_model(self, particles: np.ndarray, control: np.ndarray):
        _positions = pf_library_pybind._Vector4dVector(particles)
        return np.asarray(self.utils_provider._predict(_positions, control))

    def resample(self, particles):
        weights = particles[:, 3]
        indices = np.asarray(self.utils_provider._resample_uniform(weights))
        new_particles = particles[indices]
        new_particles[:, 3] = 1.0
        return new_particles
