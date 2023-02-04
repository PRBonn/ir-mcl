from pf_library.pf_utils import PfUtils
import numpy as np

P_prev = np.random.randn(10000, 4)**2
P_prev[:, 3] = P_prev[:, 3]/np.sum(P_prev[:, 3])
control = np.array([1, 0, 0])
utils = PfUtils()

P = utils.motion_model(P_prev, control)

P[0, 3] = 1
P[1:, 3] = 0.0
P_new = utils.resample(P)
