import numpy as np


def estimate_dvec_with_two_points(origin, target):

    vec = target - origin
    len = np.linalg.norm(vec)
    dvec = vec / len

    return dvec


