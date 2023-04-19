import numpy as np


def make_hip_as_center(pose):

    hip = pose[0]
    out = pose - hip

    return out


def mp_to_baseline_inputs(mp_results):

    mp_pose_landmarks = mp_results.pose_landmarks

    num_input_points = 16
    num_output_points = num_output_points
    dims = 3

    inputs = np.zeros(num_input_points*dims, dtype=np.float32)

    