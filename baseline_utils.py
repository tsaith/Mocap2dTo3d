import numpy as np

def make_hip_as_center(pose):

    hip = pose[0]
    out = pose - hip

    return out


def holistic_to_baseline_inputs(data):

    num_input_points = 16
    num_output_points = num_output_points
    dims = 3

    data_inputs = np.zeros(num_input_points*dims, dtype=np.float32)

    