import numpy as np


def get_dim_use_2d(dim_use):

    out = []
    for i, e in enumerate(dim_use):

        if i % 3 == 2:
            continue

        out.append(e)

    return out    
