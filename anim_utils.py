import os
from baseline_data import BaselineData
import matplotlib.pyplot as plt

from plot_utils import plot_bl_pose_2d, plot_bl_pose_3d
from file_utils import make_dir


def make_animations(dir_path, targets, is_3d=True):

    make_dir(dir_path)

    if is_3d:
        plot_func = plot_bl_pose_3d
    else:
        plot_func = plot_bl_pose_2d

    for i, target in enumerate(targets):

        filename = f"anim_{i:04d}.jpg"
        filepath = os.path.join(dir_path, filename)

        fig = plot_func(target, title="Anim plot")
        fig.savefig(filepath)
        plt.close(fig)