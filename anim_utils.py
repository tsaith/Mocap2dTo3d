import os
from baseline_data import BaselineData
import matplotlib.pyplot as plt

from plot_utils import plot_bl_pose_2d, plot_bl_pose_3d
from file_utils import make_dir


def make_animations(dir_path, targets, is_3d=True,
    title="Anim plot", xlim=None, ylim=None, zlim=None):

    make_dir(dir_path)

    for i, target in enumerate(targets):

        if is_3d:
            fig = plot_bl_pose_3d(target, title=title,
                xlim=xlim, ylim=ylim, zlim=zlim)
        else:
            fig = plot_bl_pose_2d(target, title=title,
                xlim=xlim, ylim=ylim)

        filename = f"anim_{i:04d}.jpg"
        filepath = os.path.join(dir_path, filename)

        fig.savefig(filepath)
        plt.close(fig)