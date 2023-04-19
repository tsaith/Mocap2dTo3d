import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook


def plot_baseline_data_2d(pose, xlabel="x", ylabel="y", title="title"):

    num_points = 16 # Skip hip beacause its value is always as (0, 0)
    x_arr = []
    y_arr = []
       
    for i in range(num_points):

        x = pose[2*i]
        y = pose[2*i+1]

        x_arr.append(x)
        y_arr.append(y)


    fig, ax = plt.subplots()
    ax.scatter(x_arr, y_arr)

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(title)

    ax.grid(True)
    fig.tight_layout()
    
    return fig


def plot_baseline_pose_2d(pose, xlabel="x", ylabel="y", title="title",
    mark_point_index=True):

    num_points = 17
    x_arr = []
    y_arr = []
    z_arr = []
       
    for i in range(num_points):

        x = pose[3*i]
        y = pose[3*i+1]
        z = pose[3*i+2]

        x_arr.append(x)
        y_arr.append(y)
        z_arr.append(z)


    fig, ax = plt.subplots()
    ax.scatter(x_arr, y_arr)

    # Plot the point index 
    if mark_point_index:
        for i in range(num_points):
            ax.annotate(f'{i}', xy=(x_arr[i], y_arr[i]))

    ax.invert_yaxis()

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(title)

    ax.grid(True)
    fig.tight_layout()
    
    return fig


def plot_baseline_pose_3d(pose, xlabel="x", ylabel="y", zlabel="z", title="title"):

    num_points = 17
    x_arr = []
    y_arr = []
    z_arr = []
       
    for i in range(num_points):

        x = pose[3*i]
        y = pose[3*i+1]
        z = pose[3*i+2]

        x_arr.append(x)
        y_arr.append(y)
        z_arr.append(z)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x_arr, y_arr, z_arr)

    ax.invert_yaxis()
    ax.invert_zaxis()

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_zlabel(zlabel, fontsize=15)
    ax.set_title(title)

    ax.grid(True)
    fig.tight_layout()
    
    return fig

