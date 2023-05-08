import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook


def plot_bl_inputs(pose, xlabel="x", ylabel="y", title="title",
    mark_point_index=True):
    
    num_points = 17 # Skip hip beacause its value is always as (0, 0)
    x_arr = []
    y_arr = []
       
    for i in range(num_points):

        x = pose[2*i]
        y = pose[2*i+1]

        x_arr.append(x)
        y_arr.append(y)


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


def plot_bl_pose_2d(data, xlabel="x", ylabel="y", title="title",
    mark_point_index=True):

    fig, ax = plt.subplots()

    pose = data.get_pose()
    num_points = data.num_pose_landmarks

    x_arr = []
    y_arr = []
       
    for i in range(num_points):

        x = pose[i, 0]
        y = pose[i, 1]

        x_arr.append(x)
        y_arr.append(y)

    ax.scatter(x_arr, y_arr)

    # Plot the point index 
    if mark_point_index:
        for i in range(num_points):
            ax.annotate(f'{i}', xy=(x_arr[i], y_arr[i]))
  
    # Plot the join lines
    connect_pairs = data.get_connect_pairs()

    for i in range(len(connect_pairs)):

        start_point, end_point = connect_pairs[i]

        x_arr = [start_point[0], end_point[0]]
        y_arr = [start_point[1], end_point[1]]

        ax.plot(x_arr, y_arr, color='blue')


    ax.invert_yaxis()

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(title)

    ax.grid(True)
    fig.tight_layout()
    
    return fig

def plot_bl_pose_3d(data, xlabel="x", ylabel="y", zlabel="z", title="title"):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    pose = data.get_pose()
    num_points = data.num_pose_landmarks

    x_arr = []
    y_arr = []
    z_arr = []
       
    for i in range(num_points):

        x = pose[i, 0]
        y = pose[i, 1]
        z = pose[i, 2]

        x_arr.append(x)
        y_arr.append(y)
        z_arr.append(z)

    ax.scatter(x_arr, y_arr, z_arr)

    # Plot the join lines
    connect_pairs = data.get_connect_pairs()

    for i in range(len(connect_pairs)):

        start_point, end_point = connect_pairs[i]

        x_arr = [start_point[0], end_point[0]]
        y_arr = [start_point[1], end_point[1]]
        z_arr = [start_point[2], end_point[2]]

        ax.plot(x_arr, y_arr, z_arr, color='blue')

    # Set initial view angle
    ax.view_init(elev=135, azim=90)

    ax.invert_xaxis()
    ax.invert_zaxis()


    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_zlabel(zlabel, fontsize=15)
    ax.set_title(title)

    ax.grid(True)
    fig.tight_layout()
    
    return fig
