import math
from typing import Callable

import open3d
import open3d as o3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def get_pts(filename) -> open3d.open3d.geometry.PointCloud:
    return open3d.io.read_point_cloud(filename)


def plot_ply(filename,
                formatter: Callable[[Axes3D, plt.Figure], None] = None,
                points_transform: Callable[[np.ndarray], np.ndarray] = None,
                show=False,
                labels=False):
    cloud = get_pts(filename)
    points = np.asarray(cloud.points)

    if points_transform:
        points = points_transform(points)

    x, y, z = np.transpose(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1.0, 1.0, 1.0])

    ax.plot(x, z, y, marker='.')
    if labels:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)

    formatter(ax, fig) if formatter else None

    set_axes_equal(ax)
    if show:
        plt.show(bbox_inches='tight')
    else:
        plt.savefig(filename.split(".")[0] + ".png", bbox_inches='tight')

def scatter_ply(filename,
                formatter: Callable[[Axes3D, plt.Figure], None] = None,
                points_transform: Callable[[np.ndarray], np.ndarray] = None,
                show=False,
                labels=False):
    cloud = get_pts(filename)
    points = np.asarray(cloud.points)

    if points_transform:
        points = points_transform(points)

    x, y, z = np.transpose(points)
    distance_from_center = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, z, y, c=distance_from_center, cmap='viridis', marker='.', s=0.1)
    if labels:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)

    formatter(ax, fig) if formatter else None

    set_axes_equal(ax)
    if show:
        plt.show(bbox_inches='tight')
    else:
        plt.savefig(filename.split(".")[0] + ".png", bbox_inches='tight')

def formatter(ax: Axes3D, fig: plt.Figure):
    pass
    # fig.text(0.5, 0.015, f'Noise level {noise_level}', horizontalalignment="center")
    # ax.view_init(0, 0)


if __name__ == '__main__':

    plot_ply('hull.ply', formatter)

    scatter_ply('points.ply', formatter)




