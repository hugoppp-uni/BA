from typing import Callable

import open3d

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


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
    distance_from_center = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=distance_from_center, cmap='viridis', marker='.', s=0.1)
    if labels:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)

    formatter(ax, fig) if formatter else None

    if show:
        plt.show(bbox_inches='tight')
    else:
        plt.savefig(filename.split(".")[0] + ".png", bbox_inches='tight')


if __name__ == '__main__':
    plot_ply('cube_points/data1/cube_points_04.ply', lambda ax, fig: ax.view_init(7, -75))
    # visualization.draw_geometries([cloud])  # Visualize point cloud
