import math
from typing import Callable

import open3d
import open3d as o3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


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



def generate_cube_points(edge_length, num_points_per_edge):
    range_vals = np.linspace(-edge_length / 2, edge_length / 2, num_points_per_edge)

    points = []
    for a in range_vals:
        for b in range_vals:
            # Top face
            points.append([a, b, edge_length / 2])
            # Bottom face
            points.append([a, b, -edge_length / 2])
            # Front face
            points.append([a, edge_length / 2, b])
            # Back face
            points.append([a, -edge_length / 2, b])
            # Right face
            points.append([edge_length / 2, a, b])
            # Left face
            points.append([-edge_length / 2, a, b])
    return np.array(points)


def add_noise(point_cloud, noise_level):
    noise = np.random.normal(0, noise_level, point_cloud.shape)
    noisy_point_cloud = point_cloud + noise
    return noisy_point_cloud


def simulate_missing_data(edge_length, points, multiplier=1.0):
    def distance_from_center(point, edge_length):
        distances = [edge_length / 2 - abs(coord) for coord in point]
        distances.sort()
        return math.sqrt((distances[1] - 0) ** 2 * (distances[2] - 0) ** 2)

    def missing_data_probability(point, edge_length, missing_data_rate):
        distance = distance_from_center(point, edge_length)
        # Normalize the distance to be between 0 and 1
        normalized_distance = distance / (edge_length / 2)
        return normalized_distance * missing_data_rate

    # Apply the probability function to each point to determine if it should be dropped
    filtered_points = []
    for point in points:
        # Check each coordinate (x, y, z) for the missing data probability
        if np.random.rand() > missing_data_probability(point, edge_length, multiplier):
            filtered_points.append(point)

    return np.array(filtered_points)


def insert_lines(filename, line_number, new_lines):
    # Read all lines into a list
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Split the original lines into two parts
    first_part = lines[:line_number - 1]
    second_part = lines[line_number - 1:]

    # Concatenate the first part, the new lines, and the second part
    new_lines = first_part + new_lines + second_part

    # Write the new lines back to the file
    with open(filename, 'w') as f:
        f.writelines(new_lines)


def generate_point_cloud(filename, noise_level=0.0, missing_data_level=0.0, num_points_per_edge=100, edge_length=1.0):
    sampling_resolution = 1 / num_points_per_edge / edge_length
    cube_points = generate_cube_points(edge_length, num_points_per_edge)
    cube_points = simulate_missing_data(edge_length, cube_points, missing_data_level)
    cube_points = add_noise(cube_points, noise_level)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(cube_points)
    o3d.io.write_point_cloud(filename, point_cloud, write_ascii=True, print_progress=True)
    insert_lines(filename, 3, [
        f'comment sampling resolution: {sampling_resolution}\n',
        f'comment noise level: {noise_level}\n',
    ])


def plot(filename):
    def formatter(ax: Axes3D, fig: plt.Figure):
        # fig.text(0.5, 0.015, f'Noise level {noise_level}', horizontalalignment="center")
        ax.view_init(10, -90 - 45)

    plot_ply(filename, formatter,
                        points_transform=lambda points: points + [0.5, 0.5, 0.5])


if __name__ == '__main__':
    num_points_per_edge = 100
    noise_level = 0.04  # standard deviation

    filename = f"data/cube_points.ply"
    generate_point_cloud(filename, noise_level)
    plot(filename)

    for noise_level in np.arange(0.01, 0.04, 0.01):
        filename = f"data/noise/cube_points_noise_{'{:.2f}'.format(noise_level).replace('0.', '')}.ply"
        generate_point_cloud(filename, noise_level)
        plot(filename)

    for missing_data_level in [6, 12, 24, 48]:
        filename = f"data/missing/cube_points_missing_{'{:.0f}'.format(missing_data_level).replace('.', '_')}.ply"
        generate_point_cloud(filename, 0, missing_data_level)
        plot(filename)

    for missing_data_level in [6, 12, 24, 48]:
        for noise_level in np.arange(0.01, 0.04, 0.01):
            missing = '{:.0f}'.format(missing_data_level).replace('.', '_')
            noise = '{:.2f}'.format(noise_level).replace('0.', '')
            filename = f"data/matrix/cube_points_m{missing}_n{noise}.ply"
            generate_point_cloud(filename, noise_level, missing_data_level)
            plot(filename)
