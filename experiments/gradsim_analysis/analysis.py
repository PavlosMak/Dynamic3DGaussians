import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def load_ply_file(file_path):
    """
    Load a PLY file and return vertex coordinates and faces.

    Parameters:
        file_path (str): Path to the PLY file.

    Returns:
        vertices (numpy array): Array of vertex coordinates.
        faces (numpy array): Array of faces.
    """
    ply_data = PlyData.read(file_path)
    vertices = ply_data['vertex'].data
    return vertices

# #Torus
# data_paths = {
#     "MPM": "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/torus/ground_truth/ground_truth.npz",
#     "Gradsim": "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/torus/positions_gt.npz",
#     "PAC-NeRF predicted": "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/torus/pacnerf_results/predicted.npz",
# }

# Elastic
data_paths = {
    "MPM": "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/elastic_0/ground_truth/ground_truth.npz",
    "Gradsim": "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/elastic_0/positions_gt.npz"
}




if __name__ == "__main__":
    frame_count = 12

    positions = {}
    for path in data_paths:
        positions[path] = np.load(data_paths[path])["arr_0"]
        if path == "Gaussian Trajectories":
            rot = R.from_euler("x", [180], degrees=True)
            for i, p in enumerate(positions[path]):
                positions[path][i] = rot.apply(p)

    positions["MPM"] = positions["MPM"][:len(positions["Gradsim"])]

    velocities = {}
    velocity_magnitudes = {}

    for path in data_paths:
        vs = []
        ps = positions[path]
        for i in range(1, frame_count):
            vs.append(np.mean(ps[i] - ps[i - 1], axis=0))
        velocities[path] = np.array(vs)
        velocity_magnitudes[path] = [np.linalg.norm(v) for v in vs]

    end_frame = 11
    t = [i for i in range(12)]
    plt.figure(figsize=(10, 6))
    plt.subplot(4, 1, 1)  # (rows, columns, suplot number)
    for path in data_paths:
        plt.plot(t[:end_frame], velocity_magnitudes[path][:end_frame], label=path, marker="o")
    overall_mse = np.mean((np.array(velocity_magnitudes["Gradsim"][:end_frame]) - np.array(velocity_magnitudes["MPM"][:end_frame]))**2)
    print("Overall MSE:", overall_mse)
    plt.xlabel("time")
    plt.legend()
    plt.title("Velocity Magnitudes")

    plt.subplot(4, 1, 2)
    plt.title("Velocity x")
    for path in data_paths:
        plt.plot(t[:end_frame], velocities[path][:end_frame][:, 0], label=path, marker="o")
    x_mse = np.mean((np.array(velocities["Gradsim"][:end_frame][:, 0]) - np.array(velocities["MPM"][:end_frame][:, 0]))**2)
    print("x MSE:", x_mse)
    plt.xlabel("time")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.title("Velocity y")
    for path in data_paths:
        plt.plot(t[:end_frame], velocities[path][:end_frame][:, 1], label=path, marker="o")
    y_mse = np.mean(
        (np.array(velocities["Gradsim"][:end_frame][:, 1]) - np.array(velocities["MPM"][:end_frame][:, 1])) ** 2)
    print("y MSE:", y_mse)
    plt.xlabel("time")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.title("Velocity z")
    for path in data_paths:
        plt.plot(t[:end_frame], velocities[path][:end_frame][:, 2], label=path, marker="o")
    z_mse = np.mean(
        (np.array(velocities["Gradsim"][:end_frame][:, 2]) - np.array(velocities["MPM"][:end_frame][:, 2])) ** 2)
    print("z MSE:", z_mse)
    plt.xlabel("time")
    plt.legend()

    plt.tight_layout()
    plt.show()
