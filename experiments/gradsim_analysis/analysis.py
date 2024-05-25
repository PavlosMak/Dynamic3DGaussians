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

# Elastic 0
# data_paths = {
#     "MPM": "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/elastic_0/ground_truth/ground_truth.npz",
#     "Gradsim": "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/elastic_0/positions_gt.npz",
#     "PAC-NeRF predicted": "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/elastic_0/pac_reconstruction/pac.npz"
# }

# Elastic 1
data_paths = {
    "MPM": "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/elastic_1/ground_truth/gt.npz",
    "Gradsim": "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/elastic_1/positions_gt.npz",
    "PAC-NeRF predicted": "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/elastic_1/pac_reconstruction/pac.npz"
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
    mpm = np.array(velocity_magnitudes["MPM"][:end_frame])
    overall_mse = np.mean((np.array(velocity_magnitudes["Gradsim"][:end_frame]) - mpm)**2)
    print("Overall MSE Gradsim:", overall_mse)
    if "PAC-NeRF predicted" in velocities:
        overall_mse_pac = np.mean((np.array(velocity_magnitudes["PAC-NeRF predicted"][:end_frame]) - mpm) ** 2)
        print("Overall MSE Pac:", overall_mse_pac)
    plt.xlabel("time")
    plt.legend()
    plt.title("Velocity Magnitudes")

    plt.subplot(4, 1, 2)
    plt.title("Velocity x")
    for path in data_paths:
        plt.plot(t[:end_frame], velocities[path][:end_frame][:, 0], label=path, marker="o")
    mpm = np.array(velocities["MPM"][:end_frame][:, 0])
    x_mse = np.mean((np.array(velocities["Gradsim"][:end_frame][:, 0]) - mpm)**2)
    print("")
    print("x MSE Gradsim:", x_mse)
    if "PAC-NeRF predicted" in velocities:
        x_mse_pac = np.mean((np.array(velocities["PAC-NeRF predicted"][:end_frame][:, 0]) - mpm) ** 2)
        print("x MSE PAC:", x_mse_pac)
    plt.xlabel("time")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.title("Velocity y")
    for path in data_paths:
        plt.plot(t[:end_frame], velocities[path][:end_frame][:, 1], label=path, marker="o")
    mpm = np.array(velocities["MPM"][:end_frame][:, 1])
    y_mse = np.mean(
        (np.array(velocities["Gradsim"][:end_frame][:, 1]) - mpm) ** 2)
    print("")
    print("y MSE Gradsim:", y_mse)
    if "PAC-NeRF predicted" in velocities:
        y_mse_pac = np.mean((np.array(velocities["PAC-NeRF predicted"][:end_frame][:, 1]) - mpm) ** 2)
        print("y MSE PAC:", y_mse_pac)
    plt.xlabel("time")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.title("Velocity z")
    for path in data_paths:
        plt.plot(t[:end_frame], velocities[path][:end_frame][:, 2], label=path, marker="o")
    mpm = np.array(velocities["MPM"][:end_frame][:, 2])
    z_mse = np.mean(
        (np.array(velocities["Gradsim"][:end_frame][:, 2]) - mpm) ** 2)
    print("")
    print("z MSE Gradsim:", z_mse)
    if "PAC-NeRF predicted" in velocities:
        z_mse_pac = np.mean(
            (np.array(velocities["PAC-NeRF predicted"][:end_frame][:, 2]) - mpm) ** 2)
        print("z MSE PAC:", z_mse_pac)
    plt.xlabel("time")
    plt.legend()

    plt.tight_layout()
    plt.show()

