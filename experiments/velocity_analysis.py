import numpy as np
from plyfile import PlyData


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


run_name = "morning-wave-1409"

GENERAL_PATH = "/media/pavlos/One Touch/datasets/inverse_physics_results"
DATA_PATH = f"{GENERAL_PATH}/{run_name}"

PREDICTED_PATH = f"{DATA_PATH}/predicted.npz"
UNOPTIMIZED_PATH = f"{DATA_PATH}/unoptimized.npz"
EVAL_PATH = f"{DATA_PATH}/eval.npz"
GT_FULL_PATH = f"{DATA_PATH}/pseudo_gt_positions.npz"

import matplotlib.pyplot as plt

if __name__ == "__main__":
    pseudo_gt = np.load(GT_FULL_PATH)["arr_0"]
    predicted = np.load(EVAL_PATH)["arr_0"]

    pseudo_gt_centroids = np.mean(pseudo_gt, axis=1)
    predicted_centroids = np.mean(predicted, axis=1)

    frame_count, _, _ = predicted.shape

    paths_to_gt = [f"/media/pavlos/One Touch/datasets/pac_simulation_data/elastic/0/{f}.ply" for f in
                   range(frame_count)]

    positions_gt = []
    positions_gt_centroids = []
    scale_factor = 1.0
    for path in paths_to_gt:
        verts = load_ply_file(path)
        points = scale_factor * np.array([np.array(list(v)) for v in verts])
        positions_gt.append(points)
        positions_gt_centroids.append(np.mean(points, axis=0))
    positions_gt = np.array(positions_gt)
    positions_gt_centroids = np.array(positions_gt_centroids)

    velocities_pseudo_gt = []
    velocities_predicted = []
    velocities_gt = []

    velocities_pseudo_gt_centroids_magnitudes = []
    velocities_predicted_centroids_magnitudes = []
    velocities_gt_centroids_magnitudes = []

    for i in range(frame_count - 1):
        velocities_pseudo_gt.append(
            np.mean(pseudo_gt[i + 1] - pseudo_gt[i], axis=0)
        )
        velocities_predicted.append(
            np.mean(predicted[i + 1] - predicted[i], axis=0)
        )
        velocities_gt.append(np.mean(positions_gt[i + 1] - positions_gt[i], axis=0))

        velocities_pseudo_gt_centroids_magnitudes.append(
            np.linalg.norm(pseudo_gt_centroids[i + 1] - pseudo_gt_centroids[i]))
        velocities_predicted_centroids_magnitudes.append(
            np.linalg.norm(predicted_centroids[i + 1] - predicted_centroids[i]))
        velocities_gt_centroids_magnitudes.append(
            np.linalg.norm(positions_gt_centroids[i + 1] - positions_gt_centroids[i]))

    velocities_pseudo_gt_magnitude = [np.linalg.norm(v) for v in velocities_pseudo_gt]
    velocities_predicted_magnitude = [np.linalg.norm(v) for v in velocities_predicted]
    velocities_gt = [np.linalg.norm(v) for v in velocities_gt]

    mean_velocity = np.mean(velocities_pseudo_gt_magnitude)

    end_frame = 13
    t = [i for i in range(13)]
    mean = [mean_velocity for i in t]
    plt.figure(figsize=(10, 6))
    plt.plot(t[:end_frame], velocities_gt[:end_frame], label="Simulation Data", marker="o")
    # plt.plot(t[:end_frame], velocities_pseudo_gt_magnitude[:end_frame], label="Pseudo Ground Truth", marker="o")
    plt.plot(t[:end_frame], velocities_predicted_magnitude[:end_frame], label="Predicted", marker="o")

    # plt.plot(t[:end_frame], velocities_gt_centroids_magnitudes[:end_frame], label="Simulation Data Centroid", marker="o")
    # plt.plot(t[:end_frame], velocities_pseudo_gt_centroids_magnitudes[:end_frame], label="Pseudo Ground Truth Centroid", marker="o")
    # plt.plot(t[:end_frame], velocities_predicted_centroids_magnitudes[:end_frame], label="Predicted Centroid", marker="o")

    plt.plot(t[:end_frame], mean[:end_frame], label="Mean Pseudo Ground Truth")
    plt.xlabel("Time")
    plt.ylabel("Velocity Magnitude")
    plt.title("Velocity across Time")
    plt.legend()
    plt.grid(True)
    plt.show()
