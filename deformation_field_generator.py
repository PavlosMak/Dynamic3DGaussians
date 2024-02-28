import argparse

import numpy as np
import point_cloud_utils as pcu
import torch

from helpers import load_scene_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh extraction from Dynamic 3D Gaussian model")
    parser.add_argument("-o", "--output", help="Path to output directory")
    parser.add_argument("-m", "--model_location", help="Location of the model")
    parser.add_argument("-e", "--experiment", help="Name of the experiment")
    parser.add_argument("-s", "--sequence", help="Name of the sequence")

    args = parser.parse_args()

    scene = load_scene_data(args.sequence, args.experiment, False,
                            args.model_location)
    frames = len(scene[0])

    voxel_size = 0.05

    # we get the bounds by considering all frames
    positions_tensor = torch.cat([frame["means3D"] for frame in scene[0]])
    min_point = (torch.floor(torch.min(positions_tensor, dim=0)[0])).cpu()
    max_point = (torch.ceil(torch.max(positions_tensor, dim=0)[0])).cpu()

    x_bins, y_bins, z_bins = (max_point - min_point) / voxel_size
    x_bins, y_bins, z_bins = int(x_bins), int(y_bins), int(z_bins)

    deformation_field = torch.zeros((frames, x_bins + 1, y_bins + 1, z_bins + 1, 3))

    first_positions = scene[0][0]["means3D"].cpu().numpy()
    opaque_indices = (scene[0][0]["opacities"] > 0.9).flatten().cpu().numpy()
    downsampled_gaussians = pcu.downsample_point_cloud_poisson_disk(first_positions[opaque_indices], 0.05, 500)

    np.savez("output/gaussian_centers.npz", first_positions[downsampled_gaussians])

    trajectories = []
    centers = []
    for t in range(0, len(scene[0]) - 1):
        curr_positions = scene[0][t]["means3D"].cpu().numpy()
        next_positions = scene[0][t + 1]["means3D"].cpu().numpy()
        trajectories.append((next_positions - curr_positions)[downsampled_gaussians])
        centers.append(curr_positions[downsampled_gaussians])
    trajectories = np.array(trajectories)
    np.savez("output/trajectories.npz", trajectories)
    np.savez("output/centers_in_time.npz", np.array(centers))
