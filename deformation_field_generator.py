import argparse

import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix
from tqdm import tqdm

from helpers import load_scene_data

from skimage import measure
import open3d as o3d

from scipy.ndimage import gaussian_filter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh extraction from Dynamic 3D Gaussian model")
    parser.add_argument("-o", "--output", help="Path to output directory")
    parser.add_argument("-m", "--model_location", help="Location of the model")
    parser.add_argument("-e", "--experiment", help="Name of the experiment")
    parser.add_argument("-s", "--sequence", help="Name of the sequence")
    parser.add_argument("-n", "--name", help="Filename to export")

    args = parser.parse_args()

    scene = load_scene_data(args.sequence, args.experiment, False,
                            args.model_location)
    frames = len(scene[0])

    voxel_size = 0.05

    # we get the bounds by considering all frames
    positions_tensor = torch.cat([frame["means3D"] for frame in scene[0]])
    min_point = (torch.floor(torch.min(positions_tensor, dim=0)[0]) - 1).cpu()
    max_point = (torch.ceil(torch.max(positions_tensor, dim=0)[0]) + 1).cpu()

    x_bins, y_bins, z_bins = (max_point - min_point) / voxel_size
    x_bins, y_bins, z_bins = int(x_bins), int(y_bins), int(z_bins)

    deformation_field = torch.zeros((frames, x_bins + 1, y_bins + 1, z_bins + 1, 3))

    for t in tqdm(range(frames - 1)):
        positions_current = scene[0][t]["means3D"].cpu()
        positions_next = scene[0][t + 1]["means3D"].cpu()
        deformation_gradient = positions_next - positions_current
        spatial_indices = ((positions_current - min_point) / voxel_size).int()
        for pos_index, index in enumerate(spatial_indices):
            i, j, k = index.cpu()
            deformation_field[t][i][j][k] = deformation_gradient[pos_index]

    np.savez("output/deformations.npz", deformation_field)
