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

from mesh_generation.mesh_generation import calculate_occupancies

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh extraction from Dynamic 3D Gaussian model")
    parser.add_argument("-o", "--output", help="Path to output directory")
    parser.add_argument("-m", "--model_location", help="Location of the model")
    parser.add_argument("-e", "--experiment", help="Name of the experiment")
    parser.add_argument("-s", "--sequence", help="Name of the sequence")
    parser.add_argument("-n", "--name", help="Filename to export")

    args = parser.parse_args()

    # TODO: Clean this up - Potentially extract the entire thing to a different repo
    output_file = f"/home/pavlos/Desktop/stuff/Uni-Masters/thesis/jupyters/data/volume_3000init.npz"
    load = True
    if load:

        # Load in and preprocess the data
        scene = load_scene_data(args.sequence, args.experiment, False,
                                args.model_location)
        moment = scene[0][0]

        centers = moment["means3D"].cpu()
        rotations = moment["rotations"].cpu()  # quaternions
        scales = moment["scales"].cpu()  # [sx, sy, sz] vectors
        opacities = moment["opacities"].cpu()

        occupancies = calculate_occupancies(centers, rotations, scales, opacities,
                                            output_file)
        np.savez(output_file, occupancies)
    else:
        occupancies = np.load(output_file)['arr_0']

    occupancies_smoothed = gaussian_filter(occupancies, sigma=1.0)
    np.savez(f"/home/pavlos/Desktop/stuff/Uni-Masters/thesis/jupyters/data/volume_3000init_smoothed.npz",
             occupancies_smoothed)
