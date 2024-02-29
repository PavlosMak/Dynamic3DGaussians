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

    output_file = f"/home/pavlos/Desktop/stuff/Uni-Masters/thesis/jupyters/data/volume_3000init.npz"
    load = True

    path_to_data = f"{args.model_location}/{args.experiment}/{args.sequence}/params.npz"
    data = np.load(path_to_data, allow_pickle=True)["arr_0"].tolist()

    centers = torch.tensor(data[0]["means3D"])
    rotations = torch.nn.functional.normalize(torch.tensor(data[0]["unnorm_rotations"])).cpu()
    scales = torch.exp(torch.tensor(data[0]["log_scales"])).cpu()
    opacities = torch.sigmoid(torch.tensor(data[0]["logit_opacities"])).cpu()

    occupancies = calculate_occupancies(centers, rotations, scales, opacities, output_file)
    np.savez(output_file, occupancies)
