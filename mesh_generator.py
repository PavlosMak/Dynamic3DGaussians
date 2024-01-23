import argparse

import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix
from tqdm import tqdm

from helpers import load_scene_data

from skimage import measure
import open3d as o3d


class VoxelGrid3D:
    """
    A wrapper around numpy to make some common voxel grid operation higher level
    """

    def __init__(self, min_point, max_point, voxel_size):
        self.voxel_size = voxel_size
        self.half_size = self.voxel_size / 2
        self.min_point = min_point
        self.max_point = max_point
        self.data = dict()
        self.is_dense = False
        self.occupancies = None
        self.voxel_count = 0
        self.x_resolution = -1
        self.y_resolution = -1
        self.z_resolution = -1

    def make_dense(self):
        x_bins, y_bins, z_bins = (self.max_point - self.min_point) / self.voxel_size
        x_bins, y_bins, z_bins = int(x_bins), int(y_bins), int(z_bins)
        self.occupancies = np.zeros((x_bins, y_bins, z_bins))
        self.voxel_count = x_bins * y_bins * z_bins
        self.is_dense = True
        self.x_resolution = x_bins
        self.y_resolution = y_bins
        self.z_resolution = z_bins

    def add_to_occupancy(self, point, value):
        i, j, k = torch.abs(point - self.min_point) / self.voxel_size
        i, j, k = int(i), int(j), int(k)
        self.occupancies[i][j][k] += value

    def get_occupancy(self, point):
        i, j, k = torch.abs(point - self.min_point) / self.voxel_size
        i, j, k = int(i), int(j), int(k)
        return self.occupancies[i][j][k]

    def add_data(self, point, data):
        assert len(point) == 3, "Expected point to be 3D"
        i, j, k = torch.abs(point - self.min_point) / self.voxel_size
        key = (int(i), int(j), int(k))
        if key not in self.data:
            self.data[key] = []
            self.voxel_count += 1
        self.data[key].append(data)

    def get_data(self, point):
        """Returns empty list if no data on the point's voxel"""
        i, j, k = torch.abs(point - self.min_point) / self.voxel_size
        key = (int(i), int(j), int(k))
        if key not in self.data:
            return []
        return self.data[key]

    def get_voxel_center_point(self, i: int, j: int, k: int):
        x = self.min_point[0] + i * self.voxel_size + self.half_size
        y = self.min_point[1] + j * self.voxel_size + self.half_size
        z = self.min_point[2] + k * self.voxel_size + self.half_size
        return torch.tensor([x, y, z])


def calculate_occupancies(seq: str, experiment: str, model_loc: str, output_file=None, l0_voxel_size=0.25,
                          l1_voxel_size=0.05) -> Meshes:
    """
    Impementation of the method from DreamGaussian by Tang, Jiaxiang, et al. 2023
    :param seq: Name of the sequence
    :param experiment: Name of the experiment
    :param model_loc: Path to the model output (parent folder of the experiment)
    :param output_file: Name of the output to store densities. Must be npz. If None then densities are not stored.
    :param l0_voxel_size: The voxel size for the first level.
    :param l1_voxel_size: The voxel size for the second level.
    :return: np.ndarray of shape x_resolution * y_resolution * z_resolution.
    """

    # Load in and preprocess the data
    scene = load_scene_data(seq, experiment, False,
                            model_loc)
    moment = scene[0][0]

    centers = moment["means3D"].cpu()
    rotations = moment["rotations"].cpu()  # quaternions
    scales = moment["scales"].cpu()  # [sx, sy, sz] vectors
    opacities = moment["opacities"].cpu()

    rotation_matrices = quaternion_to_matrix(rotations)
    scale_matrices = torch.diag_embed(scales, offset=0, dim1=-2, dim2=-1)

    # Calculate covariances so we can evaluate the Gaussians. Covariance calculated with: RS(S^TR^T) = RS(RS)^T
    rs = torch.bmm(rotation_matrices, scale_matrices)
    rs_t = torch.transpose(rs, 1, 2)
    covariances = torch.bmm(rs, rs_t)

    # Create the grids
    min_point = torch.floor(torch.min(centers, dim=0)[0])
    max_point = torch.ceil(torch.max(centers, dim=0)[0])


    l0_grid = VoxelGrid3D(min_point=min_point, max_point=max_point, voxel_size=l0_voxel_size)
    [l0_grid.add_data(center, gi) for gi, center in enumerate(centers)]

    l1_grid = VoxelGrid3D(min_point=min_point, max_point=max_point, voxel_size=l1_voxel_size)
    l1_grid.make_dense()

    # Calculate the densities

    pbar = tqdm(total=l1_grid.voxel_count)
    for i in range(l1_grid.x_resolution):
        for j in range(l1_grid.y_resolution):
            for k in range(l1_grid.z_resolution):
                voxel_center = l1_grid.get_voxel_center_point(i, j, k)
                for gi in l0_grid.get_data(voxel_center):
                    center = centers[gi]
                    cov = covariances[gi]
                    opacity = opacities[gi]
                    density = (opacity * torch.exp(
                        -0.5 * (voxel_center - center) @ torch.linalg.inv(cov) @ (voxel_center - center))).item()
                    l1_grid.add_to_occupancy(voxel_center, density)
                pbar.update(1)

    # Potentially save densities
    if output_file:
        np.savez(output_file, l1_grid.occupancies)

    return l1_grid.occupancies


def mesh_extractor(densities, iso_level):
    verts, faces, normals, values = measure.marching_cubes(densities, iso_level)

    # Create a TriangleMesh
    mesh = o3d.geometry.TriangleMesh()

    # Set vertices, faces, and normals
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh extraction from Dynamic 3D Gaussian model")
    parser.add_argument("-o", "--output", help="Path to output directory")
    parser.add_argument("-m", "--model_location", help="Location of the model")
    parser.add_argument("-e", "--experiment", help="Name of the experiment")
    parser.add_argument("-s", "--sequence", help="Name of the sequence")
    parser.add_argument("-n", "--name", help="Filename to export")

    args = parser.parse_args()

    occupancies = calculate_occupancies(args.sequence, args.experiment, args.model_location,
                                        f"/home/pavlos/Desktop/stuff/Uni-Masters/thesis/jupyters/data/volume.npz")
    mesh = mesh_extractor(occupancies, 0.1)
    o3d.io.write_triangle_mesh(f"{args.output}/{args.name}", mesh)
