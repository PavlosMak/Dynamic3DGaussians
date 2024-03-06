import numpy as np
import open3d as o3d
import torch
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix
from skimage import measure
from tqdm import tqdm

from scipy.spatial import KDTree

from mesh_generation.Voxel3D import VoxelGrid3D


def calculate_occupancies_kd(centers, rotations, scales, opacities, output_file=None, radius=0.1,
                             l1_voxel_size=0.025) -> Meshes:
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
    rotation_matrices = quaternion_to_matrix(rotations)
    scale_matrices = torch.diag_embed(scales, offset=0, dim1=-2, dim2=-1)

    # Calculate covariances so we can evaluate the Gaussians. Covariance calculated with: RS(S^TR^T) = RS(RS)^T
    rs = torch.bmm(rotation_matrices, scale_matrices)
    rs_t = torch.transpose(rs, 1, 2)
    covariances = torch.bmm(rs, rs_t)

    # Create the grids
    min_point = torch.floor(torch.min(centers, dim=0)[0]) - 1
    max_point = torch.ceil(torch.max(centers, dim=0)[0]) + 1

    l1_grid = VoxelGrid3D(min_point=min_point, max_point=max_point, voxel_size=l1_voxel_size)
    l1_grid.make_dense()

    # Calculate the densities

    tree = KDTree(centers.clone().detach().cpu().numpy())

    pbar = tqdm(total=l1_grid.voxel_count)
    for i in range(l1_grid.x_resolution):
        for j in range(l1_grid.y_resolution):
            for k in range(l1_grid.z_resolution):
                voxel_center = l1_grid.get_voxel_center_point(i, j, k)
                indices = tree.query_ball_point(voxel_center.numpy(), r=radius)
                for gi in indices:
                    center = centers[gi]
                    cov = covariances[gi]
                    opacity = opacities[gi]
                    density = (opacity * torch.exp(
                        -0.5 * (voxel_center - center) @ torch.linalg.inv(cov) @ (voxel_center - center))).item()
                    l1_grid.add_to_occupancy(voxel_center, density)
                pbar.update(1)

    # Potentially save densities
    if output_file:
        torch.save(l1_grid.occupancies, output_file)

    return l1_grid.occupancies


def calculate_occupancies(centers, rotations, scales, opacities, output_file=None, l0_voxel_size=0.1,
                          l1_voxel_size=0.025) -> Meshes:
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
    rotation_matrices = quaternion_to_matrix(rotations)
    scale_matrices = torch.diag_embed(scales, offset=0, dim1=-2, dim2=-1)

    # Calculate covariances so we can evaluate the Gaussians. Covariance calculated with: RS(S^TR^T) = RS(RS)^T
    rs = torch.bmm(rotation_matrices, scale_matrices)
    rs_t = torch.transpose(rs, 1, 2)
    covariances = torch.bmm(rs, rs_t)

    # Create the grids
    min_point = torch.floor(torch.min(centers, dim=0)[0]) - 1
    max_point = torch.ceil(torch.max(centers, dim=0)[0]) + 1

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
        torch.save(l1_grid.occupancies, output_file)

    return l1_grid.occupancies


# Currently works only on numpy arrays
def mesh_extractor(occupancy_volume, iso_level, geometry_to_origin=True):
    verts, faces, normals, values = measure.marching_cubes(occupancy_volume, iso_level)
    if geometry_to_origin:
        centroid = np.mean(verts, axis=0)
        verts -= centroid

    # Create a TriangleMesh
    mesh = o3d.geometry.TriangleMesh()

    # Set vertices, faces, and normals
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    return mesh
