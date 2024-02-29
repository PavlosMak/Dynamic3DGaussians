import tqdm
from pycpd import DeformableRegistration
import numpy as np
import point_cloud_utils as pcu
from numba import jit
import time
import os
import argparse
from collections import defaultdict

import pywavefront

from scipy.spatial import KDTree

import torch
from tqdm import tqdm


class AnimatedMesh:

    def __init__(self, path_to_mesh: str):
        # t = load_obj(path_to_mesh)
        scene = pywavefront.Wavefront(path_to_mesh, collect_faces=True)
        assert len(scene.meshes) == 1
        mesh = list(scene.meshes.values())[0]
        self.faces = mesh.faces
        self.frames = [torch.tensor(scene.vertices)]
        self.neighbors = defaultdict(set)
        for fi, face in enumerate(self.faces):
            v0, v1, v2 = face
            self.neighbors[v0].add(v1)
            self.neighbors[v0].add(v2)
            self.neighbors[v1].add(v0)
            self.neighbors[v1].add(v2)
            self.neighbors[v2].add(v0)
            self.neighbors[v2].add(v1)
        self.initial_diff_coords = self.get_differential_coords(0)

    def get_frame(self, frame_id):
        return self.frames[frame_id]

    def add_frame(self, vertices: np.ndarray):
        self.frames.append(vertices)

    def _export_obj(self, vertices, faces, filepath):
        """
        Export vertices and faces to a Wavefront OBJ file.

        Args:
        - vertices (list of tuples): List of vertex coordinates, each tuple representing (x, y, z) coordinates.
        - faces (list of tuples): List of faces, where each tuple contains the indices of vertices forming a face.
        - filepath (str): Filepath to save the OBJ file.
        """
        with open(filepath, 'w') as f:
            # Write vertices
            for vertex in vertices:
                f.write('v {} {} {}\n'.format(vertex[0], vertex[1], vertex[2]))

            # Write faces
            for face in faces:
                # Increment indices by 1 as OBJ format indices start from 1
                f.write('f {} {} {}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))

    def get_differential_coords(self, frame: int):
        # diff_coords = torch.zeros_like(self.frames[0])
        diff_coords = []
        for i, v in enumerate(self.neighbors):
            valence = len(self.neighbors[v])
            point = self.frames[0][v]
            d = torch.zeros(3)
            for n in self.neighbors[v]:
                d += point - self.frames[frame][n]
            d /= valence
            diff_coords.append(d)
        return torch.stack(diff_coords)

    def optimize_frame(self, frame: int):
        positions = torch.nn.Parameter(self.frames[frame])
        optimizer = torch.optim.Adam([positions], lr=0.3)
        self.frames[frame] = positions
        for t in tqdm(range(200)):
            y = self.get_differential_coords(frame)
            loss = 0.5 * torch.sum((y - self.initial_diff_coords) ** 2)
            print(f"Loss: {loss.item()}")
            loss.backward()
            optimizer.step()

    def export_frames(self, output_path: str, mesh_name: str):
        output_path = f"{output_path}/{mesh_name}"
        for i, frame in enumerate(self.frames):
            self._export_obj(frame, self.faces, f"{output_path}_{i}.obj")


@jit(forceobj=True)
def get_correspondences(source: np.ndarray, target: np.ndarray):
    """
    target: Current frame
    source: Next frame
    """
    reg = DeformableRegistration(X=target, Y=source, beta=1.5)
    result = reg.register()
    correspondeces = np.argmax(reg.P, axis=1)
    return {s_ix: target_ix for s_ix, target_ix in enumerate(correspondeces)}, reg, result


def get_deformation_vectors(source, target, correspondences):
    deformations = np.zeros((len(source), 3))
    for s_ix, src_point in enumerate(source):
        t_ix = correspondences[s_ix]
        deformations[s_ix] = target[t_ix] - src_point
    return deformations


def radius_outlier_removal(pcd: np.ndarray, radius=0.04, n=10):
    kdtree = KDTree(pcd)
    to_remove = []
    for i, point in enumerate(pcd):
        ns = kdtree.query_ball_point(point, r=radius)
        if len(ns) < n:
            to_remove.append(i)
    return np.delete(pcd, to_remove, axis=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dynamic 3D Gaussian Training")
    parser.add_argument("-o", "--output", help="Path to output directory", default="./output")
    parser.add_argument("-d", "--data", help="Path to data directory", default="./data")
    parser.add_argument("-n", "--num_samples", type=int, help="Samples to draw when downscaling",
                        default=1000)
    parser.add_argument("-e", "--exp", type=str, help="The experiment name")
    parser.add_argument("-s", "--seq", nargs="+", type=str, help="The sequence names")

    args = parser.parse_args()

    path_to_mesh = "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/generated_meshes/torus/torus_higher_res_small.obj"
    mesh = AnimatedMesh(path_to_mesh)

    print("Running")
    for seq in args.seq:
        path = f"{args.data}/{args.exp}/{seq}/params.npz"

        output_path = f"{args.output}/{args.exp}/{seq}"
        os.makedirs(output_path, exist_ok=True)

        data = np.load(path, allow_pickle=True)["arr_0"].tolist()
        frame_count = len(data)

        indices = []
        for frame in range(frame_count):
            indices.append(pcu.downsample_point_cloud_poisson_disk(data[frame]["means3D"], -1, args.num_samples))

        cross_frame_correspondences = []
        start_total = time.time()
        frame_count = 2
        for frame in range(frame_count - 1):
            print(f"Registering {frame} to {frame + 1}")
            source = np.copy(mesh.get_frame(frame))
            # source = data[frame]["means3D"][indices[frame]]
            np.savez(f"{output_path}/test.npz", source)
            target = data[frame + 5]["means3D"][indices[frame + 5]]
            target = radius_outlier_removal(target)
            np.savez(f"{output_path}/target.npz", target)
            start_local = time.time()
            correspondences, reg, reg_results = get_correspondences(source, target)
            deformations = get_deformation_vectors(source, target, correspondences)
            # reg.update_transform()
            # new = reg.transform_point_cloud(mesh.get_frame(frame))
            mesh.add_frame(torch.tensor(source + deformations))
            print(f"Optimizing")
            mesh.optimize_frame(frame + 1)
            np.savez(f"{output_path}/output.npz", source + deformations)
            end_local = time.time()
            print(f"Frame registration took: {end_local - start_local} seconds")
        end_total = time.time()
        print(f"Done registering - took {end_total - start_total} seconds")

        mesh.export_frames(output_path, "torus")
