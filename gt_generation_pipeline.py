from mesh_generation.mesh_generation import *

import numpy as np
import pymeshlab

import os
import torch
from tqdm import tqdm

import shutil
import subprocess

if __name__ == "__main__":
    exp = "morning-cloud-226"
    seq = "torus"

    path_to_data = f"/media/pavlos/One Touch/datasets/gaussian_assets_output/{exp}/{seq}/params.npz"
    output_path = f"/media/pavlos/One Touch/datasets/gt_generation/{exp}"
    data = np.load(path_to_data, allow_pickle=True)["arr_0"].tolist()

    overwrite_files = False
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/occupancies", exist_ok=True)
    os.makedirs(f"{output_path}/meshes", exist_ok=True)
    os.makedirs(f"{output_path}/filtered_meshes", exist_ok=True)
    os.makedirs(f"{output_path}/filtered_meshes/vertex_pcds", exist_ok=True)
    os.makedirs(f"{output_path}/filtered_meshes/registered_vertices", exist_ok=True)
    os.makedirs(f"{output_path}/target_gaussian_pcds", exist_ok=True)
    os.makedirs(f"{output_path}/gt", exist_ok=True)

    #TODO Set the path to where tou have installed BCPD
    path_to_bcpd = "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/bcpd/bcpd"

    iso_levels = [1.0] * len(data)
    target_face_num = 500
    print(f"Generating ground truth for {len(data)} frames")
    for i, frame_id in tqdm(enumerate(data)):
        print(f"Generating Frame: {i+1}/{len(data)}")
        # 1) Generate the occupancy grid
        frame = data[frame_id]
        centers = torch.tensor(frame["means3D"])
        rotations = torch.nn.functional.normalize(torch.tensor(frame["unnorm_rotations"]))
        scales = torch.exp(torch.tensor(frame["log_scales"]))
        opacities = torch.sigmoid(torch.tensor(frame["logit_opacities"]))

        opaque_mask = (opacities > 0.95).flatten()
        centers = centers[opaque_mask]
        rotations = rotations[opaque_mask]
        scales = scales[opaque_mask]
        opacities = opacities[opaque_mask]

        target_pcd_path = f"{output_path}/target_gaussian_pcds/target_{i}.txt"
        np.savetxt(target_pcd_path, centers.detach().cpu().numpy(), delimiter=",")

        occupancies_output_file = f"{output_path}/occupancies/occupancies_{i}.pt"
        if os.path.isfile(occupancies_output_file) and not overwrite_files:
            occupancies = torch.load(occupancies_output_file)
        else:
            occupancies = calculate_occupancies(centers, rotations, scales, opacities,
                                                output_file=occupancies_output_file, l1_voxel_size=0.025)
        # 2) Get the mesh
        mesh = mesh_extractor(occupancies.detach().cpu().numpy(), iso_levels[i])
        mesh_output_path = f"{output_path}/meshes/mesh_{i}.obj"
        o3d.io.write_triangle_mesh(mesh_output_path, mesh)

        # 3) Filter the mesh
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(mesh_output_path)
        ms.meshing_invert_face_orientation()
        ms.apply_coord_hc_laplacian_smoothing()
        ms.apply_coord_laplacian_smoothing()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_face_num)
        ms.save_current_mesh(f"{output_path}/filtered_meshes/filtered_mesh_{i}.obj")

        # 4) Rigidly Register the mesh to the gaussian point cloud that created it
        filtered_mesh_vertices = ms.current_mesh().vertex_matrix()
        filtered_vertices_pcd_path = f"{output_path}/filtered_meshes/vertex_pcds/source_{i}.txt"
        np.savetxt(filtered_vertices_pcd_path, filtered_mesh_vertices, delimiter=",")

        current_path = os.getcwd()
        if not os.path.isfile(
                f"{output_path}/filtered_meshes/registered_vertices/registered_source_{i}.txt") and not overwrite_files:
            os.chdir(f"{output_path}/filtered_meshes/registered_vertices")
            subprocess.call([path_to_bcpd, "-x", target_pcd_path, "-y", filtered_vertices_pcd_path, "-l1e9"])
            os.rename(f"{output_path}/filtered_meshes/registered_vertices/output_y.txt",
                      f"{output_path}/filtered_meshes/registered_vertices/registered_source_{i}.txt")
            os.chdir(current_path)

        # 5) Register the previous deformed mesh to the current one (or don't if first iteration)
        if i == 0:
            # Duplicate first frame to the other directory
            src = f"{output_path}/filtered_meshes/registered_vertices/registered_source_{i}.txt"
            dst = f"{output_path}/gt/gt_{i}.txt"
            shutil.copyfile(src, dst)
            continue

        os.chdir(f"{output_path}/gt")
        target_path = f"{output_path}/filtered_meshes/registered_vertices/registered_source_{i}.txt"
        source_path = f"{output_path}/gt/gt_{i - 1}.txt"
        subprocess.call([path_to_bcpd, "-x", target_path, "-y", source_path])
        os.rename(f"{output_path}/gt/output_y.txt",
                  f"{output_path}/gt/gt_{i}.txt")
        os.chdir(current_path)
