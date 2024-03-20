import os

import open3d as o3d
import torch

from helpers import load_scene_data
from mesh_generation.mesh_generation import calculate_occupancies, mesh_extractor

model_location = "/media/pavlos/One Touch/datasets/gaussian_assets_output"
exp = "royal-field-365"
seq = "thinner_torus_red"

if __name__ == "__main__":
    scene_data, is_fg = load_scene_data(seq, exp, False, model_location)
    # select_indices = scene_data[0]["opacities"] < torch.mean(scene_data[0]["opacities"])
    data = scene_data[0]

    centers = data["means3D"]
    rotations = data["rotations"]
    opacities = data["opacities"]
    scales = data["scales"]

    select_indices = (opacities > 0.99).flatten()

    centers = centers[select_indices].cpu()
    rotations = rotations[select_indices].cpu()
    scales = scales[select_indices].cpu()
    opacities = opacities[select_indices].cpu()

    output_path = "/media/pavlos/One Touch/datasets/gt_generation/royal-field"
    occupancies_output = f"{output_path}/occupancies.pt"
    if os.path.isfile(occupancies_output):
        occupancies = torch.load(occupancies_output)
    else:
        occupancies = calculate_occupancies(centers, rotations, scales, opacities,
                                            output_file=occupancies_output)

    mesh = mesh_extractor(occupancies.detach().cpu().numpy(), 0.5)
    mesh_output_path = f"{output_path}/generated_mesh.obj"
    o3d.io.write_triangle_mesh(mesh_output_path, mesh)
