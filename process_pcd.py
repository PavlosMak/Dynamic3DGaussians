import os

import open3d as o3d
import torch

import point_cloud_utils as pcu
from helpers import load_scene_data

model_location = "/media/pavlos/One Touch/datasets/gaussian_assets_output"
exp = "magic-salad-372"
seq = "thinner_torus_red_many_frames"

if __name__ == "__main__":
    scene_data, is_fg = load_scene_data(seq, exp, False, model_location)
    # select_indices = scene_data[0]["opacities"] < torch.mean(scene_data[0]["opacities"])
    data = scene_data[0]

    centers = data["means3D"]
    rotations = data["rotations"]
    opacities = data["opacities"]
    scales = data["scales"]

    select_indices = (opacities > 0.998).flatten()

    centers = centers[select_indices].cpu()
    rotations = rotations[select_indices].cpu()
    scales = scales[select_indices].cpu()
    opacities = opacities[select_indices].cpu()

    output_path = "/media/pavlos/One Touch/datasets/gt_generation/magic-salad/meshing"

    radius = 0.01
    v = centers.numpy()
    idx = pcu.downsample_point_cloud_poisson_disk(v, radius, target_num_samples=2000)

    with open(f"{output_path}/subsampled_poisson_2000.obj", "w") as f:
        for point in centers[idx]:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")