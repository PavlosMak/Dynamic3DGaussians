from helpers import load_scene_data
import torch
import numpy as np

model_location = "/media/pavlos/One Touch/datasets/gaussian_assets_output"
exp = "glamorous-valley-324"
seq = "thinner_torus"

if __name__ == "__main__":
    scene_data, is_fg = load_scene_data(seq, exp, False, model_location)
    # select_indices = scene_data[0]["opacities"] < torch.mean(scene_data[0]["opacities"])
    select_indices = (scene_data[0]["opacities"] > 0.998).flatten()
    positions = []
    for data in scene_data:
        centers = data["means3D"][select_indices]
        centers = centers.detach().cpu().numpy()
        positions.append(centers)
    positions = np.array(positions)
    np.savez("/media/pavlos/One Touch/datasets/gt_generation/glamorous-valley/pseudo_gt_positions.npz",positions)