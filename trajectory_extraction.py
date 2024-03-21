from helpers import load_scene_data
import torch
import numpy as np

model_location = "/media/pavlos/One Touch/datasets/gaussian_assets_output"
exp = "royal-field-365"
seq = "thinner_torus_red"

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

    output_mesh = False

    if output_mesh:
        output_folder = "/media/pavlos/One Touch/datasets/gt_generation/royal-field"
        np.savez(f"{output_folder}/pseudo_gt_positions_verify.npz", positions)
        with open(f"{output_folder}/first_frame.obj", "w") as f:
            for point in positions[0]:
                f.write(f"v {point[0]} {point[1]} {point[2]}\n")
    else:
        output_folder = "/media/pavlos/One Touch/datasets/gt_generation/royal-field/trajectories"
        for fi, frame_positions in enumerate(positions):
            output_path = f"{output_folder}/target_{fi}.txt"
            np.savetxt(output_path, frame_positions, delimiter=",")