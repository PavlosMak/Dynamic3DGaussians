from helpers import load_scene_data
import torch
import numpy as np

model_location = "/media/pavlos/One Touch/datasets/gaussian_assets_output"
run_name = "young-deluge"
exp = "young-deluge-628"
seq = "cow"

if __name__ == "__main__":
    scene_data, is_fg = load_scene_data(seq, exp, False, model_location)
    # select_indices = scene_data[0]["opacities"] < torch.mean(scene_data[0]["opacities"])
    select_indices = (scene_data[0]["opacities"] > 0.).flatten()
    positions = []
    for data in scene_data:
        centers = data["means3D"][select_indices]
        centers = centers.detach().cpu().numpy()
        positions.append(centers)
    positions = np.array(positions)
    print(f"Points {len(positions[0])}")

    output_mesh = True

    output_folder = f"/media/pavlos/One Touch/datasets/gt_generation/{run_name}"
    np.savez(f"{output_folder}/pseudo_gt_positions.npz", positions)

    if output_mesh:
        with open(f"{output_folder}/first_frame_dense.obj", "w") as f:
            for point in positions[0]:
                f.write(f"v {point[0]} {point[1]} {point[2]}\n")
    else:
        output_folder = f"/media/pavlos/One Touch/datasets/gt_generation/{run_name}/trajectories"
        for fi, frame_positions in enumerate(positions):
            output_path = f"{output_folder}/target_{fi}.txt"
            np.savetxt(output_path, frame_positions, delimiter=",")