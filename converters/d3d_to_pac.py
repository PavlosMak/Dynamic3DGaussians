import json

import numpy as np

import os
import shutil


def copy_and_rename_file(source_dir, destination_dir, old_filename, new_filename):
    source_path = os.path.join(source_dir, old_filename)
    destination_path = os.path.join(destination_dir, new_filename)
    shutil.copy(source_path, destination_path)


if __name__ == '__main__':
    input_base_path = "/media/pavlos/One Touch/datasets/gradsim_deformations/thinner_torus_red"
    output_base_path = "/media/pavlos/One Touch/datasets/pac_data"
    output_path = f"{output_base_path}/thinner_torus_red"

    os.makedirs(output_path)
    pac_image_path = f"{output_path}/data"
    os.makedirs(pac_image_path)

    ims_path = f"{input_base_path}/ims"
    metadata_path = f"{input_base_path}/train_meta.json"

    camera_paths = os.listdir(ims_path)
    camera_count = len(camera_paths)

    # Copy images and change directory structure
    for cam_id, camera in enumerate(camera_paths):
        base_camera_path = f"{ims_path}/{camera}"
        image_names = os.listdir(base_camera_path)
        for image_name in image_names:
            if image_name == "background.png":
                image_id = -1
            else:
                image_id = image_name.strip(".png")
                image_id = int(image_id)
            copy_and_rename_file(base_camera_path, pac_image_path, image_name, f"r_{cam_id}_{image_id}")

        # Create training data json
        with open(metadata_path) as f:
            d3d_metadata = json.load(f)

        # Indexing with frame, camera_id
        w2cs = d3d_metadata["w2c"]
        ks = d3d_metadata["k"]

        data = []
        for image_name in os.listdir(pac_image_path):
            im = image_name.split("_")
            camera_id = int(im[1])
            frame_id = int(im[2])
            file_path = f"./data/{image_name}.png"
            time = -1
            intrinsic = ks[frame_id][camera_id]
            c2w = np.linalg.inv(np.array(w2cs[frame_id][camera_id])).tolist()
            data.append({
                "file_path": file_path,
                "time": time,
                "c2w": c2w,
                "intrinsic": intrinsic
            })

        json.dump(data, open(f"{output_path}/all_data.json", "w"))
