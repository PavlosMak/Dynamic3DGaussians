import os
import shutil

import numpy as np
import json

from tqdm import tqdm

from read_cameras import *

from PIL import Image
import torchvision.transforms as transforms


scene = "potato"

path_to_input_ims = f"/media/pavlos/One Touch/datasets/spring_gauss/real_capture/static/images/{scene}"
path_to_input_seg = f"/media/pavlos/One Touch/datasets/spring_gauss/real_capture/static/masks/{scene}"
colmap_path = f"/media/pavlos/One Touch/datasets/spring_gauss/real_capture/static/colmap/{scene}"

path_to_output = f"/media/pavlos/One Touch/datasets/dynamic_spring_gauss/{scene}_static"


def copy_and_rename_file(source_file, destination_folder, new_filename):
    """
    Copy a file to a specified destination folder with a new filename.

    Args:
        source_file (str): Path to the source file.
        destination_folder (str): Path to the destination folder.
        new_filename (str): New filename for the copied file.

    Returns:
        str: Path to the copied file.
    """
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Generate the new file path
    new_file_path = os.path.join(destination_folder, new_filename)

    # Copy and rename the file
    shutil.copy(source_file, new_file_path)

    return new_file_path


if __name__ == "__main__":

    # Make output directories
    path_to_output_ims = f"{path_to_output}/ims"
    os.makedirs(path_to_output_ims, exist_ok=True)
    path_to_output_seg = f"{path_to_output}/seg"
    os.makedirs(path_to_output_seg, exist_ok=True)

    path_to_cameras = f"{colmap_path}/cameras.bin"
    # we only have one camera
    camera = read_cameras_binary(path_to_cameras)[1]

    s = 0.16835016835

    w = int(s*camera.width) + 1
    h = int(s*camera.height) + 1

    resize_transform = transforms.Resize((h, w))

    k = np.array([[s*camera.params[0], 0, s*camera.params[2]],
                  [0, s*camera.params[1], s*camera.params[3]],
                  [0, 0, 1]])


    inputs = [path_to_input_ims, path_to_input_seg]
    outputs = [path_to_output_ims, path_to_output_seg]
    for i, path in enumerate(inputs):
        base_output = outputs[i]
        extension = "jpg" if i == 0 else "png"
        total_views = len(os.listdir(path))
        for fi, file in enumerate(tqdm(os.listdir(path))):
            dst = f"{base_output}/{fi}"
            image = Image.open(f"{path}/{file}")
            os.makedirs(dst, exist_ok=True)
            image = resize_transform(image)
            # copy_and_rename_file(f"{path}/{file}", dst, f"{str(0).zfill(6)}.{extension}")
            image.save(f"{dst}/{str(0).zfill(6)}.{extension}")

    # export points
    path_to_points = f"{colmap_path}/points3D.bin"
    points = read_points3d_binary(path_to_points)
    points_array = []
    for p in points:
        point = points[p]
        points_array.append(
            [point.xyz[0], point.xyz[1], point.xyz[2], point.rgb[0] / 255, point.rgb[1] / 255, point.rgb[2] / 255, 1.0])
    points = np.array(points_array)
    np.savez(f"{path_to_output}/init_pt_cld.npz", data=points)


    path_to_image_data = f"{colmap_path}/images.bin"
    images_data = read_images_binary(path_to_image_data)
    images_data = sorted(images_data.items(), reverse=False)
    w2cs_matrices = []

    for i, (_, im) in enumerate(images_data):
        R = im.qvec2rotmat()
        t = im.tvec
        t = np.array([
            [1, 0, 0, t[0]],
            [0, 1, 0, t[1]],
            [0, 0, 1, t[2]],
            [0, 0, 0, 1]
        ])
        R = np.vstack((R, np.zeros(3)))
        last_col = np.zeros((4, 1))
        last_col[3] = 1.0
        R = np.hstack((R, last_col))
        w2cs_matrices.append(t @ R)

    total_frames = 1
    ks = np.zeros((total_frames, total_views, 3, 3))
    w2cs = np.zeros((total_frames, total_views, 4, 4))
    cam_ids = np.zeros((total_frames, total_views), dtype=int)
    fns = np.empty((total_frames, total_views), dtype=object)

    for frame_id in range(total_frames):
        for camera_id in range(total_views):
            ks[frame_id][camera_id] = k
            w2cs[frame_id][camera_id] = w2cs_matrices[camera_id]
            cam_ids[frame_id][camera_id] = camera_id
            fns[frame_id][camera_id] = f"{camera_id}/{frame_id:06d}.jpg"

    dataset_meta = {}
    dataset_meta["w"] = w
    dataset_meta["h"] = h
    dataset_meta["k"] = ks.tolist()
    dataset_meta["w2c"] = w2cs.tolist()
    dataset_meta["fn"] = fns.tolist()
    dataset_meta["cam_id"] = cam_ids.tolist()

    with open(f"{path_to_output}/train_meta.json", "w") as f:
        json.dump(dataset_meta, f)
