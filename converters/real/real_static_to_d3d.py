import os
import shutil

import numpy as np
import json

from tqdm import tqdm

from read_cameras import *

from PIL import Image
import torchvision.transforms as transforms
import rerun as rr

scene = "potato"

path_to_input_ims = f"/media/pavlos/One Touch/datasets/spring_gauss/real_capture/static/images/{scene}"
path_to_input_seg = f"/media/pavlos/One Touch/datasets/spring_gauss/real_capture/static/masks/{scene}"
colmap_path = f"/media/pavlos/One Touch/datasets/spring_gauss/real_capture/static/colmap/{scene}"

path_to_output = f"/media/pavlos/One Touch/datasets/dynamic_spring_gauss/{scene}_static"

color = [235, 155, 52]


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

    rr.init("Converter", spawn=True)

    # Make output directories
    path_to_output_ims = f"{path_to_output}/ims"
    os.makedirs(path_to_output_ims, exist_ok=True)
    path_to_output_seg = f"{path_to_output}/seg"
    os.makedirs(path_to_output_seg, exist_ok=True)

    path_to_cameras = f"{colmap_path}/cameras.bin"
    # we only have one camera
    camera = read_cameras_binary(path_to_cameras)[1]

    s = 0.16835016835
    camera.params[:] *= s
    # camera.params[2:] *= s

    w = int(s * camera.width) + 1
    h = int(s * camera.height) + 1

    resize_transform = transforms.Resize((h, w))

    k = np.array([[camera.params[0], 0, camera.params[2]],
                  [0, camera.params[1], camera.params[3]],
                  [0, 0, 1]])

    inputs = [path_to_input_ims, path_to_input_seg]
    outputs = [path_to_output_ims, path_to_output_seg]
    total_views = len(os.listdir(inputs[0]))
    for i, path in enumerate(inputs):
        base_output = outputs[i]
        extension = "jpg" if i == 0 else "png"
        for fi, file in enumerate(tqdm(os.listdir(path))):
            dst = f"{base_output}/{fi}"
            image = Image.open(f"{path}/{file}")
            os.makedirs(dst, exist_ok=True)
            image = resize_transform(image)
            image.save(f"{dst}/{str(0).zfill(6)}.{extension}")

    # export points
    path_to_points = f"{colmap_path}/points3D.bin"
    points = read_points3d_binary(path_to_points)
    # points = np.random.uniform(3.0, 4.5, size=(300, 3))
    points = np.zeros((1200, 3))
    points[:, 0] = np.random.uniform(-0.3, 0.7, size=points.shape[0])
    points[:, 1] = np.random.uniform(1, 2.5, size=points.shape[0])
    points[:, 2] = np.random.uniform(2, 3.5, size=points.shape[0])
    rr.log("initial points", rr.Points3D(points))
    points_array = []
    for p in tqdm(points):
        points_array.append([p[0], p[1], p[2], color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0])
    points = np.array(points_array)
    np.savez(f"{path_to_output}/init_pt_cld.npz", data=points)

    path_to_image_data = f"{colmap_path}/images.bin"
    images_data = read_images_binary(path_to_image_data)
    images_data = sorted(images_data.items(), reverse=False)
    w2cs_matrices = []
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    for i, (_, im) in enumerate(images_data):
        R = im.qvec2rotmat()
        Rprime = R
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
        cam_mat = t @ R
        w2cs_matrices.append(cam_mat)
        rr.log(f"cam_{i}", rr.Pinhole(
            resolution=[w, h],
            focal_length=camera.params[:2],
            principal_point=camera.params[2:],
        ))
        quat_xyzw = im.qvec[[1, 2, 3, 0]]
        rr.log(f"cam_{i}",
               rr.Transform3D(translation=im.tvec, rotation=rr.Quaternion(xyzw=quat_xyzw), from_parent=True))
        dst = f"{outputs[0]}/{i}/{str(0).zfill(6)}.jpg"
        rr.log(f"cam_{i}", rr.Image(Image.open(dst)))

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
