import json
import os

import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from matting import MattingRefine
from torchvision import transforms as T

import rembg

from converter_utilities import create_directory_if_not_exists

import cv2

base_path_veo = "/media/pavlos/One Touch/datasets/plush_dataset"
sequences = ["baby_alien"]
output_path = "/media/pavlos/One Touch/datasets/dynamic_plush"

# TODO: CHANGE THIS TO ARGUMENT LIKE FOR PAC
point_count = 100

for i, sequence in enumerate(sequences):
    print(f"Converting: {sequence}, sequence {i + 1}/{len(sequences)}")

    sequence_path = f"{base_path_veo}/{sequence}"
    veo_capture_path = f"{sequence_path}/capture"

    output_sequence_path = f"{output_path}/{sequence}"
    calibration_path = f"{veo_capture_path}/calibration.json"

    with open(calibration_path, 'r') as f:
        calibration_data = json.load(f)

    camera_count = 0
    cameras = {}
    for camera_id in tqdm(calibration_data):
        # TODO: SOMEHOW WE NEED TO TELL IF THE CAMERA SEES THE MARKER
        # or camera_id == "cam0001" or camera_id == "cam0008" or camera_id == "cam"
        # we skip if camera left or right
        if "." in camera_id:
            continue

        # initialize empty camera
        cameras[camera_count] = {}
        # read camera data
        camera_data = calibration_data[camera_id]
        k = camera_data['intrinsic']
        w2c = camera_data['extrinsic']

        # construct paths and directories needed
        path_to_video = f"{veo_capture_path}/{camera_id}.aligned.mp4"
        camera_path = f"{output_sequence_path}/ims/{camera_count}"
        seg_path = f"{output_sequence_path}/seg/{camera_count}"
        create_directory_if_not_exists(camera_path)
        create_directory_if_not_exists(seg_path)

        vidcap = cv2.VideoCapture(path_to_video)
        success, frame = vidcap.read()
        frame_id = 0

        while success:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # rotate as per instructions
            # save frame as JPEG file
            cv2.imwrite(f"{camera_path}/{frame_id:06d}.jpg",
                        frame)
            # get segmentation mask
            foreground = np.array(rembg.remove(frame))
            # if pixel is below transparency threshold we set it to black
            transparent_pixels = (foreground[:, :, 3] < 0.9 * 255)
            foreground[transparent_pixels] = np.array([0, 0, 0, 0])
            # get the mask
            foreground_pixels = foreground[:, :] != [0, 0, 0, 0]
            mask = np.clip(np.sum(foreground_pixels, axis=2), 0, 1)
            mask[mask > 0] = 255
            cv2.imwrite(f"{seg_path}/{frame_id:06d}.png", mask)

            # get camera data
            if frame_id not in cameras[camera_count]:
                cameras[camera_count][frame_id] = {}

            cameras[camera_count][frame_id]["k"] = k
            cameras[camera_count][frame_id]["w2c"] = w2c

            frame_id = frame_id + 1
            success, frame = vidcap.read()
            break

        camera_count += 1

    total_frames = len(cameras[0])
    total_cameras = camera_count

    # set width and height based on the first camera
    w, h = calibration_data["cam0001"]["resolution_w_h"]

    dataset_meta = {}
    dataset_meta["w"] = w
    dataset_meta["h"] = h

    ks = torch.zeros((total_frames, total_cameras, 3, 3))
    w2cs = torch.zeros((total_frames, total_cameras, 4, 4))
    cam_ids = torch.zeros((total_frames, total_cameras), dtype=int)
    fns = np.empty((total_frames, total_cameras), dtype=object)

    for frame_id in range(total_frames):
        for camera_id in range(total_cameras):
            ks[frame_id][camera_id] = torch.Tensor(cameras[camera_id][frame_id]["k"])
            w2cs[frame_id][camera_id] = torch.Tensor(cameras[camera_id][frame_id]["w2c"])
            cam_ids[frame_id][camera_id] = camera_id
            fns[frame_id][camera_id] = f"{camera_id}/{frame_id:06d}.jpg"

    dataset_meta["k"] = ks.tolist()
    dataset_meta["w2c"] = w2cs.tolist()
    dataset_meta["fn"] = fns.tolist()
    dataset_meta["cam_id"] = cam_ids.tolist()

    with open(f"{output_sequence_path}/train_meta.json", "w") as f:
        json.dump(dataset_meta, f)

    # Generate initial points by randomly sampling the unit cube, and initial colors from [0.5, 1.0]
    points = np.concatenate(
        (np.random.uniform(-1, 1, (point_count, 3)), np.random.uniform(0.5, 1.0, (point_count, 4))),
        axis=1)
    points[:, 6] = 1.0  # set alphas to 1.0
    points[:, 2] = np.random.uniform(-3, 1)
    file = f"{output_sequence_path}/init_pt_cld.npz"
    np.savez(file, data=points)
