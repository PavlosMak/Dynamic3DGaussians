import numpy as np
import json

from torchvision.io import read_video
import cv2
import os
from tqdm import tqdm


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

base_path_d3d = "/media/pavlos/One Touch/datasets/dynamic_3d_data/basketball"

file = f"{base_path_d3d}/init_pt_cld.npz"

# [ [x,y,z, rx, ry, rz, rw] ] (I think)
content = np.load(file)["data"]

# print(content)

# cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

base_path_veo = "/media/pavlos/One Touch/datasets/plush_dataset/serpentine"

veo_capture_path = f"{base_path_veo}/capture"
veo_calibration_path = f"{veo_capture_path}/calibration.json"

output_base_path = f"/media/pavlos/One Touch/datasets/dynamic_plush/serpentine"

with open(veo_calibration_path, 'r') as f:
    calibration = json.load(f)

cameras = [camera for camera in list(calibration.keys()) if "." not in camera]

camera_count = len(cameras)

for i, camera in enumerate(tqdm(cameras)):
    cam_output_dir = f"{output_base_path}/ims/{i}"
    create_directory_if_not_exists(cam_output_dir)
    filename = f"{veo_capture_path}/{camera}.aligned.mp4"
    vid = cv2.VideoCapture(filename)
    success, image = vid.read()
    count = 0
    while success:
        success, frame = vid.read()
        if success:
            frame_output = f"{cam_output_dir}/{count:06d}.jpg"
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(frame_output, frame)
            count += 1
