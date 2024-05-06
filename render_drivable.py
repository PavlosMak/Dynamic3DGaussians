import os

import torch
import argparse
import json

from tqdm import tqdm

from torchvision.utils import save_image
from helpers import load_scene_data_from_path, setup_camera
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

import timeit

import cv2
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic 3D Gaussian Visualizing")
    parser.add_argument("-p", "--path", help="Path", default="")
    parser.add_argument("-rm", "--render_mode", help="The rendering mode. Select from 'color', 'depth', 'centers'.",
                        default="color")
    parser.add_argument("-l", "--lines",
                        help="The additional lines to be shown. Select from none, 'trajectories' and `rotations`.",
                        default=None)
    parser.add_argument("-rb", "--remove_background", action="store_true",
                        help="Controls whether the background should be removed",
                        default=False)
    parser.add_argument("-f", "--force_loop", action="store_true", help="Controls whether to force the loop")
    parser.add_argument("-m", "--model_location", help="The location of the model to be visualized.",
                        default="./output")
    parser.add_argument("-s", "--sequences", nargs="+", type=str, help="The sequence names")
    parser.add_argument("-pt", "--path_to_test", type=str,
                        help="The path to the path where the test_cameras.json file is.")
    parser.add_argument("-o", "--output_path", type=str, help="The path where the output is stored")

    # CUDA Logging
    print(f"Cuda available: {torch.cuda.is_available()}")
    current_device = torch.cuda.current_device()
    current_device_name = torch.cuda.get_device_name(current_device)
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"Current device name: {current_device_name}")

    args = parser.parse_args()
    scene_data, is_fg = load_scene_data_from_path(args.path, args.remove_background)
    frames = len(scene_data)

    path_to_test_cameras = f"{args.path_to_test}/test_cameras.json"
    test_cameras_file = open(path_to_test_cameras)
    test_cameras = json.load(test_cameras_file)

    base_output_path = args.output_path

    render_data = []
    for i, camera in enumerate(test_cameras):
        print(f"Camera {i + 1} / {len(test_cameras)}")
        cam_id = camera["id"]
        w = camera["w"]
        h = camera["h"]
        k = camera["k"]
        near = camera["near"]
        far = camera["far"]
        w2c = camera["w2c"]
        cam = setup_camera(w, h, k, w2c, near, far, bg=[1, 1, 1])

        start_time = timeit.default_timer()

        camera_path = f"{base_output_path}/test_cam_{cam_id}"
        os.makedirs(camera_path, exist_ok=True)

        images = []
        for f, params in enumerate(tqdm(scene_data)):
            im, radius, _, = Renderer(raster_settings=cam)(**params)
            images.append((f"{camera_path}/{f}.png", im))
        elapsed = timeit.default_timer() - start_time
        seconds = elapsed
        fps = frames / elapsed
        data = {"fps": fps, "seconds": seconds, "frames": frames, "cam_id": cam_id}
        render_data.append(data)
        white_background = torch.ones_like(images[0][1])
        width, height = 400, 400
        bgcolor = [0, 0, 0]
        for path, im in images:
            # im = im.view(width, height, 3).cpu()
            # im = im
            im = im.cpu()
            # cv2_image = np.transpose(im, (1, 2, 0))
            # cv2_image[np.all(cv2_image == bgcolor, axis=2)] = [255, 255, 255]
            # cv2_image[cv2_image[:, :] == [0, 0, 0]] = 255
            # cv2_image[cv2_image[:, :] != [0, 0, 0]] *= 255
            # cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            save_image(im, path)
            # cv2.imwrite(path, cv2_image)
    output_file_path = f"{base_output_path}/render_data.json"
    with open(output_file_path, "w") as output_file:
        json.dump(render_data, output_file)
