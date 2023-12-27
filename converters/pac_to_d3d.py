import json
import os

import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from matting import MattingRefine
from torchvision import transforms as T

import cv2


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset converter from PACNerf to the Dynamic 3D Gaussians dataset")
    parser.add_argument("-i", "--input_path", help="The path where the PAC sequences are stored")
    parser.add_argument("-o", "--output_path", help="The desired output path")
    parser.add_argument("-s", "--sequences", nargs="+", type=str, help="The sequences")

    args = parser.parse_args()

    base_path_pac = args.input_path
    sequences = args.sequences
    output_path = args.output_path

    for i, sequence in enumerate(args.sequences):
        print(f"Converting: {sequence}, sequence {i + 1}/{len(args.sequences)}")
        sequence_path = f"{base_path_pac}/{sequence}"
        pac_capture_path = f"{sequence_path}/data"
        all_data = f"{sequence_path}/all_data.json"

        output_sequence_path = f"{output_path}/{sequence}"

        with open(all_data, 'r') as f:
            capture_data = json.load(f)

        # before we do anything prepare the matting model (see https://github.com/xuan-li/PAC-NeRF/blob/main/train.py)
        device = torch.device("cuda:0")
        matting_model = MattingRefine(backbone='resnet101',
                                      backbone_scale=1 / 2,
                                      refine_mode='sampling',
                                      refine_sample_pixels=100_000)
        matting_model.load_state_dict(torch.load('checkpoint/pytorch_resnet101.pth', map_location=device))
        matting_model = matting_model.eval().to(torch.float32).to(device)

        # we iterate over all images in the pac nerf data file and extract the parameters we need
        frames = {}
        highest_cam_id = -1
        highest_frame_id = -1
        for i, frame_data in enumerate(tqdm(capture_data)):
            filename = frame_data["file_path"]
            frame_name = filename[filename.index("a/r_") + 4:filename.index(".png")]
            # get the camera and frame if from the filename
            camera_id = frame_name[:frame_name.index("_")]
            frame_id = int(frame_name[frame_name.index(f"{camera_id}_") + len(camera_id) + 1:])
            if frame_id < 0:
                continue
            camera_id = int(camera_id)
            highest_frame_id = max(highest_frame_id, frame_id)
            highest_cam_id = max(highest_cam_id, camera_id)
            # create the paths we need
            camera_path = f"{output_sequence_path}/ims/{camera_id}"
            seg_path = f"{output_sequence_path}/seg/{camera_id}"
            create_directory_if_not_exists(camera_path)
            create_directory_if_not_exists(seg_path)
            full_path = f"{sequence_path}/{filename[1:]}"
            # read the frame and save it as a jpg for consistency with D3D.
            frame = Image.open(full_path)
            frame.save(f"{camera_path}/{frame_id:06d}.jpg")
            # get the camera data that we need
            w, h = frame.size
            k = frame_data["intrinsic"]
            c2w = frame_data["c2w"]
            c2w.append([0, 0, 0, 1])

            # change to OpenCV camera coordinate system
            c2w = np.array(c2w)
            c2w[0:3, 1] *= -1  # flip the y and z axis
            c2w[0:3, 2] *= -1
            c2w[2, :] *= -1  # flip whole world upside down
            reflection = np.array(
                [[1, 0, 0, 0],
                 [0, -1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

            c2w = reflection @ c2w  # reflect across x-axis

            w2c = np.linalg.inv(c2w)
            if frame_id not in frames:
                frames[frame_id] = {}
            if camera_id not in frames[frame_id]:
                frames[frame_id][camera_id] = {}
            frames[frame_id][camera_id]["k"] = k
            frames[frame_id][camera_id]["w2c"] = w2c
            # Generate the masks using the matting model segmentations
            mask = torch.zeros((h, w))
            with torch.no_grad():
                img_tensor = T.ToTensor()(frame).to(device)[None]
                bgr_image = Image.open(f"{sequence_path}/data/r_{camera_id}_-1.png")
                bgr_tensor = T.ToTensor()(bgr_image).to(device)[None]
                pha = matting_model(img_tensor, bgr_tensor)[0][0, 0].cpu().numpy()
                mask[pha >= 0.9] = 255
                cv2.imwrite(f"{seg_path}/{frame_id:06d}.png",
                            mask.numpy())  # image must be saved as one-channel to be loaded correctly

        total_frames = highest_frame_id + 1
        total_cameras = highest_cam_id + 1

        dataset_meta = {}
        dataset_meta["w"] = w
        dataset_meta["h"] = h

        ks = torch.zeros((total_frames, total_cameras, 3, 3))
        w2cs = torch.zeros((total_frames, total_cameras, 4, 4))
        cam_ids = torch.zeros((total_frames, total_cameras), dtype=int)
        fns = np.empty((total_frames, total_cameras), dtype=object)

        # w2c we need is a concatenation of everything
        # fn is a list of lists of filenames grouped by the frame number, i.e  [ ["1/000000.jpg","2/000000.jpg"], ["1/000001.jpg","2/000001.jpg"]]
        # cam_id is a list of lists with camera ids corresponding the frames - I think this is used to separate training and testing

        for frame_id in range(total_frames):
            for camera_id in range(total_cameras):
                ks[frame_id][camera_id] = torch.Tensor(frames[frame_id][camera_id]["k"])
                w2cs[frame_id][camera_id] = torch.Tensor(frames[frame_id][camera_id]["w2c"])
                cam_ids[frame_id][camera_id] = camera_id
                fns[frame_id][camera_id] = f"{camera_id}/{frame_id:06d}.jpg"

        dataset_meta["k"] = ks.tolist()
        dataset_meta["w2c"] = w2cs.tolist()
        dataset_meta["fn"] = fns.tolist()
        dataset_meta["cam_id"] = cam_ids.tolist()

        with open(f"{output_sequence_path}/train_meta.json", "w") as f:
            json.dump(dataset_meta, f)

        # Generate initial points by randomly sampling the unit cube, and initial colors from [0.5, 1.0]
        points = np.concatenate((np.random.uniform(-1, 1, (100, 3)), np.random.uniform(0.5, 1.0, (100, 4))), axis=1)
        file = f"{output_sequence_path}/init_pt_cld.npz"
        np.savez(file, data=points)
