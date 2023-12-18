import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from matting import MattingRefine
from torchvision import transforms as T
from torchvision.utils import save_image

import cv2

import struct
import collections


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


# method adapted from https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = []
    print(f"Reading from {path_to_model_file}")
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D.append(Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs))
    return points3D


base_path_d3d = "/media/pavlos/One Touch/datasets/dynamic_3d_data/basketball"

base_path_pac = "/media/pavlos/One Touch/datasets/pac_data/bird"

pac_capture_path = f"{base_path_pac}/data"
all_data = f"{base_path_pac}/all_data.json"

output_base_path = f"/media/pavlos/One Touch/datasets/dynamic_pac/bird"

with open(all_data, 'r') as f:
    capture_data = json.load(f)

# before we do anything prepare the matting model as seen in https://github.com/xuan-li/PAC-NeRF/blob/main/train.py
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
    camera_path = f"{output_base_path}/ims/{camera_id}"
    seg_path = f"{output_base_path}/seg/{camera_id}"
    create_directory_if_not_exists(camera_path)
    create_directory_if_not_exists(seg_path)
    full_path = f"{base_path_pac}/{filename[1:]}"
    # read the frame and save it as a jpg for consistency with D3D.
    frame = Image.open(full_path)
    frame.save(f"{camera_path}/{frame_id:06d}.jpg")
    # get the camera data that we need
    w, h = frame.size
    k = frame_data["intrinsic"]
    c2w = frame_data["c2w"]
    c2w.append([0, 0, 0, 1])
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
        bgr_image = Image.open(f"{base_path_pac}/data/r_{camera_id}_-1.png")
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

# w2c we need is a concatentaion of everything
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

with open(f"{output_base_path}/train_meta.json", "w") as f:
    json.dump(dataset_meta, f)

# For now the points need to be generated manually
# TODO: Find better way to get the points
path_to_colmap_points = "/media/pavlos/One Touch/datasets/dynamic_pac/bird_first_frame/dense/0/sparse/points3D.bin"
points3D = read_points3D_binary(path_to_colmap_points)
points = np.array(
    [[point.xyz[0], point.xyz[1], point.xyz[2], point.rgb[0], point.rgb[1], point.rgb[2], 1] for point in points3D])

file = f"{output_base_path}/init_pt_cld.npz"
np.savez(file, data=points)
