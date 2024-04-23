import os
import json

def get_cam_and_frame_id(file):
    filename = file[file.index("_") + 1:file.index(".png")]
    cam_id, frame_id = filename.split("_")
    cam_id = int(cam_id)
    frame_id = int(frame_id)
    return cam_id, frame_id

if __name__ == "__main__":

    path = "/media/pavlos/One Touch/datasets/pac_data/thinner_torus_red"
    path_to_metadata = f"{path}/all_data.json"
    path_to_data = f"{path}/data"

    camera_ids = set()
    frames = set()

    for file in os.listdir(path_to_data):
        cam_id, frame_id = get_cam_and_frame_id(file)
        camera_ids.add(cam_id)
        frames.add(frame_id)
        print(cam_id, frame_id)

    print(f"Cameras: {len(camera_ids)}")
    print(f"Frames: {len(frames)}")

    f = open(path_to_metadata)
    data = json.loads(f.read())

    cam_to_c2w = {}
    cam_to_intrinsic = {}
    for entry in data:
        file = entry["file_path"]
        cam_id, frame_id = get_cam_and_frame_id(file)

        if cam_id not in cam_to_c2w:
            cam_to_c2w[cam_id] = entry["c2w"]
        if cam_id not in cam_to_intrinsic:
            cam_to_intrinsic[cam_id] = entry["intrinsic"]

    new_metadata = []
    for frame in frames:
        for cam_id in cam_to_c2w:
            entry = {}
            entry["file_path"] = f"./data/r_{cam_id}_{frame}.png"
            entry["c2w"] = cam_to_c2w[cam_id]
            entry["intrinsic"] = cam_to_intrinsic[cam_id]
            new_metadata.append(entry)

    print(new_metadata)

    with open(f"{path}/all_data.json", "w") as f:
        json.dump(new_metadata, f)
