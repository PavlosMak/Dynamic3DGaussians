import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# Run this to number Pac-Nerf ground truth data in a way  that's consistent
# with the D3D ground truth

paths = []
for scene in ["whale"]:
    for cam in range(11):
        paths.append(f"/home/pavlos/Desktop/ground_truth_images/{scene}/{cam}")

if __name__ == "__main__":
    for i, p in enumerate(tqdm(paths)):
        files = os.listdir(p)
        for file in files:
            if "background" not in file:
                img = Image.open(f"{p}/{file}")
                file = file[:-4]
                file_id = str(int(file)).zfill(5)
                # print(file_id)
                img.save(f"{p}/{file_id}.png")
