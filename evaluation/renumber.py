import os
from PIL import Image
import numpy as np

# Run this to number Pac-Nerf ground truth data in a way  that's consistent
# with the D3D ground truth

paths = [
    "/media/pavlos/One Touch/datasets/dynamic_pac/elastic_0/white/0"
]

if __name__ == "__main__":
    for i, p in enumerate(paths):
        files = os.listdir(p)
        for file in files:
            img = Image.open(f"{p}/{file}")
            file = file[2:]
            file_id = file[file.index("_")+1:file.index(".")].zfill(5)
            img.save(f"{p}/{file_id}.png")