import os
from PIL import Image
import numpy as np

paths = [
    "/media/pavlos/One Touch/datasets/our_baseline/torus_recalibrated/evaluation/0"
]

if __name__ == "__main__":
    for i, p in enumerate(paths):
        files = os.listdir(p)
        bg = Image.open(f"{p}/background.png").convert('RGBA')
        for file in files:
            if "background" in file:
                continue
            img = Image.open(f"{p}/{file}").convert('RGBA')
            img = Image.alpha_composite(bg, img)
            img.save(f"{p}/{file}")