import os
from PIL import Image
import numpy as np

paths = [
    "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/results/pacnerf_baseline/ours/elastic_0/test_cam_0"
]
output_paths = [
    "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/results/pacnerf_baseline/ours/elastic_0/test_cam_0_transparent"
]

if __name__ == "__main__":
    for i, p in enumerate(paths):
        files = os.listdir(p)
        os.makedirs(f"{p}_transparent", exist_ok=True)
        for file in files:
            output_file = f"{output_paths[i]}/{file}"
            img = np.array(Image.open(f"{p}/{file}"))
            # black_pixels = np.where(
            #     (img[:, :, 0] == 0) &
            #     (img[:, :, 1] == 0) &
            #     (img[:, :, 2] == 0)
            # )
            threshold = 100
            black_pixels = np.where(
                (img[:, :, 0] <= threshold) &
                (img[:, :, 1] <= threshold) &
                (img[:, :, 2] <= threshold)
            )
            img[black_pixels] = [255, 255, 255]
            img = Image.fromarray(img)
            img.save(output_file)