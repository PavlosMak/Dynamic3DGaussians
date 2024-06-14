from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys


def open_image(image_path):
    return Image.open(image_path).convert("RGB")


def squared_difference(image1, image2):
    arr1 = np.array(image1, dtype=np.float32)
    arr2 = np.array(image2, dtype=np.float32)

    # Ensure both images are of the same size
    if arr1.shape != arr2.shape:
        raise ValueError("Images must have the same dimensions")

    # Calculate the squared difference
    diff = (arr1 - arr2) ** 2
    diff_sum = np.sum(diff, axis=2)  # Summing along the color channels to get a single value per pixel

    # Normalize the values to be in range 0-1 for applying colormap
    norm_diff = diff_sum / np.max(diff_sum)

    return norm_diff


def apply_heatmap(diff, colormap='viridis'):
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(diff)[:, :, :3]  # Apply the colormap and ignore the alpha channel
    heatmap_img = (heatmap * 255).astype(np.uint8)
    return Image.fromarray(heatmap_img)


def save_image(image, output_path):
    image.save(output_path)


if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print("Usage: python script.py <image1_path> <image2_path> <output_path>")
    #     sys.exit(1)

    image1_path = "/home/pavlos/Desktop/ground_truth_images/ball/0/00005.png"
    image2_path = "/home/pavlos/Desktop/results/our_baseline/ours/ball_gt_mesh_params/test_cam_0/5.png"
    output_path = "/home/pavlos/Desktop/Dynamic3DGaussians/output/diff_ours_gt_mesh_recovered_params.png"

    image1 = open_image(image1_path)
    image2 = open_image(image2_path)

    diff = squared_difference(image1, image2)
    heatmap_image = apply_heatmap(diff)
    save_image(heatmap_image, output_path)

    print(f"Output image saved to {output_path}")
