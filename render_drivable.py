import torch
import argparse

from torchvision.utils import save_image
from helpers import load_scene_data_from_path, params2rendervar, setup_camera
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

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

    # CUDA Logging
    print(f"Cuda available: {torch.cuda.is_available()}")
    current_device = torch.cuda.current_device()
    current_device_name = torch.cuda.get_device_name(current_device)
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"Current device name: {current_device_name}")

    args = parser.parse_args()
    scene_data, is_fg = load_scene_data_from_path(args.path, args.remove_background)

    frames = len(scene_data)

    # for now hardcoded
    w = 800
    h = 800
    k = [[965.6843872070312, 0.0, 400.0], [0.0, 965.6843872070312, 400.0], [0.0, 0.0, 1.0]]
    w2c = [[-0.9929813742637634, -4.65823825468063e-18, -0.11827089637517929, 7.127432850054195e-17],
           [-0.03442002832889557, 0.9567148089408875, 0.28898441791534424, -1.9311199761985374e-16],
           [0.11315152049064636, 0.2910270392894745, -0.949999988079071, 3.0], [0.0, 0.0, 0.0, 1.0]]
    near = 1.0
    far = 100

    cam = setup_camera(w, h, k, w2c, near, far)
    for f, params in enumerate(scene_data):
        rendervar = params  # params are already ready for rendering
        im, radius, _, = Renderer(raster_settings=cam)(**rendervar)
        save_image(im,f"output/im_{f}.png")