import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim
import json

ground_truth = ["/media/pavlos/One Touch/datasets/our_baseline/torus/white/0"]
ours = ["/home/pavlos/Desktop/stuff/Uni-Masters/thesis/results/our_baseline/ours/torus/test_cam_0_white"]
pacnerfs = ["/home/pavlos/Desktop/stuff/Uni-Masters/thesis/results/our_baseline/pac-nerf/torus/image"]
results = ["/home/pavlos/Desktop/stuff/Uni-Masters/thesis/results/our_baseline/torus"]
output_dir = "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/results"
frames = [20]


def get_psnrs(gt, our, pac):
    psnr_our = cv2.PSNR(gt, our)
    psnr_pac = cv2.PSNR(gt, pac)
    return psnr_our, psnr_pac


def get_ssims(gt, ours, pac):
    ssim_our, ssim_our_diff = ssim(gt, our, full=True, multichannel=True)
    ssim_pac, ssim_pac_diff = ssim(gt, pacnerf, full=True, multichannel=True)
    return ssim_our, ssim_our_diff, ssim_pac, ssim_pac_diff


if __name__ == "__main__":
    results = []
    for i, sequence in enumerate(ground_truth):
        sequence_results = {}
        sequence_results["gt_path"] = sequence
        sequence_results["our_path"] = ours[i]
        sequence_results["pacnerf_path"] = pacnerfs[i]

        gt_path = f"{sequence}/000000.png"
        our = f"{ours[i]}/0.png"
        pacnerf = f"{pacnerfs[i]}/000.png"

        gt = cv2.imread(gt_path)
        our = cv2.imread(our)
        pacnerf = cv2.imread(pacnerf)

        psnr_our, psnr_pac = get_psnrs(gt, our, pacnerf)
        sequence_results["first_frame_psnr_ours"] = psnr_our
        sequence_results["first_frame_psnr_pac"] = psnr_pac

        ssim_our, _, ssim_pac, _ = get_ssims(gt, our, pacnerf)
        sequence_results["first_frame_ssim_ours"] = ssim_our
        sequence_results["first_frame_ssim_pac"] = ssim_pac

        frame_count = frames[i]
        psnr_our, psnr_pac = [], []
        ssim_our, ssim_pac = [], []
        for fi in range(frame_count):
            gt = f"{sequence}/{str(fi).zfill(6)}.png"  # format 000000
            our = f"{ours[i]}/{fi}.png"
            pacnerf = f"{pacnerfs[i]}/{str(fi).zfill(3)}.png"  # format 000

            gt = cv2.imread(gt)
            our = cv2.imread(our)
            pacnerf = cv2.imread(pacnerf)

            psnrs = get_psnrs(gt, our, pacnerf)
            psnr_our.append(psnrs[0])
            psnr_pac.append(psnrs[1])

            ssims = get_ssims(gt, our, pacnerf)
            ssim_our.append(ssims[0])
            ssim_pac.append(ssims[2])

        psnr_our = np.mean(psnr_our)
        psnr_pac = np.mean(psnr_pac)
        ssim_our = np.mean(ssim_our)
        ssim_pac = np.mean(ssim_pac)

        sequence_results["mean_psnr_our"] = psnr_our
        sequence_results["mean_psnr_pac"] = psnr_pac
        sequence_results["mean_ssim_our"] = ssim_our
        sequence_results["mean_ssim_pac"] = ssim_pac

        results.append(sequence_results)
        # TODO: Overlap with the mask?
        # TODO: Chamfer distance - probably better at different spot

    with open(f"{output_dir}/image_metrics.json", "w") as f:
        json.dump(results, f)
