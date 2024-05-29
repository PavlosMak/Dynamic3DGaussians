import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim
import lpips
import json
import torch
from torchvision.transforms import Normalize

ground_truth = ["/media/pavlos/One Touch/datasets/our_baseline/torus/evaluation/0",
                "/media/pavlos/One Touch/datasets/dynamic_pac/elastic_0/evaluation/0"
                "/media/pavlos/One Touch/datasets/dynamic_pac/torus/evaluation/0"]
ours = [
    "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/results/our_baseline/ours/torus/test_cam_0",
    "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/results/pacnerf_baseline/ours/elastic_0/test_cam_0"]
pacnerfs = [
    "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/results/our_baseline/pac-nerf/torus/image",
    "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/results/pacnerf_baseline/pacnerf/0/image",
    ""
]
output_dir = "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/results"
frames = [20, 14]

loss_fn_alex = lpips.LPIPS(net='alex')


def get_psnrs(gt, our, pac):
    psnr_our = cv2.PSNR(gt, our)
    psnr_pac = cv2.PSNR(gt, pac)
    return psnr_our, psnr_pac


def get_ssims(gt, ours, pac):
    ssim_our, ssim_our_diff = ssim(gt, our, full=True, multichannel=True)
    ssim_pac, ssim_pac_diff = ssim(gt, pacnerf, full=True, multichannel=True)
    return ssim_our, ssim_our_diff, ssim_pac, ssim_pac_diff


def get_lpips(gt, ours, pac):
    transform = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    gt = transform(torch.permute(torch.Tensor(gt), [2, 0, 1]))
    ours = cv2.cvtColor(ours, cv2.COLOR_BGR2RGB)
    ours = transform(torch.permute(torch.Tensor(ours), [2, 0, 1]))
    pac = cv2.cvtColor(pac, cv2.COLOR_BGR2RGB)
    pac = transform(torch.permute(torch.Tensor(pac), [2, 0, 1]))

    lpips_ours = loss_fn_alex(gt, ours).detach().cpu().item()
    lpips_pacs = loss_fn_alex(gt, pac).detach().cpu().item()

    return lpips_ours, lpips_pacs


if __name__ == "__main__":
    results = []
    for i, sequence in enumerate(ground_truth):
        sequence_results = {}
        sequence_results["gt_path"] = sequence
        sequence_results["our_path"] = ours[i]
        sequence_results["pacnerf_path"] = pacnerfs[i]

        gt_path = f"{sequence}/00000.png"
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

        lpips_our, lpips_pac = get_lpips(gt, our, pacnerf)
        sequence_results["first_frame_lpips_ours"] = lpips_our
        sequence_results["first_frame_lpips_pac"] = lpips_pac

        frame_count = frames[i]
        psnr_our, psnr_pac = [], []
        ssim_our, ssim_pac = [], []
        lpips_our, lpips_pac = [], []
        for fi in range(frame_count):
            gt = f"{sequence}/{str(fi).zfill(5)}.png"  # format 00000
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

            lpips = get_lpips(gt, our, pacnerf)
            lpips_our.append(lpips[0])
            lpips_pac.append(lpips[1])

        psnr_our = np.mean(psnr_our)
        psnr_pac = np.mean(psnr_pac)
        ssim_our = np.mean(ssim_our)
        ssim_pac = np.mean(ssim_pac)
        lpips_our = np.mean(lpips_our)
        lpips_pac = np.mean(lpips_pac)

        sequence_results["mean_psnr_our"] = psnr_our
        sequence_results["mean_psnr_pac"] = psnr_pac
        sequence_results["mean_ssim_our"] = ssim_our
        sequence_results["mean_ssim_pac"] = ssim_pac
        sequence_results["mean_lpips_our"] = lpips_our
        sequence_results["mean_lpips_pac"] = lpips_pac

        results.append(sequence_results)
        # TODO: Overlap with the mask?
        # TODO: Chamfer distance - probably better at different spot

    print(f"Saving at {output_dir}")
    with open(f"{output_dir}/image_metrics.json", "w") as f:
        json.dump(results, f)
