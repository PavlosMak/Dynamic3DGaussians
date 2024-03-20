import argparse
import copy
import json
import os
from random import randint

import wandb

import numpy as np
import torch
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from tqdm import tqdm

from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer, remove_transparent
from helpers import *


def get_dataset(t, md, seq, data_dir: str):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        np_im = np.array(copy.deepcopy(Image.open(f"{data_dir}/{seq}/ims/{fn}")))
        im = torch.tensor(np_im).float().cuda().permute(2, 0, 1) / 255
        segmented_image = torch.tensor(np_im).float() / 255
        seg = np.array(copy.deepcopy(Image.open(f"{data_dir}/{seq}/seg/{fn.replace('.jpg', '.png')}"))).astype(
            np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        # TODO: There must be a better way to do the masking
        segmented_image[seg == 0.0] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        segmented_image = torch.permute(segmented_image, (2, 0, 1)).cuda()
        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c, 'seg_im': segmented_image})
    return dataset


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data, todo_dataset


def initialize_params(seq, md, data_dir: str):
    init_pt_cld = np.load(f"{data_dir}/{seq}/init_pt_cld.npz")["data"]
    seg = init_pt_cld[:, 6]
    max_cams = 50
    sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    params = {
        'means3D': init_pt_cld[:, :3],
        'rgb_colors': init_pt_cld[:, 3:6],
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
        'logit_opacities': np.zeros((seg.shape[0], 1)),
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        'cam_m': np.zeros((max_cams, 3)),
        'cam_c': np.zeros((max_cams, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))

    min_point = np.min(init_pt_cld[:, :3], axis=0)
    max_point = np.max(init_pt_cld[:, :3], axis=0)
    half_extends = 0.5 * (max_point - min_point)
    scene_radius = np.max(half_extends)

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'split_into': 2}
    return params, variables


def initialize_optimizer(params, variables):
    lrs = {
        # 'means3D': 0.00016 * variables['scene_radius'],
        'means3D': 0.001 * variables['scene_radius'],
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_loss(params, curr_data, variables, is_initial_timestep, use_entropy_loss=False):
    losses = {}

    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    # losses['im'] = 0.8 * l1_loss_v1(im, curr_data['seg_im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['seg_im']))
    losses['im'] = l1_loss_masked(im, curr_data['seg_im'], curr_data['seg'][0, :])
    # losses['im'] = l2_loss_masked(im, curr_data['seg_im'], curr_data['seg'][0, :])

    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))

    is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
    if not is_initial_timestep:
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]

        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
        losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                              variables["neighbor_weight"])

        losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None],
                                            variables["neighbor_weight"])

        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

        losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()

        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]
        losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])

    loss_weights = {'im': 70.0, 'seg': 3.0, 'rigid': 15.0, 'rot': 4.0, 'iso': 20, 'floor': 2.0, 'bg': 20.0,
                    'soft_col_cons': 0.01}

    wandb.log(losses)
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, variables


def initialize_per_timestep(params, variables, optimizer):
    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    new_pts = pts + (pts - variables["prev_pts"])
    new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    variables["prev_pts"] = pts.detach()
    variables["prev_rot"] = rot.detach()

    new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)

    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()

    variables["init_volume"] = get_volume(init_fg_pts).detach()
    variables["init_mean"] = torch.mean(init_fg_pts).detach()
    variables["init_variance"] = torch.var(init_fg_pts, axis=0).detach()

    entropy_x, entropy_y, entropy_z = get_entropies(init_fg_pts)
    variables["init_entropy_x"] = entropy_x.detach()
    variables["init_entropy_y"] = entropy_y.detach()
    variables["init_entropy_z"] = entropy_z.detach()

    # params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c', "rgb_colors"]
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c', "rgb_colors"]
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables


def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        wandb.log({"renders": wandb.Image(im)})
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        wandb.log({"PSNR": psnr})
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)


def log_dataset(dataset):
    with open("camera_log_gaussians.txt", "w") as file:
        for view in dataset:
            camera = view["cam"]
            camera_log = {"pos": camera.campos.tolist(), "viewmatrix": camera.viewmatrix.tolist()}
            file.write(str(camera_log) + "\n")
    file.close()


def train(seq, exp, args: argparse.Namespace):
    if os.path.exists(f"{args.output}/{exp}/{seq}"):
        print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
        return
    os.makedirs(f"{args.output}/{exp}/{seq}")
    md = json.load(open(f"{args.data}/{seq}/train_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])
    num_timesteps = min(num_timesteps, args.timesteps)
    params, variables = initialize_params(seq, md, args.data)
    optimizer = initialize_optimizer(params, variables)
    output_params = []
    for t in range(num_timesteps):
        dataset = get_dataset(t, md, seq, args.data)
        log_dataset(dataset)
        todo_dataset = []
        is_initial_timestep = (t == 0)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
        num_iter_per_timestep = args.initial_iters if is_initial_timestep else args.rest_iters
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            curr_data, todo_dataset = get_batch(todo_dataset, dataset)
            loss, variables = get_loss(params, curr_data, variables, is_initial_timestep)
            loss.backward()
            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                if is_initial_timestep:
                    params, variables = densify(params, variables, optimizer, i)
                    if i == num_iter_per_timestep - 1:
                        params, variables = remove_transparent(params, variables, optimizer)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if i == num_iter_per_timestep - 1:
                opacities = torch.sigmoid(params["logit_opacities"].clone().detach()).flatten().tolist()
                plot_histogram(opacities, xlabel="opacity", title=f"Opacity counts - frame {t}",
                               save=f"{args.output}/{exp}/{seq}/frame_{t}.pdf", bins=500)
                wandb.log({"gaussian centers": wandb.Object3D(
                    np.concatenate(
                        (np.array(params["means3D"].tolist()), 255 * np.array(params["rgb_colors"].tolist())),
                        axis=1))})

        progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))
        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)
    save_params(output_params, seq, exp, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic 3D Gaussian Training")
    parser.add_argument("-o", "--output", help="Path to output directory", default="./output")
    parser.add_argument("-d", "--data", help="Path to data directory", default="./data")
    parser.add_argument("-i", "--initial_iters", type=int, help="Optimization iterations for the original frame",
                        default=10000)
    parser.add_argument("-r", "--rest_iters", type=int,
                        help="Optimization iterations for the all but the original frame",
                        default=2000)
    parser.add_argument("-t", "--timesteps", type=int, help="Timesteps to optimize for",
                        default=1000000000)  # default big value, so that we use all available ones
    parser.add_argument("-s", "--sequences", nargs="+", type=str, help="The sequence names")

    args = parser.parse_args()

    # CUDA Logging
    print(f"Cuda available: {torch.cuda.is_available()}")
    current_device = torch.cuda.current_device()
    current_device_name = torch.cuda.get_device_name(current_device)
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"Current device name: {current_device_name}")
    print(f"Current device memory: {torch.cuda.get_device_properties(current_device).total_memory}")

    run = wandb.init(project="Gaussian Assets", config={
        "dataset": args.data,
        "sequences": args.sequences,
        "initial_iters": args.initial_iters,
        "rest_iters": args.rest_iters
    })

    print(f"Running {run.name}")

    # TODO Make loss weights a parameter so that we can log and sweep it.
    for sequence in args.sequences:
        train(sequence, run.name, args)
        torch.cuda.empty_cache()
