import torch
import os
import open3d as o3d
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
import matplotlib.pyplot as plt


def setup_camera(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam


def params2rendervar(params):
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_masked(x, y, mask: torch.Tensor):
    return torch.abs((x - y)).sum() / torch.count_nonzero(mask)


def opacity_loss(opacity_logits: torch.Tensor):
    exp = torch.exp(opacity_logits)
    alphas = exp / (1 + exp)
    return (1 - alphas).mean()


def opacity_entropy_loss(opacity_logits: torch.Tensor):
    alphas = torch.sigmoid(opacity_logits)
    log_alphas = torch.log2(alphas + 1e-8)
    return -torch.sum(alphas * log_alphas)


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)


def params2cpu(params, is_initial_timestep):
    if is_initial_timestep:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items()}
    else:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items() if
               k in ['means3D', 'rgb_colors', 'unnorm_rotations']}
    return res


def save_params(output_params, seq, exp, output_dir: str):
    to_save = {}
    for k in output_params[0].keys():
        if len(output_params) > 1 and k in output_params[1].keys():
            to_save[k] = np.stack([params[k] for params in output_params])
        else:
            to_save[k] = output_params[0][k]
    os.makedirs(f"{output_dir}/{exp}/{seq}", exist_ok=True)
    np.savez(f"{output_dir}/{exp}/{seq}/params", **to_save)


def load_scene_data(seq, exp, remove_background: bool, model_location: str, seg_as_col=False):
    params = dict(np.load(f"{model_location}/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        if remove_background:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if remove_background:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg


def get_volume(centers):
    AABB_min = torch.min(centers, axis=-2)[0]
    AABB_max = torch.max(centers, axis=-2)[0]
    return torch.prod(AABB_max - AABB_min)


def get_entropies(centers):
    xs = centers[:, 0]
    ys = centers[:, 1]
    zs = centers[:, 2]

    xs = (xs - torch.min(xs)) / (torch.max(xs) - torch.min(xs)) + 1e-7
    ys = (ys - torch.min(ys)) / (torch.max(ys) - torch.min(ys)) + 1e-7
    zs = (zs - torch.min(zs)) / (torch.max(zs) - torch.min(zs)) + 1e-7

    return -torch.sum(torch.log(xs) * xs), -torch.sum(torch.log(ys) * ys), -torch.sum(torch.log(zs) * zs)



def plot_histogram(data, bins=500, xlabel="Values", ylabel="Frequency", title="Histogram", save=""):
    """
    Plots a histogram from a list of data.

    Parameters:
    - data: List of numerical data.
    - bins: Number of bins for the histogram. Default is 10.
    - xlabel: Label for the x-axis. Default is "Values".
    - ylabel: Label for the y-axis. Default is "Frequency".
    - title: Title of the histogram plot. Default is "Histogram".
    """
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel(xlabel)
    plt.xlim(0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if len(save) == 0:
        plt.show()
    else:
        plt.savefig(save)
