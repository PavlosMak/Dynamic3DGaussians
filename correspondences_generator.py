from pycpd import DeformableRegistration
import numpy as np
import point_cloud_utils as pcu
from numba import jit
import time
import os
import argparse


@jit(forceobj=True)
def get_correspondences(source: np.ndarray, target: np.ndarray):
    """
    target: Current frame
    source: Next frame
    """
    reg = DeformableRegistration(X=target, Y=source, low_rank=True)
    result = reg.register()
    correspondeces = np.argmax(reg.P, axis=1)
    return {s_ix: target_ix for s_ix, target_ix in enumerate(correspondeces)}, reg.P, result


def get_deformation_vectors(source, target, correspondences):
    deformations = np.zeros((len(source), 3))
    for s_ix, src_point in enumerate(source):
        t_ix = correspondences[s_ix]
        deformations[s_ix] = target[t_ix] - src_point
    return deformations


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dynamic 3D Gaussian Training")
    parser.add_argument("-o", "--output", help="Path to output directory", default="./output")
    parser.add_argument("-d", "--data", help="Path to data directory", default="./data")
    parser.add_argument("-n", "--num_samples", type=int, help="Samples to draw when downscaling",
                        default=1000)
    parser.add_argument("-e", "--exp", type=str, help="The experiment name")
    parser.add_argument("-s", "--seq", nargs="+", type=str, help="The sequence names")

    args = parser.parse_args()

    for seq in args.seq:
        path = f"{args.data}/{args.exp}/{seq}/params.npz"

        output_path = f"{args.output}/{args.exp}/{seq}"
        os.makedirs(output_path, exist_ok=True)

        data = np.load(path, allow_pickle=True)["arr_0"].tolist()
        frame_count = len(data)

        indices = []
        for frame in range(frame_count):
            indices.append(pcu.downsample_point_cloud_poisson_disk(data[frame]["means3D"], -1, args.num_samples))

        cross_frame_correspondences = []
        start_total = time.time()
        frame_count = 2
        for frame in range(frame_count - 1):
            print(f"Registering {frame} to {frame + 1}")
            source = data[frame]["means3D"][indices[frame]]
            target = data[frame + 1]["means3D"][indices[frame + 1]]
            start_local = time.time()
            correspondences, P, reg_results = get_correspondences(source, target)
            cross_frame_correspondences.append(correspondences)
            deformations = get_deformation_vectors(source, target, correspondences)
            end_local = time.time()
            np.savez(f"{output_path}/source_{frame}.npz", source)
            np.savez(f"{output_path}/target_{frame + 1}.npz", target)
            np.savez(f"{output_path}/deformations_{frame}-{frame + 1}.npz", deformations)
            np.savez(f"{output_path}/P_{frame}-{frame + 1}.npz", P)
            print(f"Frame registration took: {end_local - start_local} seconds")
        end_total = time.time()
        print(f"Done registering - took {end_total - start_total} seconds")
