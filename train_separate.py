import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

import wandb
from external import densify
from helpers import params2cpu
from train import initialize_params, initialize_optimizer, get_dataset, log_dataset, get_batch, \
    get_loss, report_progress


def save_params(output_params, seq, exp, output_dir: str):
    to_save = {}
    for t, params in enumerate(output_params):
        to_save[t] = {p : np.array(params[p]) for p in params}
    os.makedirs(f"{output_dir}/{exp}/{seq}", exist_ok=True)
    np.savez(f"{output_dir}/{exp}/{seq}/params", to_save)

def train(seq, exp, args: argparse.Namespace):
    if os.path.exists(f"{args.output}/{exp}/{seq}"):
        print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
        return
    md = json.load(open(f"{args.data}/{seq}/train_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])
    num_timesteps = min(num_timesteps, args.timesteps)
    output_params = []
    for t in range(num_timesteps):
        params, variables = initialize_params(seq, md, args.data)
        optimizer = initialize_optimizer(params, variables)
        dataset = get_dataset(t, md, seq, args.data)
        log_dataset(dataset)
        todo_dataset = []
        progress_bar = tqdm(range(args.initial_iters), desc=f"timestep {t}")
        for i in range(args.initial_iters):
            curr_data, todo_dataset = get_batch(todo_dataset, dataset)
            loss, variables = get_loss(params, curr_data, variables, True)
            loss.backward()
            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                params, variables = densify(params, variables, optimizer, i)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if i == args.initial_iters - 1:
                wandb.log({"gaussian centers": wandb.Object3D(
                    np.concatenate(
                        (np.array(params["means3D"].tolist()), 255 * np.array(params["rgb_colors"].tolist())),
                        axis=1))})
        progress_bar.close()
        output_params.append(params2cpu(params, True))
    save_params(output_params, seq, exp, args.output)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic 3D Gaussian Training")
    parser.add_argument("-o", "--output", help="Path to output directory", default="./output")
    parser.add_argument("-d", "--data", help="Path to data directory", default="./data")
    parser.add_argument("-i", "--initial_iters", type=int, help="Optimization iterations for the original frame",
                        default=10000)
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
        "initial_iters": args.initial_iters
    })

    print(f"Running {run.name}")

    # TODO Make loss weights a parameter so that we can log and sweep it.
    for sequence in args.sequences:
        train(sequence, run.name, args)
        torch.cuda.empty_cache()
