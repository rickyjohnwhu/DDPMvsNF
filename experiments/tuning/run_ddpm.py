# import packages and filter version warnings
import numpy as np
import argparse
from sklearn.decomposition import PCA
from argparse import Namespace
import yaml
import utils_ddpm
import sys
import os
import warnings
import torch
import random
import subprocess

warnings.filterwarnings("ignore", message=".*functorch.*deprecated.*vmap.*")


# main run args function, trains and samples one model for fixed hyperparameters
def run(args):

    # set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    # preprocess samples
    train_loader, train_np, valid_np, test_np, num_set = utils_ddpm.preprocess_samples(
        args.input, args.split, args.batch_size, args.dataset_limiter
    )
    print("Preprocessed samples.")

    # initialize diffusion architecture and associated neural network model
    diffusion = utils_ddpm.VPDiffusion(num_diffusion_timesteps=args.diffusion_steps)
    model, backbone, num_torsions = utils_ddpm.init_model_backbone(
        train_loader,
        args.resnet_block_groups,
        args.learned_sinusoidal_cond,
        args.learned_sinusoidal_dim,
        args.dim_mults,
        num_set,
        args.lr,
        args.model_dim,
    )
    print("Initialized model.")

    # construct and fit PCA
    pca = PCA(n_components=2, random_state=args.seed)
    print("Initialized PCA.")
    trainingpca = pca.fit_transform(train_np)
    print("Fitted PCA.")

    # train and sample from model, compute performance metrics
    generated_data, info = utils_ddpm.train_and_sample(
        model,
        train_loader,
        valid_np,
        test_np,
        train_np,
        pca,
        args.bounds,
        diffusion,
        backbone,
        num_torsions,
        num_set,
        args.sample_batching,
        args.epoch_max,
        args.aib9_status,
    )
    print("Trained and sampled model.")
    args_dict = {key: getattr(args, key) for key in vars(args)}

    # designate output path, save input hyperparameters, performance metrics, and generated data
    output_path = f"output/ddpm/tune={args.model_dim}_seed={args.seed}"
    os.makedirs(output_path, exist_ok=True)

    hyperparameters_txt_path = os.path.join(output_path, "hyperparameters.txt")
    with open(hyperparameters_txt_path, "w") as file:
        for key, value in args_dict.items():
            file.write(f"{key}: {value}\n")

    metrics_txt_path = os.path.join(output_path, "metrics.txt")
    with open(metrics_txt_path, "w") as file:
        for key, value in info.items():
            file.write(f"{key}: {value}\n")

    np.save(os.path.join(output_path, "generated_data.npy"), generated_data)
    print("Saved data, hyperparameters, and metrics.")


# load config file
def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config["config"]


# perform experiment on a slurm-enabled cluster with a temporary script file, used when --use_slurm is appended to model run command
def create_slurm_script(model_dim, seed, config):
    script = f"""#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=NONE
#SBATCH --job-name=ddpm_tune
#SBATCH --output=logs/ddpm/tune-{model_dim}-seed-{seed}.out

source $MAMBA_ROOT_PREFIX/etc/profile.d/micromamba.sh
micromamba activate compenv

python run_ddpm.py --config config/config_ddpm.yml --model_dim {model_dim} --seed {seed} --slurm_run
"""

    slurm_script_filename = f"slurm_{model_dim}_{seed}.sh"
    with open(slurm_script_filename, "w") as f:
        f.write(script)

    return slurm_script_filename


# main command to train and sample models corresponding to a set of experimental hyperparameters, can handle --use_slurm or run locally
def main():
    config = load_config("config/config_ddpm.yml")
    print("Loaded config.")

    args = Namespace(**config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--model_dim", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use_slurm", action="store_true")
    parser.add_argument("--slurm_run", action="store_true")
    args = parser.parse_args(namespace=args)
    print("Parsed arguments.")

    # experimental parameters over which to iterate
    model_dims = [4, 8, 16, 32, 48, 64, 96, 128, 256]
    seeds = [1, 2, 3]

    # create, run, and delete temporary slurm scripts
    if args.use_slurm:

        for model_dim in model_dims:
            for seed in seeds:
                print("Created and submitted SLURM job.")
                slurm_script = create_slurm_script(model_dim, seed, config)
                subprocess.run(["sbatch", slurm_script])
                os.remove(slurm_script)
                print("Deleted SLURM script.")

    elif args.slurm_run:
        run(args)
    else:

        for model_dim in model_dims:
            for seed in seeds:
                args.model_dim = model_dim
                args.seed = seed
                run(args)


if __name__ == "__main__":
    main()
