# import packages
import numpy as np
import argparse
from sklearn.decomposition import PCA
from argparse import Namespace
import yaml
import utils_cfm
import sys
import os
import warnings
import torch
import random
import subprocess

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
    train_loader, train_np, valid_np, test_np = utils_cfm.preprocess_samples(
        args.input, args.split, args.batch_size, args.dataset_limiter, args.device
    )
    print("Preprocessed samples.")

    # initialize flow matching model, prior, and interpolation function
    model, FM, base = utils_cfm.init_model(
        args.path_dim,
        args.w,
        args.sigma,
        args.model_style,
        args.time_varying,
        args.resnet_block_groups,
        args.learned_sinusoidal_cond,
        args.learned_sinusoidal_dim,
        args.dim_mults,
        args.model_dim,
        args.device,
    )
    interp = utils_cfm.Interpolater((args.path_dim,), (args.model_dim,))
    print("Initialized model.")

    # construct and fit PCA
    pca = PCA(n_components=2, random_state=args.seed)
    print("Initialized PCA.")
    trainingpca = pca.fit_transform(train_np)
    print("Fitted PCA.")

    # train and sample from model, compute performance metrics
    generated_data, info = utils_cfm.train_and_sample(
        model,
        FM,
        base,
        interp,
        train_loader,
        valid_np,
        test_np,
        train_np,
        pca,
        args.bounds,
        args.lr,
        args.epoch_max,
        args.aib9_status,
    )
    print("Trained and sampled model.")
    args_dict = {key: getattr(args, key) for key in vars(args)}

    # designate output path, save input hyperparameters, performance metrics, and generated data
    output_path = f"output/cfm/dim={args.path_dim}_seed={args.seed}"
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
def create_slurm_script(path_dim, input, seed, config):
    script = f"""#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=NONE
#SBATCH --job-name=cfm_dim
#SBATCH --output=logs/cfm/dim-{path_dim}-seed-{seed}.out

source $MAMBA_ROOT_PREFIX/etc/profile.d/micromamba.sh
micromamba activate compenv

python run_cfm.py --config config/config_cfm.yml --path_dim {path_dim} --input {input} --seed {seed} --slurm_run
"""

    slurm_script_filename = f"slurm_{path_dim}_{seed}.sh"
    with open(slurm_script_filename, "w") as f:
        f.write(script)

    return slurm_script_filename


# main command to train and sample models corresponding to a set of experimental hyperparameters, can handle --use_slurm or run locally
def main():
    config = load_config("config/config_cfm.yml")
    print("Loaded config.")

    args = Namespace(**config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--path_dim", type=int)
    parser.add_argument("--input", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use_slurm", action="store_true")
    parser.add_argument("--slurm_run", action="store_true")
    args = parser.parse_args(namespace=args)
    print("Parsed arguments.")

    # experimental parameters over which to iterate
    inputs = ['../../datasets/data_input/gmm_dim_10_modes_4.npy', '../../datasets/data_input/gmm_dim_20_modes_4.npy', '../../datasets/data_input/gmm_dim_30_modes_4.npy', '../../datasets/data_input/gmm_dim_40_modes_4.npy', '../../datasets/data_input/gmm_dim_50_modes_4.npy', '../../datasets/data_input/gmm_dim_60_modes_4.npy', '../../datasets/data_input/gmm_dim_70_modes_4.npy', '../../datasets/data_input/gmm_dim_80_modes_4.npy', '../../datasets/data_input/gmm_dim_90_modes_4.npy', '../../datasets/data_input/gmm_dim_100_modes_4.npy']
    path_dims = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    seeds = [1, 2, 3]

    # create, run, and delete temporary slurm scripts
    if args.use_slurm:

        for i in range(len(path_dims)):
            for seed in seeds:
                print("Created and submitted SLURM job.")
                slurm_script = create_slurm_script(path_dims[i], inputs[i], seed, config)
                subprocess.run(["sbatch", slurm_script])
                os.remove(slurm_script)
                print("Deleted SLURM script.")

    elif args.slurm_run:
        run(args)
    else:

        for i in range(len(path_dims)):
            for seed in seeds:
                args.path_dim = path_dims[i]
                args.input = inputs[i]
                args.seed = seed
                run(args)


if __name__ == "__main__":
    main()
