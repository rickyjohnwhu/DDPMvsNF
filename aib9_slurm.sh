#!/usr/bin/env bash
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=aib9
#SBATCH --mail-type=NONE
#SBATCH --output=aib9.out

source /home/rjohn123/scratch/miniconda3/etc/profile.d/conda.sh 
conda activate aib9
module load cuda/12.1.1/
python run_aib9.py