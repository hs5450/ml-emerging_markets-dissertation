#!/bin/bash
#SBATCH --job-name=jupyter_gpu
#SBATCH --partition=gpu            # GPU partition
#SBATCH --gres=gpu:1               # request 1 GPU
#SBATCH --cpus-per-task=8          # CPUs for data loading
#SBATCH --mem=32G                  # RAM
#SBATCH --time=04:00:00            # walltime
#SBATCH --output=slurm-jupyter-%j.out

module purge
module load CUDA/11.7 Anaconda3
conda activate tfdis              # your env with torch, transformers, jupyterlab

cd ~/transformer_dissertation
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
