#!/bin/bash
#SBATCH --partition=gpu_irmb
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=1D
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere:1

## Build command
## singularity build --fakeroot --force parametric_nn.sif app/.devcontainer/container_conda.def

SCRIPT=Parametric_FEM_NN_training_time.py

srun singularity run \
 --cleanenv \
 --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
 --nv \
 --nvccli \
 --app Parametric_FEM_NN \
 parametric_nn.sif \