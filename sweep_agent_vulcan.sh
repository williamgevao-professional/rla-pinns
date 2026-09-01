#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --account=aip-fdangel
module load python/3.11
source venv/bin/activate
export PYTHONPATH=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
wandb agent $SWEEP_ID
