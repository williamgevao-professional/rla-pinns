#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --account=aip-fdangel

module load python/3.11
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/scratch/wgevao/rla-pinns

wandb agent $SWEEP_ID