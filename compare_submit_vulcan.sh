#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --account=aip-fdangel

module load python/3.11
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/scratch/wgevao/rla-pinns
export WANDB_MODE=online

python -u rla_pinns/train.py \
    --equation black-scholes-logS \
    --boundary_condition call_payoff \
    --dim_Omega 1 \
    --model mlp-tanh-64 \
    --dtype float64 \
    --num_seconds 6000 \
    --wandb \
    --wandb_project deep-hedging-compare \
    $OPT_ARGS
