#!/bin/bash
# Hessian-free diagnostic, stage 2, on 100-d BSB (600 s each).
#
# Stage 1 showed HF drives the loss on a FIXED 3500-point batch to 1e-9 while
# the path error stays at 0.043: it overfits the sampled points. With a fresh
# batch per step it instead oscillates (0.035..0.10). Two ways out:
#
#   hf_diag_bigbatch  fixed batch, 10x more points (30000 interior / 5000 IC)
#   hf_diag_cg20      fresh batch per step, but only 20 CG iterations per step,
#                     so each step is a damped partial solve, not an exact one
#
# Usage, on Vulcan from ~/scratch/rla-pinns:  bash sweeps/bsb/hf_diag2.sh
set -euo pipefail

BASE="--equation bsb-logS --boundary_condition bsb_payoff --dim_Omega 100 \
--model mlp-tanh-64-64-48-48 --dtype float64 --num_seconds 600 \
--interior_sampling path --wandb --wandb_project bsb-100d-sweeps \
--model_seed 1 --data_seed 0 --optimizer HessianFree --HessianFree_damping 1e-3"

submit() {  # id, extra flags
    sbatch --job-name="$1" --gres=gpu:l40s:1 --time=00:20:00 \
        --export=ALL,OPT_ARGS="$BASE $2 --wandb_id $1" compare_submit_vulcan.sh
}
submit hf_diag_bigbatch "--batch_frequency 0 --N_Omega 30000 --N_dOmega 5000 --HessianFree_cg_max_iter 250"
submit hf_diag_cg20     "--batch_frequency 1 --N_Omega 3000 --N_dOmega 500 --HessianFree_cg_max_iter 20"
