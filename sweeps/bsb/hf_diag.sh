#!/bin/bash
# Hessian-free diagnostic on 100-d BSB: three 600 s runs that isolate why every
# sweep configuration plateaus at rl2_error_path ~ 0.076 after an early minimum.
#
#   hf_diag_base      the sweep setting (resampled batch each step, CG warm start)
#   hf_diag_fixbatch  same but batch_frequency 0: the objective is fixed between steps
#   hf_diag_nowarm    same as base but CG warm start off (cg_decay_x0 0)
#
# Usage, on Vulcan from ~/scratch/rla-pinns:  bash sweeps/bsb/hf_diag.sh
set -euo pipefail

BASE="--equation bsb-logS --boundary_condition bsb_payoff --dim_Omega 100 \
--model mlp-tanh-64-64-48-48 --dtype float64 --num_seconds 600 \
--N_Omega 3000 --N_dOmega 500 --interior_sampling path --wandb \
--wandb_project bsb-100d-sweeps --model_seed 1 --data_seed 0 \
--optimizer HessianFree --HessianFree_damping 1e-3 --HessianFree_cg_max_iter 250"

submit() {  # id, extra flags
    sbatch --job-name="$1" --gres=gpu:l40s:1 --time=00:20:00 \
        --export=ALL,OPT_ARGS="$BASE $2 --wandb_id $1" compare_submit_vulcan.sh
}
submit hf_diag_base     "--batch_frequency 1"
submit hf_diag_fixbatch "--batch_frequency 0"
submit hf_diag_nowarm   "--batch_frequency 1 --HessianFree_cg_decay_x0 0"
