#!/bin/bash
# Submit the 100-d Black-Scholes-Barenblatt optimizer comparison on Vulcan.
#
# Five arms (wandb ids), all: bsb-logS equation, dim_Omega 100, mlp-tanh-64-64-48-48
# (D = 16209), float64, path sampling, N_Omega 3000, N_dOmega 500, 3000 s.
# Hyperparameters are the winners of the 3000 s sweeps in wandb project
# bsb-100d-sweeps (16 random draws each; sweeps/bsb/best_configs.py):
#   bsb_sgd    SGD, lr 1.79e-2, momentum 0.902                (astral-sweep-4)
#   bsb_adam   Adam, lr 1.95e-3                               (glorious-sweep-14)
#   bsb_kfac   KFAC, damping 8.56e-6, grid line search        (driven-sweep-14)
#   bsb_hf     Hessian-free, damping 0.606, CG 175, adaptive  (charmed-sweep-16;
#              all 16 HF configs plateau at ~0.076, see sweeps/bsb/hf_diag*.sh)
#   bsb_rngd   ENGD-Woodbury (RNGD), damping 7.99e-11, exact  (radiant-sweep-6)
#
# Usage, on Vulcan from ~/scratch/rla-pinns after `git pull`:
#   bash launch_bsb.sh                    # all five
#   bash launch_bsb.sh bsb_kfac           # one arm
#   TAG=_s2 MODEL_SEED=2 bash launch_bsb.sh
#   DRY_RUN=1 bash launch_bsb.sh
set -euo pipefail

TAG="${TAG:-}"
MODEL_SEED="${MODEL_SEED:-1}"
DATA_SEED="${DATA_SEED:-0}"
KFAC_DAMPING="${KFAC_DAMPING:-8.564804157942645e-06}"
ARMS="${*:-bsb_sgd bsb_adam bsb_kfac bsb_hf bsb_rngd}"

BASE="--equation bsb-logS --boundary_condition bsb_payoff --dim_Omega 100 \
--model mlp-tanh-64-64-48-48 --dtype float64 \
--num_seconds 3000 --N_Omega 3000 --N_dOmega 500 \
--interior_sampling path --batch_frequency 1 --wandb \
--wandb_project deep-hedging-compare \
--model_seed $MODEL_SEED --data_seed $DATA_SEED"

for arm in $ARMS; do
    case "$arm" in
        bsb_sgd)  OPT="--optimizer SGD --SGD_lr 0.017867660873741583 --SGD_momentum 0.9020533465633788" ;;
        bsb_adam) OPT="--optimizer Adam --Adam_lr 0.0019527545948944008" ;;
        bsb_kfac) OPT="--optimizer KFAC --KFAC_damping $KFAC_DAMPING" ;;
        # adaptive damping, line search and CG backtracking are on by default
        bsb_hf)   OPT="--optimizer HessianFree --HessianFree_damping 0.6058080027028993 \
--HessianFree_curvature_opt ggn --HessianFree_cg_max_iter 175 --HessianFree_cg_decay_x0 0.95" ;;
        bsb_rngd) OPT="--optimizer RNGD --RNGD_damping 7.9948799195362e-11 --RNGD_approximation exact" ;;
        *) echo "unknown arm: $arm" >&2; exit 1 ;;
    esac
    ID="${arm}${TAG}"
    CMD=(sbatch --job-name="$ID" --gres=gpu:l40s:1 --time=01:30:00
         --export=ALL,OPT_ARGS="$BASE $OPT --wandb_id $ID" compare_submit_vulcan.sh)
    echo "== $ID"
    if [[ -n "${DRY_RUN:-}" ]]; then
        printf '  %q' "${CMD[@]}"; echo
    else
        "${CMD[@]}"
    fi
done
