#!/bin/bash
# Submit the 100-d Black-Scholes-Barenblatt optimizer comparison on Vulcan.
#
# Five arms (wandb ids), all: bsb-logS equation, dim_Omega 100, mlp-tanh-64-64-48-48
# (D = 16209), float64, path sampling, N_Omega 3000, N_dOmega 500, 3000 s.
#   bsb_sgd    SGD, lr 2.56e-3, momentum 0.9      (best 1-D BS 3000 s sweep)
#   bsb_adam   Adam, lr 1e-3                       (best 1-D BS path-sampling sweep)
#   bsb_kfac   KFAC, damping $KFAC_DAMPING, grid line search  (never swept on BS;
#              1e-4..1e-2 were equally good, 1e-6/1e-8 worse, in a 40 s CPU check)
#   bsb_hf     Hessian-free, damping 1.38e-3 adaptive, GGN, CG 250, line search
#              (best 1-D BS 3000 s sweep)
#   bsb_rngd   ENGD-Woodbury (RNGD), damping 7.98e-7, exact, grid line search
#              (same as the 1-D FS-PINN baseline)
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
KFAC_DAMPING="${KFAC_DAMPING:-1e-3}"
ARMS="${*:-bsb_sgd bsb_adam bsb_kfac bsb_hf bsb_rngd}"

BASE="--equation bsb-logS --boundary_condition bsb_payoff --dim_Omega 100 \
--model mlp-tanh-64-64-48-48 --dtype float64 \
--num_seconds 3000 --N_Omega 3000 --N_dOmega 500 \
--interior_sampling path --batch_frequency 1 --wandb \
--wandb_project deep-hedging-compare \
--model_seed $MODEL_SEED --data_seed $DATA_SEED"

for arm in $ARMS; do
    case "$arm" in
        bsb_sgd)  OPT="--optimizer SGD --SGD_lr 2.5614826751287644e-3 --SGD_momentum 0.9" ;;
        bsb_adam) OPT="--optimizer Adam --Adam_lr 1e-3" ;;
        bsb_kfac) OPT="--optimizer KFAC --KFAC_damping $KFAC_DAMPING" ;;
        # adaptive damping, line search and CG backtracking are on by default
        bsb_hf)   OPT="--optimizer HessianFree --HessianFree_damping 1.3804252810122471e-3 \
--HessianFree_curvature_opt ggn --HessianFree_cg_max_iter 250 --HessianFree_cg_decay_x0 0.95" ;;
        bsb_rngd) OPT="--optimizer RNGD --RNGD_damping 7.98e-7 --RNGD_approximation exact" ;;
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
