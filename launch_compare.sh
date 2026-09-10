#!/bin/bash
# Submit the 3000 s loss/optimiser comparison on Vulcan (one SLURM job per arm).
#
# Arms (wandb ids):
#   pinn_adam   FS-PINN residual, Adam lr 1e-3
#   pinn_rngd   FS-PINN residual, ENGD-Woodbury (RNGD, damping 7.98e-7, exact,
#               grid line search) -- same flags as run `pinn_rate_baseline`
#   em_adam     EM-BSDE, Adam lr 1e-3
#   heun_adam   Heun-BSDE, Adam lr 1e-3
#   unem_adam   Un-EM-BSDE (main/sub stack 5, p = 2), Adam lr 1e-3
#
# Usage, on Vulcan from ~/scratch/rla-pinns after `git pull`:
#   bash launch_compare.sh                      # all five arms
#   bash launch_compare.sh unem_adam            # just the missing arm
#   TAG=_s2 MODEL_SEED=2 bash launch_compare.sh # a second seed; TAG keeps ids unique
#   DRY_RUN=1 bash launch_compare.sh            # print the sbatch commands only
#
# wandb ids must be unique inside a project, so re-running an arm that already
# exists needs a TAG. Existing ids: pinn_adam, em_adam, heun_adam,
# pinn_rate_baseline (= pinn_rngd's settings).
set -euo pipefail

TAG="${TAG:-}"
MODEL_SEED="${MODEL_SEED:-1}"
DATA_SEED="${DATA_SEED:-0}"
ARMS="${*:-pinn_adam pinn_rngd em_adam heun_adam unem_adam}"

BASE="--num_seconds 3000 --N_Omega 3000 --N_dOmega 500 --dim_Omega 1 \
--equation black-scholes-logS --boundary_condition call_payoff \
--interior_sampling path --batch_frequency 1 --wandb \
--wandb_project deep-hedging-compare \
--model_seed $MODEL_SEED --data_seed $DATA_SEED"

ADAM="--optimizer Adam --Adam_lr 1e-3"
RNGD="--optimizer RNGD --RNGD_damping 7.98e-7 --RNGD_approximation exact"

for arm in $ARMS; do
    case "$arm" in
        pinn_adam) OPT="$ADAM --loss_type residual" ;;
        pinn_rngd) OPT="$RNGD" ;;
        em_adam)   OPT="$ADAM --loss_type bsde_em" ;;
        heun_adam) OPT="$ADAM --loss_type bsde_heun" ;;
        unem_adam) OPT="$ADAM --loss_type bsde_unem --unem_main_stack 5 --unem_sub_stack 5 --unem_p 2" ;;
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
