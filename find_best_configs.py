"""
Find the best hyperparameter configuration for each optimizer sweep.

Reads ALL 5 finished sweeps, groups runs by their actual 'optimizer' config
field (so we don't need to know which sweep is which optimizer), finds the
run with the lowest l2_error per optimizer, and prints the winning config
PLUS a ready-to-paste sbatch command for the 6000s comparison run.

Usage (on the cluster, venv active):
    python find_best_configs.py
"""

import wandb

ENTITY = "williamgevao202-concordia-university"
PROJECT = "deep-hedging-optim"

# The 5 finished sweep IDs (from the wandb sweeps page).
SWEEP_IDS = ["uyeatfwl", "1zup9fnw", "njqcxug1", "ruy7v5qb", "aymr09qr"]

# Which config keys to report + pass to the 6000s run, per optimizer.
RELEVANT_KEYS = {
    "Adam":        ["Adam_lr"],
    "SGD":         ["SGD_lr"],
    "LBFGS":       ["LBFGS_lr", "LBFGS_history_size"],
    "HessianFree": ["HessianFree_damping", "HessianFree_cg_max_iter"],
    "RNGD":        ["RNGD_damping", "RNGD_approximation"],
}

# CLI flag name for each config key (usually identical, but explicit is safe).
KEY_TO_FLAG = {
    "Adam_lr": "--Adam_lr",
    "SGD_lr": "--SGD_lr",
    "LBFGS_lr": "--LBFGS_lr",
    "LBFGS_history_size": "--LBFGS_history_size",
    "HessianFree_damping": "--HessianFree_damping",
    "HessianFree_cg_max_iter": "--HessianFree_cg_max_iter",
    "RNGD_damping": "--RNGD_damping",
    "RNGD_approximation": "--RNGD_approximation",
}


def main():
    api = wandb.Api()
    print(f"Reading sweeps from {ENTITY}/{PROJECT}\n" + "=" * 70)

    # Gather every run across all sweeps, tagged with its optimizer.
    # best_per_opt[optimizer] = (l2_error, config_dict, run_name)
    best_per_opt = {}
    all_by_opt = {}  # optimizer -> list of l2_errors (to show spread)

    for sid in SWEEP_IDS:
        try:
            sweep = api.sweep(f"{ENTITY}/{PROJECT}/{sid}")
        except Exception as e:
            print(f"Sweep {sid}: could not load ({e})")
            continue

        runs = list(sweep.runs)
        for run in runs:
            opt = run.config.get("optimizer", "UNKNOWN")
            l2 = run.summary.get("l2_error", None)
            if l2 is None or not isinstance(l2, (int, float)):
                continue  # skip runs with no valid l2_error
            all_by_opt.setdefault(opt, []).append(l2)
            prev = best_per_opt.get(opt)
            if prev is None or l2 < prev[0]:
                cfg = {k: run.config.get(k) for k in RELEVANT_KEYS.get(opt, [])}
                best_per_opt[opt] = (l2, cfg, run.name)

    if not best_per_opt:
        print("\nNO valid runs found with l2_error. Something is wrong.")
        return

    # Report best per optimizer
    print("\nBEST CONFIG PER OPTIMIZER:\n")
    for opt in sorted(best_per_opt):
        l2, cfg, name = best_per_opt[opt]
        spread = sorted(all_by_opt[opt])
        print(f"{opt}:")
        print(f"   best l2_error = {l2:.4e}   (run: {name})")
        print(f"   config: {cfg}")
        print(f"   grid spread: {spread[0]:.3e} (best) -> {spread[-1]:.3e} (worst), "
              f"{len(spread)} runs")
        print()

    # Emit ready-to-paste sbatch commands for the 6000s comparison runs
    print("=" * 70)
    print("\nREADY-TO-PASTE 6000s COMPARISON COMMANDS:\n")
    print("# Run these on the cluster (venv active) to launch the head-to-head.")
    print("# They are sbatch jobs - they survive closing your laptop.\n")
    for opt in sorted(best_per_opt):
        l2, cfg, name = best_per_opt[opt]
        flag_str = " ".join(f"{KEY_TO_FLAG[k]} {v}" for k, v in cfg.items())
        opt_args = f"--optimizer {opt} {flag_str}".strip()
        print(f'sbatch --export=ALL,OPT_ARGS="{opt_args}" compare_submit.sh')
    print()


if __name__ == "__main__":
    main()