"""Two-panel optimizer comparison on 100-d Black-Scholes-Barenblatt (bsb-logS).

Relative L2 error along forward SDE paths vs optimizer step (left) and vs
wall-clock time (right), one line per optimizer, shared legend below. Same
layout as the rla-pinns paper's Poisson figure.

Pulls the runs from wandb (cached in results/ after the first fetch) and writes
plots/bsb_optimizers.png and plots/bsb_optimizers_table.md.

Run on the Mac:  python compare_bsb.py [--refresh] [--metric rl2_error_path]
"""
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENTITY = "williamgevao202-concordia-university"
PROJECT = "deep-hedging-compare"
SWEEP_PROJECT = "bsb-100d-sweeps"   # --from_sweeps: best finished run per optimizer
OPTIMIZER_OF = {"bsb_sgd": "SGD", "bsb_adam": "Adam", "bsb_kfac": "KFAC",
                "bsb_hf": "HessianFree", "bsb_rngd": "RNGD"}

# wandb run id -> (legend label, color). Fixed order and fixed colors.
RUNS = [
    ("bsb_sgd",  "SGD",             "#ff7f0e"),
    ("bsb_adam", "Adam",            "#d62728"),
    ("bsb_kfac", "KFAC",            "#2ca02c"),
    ("bsb_hf",   "Hessian-free",    "#000000"),
    ("bsb_rngd", "ENGD (Woodbury)", "#1f77b4"),
]
KEYS = ["step", "time", "rl2_error_path", "l2_error_path", "loss", "loss_interior", "loss_boundary"]
CACHE = Path("results/compare_bsb_cache.json")


def _best_sweep_runs(api) -> Dict[str, object]:
    """Lowest final rl2_error_path among finished sweep runs, per optimizer."""
    import math
    best = {}
    for r in api.runs(f"{ENTITY}/{SWEEP_PROJECT}", per_page=300):
        opt = r.config.get("optimizer")
        try:
            v = float(r.summary.get("rl2_error_path"))
        except (TypeError, ValueError):
            continue
        if r.state != "finished" or math.isnan(v) or r.name.startswith("hf_diag"):
            continue
        if opt not in best or v < best[opt][0]:
            best[opt] = (v, r)
    return {name: best[opt][1] for name, opt in OPTIMIZER_OF.items() if opt in best}


def fetch(refresh: bool, from_sweeps: bool = False) -> Dict:
    cache = CACHE.with_name("compare_bsb_sweeps_cache.json") if from_sweeps else CACHE
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    import wandb
    api = wandb.Api(timeout=120)
    if from_sweeps:
        picked = _best_sweep_runs(api)
    else:
        all_runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
        picked = {name: rs[0] for name, _, _ in RUNS
                  if (rs := [r for r in all_runs if r.name == name])}
    data = {}
    for name, _, _ in RUNS:
        if name not in picked:
            print(f"!! run {name!r} not found, skipping")
            continue
        r = picked[name]
        cols = {k: [] for k in KEYS}
        for row in r.scan_history(keys=KEYS):
            if row.get("rl2_error_path") is None or row.get("time") is None:
                continue
            for k in KEYS:
                cols[k].append(row.get(k))
        cols["n_params"] = r.config.get("n_params") or r.summary.get("n_params")
        cols["model"] = r.config.get("model")
        cols["dim_Omega"] = r.config.get("dim_Omega")
        cols["summary_steps"] = r.summary.get("step")
        cols["source"] = f"{r.project}/{r.name} ({r.id})"
        cols["hparams"] = {k: v for k, v in r.config.items()
                           if k.startswith(f"{r.config.get('optimizer')}_")
                           and not isinstance(v, (list, dict))}
        data[name] = cols
        print(f"fetched {name} <- {cols['source']}: {len(cols['time'])} rows")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def _arr(d: Dict, k: str) -> np.ndarray:
    return np.asarray([np.nan if v is None else v for v in d[k]], dtype=float)


def n_params(model: str, d: int) -> int:
    widths = [int(w) for w in model.split("-")[2:]]
    dims = [d + 1] + widths + [1]
    return sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))


def main() -> None:
    ap = ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--from_sweeps", action="store_true",
                    help="use the best finished sweep run per optimizer instead of the named runs")
    ap.add_argument("--metric", default="rl2_error_path",
                    choices=["rl2_error_path", "l2_error_path", "loss"])
    ap.add_argument("--out", default="plots")
    args = ap.parse_args()

    data = fetch(args.refresh, args.from_sweeps)
    if not data:
        raise SystemExit("no runs found on wandb yet; launch them with launch_bsb.sh")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    any_run = next(iter(data.values()))
    D = n_params(any_run["model"], int(any_run["dim_Omega"])) if any_run.get("model") else None
    title = f"100d BSB" + (f" ($D = {D}$)" if D else "")
    ylabel = {"rl2_error_path": "relative $L_2$ error", "l2_error_path": "$L_2$ error",
              "loss": "loss"}[args.metric]

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    rows = []
    for name, label, color in RUNS:
        if name not in data:
            continue
        d = data[name]
        step, t, y = _arr(d, "step"), _arr(d, "time"), _arr(d, args.metric)
        ok = ~np.isnan(y) & (y > 0)
        step, t, y = step[ok], t[ok], y[ok]
        step = np.maximum(step, 1)      # log axis: show step 0 at 1
        axes[0].plot(step, y, color=color, linewidth=2, label=label)
        axes[1].plot(np.maximum(t, 1e-2), y, color=color, linewidth=2, label=label)
        rows.append((label, y[-1], y.min(), int(d["summary_steps"] or step[-1]), t[-1]))

    for ax, xlabel in zip(axes, ("Iteration", "Time (s)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, which="major", color="#e6e6e6", linewidth=0.8)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    suffix = "_sweepbest" if args.from_sweeps else ""
    path = out / f"bsb_optimizers_{args.metric}{suffix}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")

    lines = [f"| optimizer | final {args.metric} | best | steps | time (s) | run | hyperparameters |",
             "|---|---|---|---|---|---|---|"]
    for (label, fin, best, steps, tend), (name, _, _) in zip(rows, [x for x in RUNS if x[0] in data]):
        d = data[name]
        hp = ", ".join(f"{k.split('_', 1)[1]}={v:.3g}" if isinstance(v, float) else f"{k.split('_', 1)[1]}={v}"
                       for k, v in d.get("hparams", {}).items() if k.split("_", 1)[1] not in ("equation",))
        lines.append(f"| {label} | {fin:.3e} | {best:.3e} | {steps} | {tend:.0f} | {d.get('source', name)} | {hp} |")
    table = "\n".join(lines)
    print(table)
    (out / f"bsb_optimizers_{args.metric}{suffix}_table.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
