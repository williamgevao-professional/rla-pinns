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


def fetch(refresh: bool) -> Dict:
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())
    import wandb
    api = wandb.Api(timeout=120)
    all_runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    data = {}
    for name, _, _ in RUNS:
        rs = [r for r in all_runs if r.name == name]
        if not rs:
            print(f"!! run {name!r} not found, skipping")
            continue
        r = rs[0]
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
        data[name] = cols
        print(f"fetched {name}: {len(cols['time'])} rows")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data))
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
    ap.add_argument("--metric", default="rl2_error_path",
                    choices=["rl2_error_path", "l2_error_path", "loss"])
    ap.add_argument("--out", default="plots")
    args = ap.parse_args()

    data = fetch(args.refresh)
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
    path = out / f"bsb_optimizers_{args.metric}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")

    lines = [f"| optimizer | final {args.metric} | best | steps | time (s) |", "|---|---|---|---|---|"]
    for label, fin, best, steps, tend in rows:
        lines.append(f"| {label} | {fin:.3e} | {best:.3e} | {steps} | {tend:.0f} |")
    table = "\n".join(lines)
    print(table)
    (out / f"bsb_optimizers_{args.metric}_table.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
