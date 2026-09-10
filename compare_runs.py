"""Compare the five 3000 s runs: FS-PINN (ENGD-Woodbury, Adam) and the three
BSDE losses (EM, Heun, Un-EM) with Adam, on 1D log-space Black-Scholes.

Pulls the runs from wandb (cached locally after the first fetch), prints a
summary table, and writes three figures to plots/:

  compare_rl2_path_time.png   rl2_error_path vs wall-clock time   (log-log)
  compare_rl2_path_step.png   rl2_error_path vs optimizer step    (log-log)
  compare_loss_interior.png   interior loss vs time (symlog: Un-EM can be < 0)

Run on the Mac:  python compare_runs.py [--refresh] [--out plots]
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

# Fixed order and fixed colors (validated categorical palette; never cycled).
RUNS = [
    # wandb run name,      label,                         color
    ("pinn_rate_baseline", "FS-PINN, ENGD-Woodbury", "#2a78d6"),
    ("pinn_adam",          "FS-PINN, Adam",          "#eb6834"),
    ("em_adam",            "EM-BSDE, Adam",          "#1baf7a"),
    ("heun_adam",          "Heun-BSDE, Adam",        "#eda100"),
    ("unem_adam",          "Un-EM-BSDE, Adam",       "#e87ba4"),
]
KEYS = ["step", "time", "rl2_error_path", "l2_error_path",
        "loss", "loss_interior", "loss_boundary"]
CACHE = Path("results/compare_runs_cache.json")
THRESHOLDS = [1e-2, 5e-3, 2e-3]   # rl2_error_path levels for time-to-reach

TEXT = "#333333"
MUTED = "#888888"
GRID = "#e6e6e6"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch(refresh: bool) -> Dict[str, Dict[str, List[float]]]:
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
        cols["config"] = {k: r.config.get(k) for k in
                          ("optimizer", "loss_type", "num_seconds", "model_seed",
                           "data_seed", "N_bsde_paths", "unem_main_stack")}
        cols["summary_steps"] = r.summary.get("step")
        data[name] = cols
        print(f"fetched {name}: {len(cols['time'])} logged rows")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data))
    return data


def _arr(d: Dict, k: str) -> np.ndarray:
    return np.asarray([np.nan if v is None else v for v in d[k]], dtype=float)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def time_to(t: np.ndarray, y: np.ndarray, level: float) -> float:
    hit = np.nonzero(y < level)[0]
    return float(t[hit[0]]) if hit.size else float("nan")


def summarise(data: Dict) -> List[Dict]:
    rows = []
    for name, label, _ in RUNS:
        if name not in data:
            continue
        d = data[name]
        t, y = _arr(d, "time"), _arr(d, "rl2_error_path")
        steps = _arr(d, "step")
        li = _arr(d, "loss_interior")
        row = dict(
            run=name, arm=label,
            final_rl2_path=y[-1], best_rl2_path=np.nanmin(y),
            steps=int(d["summary_steps"] or steps[-1]),
            steps_per_s=(d["summary_steps"] or steps[-1]) / t[-1],
            final_loss_interior=li[-1],
        )
        for lvl in THRESHOLDS:
            row[f"t@{lvl:g}"] = time_to(t, y, lvl)
        rows.append(row)
    return rows


def print_table(rows: List[Dict]) -> str:
    cols = ["arm", "final_rl2_path", "best_rl2_path", "steps", "steps_per_s",
            "final_loss_interior"] + [f"t@{lvl:g}" for lvl in THRESHOLDS]
    fmt = {
        "arm": lambda v: f"{v:<24}",
        "final_rl2_path": lambda v: f"{v:.4f}",
        "best_rl2_path": lambda v: f"{v:.4f}",
        "steps": lambda v: f"{v:d}",
        "steps_per_s": lambda v: f"{v:.1f}",
        "final_loss_interior": lambda v: f"{v: .2e}",
    }
    for lvl in THRESHOLDS:
        fmt[f"t@{lvl:g}"] = lambda v: "never" if np.isnan(v) else f"{v:.0f}s"
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt[c](r[c]) for c in cols) + " |")
    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(False, which="minor")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=TEXT, labelsize=9)


def _end_labels(ax, ends: List[tuple]) -> None:
    """Direct labels at the right end of each line, nudged apart in pixel space
    so the spacing is correct on log, symlog and linear axes alike."""
    ax.figure.canvas.draw()  # scales must be final before transforming
    to_px, to_data = ax.transData.transform, ax.transData.inverted().transform
    px = [to_px((x, y)) for x, y, _, _ in ends]
    order = np.argsort([p[1] for p in px])
    ys = [px[i][1] for i in order]
    gap = 13.0  # pixels, about one 9pt line
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    for i, y_px in zip(order, ys):
        x, _, label, _ = ends[i]
        _, y = to_data((px[i][0], y_px))
        ax.annotate(label, (x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9, color=TEXT)


def plot_metric(data: Dict, xkey: str, ykey: str, xlabel: str, ylabel: str,
                title: str, path: Path, log_x: bool = True, log_y: bool = True,
                symlog_y: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ends = []
    for name, label, color in RUNS:
        if name not in data:
            continue
        x, y = _arr(data[name], xkey), _arr(data[name], ykey)
        ok = ~np.isnan(x) & ~np.isnan(y) & (x > 0 if log_x else True)
        if log_y:
            ok &= y > 0
        x, y = x[ok], y[ok]
        ax.plot(x, y, color=color, linewidth=2, label=label, solid_capstyle="round")
        ends.append((x[-1], y[-1], label, color))
    if log_x:
        ax.set_xscale("log")
    if symlog_y:
        ax.set_yscale("symlog", linthresh=1e-4)
    elif log_y:
        ax.set_yscale("log")
    _style(ax, xlabel, ylabel, title)
    _end_labels(ax, ends)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=TEXT)
    ax.margins(x=0.02)
    fig.subplots_adjust(right=0.78)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true", help="re-download from wandb")
    ap.add_argument("--out", type=str, default="plots")
    args = ap.parse_args()

    data = fetch(args.refresh)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = summarise(data)
    table = print_table(rows)
    (out / "compare_runs_table.md").write_text(table + "\n")

    plot_metric(data, "time", "rl2_error_path", "wall-clock time (s)",
                "relative L2 error on paths",
                "Relative L2 error along forward SDE paths, 3000 s on one L40s",
                out / "compare_rl2_path_time.png")
    plot_metric(data, "step", "rl2_error_path", "optimizer step",
                "relative L2 error on paths",
                "Relative L2 error along forward SDE paths, per optimizer step",
                out / "compare_rl2_path_step.png")
    plot_metric(data, "time", "loss_interior", "wall-clock time (s)",
                "interior loss (symlog)",
                "Interior training loss; Un-EM is unbiased so it can go negative",
                out / "compare_loss_interior.png", log_y=False, symlog_y=True)


if __name__ == "__main__":
    main()
