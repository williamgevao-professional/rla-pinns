"""
Overlay all optimizers' convergence — the paper's Figure 2 / 3 format.

Pulls the five runs from the wandb project and produces:

  comparison_l2_error.png : L2 error vs Steps  |  L2 error vs Time   (log-log)
  comparison_loss.png     : Loss vs Steps      |  Loss vs Time       (log-log)

One colored line per optimizer (Adam, SGD, LBFGS, HessianFree, RNGD).

Run anywhere with internet + wandb access (login node, or your Mac):
    python plot_optimizer_comparison.py

No pandas required (uses scan_history).
"""

import wandb
import matplotlib.pyplot as plt
import numpy as np

ENTITY = "williamgevao202-concordia-university"
PROJECT = "deep-hedging-compare"

# consistent color + order per optimizer (so both figures match)
OPT_ORDER = ["Adam", "SGD", "LBFGS", "HessianFree", "RNGD"]
OPT_COLORS = {
    "Adam":        "#d62728",  # red
    "SGD":         "#ff7f0e",  # orange
    "LBFGS":       "#2ca02c",  # green
    "HessianFree": "#000000",  # black
    "RNGD":        "#1f77b4",  # blue (the second-order star, like ENGD-W in the paper)
}


def fetch_runs():
    """Pull each run's history into arrays keyed by optimizer."""
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    data = {}
    for run in runs:
        opt = run.config.get("optimizer", None)
        if opt is None:
            continue
        steps, times, l2, loss = [], [], [], []
        for row in run.scan_history(keys=["step", "time", "l2_error", "loss"]):
            # some rows may miss a key; guard each
            if row.get("l2_error") is None or row.get("loss") is None:
                continue
            steps.append(row.get("step"))
            times.append(row.get("time"))
            l2.append(row.get("l2_error"))
            loss.append(row.get("loss"))
        if not steps:
            print(f"  WARNING: no usable history for {opt} ({run.name})")
            continue
        data[opt] = {
            "step": np.array(steps, float),
            "time": np.array(times, float),
            "l2_error": np.array(l2, float),
            "loss": np.array(loss, float),
        }
        print(f"  {opt:12s}: {len(steps)} points, "
              f"final l2={l2[-1]:.2e}, final loss={loss[-1]:.2e}, "
              f"max time={max(t for t in times if t is not None):.0f}s")
    return data


def two_panel(data, metric, ylabel, outfile, title):
    """Left: metric vs step (log-log). Right: metric vs time (log-log)."""
    fig, (ax_step, ax_time) = plt.subplots(1, 2, figsize=(13, 5))

    for opt in OPT_ORDER:
        if opt not in data:
            continue
        d = data[opt]
        c = OPT_COLORS[opt]
        # sort by x just in case history isn't ordered
        si = np.argsort(d["step"])
        ax_step.plot(d["step"][si], d[metric][si], color=c, lw=1.6, label=opt)
        ti = np.argsort(d["time"])
        ax_time.plot(d["time"][ti], d[metric][ti], color=c, lw=1.6, label=opt)

    for ax, xlabel in [(ax_step, "Iteration (step)"), (ax_time, "Time [s]")]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)

    # single shared legend below
    handles, labels = ax_step.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outfile}")


def main():
    print(f"Fetching runs from {ENTITY}/{PROJECT}...")
    data = fetch_runs()
    if not data:
        print("No runs with usable history found.")
        return

    print("\nBuilding figures...")
    two_panel(data, "l2_error", "L2 error",
              "comparison_l2_error.png",
              "Optimizer comparison on Black-Scholes PINN: L2 error")
    two_panel(data, "loss", "loss",
              "comparison_loss.png",
              "Optimizer comparison on Black-Scholes PINN: loss")
    print("\nDone. Two figures, each with steps panel + time panel.")


if __name__ == "__main__":
    main()