"""Five L2-curve comparison plots, x-axis = TRUE training step.
Run on Mac (.venv): python plot_path_comparisons.py
"""
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENTITY = "williamgevao202-concordia-university"
PROJECT = "deep-hedging-compare"

api = wandb.Api()

def fetch(name):
    """Return (steps, l2) using the real logged step, not the row index."""
    rs = [r for r in api.runs(f"{ENTITY}/{PROJECT}") if r.name == name or r.id == name]
    if not rs:
        print(f"  !! no run '{name}'")
        return None, None
    r = rs[0]
    # try common step keys; fall back to _step
    step_keys = ["step", "Step", "_step"]
    steps, l2 = [], []
    for row in r.scan_history():
        if row.get("l2_error") is None:
            continue
        s = None
        for k in step_keys:
            if row.get(k) is not None:
                s = row[k]; break
        if s is None:
            s = len(steps)
        steps.append(s); l2.append(row["l2_error"])
    return steps, l2

def plot(pairs, title, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, label in pairs:
        s, l2 = fetch(name)
        if s:
            ax.plot(s, l2, label=label, lw=1.3)
    ax.set_yscale("log")
    ax.set_xscale("log")               # log-x so the real step range is readable
    ax.set_xlabel("training step")
    ax.set_ylabel("L2 error (whole-domain, 30k uniform eval)")
    ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fname}")

print("1. mu sweep")
plot([("path_mu_neg05","mu=-0.5"),("path_mu_neg02","mu=-0.2"),
      ("rngd_path_k1","mu=0.0"),("path_mu_pos02","mu=+0.2"),
      ("path_mu_pos05","mu=+0.5")],
     "Path sampling: effect of drift (mu)", "cmp_mu_sweep.png")

print("2. path vs uniform, FIXED")
plot([("rngd_path_fixed","path, fixed"),("rngd_uniform_fixed","uniform, fixed")],
     "Path vs Uniform (fixed points)", "cmp_path_vs_uniform_fixed.png")

print("3. path vs uniform, K1")
plot([("rngd_path_k1","path, resample"),("rngd_uniform_k1","uniform, resample")],
     "Path vs Uniform (resample every step)", "cmp_path_vs_uniform_k1.png")

print("4. uniform: K1 vs fixed")
plot([("rngd_uniform_k1","uniform, resample"),("rngd_uniform_fixed","uniform, fixed")],
     "Uniform: resample vs fixed", "cmp_uniform_k1_vs_fixed.png")

print("5. path: K1 vs fixed")
plot([("rngd_path_k1","path, resample"),("rngd_path_fixed","path, fixed")],
     "Path: resample vs fixed", "cmp_path_k1_vs_fixed.png")
print("done")