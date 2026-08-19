import wandb
import matplotlib.pyplot as plt

PROJECT = "williamgevao202-concordia-university/deep-hedging-compare"
RUNS = {"uniform": "uni_x0fixed", "path": "path_x0fixed"}
COLORS = {"uniform": "tab:orange", "path": "tab:blue"}

api = wandb.Api()
hist = {}
for label, rid in RUNS.items():
    r = api.run(f"{PROJECT}/{rid}")
    h = r.history(keys=["step", "l2_error", "l2_error_path"], pandas=True)
    hist[label] = h.dropna(subset=["l2_error"]).sort_values("step")
    print(f"{label}: final l2={hist[label]['l2_error'].iloc[-1]:.3e}, "
          f"final l2_path={hist[label]['l2_error_path'].iloc[-1]:.3e}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
panels = [("l2_error", "Uniform-domain eval"),
          ("l2_error_path", "Path-region eval")]

for ax, (key, title) in zip(axes, panels):
    for label, h in hist.items():
        ax.plot(h["step"], h[key], label=f"{label}-trained",
                color=COLORS[label], lw=1.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
axes[0].set_ylabel(r"$L^2$ error"); axes[0].legend()

fig.tight_layout()
fig.savefig("x0fixed_comparison.png", dpi=200)
print("saved x0fixed_comparison.png")
