import wandb
import matplotlib.pyplot as plt

PROJECT = "williamgevao202-concordia-university/deep-hedging-compare"
RUNS = {
    "path":            ("path_nobound",  "tab:blue"),
    "Gaussian s=0.400": ("gauss_nobound", "tab:orange"),
    "Gaussian s=0.141": ("gauss_fitted",  "tab:green"),
}

api = wandb.Api()
hist = {}
for label, (rid, _) in RUNS.items():
    r = api.run(f"{PROJECT}/{rid}")
    h = r.history(keys=["step", "l2_error_path"], pandas=True)
    hist[label] = h.dropna(subset=["l2_error_path"]).sort_values("step")
    print(f"{label:18s}: final l2_path={hist[label]['l2_error_path'].iloc[-1]:.3e}")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for label, (_, colour) in RUNS.items():
    h = hist[label]
    ax.plot(h["step"], h["l2_error_path"], label=label, color=colour, lw=1.3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("training step"); ax.set_ylabel(r"$L^2$ error (path region)")
ax.grid(True, which="both", alpha=0.25); ax.legend()
fig.tight_layout()
fig.savefig("three_arm_comparison.png", dpi=200)
print("saved three_arm_comparison.png")
