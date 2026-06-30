"""
Visualize the PINN learning the Black-Scholes value surface.

Produces FOUR figures so you can compare 2D vs 3D:

  2D versions (option value as CURVES vs stock price - easy to read):
    (1) convergence_2d.png     - value vs S at a fixed time; network curve at
                                 each checkpoint overlaid on the true curve.
    (2) final_fit_2d.png       - value vs S at several fixed times; final network
                                 curve (solid) vs analytic (black dashed) + payoff.

  3D versions (the full value SURFACE - looks impressive, harder to read):
    (3) convergence_3d.png     - the surface at each checkpoint + analytic, labelled.
    (4) final_vs_analytic_3d.png - final surface vs analytic vs error, in (t, S).

Run from repo root (venv active), after a checkpointed training run:
    python visualize_bs_surface.py

Coordinates:
  - Network + bs_call_price work in TRANSFORMED coords (tau, x):
    tau = T - t (time to maturity), x = log S (log price).
  - Figures are shown in ORIGINAL (t, S) coords: S = exp(x), t = T - tau.
  - The VALUE is unchanged by the relabel; only axis names change.
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

CHECKPOINT_DIR = "checkpoints_rngd"        # change to "checkpoints_rngd" for the RNGD run
CHECKPOINT_GLOB = "black-scholes-logS_1d_call_payoff_mlp-tanh-64_*_step*.pt"

try:
    from rla_pinns.black_scholes_logS_equation import (
        SIGMA, STRIKE, MATURITY, X_MIN, X_MAX, bs_call_price,
    )
    print("Imported constants + analytic price from rla_pinns.")
except Exception as e:
    print(f"(could not import equation module: {e})")
    print("(using fallback constants + local analytic price)")
    SIGMA, STRIKE, MATURITY = 0.2, 1.0, 1.0
    X_MIN, X_MAX = -3.0, 3.0

    def bs_call_price(X: Tensor) -> Tensor:
        from torch import clamp, exp, log, distributions
        tau, x = X[:, [0]], X[:, [1]]
        S = exp(x)
        sqrt_tau = clamp(tau, min=0.0).sqrt()
        at_init = tau <= 0.0
        safe = clamp(sqrt_tau, min=1e-12)
        d1 = (log(clamp(S, min=1e-12) / STRIKE) + 0.5 * SIGMA**2 * tau) / (SIGMA * safe)
        d2 = d1 - SIGMA * safe
        N = distributions.Normal(0.0, 1.0)
        value = S * N.cdf(d1) - STRIKE * N.cdf(d2)
        payoff = clamp(S - STRIKE, min=0.0)
        return at_init.to(value.dtype) * payoff + (~at_init).to(value.dtype) * value

DTYPE = torch.float64
GRID_N = 80


def build_model():
    return nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 1)).to(DTYPE)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = build_model()
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("step", None)


def find_checkpoints():
    paths = glob.glob(os.path.join(CHECKPOINT_DIR, CHECKPOINT_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"No checkpoints in {CHECKPOINT_DIR!r} matching {CHECKPOINT_GLOB!r}."
        )
    step_of = lambda p: int(re.search(r"step(\d+)\.pt$", p).group(1))
    paths = sorted(paths, key=step_of)
    return paths, [step_of(p) for p in paths]


def surface_grid():
    taus = np.linspace(0.0, MATURITY, GRID_N)
    xs = np.linspace(X_MIN, X_MAX, GRID_N)
    TAU, Xg = np.meshgrid(taus, xs, indexing="ij")
    X = torch.tensor(np.stack([TAU.ravel(), Xg.ravel()], 1), dtype=DTYPE)
    return TAU, Xg, X


def eval_grid(fn, X):
    with torch.no_grad():
        return fn(X).reshape(GRID_N, GRID_N).cpu().numpy()


def line_at_time(fn, tau_value, n=200):
    xs = np.linspace(X_MIN, X_MAX, n)
    X = torch.tensor(np.stack([np.full(n, tau_value), xs], 1), dtype=DTYPE)
    with torch.no_grad():
        V = fn(X).cpu().numpy().ravel()
    return np.exp(xs), V


def fig_convergence_2d(paths, steps):
    tau_slice = MATURITY * 0.5
    t_label = MATURITY - tau_slice
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.cm.viridis
    for i, (path, step) in enumerate(zip(paths, steps)):
        model, _ = load_model(path)
        S, V = line_at_time(model, tau_slice)
        ax.plot(S, V, color=cmap(i / max(1, len(paths) - 1)), lw=2,
                label=f"network, step {step}")
    S, Vtrue = line_at_time(bs_call_price, tau_slice)
    ax.plot(S, Vtrue, "k--", lw=2.5, label="analytic (truth)")
    ax.plot(S, np.clip(S - STRIKE, 0, None), color="grey", ls=":", lw=1.5,
            label="payoff max(S-K,0)")
    ax.set_xlabel("stock price  S")
    ax.set_ylabel("option value  V")
    ax.set_title(f"Network learning the option value (slice at t = {t_label:.2f})\n"
                 "each colored line is a later training step; black dashed = true value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("convergence_2d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  wrote convergence_2d.png")


def fig_final_fit_2d(paths, steps):
    model, last_step = load_model(paths[-1])
    tau_slices = [0.0, 0.25 * MATURITY, 0.5 * MATURITY, MATURITY]
    fig, axes = plt.subplots(1, len(tau_slices), figsize=(4 * len(tau_slices), 4.2),
                             sharey=True)
    for ax, tau in zip(axes, tau_slices):
        t_label = MATURITY - tau
        S, Vnn = line_at_time(model, tau)
        _, Vtrue = line_at_time(bs_call_price, tau)
        ax.plot(S, Vnn, "C0-", lw=2, label="network")
        ax.plot(S, Vtrue, "k--", lw=2, label="analytic")
        ax.axvline(STRIKE, color="grey", ls=":", lw=1, alpha=0.7)
        ax.set_title(f"t = {t_label:.2f}" + ("  (expiry)" if tau == 0.0 else ""))
        ax.set_xlabel("stock price  S")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("option value  V")
    axes[0].legend(fontsize=9)
    fig.suptitle(f"Final network (step {last_step}) vs analytic Black-Scholes, at several times\n"
                 "solid = network, dashed = truth, dotted vertical = strike K", fontsize=12)
    fig.tight_layout()
    fig.savefig("final_fit_2d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  wrote final_fit_2d.png")


def fig_convergence_3d(paths, steps, TAU, Xg, X, analytic):
    S = np.exp(Xg)
    t = MATURITY - TAU
    vmin, vmax = float(analytic.min()), float(analytic.max())
    n = len(paths) + 1
    fig = plt.figure(figsize=(3.4 * n, 3.8))
    for i, (path, step) in enumerate(zip(paths, steps)):
        model, _ = load_model(path)
        V = eval_grid(model, X)
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        ax.plot_surface(t, S, V, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"step {step}", fontsize=10)
        ax.set_xlabel("time t", fontsize=8)
        ax.set_ylabel("stock S", fontsize=8)
        ax.set_zlabel("value V", fontsize=8)
        ax.set_zlim(vmin, vmax)
        ax.tick_params(labelsize=6)
    ax = fig.add_subplot(1, n, n, projection="3d")
    ax.plot_surface(t, S, analytic, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title("analytic (truth)", fontsize=10)
    ax.set_xlabel("time t", fontsize=8)
    ax.set_ylabel("stock S", fontsize=8)
    ax.set_zlabel("value V", fontsize=8)
    ax.set_zlim(vmin, vmax)
    ax.tick_params(labelsize=6)
    fig.suptitle("Value surface as training proceeds (height = option value)", fontsize=12)
    fig.tight_layout()
    fig.savefig("convergence_3d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  wrote convergence_3d.png")


def fig_final_3d(paths, TAU, Xg, X, analytic):
    model, last_step = load_model(paths[-1])
    Vnn = eval_grid(model, X)
    err = np.abs(Vnn - analytic)
    S = np.exp(Xg)
    t = MATURITY - TAU
    fig = plt.figure(figsize=(15, 4.5))
    specs = [
        (Vnn, f"network surface (step {last_step})", "viridis", "value V"),
        (analytic, "analytic Black-Scholes", "viridis", "value V"),
        (err, f"|error|  (max {err.max():.2e})", "magma", "|error|"),
    ]
    for i, (Z, title, cmap, zlab) in enumerate(specs):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot_surface(t, S, Z, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("time t")
        ax.set_ylabel("stock S")
        ax.set_zlabel(zlab)
    fig.suptitle("Final network vs analytic vs error, in original (t, S) coords", fontsize=13)
    fig.tight_layout()
    fig.savefig("final_vs_analytic_3d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  wrote final_vs_analytic_3d.png")
    print(f"  max |error| = {err.max():.3e}, mean |error| = {err.mean():.3e}")


def main():
    print("Discovering checkpoints...")
    paths, steps = find_checkpoints()
    print(f"  {len(paths)} checkpoints at steps {steps}")
    TAU, Xg, X = surface_grid()
    analytic = eval_grid(bs_call_price, X)
    print("2D figures (curves):")
    fig_convergence_2d(paths, steps)
    fig_final_fit_2d(paths, steps)
    print("3D figures (surfaces):")
    fig_convergence_3d(paths, steps, TAU, Xg, X, analytic)
    fig_final_3d(paths, TAU, Xg, X, analytic)
    print("Done. Compare the 2D vs 3D versions and keep whichever reads better.")


if __name__ == "__main__":
    main()