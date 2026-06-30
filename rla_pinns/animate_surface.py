"""
Animate the PINN learning the Black-Scholes value surface.

Produces a smooth GIF (and optional MP4) where the network's output morphs
from its initial state into the true Black-Scholes surface as training proceeds.

Two animation styles (set STYLE):
  - "2d"      : value-vs-stock-price curve morphing onto the analytic curve.
                Clearest for *seeing* the fit happen.
  - "3d"      : the full value surface morphing. More impressive, busier.
  - "both"    : side-by-side 2D curve + 3D surface in one animation.

Run from repo root (venv active), after a FINE-checkpointed training run:
    python animate_surface.py

Needs many checkpoints for smoothness (e.g. 30 frames). Point CHECKPOINT_DIR
at the fine-checkpoint run.

Coordinates: network works in (tau, x); shown in (t, S) where S=exp(x), t=T-tau.
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from torch import Tensor, nn

# ----------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints_rngd_fine"   # the fine-checkpoint run
CHECKPOINT_GLOB = "black-scholes-logS_1d_call_payoff_mlp-tanh-64_*_step*.pt"
STYLE = "both"        # "2d", "3d", or "both"
FPS = 6               # frames per second in the output
OUT_GIF = "surface_learning.gif"
OUT_MP4 = "surface_learning.mp4"   # written only if ffmpeg available
TAU_SLICE_FRAC = 0.5  # which time slice for the 2D curve (0=expiry, 1=far)

try:
    from rla_pinns.black_scholes_logS_equation import (
        SIGMA, STRIKE, MATURITY, X_MIN, X_MAX, bs_call_price,
    )
    print("Imported constants + analytic price from rla_pinns.")
except Exception as e:
    print(f"(could not import equation module: {e}; using fallback)")
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
GRID_N = 60


def build_model():
    return nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 1)).to(DTYPE)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    m = build_model()
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m, ckpt.get("step", None)


def find_checkpoints():
    paths = glob.glob(os.path.join(CHECKPOINT_DIR, CHECKPOINT_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"No checkpoints in {CHECKPOINT_DIR!r}. Run a fine-checkpoint training first."
        )
    step_of = lambda p: int(re.search(r"step(\d+)\.pt$", p).group(1))
    paths = sorted(paths, key=step_of)
    return paths, [step_of(p) for p in paths]


# grids
def surface_grid():
    taus = np.linspace(0.0, MATURITY, GRID_N)
    xs = np.linspace(X_MIN, X_MAX, GRID_N)
    TAU, Xg = np.meshgrid(taus, xs, indexing="ij")
    X = torch.tensor(np.stack([TAU.ravel(), Xg.ravel()], 1), dtype=DTYPE)
    return TAU, Xg, X


def eval_grid(fn, X):
    with torch.no_grad():
        return fn(X).reshape(GRID_N, GRID_N).cpu().numpy()


def line_grid():
    tau = MATURITY * TAU_SLICE_FRAC
    xs = np.linspace(X_MIN, X_MAX, 200)
    X = torch.tensor(np.stack([np.full(200, tau), xs], 1), dtype=DTYPE)
    return np.exp(xs), X, MATURITY - tau


def eval_line(fn, X):
    with torch.no_grad():
        return fn(X).cpu().numpy().ravel()


def main():
    print("Discovering checkpoints...")
    paths, steps = find_checkpoints()
    print(f"  {len(paths)} frames at steps {steps}")

    # precompute analytic + per-frame network outputs (so animation is fast)
    TAU, Xg, Xsurf = surface_grid()
    S_surf = np.exp(Xg)
    t_surf = MATURITY - TAU
    analytic_surf = eval_grid(bs_call_price, Xsurf)
    vmin, vmax = float(analytic_surf.min()), float(analytic_surf.max())

    S_line, Xline, t_line = line_grid()
    analytic_line = eval_line(bs_call_price, Xline)

    print("Pre-evaluating network at each checkpoint...")
    nn_surfs, nn_lines = [], []
    for p in paths:
        m, _ = load_model(p)
        nn_surfs.append(eval_grid(m, Xsurf))
        nn_lines.append(eval_line(m, Xline))

    # ---- build the animation ----
    if STYLE == "2d":
        fig, ax2d = plt.subplots(figsize=(7, 5))
        ax3d = None
    elif STYLE == "3d":
        fig = plt.figure(figsize=(7, 5.5))
        ax3d = fig.add_subplot(111, projection="3d")
        ax2d = None
    else:  # both
        fig = plt.figure(figsize=(13, 5.5))
        ax2d = fig.add_subplot(1, 2, 1)
        ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    def draw(frame):
        step = steps[frame]
        if ax2d is not None:
            ax2d.clear()
            ax2d.plot(S_line, analytic_line, "k--", lw=2.5, label="analytic (truth)")
            ax2d.plot(S_line, nn_lines[frame], "C0-", lw=2.5, label="network")
            ax2d.plot(S_line, np.clip(S_line - STRIKE, 0, None),
                      color="grey", ls=":", lw=1.2, label="payoff")
            ax2d.set_xlabel("stock price  S")
            ax2d.set_ylabel("option value  V")
            ax2d.set_ylim(-1, vmax * 1.05)
            ax2d.set_title(f"value curve (t = {t_line:.2f})")
            ax2d.legend(loc="upper left", fontsize=8)
            ax2d.grid(alpha=0.3)

        if ax3d is not None:
            ax3d.clear()
            ax3d.plot_surface(t_surf, S_surf, nn_surfs[frame],
                              cmap="viridis", vmin=vmin, vmax=vmax)
            ax3d.set_xlabel("time t", fontsize=8)
            ax3d.set_ylabel("stock S", fontsize=8)
            ax3d.set_zlabel("value V", fontsize=8)
            ax3d.set_zlim(vmin, vmax)
            ax3d.set_title("value surface")
            ax3d.tick_params(labelsize=6)

        fig.suptitle(f"PINN learning Black-Scholes — training step {step}",
                     fontsize=13)
        return []

    print(f"Rendering {len(paths)} frames...")
    anim = animation.FuncAnimation(
        fig, draw, frames=len(paths), interval=1000 / FPS, blit=False
    )

    # save GIF (pillow)
    anim.save(OUT_GIF, writer=animation.PillowWriter(fps=FPS))
    print(f"  wrote {OUT_GIF}")

    # try MP4 (ffmpeg) - optional, nicer quality/smaller
    try:
        anim.save(OUT_MP4, writer=animation.FFMpegWriter(fps=FPS, bitrate=1800))
        print(f"  wrote {OUT_MP4}")
    except Exception as e:
        print(f"  (skipped MP4 - ffmpeg not available: {e})")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()