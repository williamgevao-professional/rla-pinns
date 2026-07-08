"""
Step 1: extract delta (dV/dS) from a trained PINN and validate vs analytic.

Network is in log-coords (tau, x), x = log S. Chain rule:
    dV/dS = (1/S) * dV/dx
Analytic call delta (r=0):  Delta = N(d1),  d1 as in bs_call_price.

Produces validate_delta.png: PINN delta vs analytic N(d1) across S, at several taus.

Run from repo root, venv active:
    python validate_delta.py
"""

import glob, os, re
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints_rngd_fine"   # or checkpoints_rngd
CHECKPOINT_GLOB = "black-scholes-logS_1d_call_payoff_mlp-tanh-64_*_step*.pt"
TAUS = [1.0, 0.5, 0.25]                     # time slices to check
DTYPE = torch.float64

from rla_pinns.black_scholes_logS_equation import SIGMA, STRIKE, X_MIN, X_MAX


def build_model():
    return nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 1)).to(DTYPE)


def load_latest():
    paths = glob.glob(os.path.join(CHECKPOINT_DIR, CHECKPOINT_GLOB))
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {CHECKPOINT_DIR!r}")
    step = lambda p: int(re.search(r"step(\d+)\.pt$", p).group(1))
    path = max(paths, key=step)
    ckpt = torch.load(path, map_location="cpu")
    m = build_model(); m.load_state_dict(ckpt["model"]); m.eval()
    print(f"loaded {os.path.basename(path)} (step {ckpt.get('step','?')})")
    return m


def pinn_delta(model, tau, S):
    """dV/dS = (1/S) dV/dx via autodiff. tau, S: 1D numpy arrays."""
    x = np.log(S)
    X = torch.tensor(np.stack([np.full_like(x, tau), x], 1),
                     dtype=DTYPE, requires_grad=True)
    V = model(X)
    dV = torch.autograd.grad(V.sum(), X)[0][:, 1]      # dV/dx
    dVdx = dV.detach().numpy()
    return dVdx / S                                     # * (1/S) -> dV/dS


def analytic_delta(tau, S):
    """Call delta (r=0) = N(d1)."""
    from scipy.stats import norm
    if tau <= 0:
        return (S > STRIKE).astype(float)               # step at expiry
    d1 = (np.log(S / STRIKE) + 0.5 * SIGMA**2 * tau) / (SIGMA * np.sqrt(tau))
    return norm.cdf(d1)


def main():
    model = load_latest()
    S = np.linspace(0.3, 3.0, 300)

    fig, axes = plt.subplots(1, len(TAUS), figsize=(5 * len(TAUS), 4.5), squeeze=False)
    for ax, tau in zip(axes[0], TAUS):
        d_net = pinn_delta(model, tau, S)
        d_true = analytic_delta(tau, S)
        err = np.abs(d_net - d_true).max()
        ax.plot(S, d_true, "k--", lw=2.5, label="analytic N(d1)")
        ax.plot(S, d_net, "C0-", lw=2, label="PINN dV/dS")
        ax.axvline(STRIKE, color="grey", ls=":", lw=1)
        ax.set_xlabel("stock price S"); ax.set_ylabel("delta")
        ax.set_title(f"tau = {tau}  (max err {err:.2e})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        print(f"  tau={tau}: max|delta_net - delta_true| = {err:.3e}")

    fig.suptitle("PINN delta vs analytic Black-Scholes delta", fontsize=13)
    fig.tight_layout()
    fig.savefig("validate_delta.png", dpi=140, bbox_inches="tight")
    print("wrote validate_delta.png")


if __name__ == "__main__":
    main()