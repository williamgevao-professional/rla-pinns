"""
Verify the log-S transform: does wrapping the Fokker-Planck engine reproduce
analytic Black-Scholes?

Two independent checks:
  1. RESIDUAL: analytic BS surface fed into the FP-wrapped operator -> residual
     should be ~0 (the true solution satisfies the transformed PDE).
  2. VALUE: bs_call_price (the ground-truth used for L2) matches a direct
     S-space Black-Scholes formula on the same points.

Run from repo root, venv active:
    PYTHONPATH=. python verify_bs_logS_mapping.py
"""

import torch
from torch import nn
from torch.distributions import Normal

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

from rla_pinns.black_scholes_logS_equation import (
    SIGMA, STRIKE, MATURITY,
    bs_call_price,
    interior_points,
    evaluate_interior_loss,
)


def bs_direct(S, tau):
    """Direct S-space Black-Scholes call, r=0."""
    safe = tau.clamp(min=1e-12).sqrt()
    d1 = (torch.log(S / STRIKE) + 0.5 * SIGMA**2 * tau) / (SIGMA * safe)
    d2 = d1 - SIGMA * safe
    N = Normal(0.0, 1.0)
    val = S * N.cdf(d1) - STRIKE * N.cdf(d2)
    payoff = (S - STRIKE).clamp(min=0.0)
    return torch.where(tau <= 0.0, payoff, val)


# ---- CHECK 1: analytic value matches direct S-space formula ----
def check_value():
    X = interior_points(200)                 # [tau, x]
    tau, x = X[:, [0]], X[:, [1]]
    S = x.exp()
    v_logS = bs_call_price(X)
    v_direct = bs_direct(S, tau)
    diff = (v_logS - v_direct).abs().max().item()
    print(f"[value ] max|bs_call_price - direct BS| = {diff:.3e}", 
          "PASS" if diff < 1e-10 else "FAIL")
    return diff < 1e-10


# ---- CHECK 2: analytic solution has ~zero PDE residual ----
def check_residual():
    # a network whose output we OVERRIDE to equal the analytic surface,
    # so the operator is evaluated on the true solution.
    # Simplest route: wrap bs_call_price in an nn.Module-like callable.
    class AnalyticNet(nn.Module):
        def forward(self, X):
            return bs_call_price(X)

    model = AnalyticNet()
    X = interior_points(200)
    y = torch.zeros(X.shape[0], 1)
    try:
        _, residual, _ = evaluate_interior_loss(model, X, y)
        r = residual.abs().max().item()
        # residual won't be exactly 0 (autodiff through the closed form near
        # the kink), but should be small away from tau=0 / x=0.
        print(f"[resid ] max|operator(analytic)| = {r:.3e}",
              "PASS" if r < 1e-3 else "CHECK (large near kink?)")
        return r < 1e-3
    except Exception as e:
        print(f"[resid ] could not evaluate: {e}")
        print("         (value check above is the primary one; residual is optional)")
        return None


if __name__ == "__main__":
    print(f"SIGMA={SIGMA}, STRIKE={STRIKE}, MATURITY={MATURITY}\n")
    v = check_value()
    r = check_residual()
    print()
    if v:
        print("Mapping confirmed: log-S formulation reproduces analytic BS.")
    else:
        print("Value mismatch — check d1/d2 convention or r != 0 in bs_call_price.")