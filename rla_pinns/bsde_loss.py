"""Standalone BSDE self-consistency losses for log-space Black-Scholes.

Not yet wired into training. Run this file directly to test against the
analytic solution.
"""
import torch
from torch import Tensor, randn, zeros, cat, arange, linspace
from typing import Callable, Tuple

from rla_pinns.black_scholes_logS_equation import (
    SIGMA, RATE, MATURITY, PATH_X0, PATH_STEPS,
)


def sample_paths(n_paths: int, steps: int = None, dtype=torch.float64,
                 device=None) -> Tuple[Tensor, Tensor]:
    """Simulate log-price paths, keeping the Brownian increments.

    Returns:
        X:  [n_paths, steps+1, 2] with columns (tau, x)
        dW: [n_paths, steps]      the increments that generated each step
    """
    steps = PATH_STEPS if steps is None else steps
    dt = MATURITY / steps

    dW = (dt ** 0.5) * randn(n_paths, steps, dtype=dtype, device=device)
    incr = (RATE - 0.5 * SIGMA**2) * dt + SIGMA * dW
    x = PATH_X0 + cat([zeros(n_paths, 1, dtype=dtype, device=device),
                incr.cumsum(dim=1)], dim=1)# [n_paths, steps+1]

    # paths run forward in t; tau = T - t, so tau goes T -> 0
    t = arange(steps + 1, dtype=dtype, device=device) * dt
    tau = (MATURITY - t).unsqueeze(0).expand(n_paths, -1)   # [n_paths, steps+1]

    X = torch.stack([tau, x], dim=-1)                        # [n_paths, steps+1, 2]
    return X, dW


def _eval_network(model: Callable, X_flat: Tensor) -> Tuple[Tensor, Tensor]:
    """Evaluate u and du/dx at a batch of (tau, x) points."""
    X_flat = X_flat.detach().requires_grad_(True)
    u = model(X_flat)
    grad_u = torch.autograd.grad(u.sum(), X_flat, create_graph=True)[0]
    return u, grad_u[:, [1]]        # value, d/dx

def _eval_network_2nd(model: Callable, X_flat: Tensor
                      ) -> Tuple[Tensor, Tensor, Tensor]:
    """Evaluate u, du/dx and d2u/dx2 at a batch of (tau, x) points."""
    X_flat = X_flat.detach().requires_grad_(True)
    u = model(X_flat)
    g = torch.autograd.grad(u.sum(), X_flat, create_graph=True)[0]
    u_x = g[:, [1]]
    u_xx = torch.autograd.grad(u_x.sum(), X_flat, create_graph=True)[0][:, [1]]
    return u, u_x, u_xx


def bsde_loss_heun(model: Callable, X: Tensor, dW: Tensor) -> Tensor:
    """Heun (predictor-corrector) self-consistency loss.

    Uses the Stratonovich-corrected driver h* = r*Y - 0.5*sigma^2*u_xx.
    """
    n_paths, n_pts, _ = X.shape
    steps = n_pts - 1
    dt = MATURITY / steps

    u, u_x, u_xx = _eval_network_2nd(model, X.reshape(-1, 2))
    Y = u.view(n_paths, n_pts)
    Z = u_x.view(n_paths, n_pts)
    Yxx = u_xx.view(n_paths, n_pts)

    h_star = RATE * Y - 0.5 * SIGMA**2 * Yxx          # corrected driver

    predicted = (0.5 * (h_star[:, :-1] + h_star[:, 1:]) * dt
                 + 0.5 * SIGMA * (Z[:, :-1] + Z[:, 1:]) * dW)
    claimed = Y[:, 1:] - Y[:, :-1]

    residual = (claimed - predicted) / dt
    return 0.5 * (residual ** 2).mean()


def bsde_loss_em(model: Callable, X: Tensor, dW: Tensor) -> Tensor:
    """Euler-Maruyama one-step self-consistency loss."""
    n_paths, n_pts, _ = X.shape
    steps = n_pts - 1
    dt = MATURITY / steps

    u, u_x = _eval_network(model, X.reshape(-1, 2))
    Y = u.view(n_paths, n_pts)
    Z = u_x.view(n_paths, n_pts)

    # driver h = r * Y, evaluated at the start of each step
    predicted = RATE * Y[:, :-1] * dt + SIGMA * Z[:, :-1] * dW
    claimed = Y[:, 1:] - Y[:, :-1]

    residual = (claimed - predicted) / dt
    return 0.5 * (residual ** 2).mean()


if __name__ == "__main__":
    from torch.nn import Module
    from rla_pinns.black_scholes_logS_equation import bs_call_price

    class Analytic(Module):
        def forward(self, X):
            return bs_call_price(X)

    torch.manual_seed(0)

    print("EM vs Heun loss of the analytic solution, by step count:")
    for steps in [10, 25, 50, 100, 200]:
        X, dW = sample_paths(10000, steps=steps)
        em = bsde_loss_em(Analytic(), X, dW)
        heun = bsde_loss_heun(Analytic(), X, dW)
        print(f"  steps={steps:4d}  dt={MATURITY/steps:.4f}  "
              f"EM={em.item():.6e}  Heun={heun.item():.6e}")