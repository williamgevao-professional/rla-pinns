"""Standalone BSDE self-consistency losses for log-space Black-Scholes.

Not yet wired into training. Run this file directly to test against the
analytic solution.
"""
import torch
from torch import Tensor, randn, zeros, cat, arange, linspace, stack
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


# ---------------------------------------------------------------------------
# Un-EM-BSDE (Seo et al., "Unbiased and Second-Order-Free Training for
# High-Dimensional PDEs"). Port of `UnEMBSDE_loss` in their JAX solver.
#
# Per path and step, draw K = main_stack + (p-1)*sub_stack independent noises
# eta_k ~ N(0, dt) (the first one is the path's own increment dW), form the EM
# one-step error e_k = Y + r*Y*dt + sigma*Z*eta_k - u(tau - dt, x_k) at each
# shifted point x_k, average e_k within blocks, and multiply the block means.
# Conditional on x the blocks are independent, so E[main * prod sub] = r(x)^p
# with r(x) the true local residual of the network: the O(dt^2) variance term
# that biases the plain EM loss is gone, and no u_xx is needed.
# ---------------------------------------------------------------------------

def sample_unem_noise(dW: Tensor, K: int) -> Tensor:
    """Noises for the Un-EM loss: [n_paths, steps, K], eta[..., 0] = dW."""
    n_paths, steps = dW.shape
    dt = MATURITY / steps
    eta = (dt ** 0.5) * randn(n_paths, steps, K, dtype=dW.dtype, device=dW.device)
    eta[..., 0] = dW
    return eta


def unem_components(model: Callable, X: Tensor, Y: Tensor, Z: Tensor, eta: Tensor,
                    main_stack: int = 5, sub_stack: int = 5, p: int = 2
                    ) -> Tuple[Tensor, Tensor]:
    """Block-averaged one-step errors of the Un-EM loss.

    Args:
        model: maps [N, 2] points (tau, x) to [N, 1] values.
        X:     [n_paths, steps+1, 2] path points (tau, x).
        Y, Z:  [n_paths, steps] network value and d/dx at the step starts.
        eta:   [n_paths, steps, K] noises, K = main_stack + (p-1)*sub_stack.

    Returns:
        main: [n_paths, steps] mean of e_k/dt over the main block.
        sub:  [n_paths, steps] product over the p-1 sub blocks of their means.
    """
    n_paths, steps = Y.shape
    K = main_stack + (p - 1) * sub_stack
    assert eta.shape == (n_paths, steps, K), (eta.shape, (n_paths, steps, K))
    dt = MATURITY / steps

    tau_next = X[:, 1:, 0].unsqueeze(-1).expand(-1, -1, K)
    x_k = X[:, :-1, 1].unsqueeze(-1) + (RATE - 0.5 * SIGMA**2) * dt + SIGMA * eta
    u_hat = model(stack([tau_next, x_k], dim=-1).reshape(-1, 2)).view(n_paths, steps, K)

    predicted = Y.unsqueeze(-1) * (1.0 + RATE * dt) + SIGMA * Z.unsqueeze(-1) * eta
    e = (predicted - u_hat) / dt

    main = e[..., :main_stack].mean(dim=-1)
    if p == 1:  # single block: plain (biased-free but noisy) residual estimate
        return main, torch.ones_like(main)
    sub = e[..., main_stack:].reshape(n_paths, steps, p - 1, sub_stack).mean(dim=-1).prod(dim=-1)
    return main, sub


def bsde_loss_unem(model: Callable, X: Tensor, dW: Tensor, main_stack: int = 5,
                   sub_stack: int = 5, p: int = 2) -> Tensor:
    """Unbiased EM (Un-EM-BSDE) loss, 0.5 * mean(main * prod sub).

    Scaled by 1/dt^p so that for p = 2 it estimates the same quantity as
    `bsde_loss_em` minus the EM variance term. Note the sample value can be
    negative; only its expectation is >= 0.
    """
    n_paths, n_pts, _ = X.shape
    steps = n_pts - 1
    K = main_stack + (p - 1) * sub_stack

    u, u_x = _eval_network(model, X.reshape(-1, 2))
    Y = u.view(n_paths, n_pts)[:, :-1]
    Z = u_x.view(n_paths, n_pts)[:, :-1]

    eta = sample_unem_noise(dW, K)
    main, sub = unem_components(model, X, Y, Z, eta, main_stack, sub_stack, p)
    return 0.5 * (main * sub).mean()


if __name__ == "__main__":
    from torch.nn import Module
    from rla_pinns.black_scholes_logS_equation import bs_call_price

    class Analytic(Module):
        def forward(self, X):
            return bs_call_price(X)

    torch.manual_seed(0)

    print("EM vs Heun vs Un-EM loss of the analytic solution, by step count:")
    for steps in [10, 25, 50, 100, 200]:
        X, dW = sample_paths(10000, steps=steps)
        em = bsde_loss_em(Analytic(), X, dW)
        heun = bsde_loss_heun(Analytic(), X, dW)
        unem = bsde_loss_unem(Analytic(), X, dW)
        print(f"  steps={steps:4d}  dt={MATURITY/steps:.4f}  "
              f"EM={em.item():.6e}  Heun={heun.item():.6e}  UnEM={unem.item(): .6e}")