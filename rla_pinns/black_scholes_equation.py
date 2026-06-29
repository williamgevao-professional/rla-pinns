"""Functionality for solving the (simplified) Black-Scholes equation.

This module mirrors the public interface of ``heat_equation.py`` so it plugs
into the universal ``train.py`` training script. It solves the simplified
Black-Scholes PDE for a European call (no interest rate, no drift):

    V_t + 1/2 * sigma^2 * S^2 * V_SS = 0,    V(T, S) = max(S - K, 0).

Inputs follow the repo convention ``X = [t, S]`` (time first, then the single
spatial/price coordinate), so ``dim_Omega = 1`` and the full input dim is 2.

IMPORTANT - scope of this module:
- Only the autograd path is implemented (``evaluate_interior_loss``). This is
  what Adam / SGD / LBFGS / plain HessianFree use.
- The KFAC path (``evaluate_interior_loss_and_kfac`` and friends) is NOT
  implemented, because the repo's forward-Laplacian / KFAC machinery is built
  for the *unweighted* Laplacian. The S^2-weighting in the Black-Scholes
  operator breaks that assumption, so a naive copy would silently produce a
  wrong curvature matrix. The PDE-aware optimizers (KFAC, ENGD, SPRING, RNGD,
  HessianFreeCached) are therefore unsupported for this equation for now.
"""

from math import sqrt
from typing import Dict, List, Tuple, Union

from einops import einsum
from torch import Tensor, clamp, distributions, log, rand, zeros
from torch.nn import Module
from os import environ

from rla_pinns.autodiff_utils import (
    autograd_input_hessian,  # gives the second derivatives of V w.r.t the input coordinates (t, S) e.g. V_tt, V_tS, V_SS
    autograd_input_jacobian,  # gives the first derivatives of V w.r.t the input coordinates (t, S) e.g V_t and V_S
)


SIGMA = 0.2  # volatility
STRIKE = 1.0  # strike price K - can buy at time T
MATURITY = float(environ.get("BS_MATURITY", 1.0))  # maturity T (time runs over [0, T])
S_MAX = 3.0  # upper end of the truncated price domain [0, S_MAX] - can't be infinite


def _standard_normal_cdf(x: Tensor) -> Tensor:
    """Cumulative distribution function of the standard normal at ``x``."""
    normal = distributions.Normal(0.0, 1.0)
    return normal.cdf(x)


def bs_call_price(X: Tensor) -> Tensor:
    # Analytic European-call price under the simplified Black-Scholes PDE - used to measure test error and to produce terminal condition targets
   
    t = X[:, [0]]  # all rows, column 0 (time)
    S = X[:, [1]] # all rows, column 1 (price/stock)
    tau = MATURITY - t  # time-to-maturity, shape (N, 1)

    S_clamped = clamp(S, min=1e-12)
    sqrt_tau = clamp(tau, min=0.0).sqrt()

    # Where tau == 0 we return the intrinsic payoff directly to avoid 0/0.
    at_maturity = tau <= 0.0

    # Safe sqrt_tau for the division (the masked entries are overwritten below).
    safe_sqrt_tau = clamp(sqrt_tau, min=1e-12)
    d1 = (log(S_clamped / STRIKE) + 0.5 * SIGMA**2 * tau) / (SIGMA * safe_sqrt_tau)  # intermediates to help compute option price
    d2 = d1 - SIGMA * safe_sqrt_tau

    value = S * _standard_normal_cdf(d1) - STRIKE * _standard_normal_cdf(d2)  # returns the option price at any (t, S)
    payoff = clamp(S - STRIKE, min=0.0)

    return at_maturity.to(value.dtype) * payoff + (~at_maturity).to(value.dtype) * value  # at expiry, return payoff; otherwise return the Black-Scholes price


def bs_call_payoff(X: Tensor) -> Tensor:  # terminal payoff
    S = X[:, [1]]
    return clamp(S - STRIKE, min=0.0)



def interior_points(N: int) -> Tensor:
    """Draw interior collocation points ``(t, S)`` with t in [0, T), S in [0, S_MAX].

    Args:
        N: Number of points to draw.

    Returns:
        Tensor of shape ``(N, 2)`` with columns ``[t, S]``.
    """
    # rand(N, 1) gives N random numbers in [0, 1), so t is in [0, T) and S is in [0, S_MAX)
    t = MATURITY * rand(N, 1)
    S = S_MAX * rand(N, 1)
    return _cat_time_space(t, S)


def terminal_points(N: int) -> Tensor:
    # Draw points on the terminal condition: t = T, S in [0, S_MAX].

    # This is where the payoff kink at S = K lives. We deliberately oversample
    # near the strike so the network actually sees the stiff transition that
    # drives the ill-conditioning we want to study.

    
    
    from torch import full

    # Half uniform over the domain, half concentrated in a band around the strike 
    # N = number of sampled points
    n_strike = N // 2
    n_uniform = N - n_strike

    S_uniform = S_MAX * rand(n_uniform, 1)
    # band of +/- 0.25 around the strike, clipped to the domain
    S_strike = clamp(STRIKE + 0.5 * (rand(n_strike, 1) - 0.5), min=0.0, max=S_MAX)

    from torch import cat

    S = cat([S_uniform, S_strike], dim=0)
    t = full((N, 1), float(MATURITY)) # all of these points are at the terminal time T
    return _cat_time_space(t, S)


def spatial_boundary_points(N: int) -> Tensor:  
    # Draw points on the two price edges of the rectangle - spatial boundary S in {0, S_MAX}, random time in [0, T]
    
    from torch import cat, full

    n_low = N // 2
    n_high = N - n_low
    S_low = zeros(n_low, 1) # half the points at price = 0 (bottom edge)
    S_high = full((n_high, 1), float(S_MAX)) # half the points at price = S_MAX (top edge)
    S = cat([S_low, S_high], dim=0) # stack them into one column
    t = MATURITY * rand(N, 1) # at random times along each edge
    return _cat_time_space(t, S)


def _cat_time_space(t: Tensor, S: Tensor) -> Tensor:  # concatenate a time column and a price column into [t, S]
    from torch import cat
    return cat([t, S], dim=1)


# ----------------------------------------------------------------------------
# Interior loss (autograd only).
# ----------------------------------------------------------------------------
def evaluate_interior_loss(
    model: Union[Module, List[Module]], X: Tensor, y: Tensor
) -> Tuple[Tensor, Tensor, None]:
    """Evaluate the interior (PDE-residual) loss for Black-Scholes.

    Residual:  V_t + 1/2 * sigma^2 * S^2 * V_SS  -  y   (with y = 0).

    Both a ``Module`` and a ``list`` of layers are accepted; in either case the
    derivatives are computed with autograd. (Unlike heat, the ``list`` branch
    does NOT use the forward-Laplacian framework, because that path feeds KFAC,
    which is unsupported here.) This keeps LBFGS - which passes ``layers`` -
    working.

    Args:
        model: The model, or the list of layers forming the sequential model.
        X: Interior inputs of shape ``(N, 2)`` with columns ``[t, S]``.
        y: Target tensor of shape ``(N, 1)`` (all zeros).

    Returns:
        ``(loss, residual, None)``. The trailing ``None`` matches the heat
        signature (no curvature intermediates are produced).

    Raises:
        ValueError: If ``model`` is neither a Module nor a list of Modules.
    """
    if isinstance(model, Module): # different optimisers pass the model in different formats; this handles both cases
        net = model
    elif isinstance(model, list) and all(isinstance(layer, Module) for layer in model):
        from torch.nn import Sequential

        net = Sequential(*model)
    else:
        raise ValueError(
            "Model must be a Module or a list of Modules that form a sequential model."
            f" Got: {model}."
        )

    # Spatial coordinate index is 1 (column 0 is time).
    spatial = [1]

    # V_t : time derivative, column 0 of the input Jacobian.
    time_jacobian = autograd_input_jacobian(net, X).squeeze(1)[:, [0]]

    # V_SS : second derivative w.r.t. S. With a single spatial coordinate the
    # spatial Hessian is 1x1; take its only entry. (Using the trace via einsum
    # generalises cleanly and matches the heat-equation style.)
    spatial_hessian = autograd_input_hessian(net, X, coordinates=spatial)
    V_SS = einsum(spatial_hessian, "batch i i -> batch").unsqueeze(1)

    S = X[:, [1]]
    residual = time_jacobian + 0.5 * SIGMA**2 * (S**2) * V_SS - y # y is tensor of zeros, so this is just the PDE residual at each point

    return 0.5 * (residual**2).mean(), residual, None # 0.5 for mathematical convenience in derivatives

 

    

    
