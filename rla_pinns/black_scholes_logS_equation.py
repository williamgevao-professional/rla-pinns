"""Black-Scholes in log-space, mapped onto the (standard, linear) Fokker-Planck engine.

Transforms applied to the simplified Black-Scholes PDE
    V_t + 0.5 sigma^2 S^2 V_SS = 0,   V(T, S) = max(S - K, 0):

    x   = log(S)        removes the S^2 weighting
    tau = T - t         flips backward -> forward (fixes the diffusion sign)

In (tau, x) coordinates this becomes the standard isotropic Fokker-Planck operator
    V_tau + div(V * mu) - 0.5 Tr(sigma sigma^T Hess V)
  = V_tau + mu * V_x    - 0.5 sigma_fp^2 V_xx                          (LINEAR)
with CONSTANT coefficients
    mu       = +0.5 * sigma^2   (= 0.02)     [constant => div_mu = 0]
    sigma_fp =  sigma           (= 0.2)

This mapping is verified numerically (see verify_bs_logS_mapping.py and
which_fp_engine.py): the standard FP engine matches BS exactly; the LOG FP engine
does NOT (it has a spurious nonlinear term), so we wrap the STANDARD engine.

Input convention: X = [tau, x] (time first, then the single spatial coord),
so dim_Omega = 1 and full input dim = 2.
"""

from functools import partial
from math import sqrt as _sqrt

from torch import Tensor, clamp, full, ones, zeros, cat, rand, log, exp
from os import environ

# IMPORTANT: wrap the STANDARD (linear) engine, NOT log_fokker_planck_equation.
from rla_pinns import fokker_planck_equation


SIGMA = 0.2                                       # volatility
STRIKE = 1.0                                      # strike K
MATURITY = float(environ.get("BS_MATURITY", 1.0)) # T (tau runs over [0, T])
X_MIN = float(environ.get("BS_XMIN", -3.0))       # log-price domain lower bound
X_MAX = float(environ.get("BS_XMAX", 3.0))        # log-price domain upper bound

_MU_CONST = 0.5 * SIGMA**2                         # 0.02
_SIGMA_FP = SIGMA                                  # 0.2


# ---------------------------------------------------------------------------
# Coefficient functions (constant), matching the engine's expected shapes.
# mu(x):      un-batched x -> (dim_Omega,);  batched X -> (batch, dim_Omega)
# div_mu(X):  X -> (batch, 1)   (zero, since mu is constant)
# sigma(X):   X -> (batch, dim_Omega, dim_Omega), here sigma_fp * I
# ---------------------------------------------------------------------------
def mu_bs(x: Tensor) -> Tensor:
    dim_Omega = x.shape[-1] - 1
    return _MU_CONST * ones(x.shape[:-1] + (dim_Omega,), dtype=x.dtype, device=x.device)


def div_mu_bs(X: Tensor) -> Tensor:
    return zeros(X.shape[:-1] + (1,), dtype=X.dtype, device=X.device)


def sigma_bs(X: Tensor) -> Tensor:
    batch_size, dim = X.shape
    dim_Omega = dim - 1
    from torch import eye
    return (
        _SIGMA_FP * eye(dim_Omega, dtype=X.dtype, device=X.device).unsqueeze(0)
    ).expand(batch_size, dim_Omega, dim_Omega)


# ---------------------------------------------------------------------------
# Analytic solution (ground truth), in (tau, x) coordinates.
# V(tau, x) = Black-Scholes call at (t = T - tau, S = exp(x)), r = 0.
# ---------------------------------------------------------------------------
def bs_call_price(X: Tensor) -> Tensor:
    from torch import distributions
    tau = X[:, [0]]
    x = X[:, [1]]
    S = exp(x)
    # remaining time to maturity in original coords equals tau here
    sqrt_tau = clamp(tau, min=0.0).sqrt()
    at_init = tau <= 0.0
    safe = clamp(sqrt_tau, min=1e-12)
    d1 = (log(clamp(S, min=1e-12) / STRIKE) + 0.5 * SIGMA**2 * tau) / (SIGMA * safe)
    d2 = d1 - SIGMA * safe
    N = distributions.Normal(0.0, 1.0)
    value = S * N.cdf(d1) - STRIKE * N.cdf(d2)
    payoff = clamp(S - STRIKE, min=0.0)
    return at_init.to(value.dtype) * payoff + (~at_init).to(value.dtype) * value


def bs_call_payoff(X: Tensor) -> Tensor:
    """Initial condition at tau = 0: max(exp(x) - K, 0)."""
    x = X[:, [1]]
    return clamp(exp(x) - STRIKE, min=0.0)


# ---------------------------------------------------------------------------
# Collocation samplers. Domain: tau in [0, T], x in [X_MIN, X_MAX].
# ---------------------------------------------------------------------------
def _cat_time_space(t: Tensor, x: Tensor) -> Tensor:
    return cat([t, x], dim=1)


def interior_points(N: int) -> Tensor:
    tau = MATURITY * rand(N, 1)
    x = X_MIN + (X_MAX - X_MIN) * rand(N, 1)
    return _cat_time_space(tau, x)


def terminal_points(N: int) -> Tensor:
    """Initial condition lives at tau = 0 (was the terminal payoff at t = T)."""
    # oversample near the strike x = log(K) = 0 where the payoff kink lives
    n_strike = N // 2
    n_uniform = N - n_strike
    x_uniform = X_MIN + (X_MAX - X_MIN) * rand(n_uniform, 1)
    x_strike = clamp(0.0 + 0.5 * (rand(n_strike, 1) - 0.5), min=X_MIN, max=X_MAX)
    x = cat([x_uniform, x_strike], dim=0)
    tau = zeros(N, 1)              # tau = 0 initial condition
    return _cat_time_space(tau, x)


def spatial_boundary_points(N: int) -> Tensor:
    """Price edges x in {X_MIN, X_MAX}, random tau in [0, T]."""
    n_low = N // 2
    n_high = N - n_low
    x_low = full((n_low, 1), float(X_MIN))
    x_high = full((n_high, 1), float(X_MAX))
    x = cat([x_low, x_high], dim=0)
    tau = MATURITY * rand(N, 1)
    return _cat_time_space(tau, x)


# ---------------------------------------------------------------------------
# Wire the STANDARD Fokker-Planck engine with the BS coefficients.
# These three names are what train.py / EVAL_FNS / LOSS_EVALUATORS import.
# ---------------------------------------------------------------------------
evaluate_interior_loss = partial(
    fokker_planck_equation.evaluate_interior_loss,
    mu=mu_bs,
    sigma=sigma_bs,
    div_mu=div_mu_bs,
    sigma_isotropic=True,
)

evaluate_interior_loss_and_kfac = partial(
    fokker_planck_equation.evaluate_interior_loss_and_kfac,
    mu=mu_bs,
    sigma=sigma_bs,
    div_mu=div_mu_bs,
    sigma_isotropic=True,
)

evaluate_interior_loss_with_layer_inputs_and_grad_outputs = partial(
    fokker_planck_equation.evaluate_interior_loss_with_layer_inputs_and_grad_outputs,
    mu=mu_bs,
    sigma=sigma_bs,
    div_mu=div_mu_bs,
    sigma_isotropic=True,
)