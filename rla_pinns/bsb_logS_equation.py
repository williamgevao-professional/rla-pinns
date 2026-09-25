"""Black-Scholes-Barenblatt (BSB) in log-price coordinates, on the Fokker-Planck engine.

The d-dimensional BSB benchmark of Seo et al. (Un-EM-BSDE, App. E.2), Han et al.:

    u_t + 0.5 * alpha^2 * sum_i x_i^2 u_{x_i x_i} = r (u - x . grad u),
    u(T, x) = |x|^2,   X0 = (1, 1/2, 1, 1/2, ...),   alpha = 0.4, r = 0.05, d = 100,
    exact:  u(t, x) = exp((r + alpha^2)(T - t)) |x|^2.

With y = log x (component-wise) and tau = T - t this is constant-coefficient:

    u_tau + (0.5 alpha^2 - r) sum_i u_{y_i} - 0.5 alpha^2 Laplacian_y u + r u = 0,

i.e. the same isotropic Fokker-Planck form as `black_scholes_logS_equation`
(mu = (0.5 alpha^2 - r) * 1, sigma_fp = alpha, potential = r), in d dimensions,
with initial condition g(y) = sum_i exp(2 y_i) at tau = 0.

Scale: |X0|^2 = 62.5, so by default the condition and solution are divided by d
(`BSB_NORMALISE=1`). The PDE is linear, so this rescales u by 1/d and leaves the
relative L2 error unchanged; it only keeps the network output O(1).

Input convention: X = [tau, y_1, ..., y_d], so dim_Omega = d = BSB_DIM (env, 100).
Run train.py with `--dim_Omega` equal to BSB_DIM; the samplers assert this.
"""

from functools import partial
from math import log as _log
from os import environ

from torch import Tensor, arange, cat, exp, full, ones, rand, randn, randperm, tensor, zeros

from rla_pinns import fokker_planck_equation

DIM = int(environ.get("BSB_DIM", 100))               # spatial dimension d
ALPHA = float(environ.get("BSB_ALPHA", 0.4))         # volatility
RATE = float(environ.get("BSB_RATE", 0.05))          # interest rate r
MATURITY = float(environ.get("BSB_MATURITY", 1.0))   # T; tau runs over [0, T]
NORMALISE = bool(int(environ.get("BSB_NORMALISE", 1)))
PATH_STEPS = int(environ.get("BSB_PATH_STEPS", 100))  # time steps per path (paper: N = 100)
GAUSS_STD = ALPHA * (MATURITY / 2) ** 0.5

_MU_CONST = 0.5 * ALPHA**2 - RATE                    # +0.03
_SCALE = 1.0 / DIM if NORMALISE else 1.0


def path_x0(dtype=None, device=None) -> Tensor:
    """log X0 = log(1, 1/2, 1, 1/2, ...), shape [d]."""
    y0 = [0.0 if i % 2 == 0 else -_log(2.0) for i in range(DIM)]
    return tensor(y0, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Coefficient functions (constant), same conventions as black_scholes_logS
# ---------------------------------------------------------------------------
def mu_bsb(x: Tensor) -> Tensor:
    dim_Omega = x.shape[-1] - 1
    return _MU_CONST * ones(x.shape[:-1] + (dim_Omega,), dtype=x.dtype, device=x.device)


def div_mu_bsb(X: Tensor) -> Tensor:
    return zeros(X.shape[:-1] + (1,), dtype=X.dtype, device=X.device)


def sigma_bsb(X: Tensor) -> Tensor:
    batch_size, dim = X.shape
    dim_Omega = dim - 1
    from torch import eye
    return (
        ALPHA * eye(dim_Omega, dtype=X.dtype, device=X.device).unsqueeze(0)
    ).expand(batch_size, dim_Omega, dim_Omega)


# ---------------------------------------------------------------------------
# Exact solution and initial condition, in (tau, y)
# ---------------------------------------------------------------------------
def bsb_terminal_payoff(X: Tensor) -> Tensor:
    """g(y) = sum_i exp(2 y_i) (= |x|^2), scaled by 1/d if NORMALISE."""
    y = X[..., 1:]
    return _SCALE * exp(2.0 * y).sum(dim=-1, keepdim=True)


def bsb_solution(X: Tensor) -> Tensor:
    tau = X[..., [0]]
    return exp((RATE + ALPHA**2) * tau) * bsb_terminal_payoff(X)


# ---------------------------------------------------------------------------
# Collocation samplers
# ---------------------------------------------------------------------------
def _check_dim(x: Tensor) -> None:
    assert x.shape[-1] == DIM, (
        f"BSB module is compiled for d = {DIM} (env BSB_DIM); got {x.shape[-1]}. "
        "Pass --dim_Omega equal to BSB_DIM."
    )


def _cat_time_space(t: Tensor, x: Tensor) -> Tensor:
    _check_dim(x)
    return cat([t, x], dim=1)


def gaussian_interior_points(N: int) -> Tensor:
    """Gaussian approximation to the path measure around the drifted start."""
    tau = MATURITY * rand(N, 1)
    t = MATURITY - tau
    y = path_x0() + (RATE - 0.5 * ALPHA**2) * t + ALPHA * t.sqrt() * randn(N, DIM)
    return _cat_time_space(tau, y)


def interior_points(N: int) -> Tensor:
    """No bounded box makes sense in 100-D; 'uniform' falls back to the Gaussian."""
    return gaussian_interior_points(N)


def terminal_points(N: int) -> Tensor:
    """tau = 0 points drawn from the path endpoint distribution (t = T)."""
    z = randn(N, DIM)
    y = path_x0() + (RATE - 0.5 * ALPHA**2) * MATURITY + ALPHA * MATURITY**0.5 * z
    tau = zeros(N, 1)
    return _cat_time_space(tau, y)


def sample_log_paths(n_paths: int, steps: int = None, dtype=None, device=None) -> Tensor:
    """Log-price GBM paths from X0. Returns [n_paths, steps+1, 1+d] rows (tau, y)."""
    steps = PATH_STEPS if steps is None else steps
    dt = MATURITY / steps
    y0 = path_x0(dtype=dtype, device=device)
    Z = randn(n_paths, steps, DIM, dtype=dtype, device=device)
    incr = (RATE - 0.5 * ALPHA**2) * dt + ALPHA * dt**0.5 * Z
    y = y0 + cat([zeros(n_paths, 1, DIM, dtype=dtype, device=device),
                  incr.cumsum(dim=1)], dim=1)                       # [n, steps+1, d]
    t = arange(steps + 1, dtype=y.dtype, device=device) * dt
    tau = (MATURITY - t).view(1, -1, 1).expand(n_paths, -1, 1)      # [n, steps+1, 1]
    return cat([tau, y], dim=-1)


def path_interior_points(N: int) -> Tensor:
    """N collocation points taken from simulated log-price paths (unbounded domain)."""
    n_paths = N // (PATH_STEPS + 1) + 1
    X = sample_log_paths(n_paths).reshape(-1, 1 + DIM)
    return X[randperm(X.shape[0])[:N]]


# ---------------------------------------------------------------------------
# Loss evaluators: the linear Fokker-Planck engine with BSB coefficients
# ---------------------------------------------------------------------------
evaluate_interior_loss = partial(
    fokker_planck_equation.evaluate_interior_loss,
    mu=mu_bsb,
    sigma=sigma_bsb,
    div_mu=div_mu_bsb,
    sigma_isotropic=True,
    potential=RATE,
)

evaluate_interior_loss_and_kfac = partial(
    fokker_planck_equation.evaluate_interior_loss_and_kfac,
    mu=mu_bsb,
    sigma=sigma_bsb,
    div_mu=div_mu_bsb,
    sigma_isotropic=True,
    potential=RATE,
)

evaluate_interior_loss_with_layer_inputs_and_grad_outputs = partial(
    fokker_planck_equation.evaluate_interior_loss_with_layer_inputs_and_grad_outputs,
    mu=mu_bsb,
    sigma=sigma_bsb,
    div_mu=div_mu_bsb,
    sigma_isotropic=True,
    potential=RATE,
)
