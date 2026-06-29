"""Tests and commented simplified Black–Scholes for European call options.

This file is intentionally verbose with comments to explain the structure
and connect to deep hedging (learning the hedge delta = ∂V/∂S).

SIMPLIFIED PDE (no interest rate, no drift):
  ∂V/∂t + (1/2) σ² S² ∂²V/∂S² = 0
  V(T, S) = max(S - K, 0)

The stock price follows pure diffusion (no drift):
  dS_t = σ S_t dW_t

The hedge (the "delta" = ∂V/∂S) is what a deep hedging network learns to
approximate at each time step. The network's job: given (t, S), output the
correct number of shares to hold to offset the option liability.

This file contains:
- Analytic solutions to the simplified PDE (closed-form still exists).
- Tests verifying the analytic price satisfies the PDE (residual ≈ 0).
- Tests verifying delta (∂V/∂S) matches both analytic formula and autograd.
- Parametrization by horizon N ∈ {30, 250}, which defines dt = T/N when
  simulating paths (but here we test the continuous-time formula).

Notes for reading:
- Inputs X = [t, S]: current time and stock price.
- Maturity T = 1.0 year (fixed).
- Time-to-maturity: tau = T - t.
- Derivatives computed per-sample (loop over batch) for clarity.
"""

from math import sqrt

import pytest
import torch
from torch import float64, manual_seed


def bs_call_price_simplified(S, K, tau, sigma):
    """Analytic European call price under simplified PDE (no interest, no drift).

    PDE: ∂V/∂t + (1/2) σ² S² ∂²V/∂S² = 0, V(T,S) = max(S-K, 0)
    
    With no interest rate (r=0) and no drift, the solution is still Black–Scholes
    but simplified:
      V(t, S) = S * N(d1) - K * N(d2)
    
    ========== WHAT ARE d1, d2, N(d1), N(d2)? ==========
    
    • d1 and d2 are **intermediate calculations** (standardized normal variables).
      They capture two things: (1) moneyness (how far in/out of the money),
      (2) time value (volatility * sqrt(time)).
      
    • N(·) is the **cumulative normal CDF** (standard normal, mean=0, std=1).
      It maps a real number to a probability in [0, 1].
      
    • N(d1) is the **probability** that S_T > K (roughly; more precisely, it's
      the option delta).
      
    • N(d2) is the **risk-neutral probability** that the option expires in-the-money.
    
    • The formula V = S*N(d1) - K*N(d2) is a **weighted average** of the payoff
      S*1 - K*1 (if in-the-money) vs 0 (if out-of-the-money).
    
    ========== WHY USE tau, NOT dt? ==========
    
    • dt = 1/N is the **step size for discretizing time** (e.g., 1/30 if N=30).
      It's only needed if you simulate paths step-by-step.
      
    • tau = T - t is the **time remaining** at a given state (t, S).
      No matter how you slice time, tau tells you "how many years until expiration."
      
    • These are orthogonal: dt is about simulation resolution; tau is about
      state-dependent time-to-expiration.
      
    • Example: T = 1 year, N = 30 (so dt = 1/30), current time t = 0.5 year.
      Then tau = 1 - 0.5 = 0.5 years remaining.
      dt = 1/30 is irrelevant to this particular pricing calculation—we only
      care that we have 0.5 years left.

    Args:
        S: spot price (tensor scalar)
        K: strike price (float)
        tau: time-to-maturity = T - t (float >= 0); **not** dt
        sigma: volatility (constant)

    Returns:
        torch scalar tensor of option value
    """
    # Ensure consistent dtype for numerical stability
    S = torch.as_tensor(S, dtype=float64)
    K = float(K)
    tau = float(tau)
    sigma = float(sigma)

    if tau == 0.0:
        # At maturity: intrinsic value (payoff)
        return torch.clamp(S - K, min=0.0).to(dtype=float64)

    # Avoid log(0) by clamping S
    S_clamped = torch.clamp(S, min=1e-12)
    sqrt_tau = sqrt(tau)
    
    # ===== COMPUTING d1 and d2 =====
    # d1 = (ln(S/K) + (σ²/2) * tau) / (σ√tau)
    # d2 = d1 - σ√tau
    #
    # Intuition:
    #   • ln(S/K) ≈ log-moneyness (how far in/out of money)
    #   • (σ²/2) * tau ≈ time value (more time = more volatility = higher value)
    #   • σ√tau ≈ standard deviation of log-price moves over tau
    #   • d1 is roughly "how many standard deviations above strike?"
    #   • d2 shifts by one std-dev (the "jump" from probability of call being ITM
    #     to the adjusted strike)
    #
    # The two are used in TWO probabilities:
    #   • N(d1) → probability-like quantity (related to delta, the hedge)
    #   • N(d2) → risk-neutral prob. option ends ITM
    
    d1 = (torch.log(S_clamped / K) + 0.5 * sigma ** 2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    # ===== EVALUATING THE CDF =====
    # N(·) = cumulative normal CDF → maps any real number to [0, 1]
    # Normal(0, 1) is standard normal: mean 0, std 1
    # cdf(x) = P(Z <= x) where Z ~ N(0, 1)
    #
    # Example: N(0) = 0.5 (50% of mass below zero)
    #          N(2) ≈ 0.977 (97.7% below two std-devs)
    
    normal = torch.distributions.Normal(0.0, 1.0)
    Nd1 = normal.cdf(d1)  # N(d1) in [0, 1]
    Nd2 = normal.cdf(d2)  # N(d2) in [0, 1]

    # Option value: S * N(d1) - K * N(d2)  [no discount factor since r=0]
    #
    # What does this mean?
    #   • S * N(d1)    = Expected value of owning stock, probability-weighted
    #   • K * N(d2)    = Expected strike you'd pay, probability-weighted
    #   • Difference   = Fair price to charge for the call option
    #
    # In simpler terms: if the option is likely to end ITM (N(d2) high),
    # you charge more. If volatility is high (tau is large), uncertainty
    # grows, and both N(d1) and N(d2) move, changing the price.
    
    return (S * Nd1 - K * Nd2).to(dtype=float64)


def bs_call_delta_simplified(S, K, tau, sigma):
    """Analytic delta (∂V/∂S) of the European call under simplified PDE.

    The delta is the hedge: how many shares to hold at time (t, S) to be
    perfectly hedged against infinitesimal price moves.

    Delta = N(d1), where d1 comes from the Black–Scholes formula above.
    
    ========== WHY IS DELTA = N(d1)? ==========
    
    • Remember: V = S*N(d1) - K*N(d2)
    • Taking ∂V/∂S:
        ∂V/∂S = N(d1) + S * (∂N(d1)/∂S) - K * (∂N(d2)/∂S)
    • After calculus (chain rule on the normal CDF), the two correction terms
      cancel out exactly—a beautiful property!
    • Result: ∂V/∂S = N(d1)
    
    • Intuition: N(d1) tells you "how sensitive the option is to stock moves."
      If N(d1) = 0.3, a $1 move in stock changes option value by ~$0.30.
      If N(d1) = 1.0 (deep in the money), a $1 move changes it by $1.00.
    
    At tau = 0 (maturity):
      - delta = 0 if S <= K (option expires worthless, no delta)
      - delta = 1 if S > K (you own one share per option, delta = 1)
    
    This discontinuity at S=K is the "kink" mentioned in the notes—the deep
    hedging network must learn to approximate this sharp transition when
    trained on finite-horizon paths.

    Args:
        S: spot price
        K: strike price
        tau: time-to-maturity
        sigma: volatility

    Returns:
        torch scalar, the hedge ratio (number of shares to hold per option)
    """
    S = torch.as_tensor(S, dtype=float64)
    tau = float(tau)
    sigma = float(sigma)

    if tau == 0.0:
        # Right-continuous payoff slope: 0 for S<=K, 1 for S>K
        return (S > K).to(dtype=float64)

    sqrt_tau = sqrt(tau)
    S_clamped = torch.clamp(S, min=1e-12)
    
    # d1 (simplified, r=0)
    d1 = (torch.log(S_clamped / K) + 0.5 * sigma ** 2 * tau) / (sigma * sqrt_tau)
    
    # Delta = N(d1)
    normal = torch.distributions.Normal(0.0, 1.0)
    return normal.cdf(d1).to(dtype=float64)


@pytest.mark.parametrize("n_steps", [30, 250], ids=["N=30", "N=250"])
def test_black_scholes_pde_simplified(n_steps: int):
    """Verify analytic price satisfies the simplified PDE (no interest, no drift).

    PDE: ∂V/∂t + (1/2) σ² S² ∂²V/∂S² = 0, V(T,S) = max(S-K, 0)
    
    We compute the residual:
      residual(t, S) = V_t + (1/2) σ² S² V_SS

    For the analytic solution, this should equal zero (up to machine precision).

    Args:
        n_steps: Number of time steps if we were simulating (30 or 250).
                 Defines dt = T / n_steps when discretizing [0,T].
                 Here we use continuous PDE but document the choice.
    
    Note: We use a simplified direct derivative check: V_t = -∂V/∂tau * ∂tau/∂t = -∂V/∂tau
    since tau = T - t, so ∂tau/∂t = -1.
    """
    manual_seed(0)
    dtype = float64

    # Problem setup
    T = torch.tensor(1.0, dtype=dtype)
    dt = 1.0 / n_steps
    sigma = 0.2
    K = 1.0

    # Test points: (t, S) from before maturity
    batch = 8
    t_vals = torch.linspace(0.0, T.item() - dt, batch, dtype=dtype)
    S_vals = torch.linspace(0.2, 1.8, batch, dtype=dtype)

    residuals = []
    for i in range(batch):
        t_i = t_vals[i : i + 1].requires_grad_(True)
        S_i = S_vals[i : i + 1].requires_grad_(True)
        
        tau_i = T - t_i  # time-to-maturity, shape (1,)

        # ===== Compute V(t, S) via tau = T - t =====
        # We use the closed form: V = S * N(d1) - K * N(d2)
        # where d1 = (ln(S/K) + σ²/2 * tau) / (σ√tau)
        #       d2 = d1 - σ√tau
        
        sqrt_tau = torch.sqrt(tau_i)
        d1 = (torch.log(S_i / K) + 0.5 * sigma ** 2 * tau_i) / (sigma * sqrt_tau)
        d2 = d1 - sigma * sqrt_tau
        
        normal = torch.distributions.Normal(0.0, 1.0)
        Nd1 = normal.cdf(d1)
        Nd2 = normal.cdf(d2)
        
        Vi = S_i * Nd1 - K * Nd2

        # ===== Compute ∂V/∂t =====
        # Since V depends on t through tau, we get ∂V/∂t via chain rule
        dV_dtau = torch.autograd.grad(Vi.sum(), tau_i, create_graph=True)[0]
        V_t = -dV_dtau  # ∂V/∂t = -∂V/∂tau (since tau = T - t)

        # ===== Compute ∂V/∂S =====
        dV_dS = torch.autograd.grad(Vi.sum(), S_i, create_graph=True, retain_graph=True)[0]
        
        # ===== Compute ∂²V/∂S² =====
        dV2_dS2 = torch.autograd.grad(dV_dS.sum(), S_i, create_graph=False)[0]

        # ===== PDE residual =====
        residual = V_t + 0.5 * sigma ** 2 * (S_i ** 2) * dV2_dS2
        residuals.append(residual.detach())

    residuals = torch.stack(residuals).squeeze()

    # The analytic solution satisfies the PDE to machine precision
    # Allow some numerical error from second derivatives
    assert torch.allclose(residuals, torch.zeros_like(residuals), rtol=1e-3, atol=1e-4)


def test_black_scholes_delta_analytic_vs_autograd():
    """Verify analytic delta matches autograd derivative ∂V/∂S.

    The HEDGE is the delta—the number of shares to hold.
    
    In practice, a deep hedging network learns to approximate this delta
    from finite-horizon simulated paths. Here we simply verify that both
    methods (analytic formula and autograd) agree on the hedge value.
    
    This is important because:
    - The network must learn to output a delta-like quantity.
    - We can use analytic deltas to validate training or as a loss reference.
    - Near S=K (the strike), delta changes sharply—this is the "kink"
      the network finds hardest to learn.
    """
    manual_seed(1)
    dtype = float64
    T = torch.tensor(1.0, dtype=dtype)
    sigma = 0.25
    K = 1.2

    batch = 6
    t_vals = torch.linspace(0.0, 0.9, batch, dtype=dtype)
    S_vals = torch.linspace(0.3, 2.0, batch, dtype=dtype)

    deltas_analytic = []
    deltas_autograd = []

    for i in range(batch):
        t_i = t_vals[i].item()
        S_i = S_vals[i : i + 1].requires_grad_(True)
        tau = T.item() - t_i

        # Compute V directly without function call to preserve autograd
        sqrt_tau = torch.sqrt(torch.tensor(tau, dtype=dtype))
        d1 = (torch.log(S_i / K) + 0.5 * sigma ** 2 * tau) / (sigma * sqrt_tau)
        d2 = d1 - sigma * sqrt_tau
        
        normal = torch.distributions.Normal(0.0, 1.0)
        Nd1 = normal.cdf(d1)
        
        Vi = S_i * Nd1 - K * normal.cdf(d2)
        
        # Autograd derivative: ∂V/∂S
        delta_auto = torch.autograd.grad(Vi.sum(), S_i, create_graph=False)[0]
        deltas_autograd.append(delta_auto.detach())
        
        # Analytic delta
        deltas_analytic.append(bs_call_delta_simplified(S_i.detach(), K, tau, sigma))

    deltas_autograd = torch.stack(deltas_autograd)
    deltas_analytic = torch.stack(deltas_analytic)

    # They should match closely (away from tau=0 and S=K where discontinuities exist)
    assert torch.allclose(deltas_autograd, deltas_analytic, rtol=1e-6, atol=1e-6)
