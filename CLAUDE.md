# rla-pinns — project context

Research placement at Mila (William Gevao, supervised by Felix Dangel).
Fork of Felix's group's `rla-pinns`. PINN / BSDE conditioning study on a 1D
Black-Scholes testbed.

---

## HARD CONSTRAINTS — read before editing anything

1. **Never edit tracked files on Vulcan.** Workflow is: edit on Mac → commit →
   push → `git pull` on Vulcan → run. Three merge conflicts have already come
   from breaking this.
2. **Never clone reference repos inside this repo.** External code (e.g.
   `Un-EM-BSDE`, `heunbsde`) lives in `~/Desktop/refs/`, outside the git tree.
3. **`--dim_Omega 1` is mandatory** on every run. It defaults to 2 and gives a
   shape error.
4. Do not re-litigate settled work — see "Done, do not redo" below.

---

## Environments

| | path | venv |
|---|---|---|
| Mac (edit here) | `~/Desktop/rla-pinns` | `.venv` |
| Vulcan (run here) | `~/scratch/rla-pinns` | `venv` |

- GitHub: `git@github.com:williamgevao-professional/rla-pinns.git`
- Cluster: Vulcan (Alliance Canada / Amii), SLURM, L40s GPUs, account `aip-fdangel`
- wandb: `williamgevao202-concordia-university/deep-hedging-compare`

---

## The PDE

In log-price `x = log S`, time-to-maturity `tau = T - t`:

```
∂_τ u + (½σ² − r) ∂_x u − ½σ² ∂_xx u + r u = 0
```

σ = 0.2, r = 0.05 → coefficients `−0.03`, `−0.02`, `+0.05`.

- Initial condition at τ=0: `max(e^x − K, 0)`, K = 1
- Forward SDE: `dX = (r − ½σ²) dt + σ dW`
- Analytic: `u = e^x Φ(d₁) − K e^{−rτ} Φ(d₂)`,
  `d₁,₂ = (x + (r ± ½σ²)τ) / (σ√τ)`
- **Domain is unbounded.** No spatial boundary conditions. Uniqueness comes from
  Tychonov's growth condition `|u| ≤ M e^{ax²}`, satisfied since `u ≤ e^x`.
  Do not reintroduce walls or Dirichlet conditions — see "Done" below.

Verified to 1e-17 through the engine.

---

## Code layout

- `black_scholes_logS_equation.py` — the residual is the **Fokker-Planck
  evaluator** with coefficients partialled in (three `partial(...)` blocks at
  the bottom). `_MU_CONST = 0.5*SIGMA**2 - RATE` (= −0.03).
- `fokker_planck_equation.py` — adding `r` required a new `potential` parameter
  threaded through all three evaluator functions. The residual is assembled in
  exactly one place (the other two delegate) and gains `+ potential * p`.
- `rla_pinns/bsde_loss.py` — `sample_paths`, `bsde_loss_em`, `bsde_loss_heun`.
- `train.py` — `--loss_type {residual, bsde_em, bsde_heun}`, `--N_bsde_paths`
  (default 60). **BSDE losses are wired into the first-order branch only**;
  an assertion enforces Adam/SGD/LBFGS.
- Both `l2_error` and `l2_error_path` print to stdout and log to wandb.
  `l2_error` (uniform over [-3,3]) is uninformative under path sampling —
  ignore it. `rl2_error_path` is the metric that matters.
- Profiler code has been removed.

### Standard sbatch pattern

```bash
E="ALL,OPT_ARGS"
B="--num_seconds 3000 --N_Omega 3000 --N_dOmega 500 --dim_Omega 1 \
   --equation black-scholes-logS --boundary_condition call_payoff \
   --interior_sampling path --batch_frequency 1 --wandb \
   --wandb_project deep-hedging-compare"

sbatch --export=$E="$B --optimizer Adam --Adam_lr 1e-3 \
  --loss_type bsde_em --wandb_id em_adam" \
  --gres=gpu:l40s:1 --time=01:30:00 compare_submit_vulcan.sh
```

---

## Results established so far

- **RNGD (ENGD-Woodbury) on the FS-PINN loss beats Adam by ~5×**:
  `rl2_error_path` 0.0014 vs 0.0070 at 3000s, and gets there much earlier.
  (FS-PINNs here = Park & Tu's: the full PINN residual, second derivatives
  included, evaluated along forward SDE trajectories, plus the IC at τ=0.)
- Heun-BSDE + Adam ≈ PINN + Adam (0.0078 vs 0.0070, within seed noise).
- EM-BSDE plateaus at 0.146 — that is the discretisation bias floor, not an
  optimisation failure. Flat across a 100× learning-rate range.
- Adam lr sweep: 1e-3 is best for both residual and Heun.
- Profiling RNGD (20 steps, L40s): Cholesky (`getrf_wo_pivot`) is 51% of self
  CUDA time, 15.4ms/step, float64. `cutlass_d884gemm` 19%.

**Caveat: single seeds throughout.** Any claim of parity or difference needs
2–3 seeds before it goes anywhere.

---

## Current task

Port Seo et al.'s **Un-EM-BSDE** loss into this codebase, then compute its
Hessian spectrum and check for negative eigenvalues.

- Paper: "Unbiased and Second-Order-Free Training for High-Dimensional PDEs",
  ICML 2026, arXiv 2605.14643
- Reference code: `github.com/seojaemin22/Un-EM-BSDE` → clone to
  `~/Desktop/refs/Un-EM-BSDE` (**not** inside this repo)
- Check what framework they use before porting — Park & Tu's Heun repo is JAX,
  so this may be a rewrite rather than a copy.

**Why:** their loss is a *product of two independent one-step errors*, not a
sum of squares. So it has no Gauss-Newton structure and can be negative. If the
curvature matrix is indefinite, second-order optimisers can't be applied to it —
which would be a clean asymmetry, since RNGD demonstrably does work on
FS-PINNs.

Read their loss implementation before writing any code.

---

## Done, do not redo

- Path sampling and the underdetermination diagnosis. On a bounded domain
  `[-3,3]` with Dirichlet walls there was an unsampled band between the cone
  (|x| ≤ 0.75) and the walls; loss went 15× *lower* while L2 went 7600× *worse*.
  Explanation is the maximum principle — it eliminates interior maxima pointwise
  using the PDE at each point, so unsampled points are never eliminated and
  uniqueness breaks. Fixed by going unbounded. Written up.
- The `r ≠ 0` extension and its verification (machine precision, confirmed twice).
- EM and Heun implementations, verified against the analytic solution; bias
  behaviour reproduces Park & Tu (EM flat, Heun falls at ~Δt^½).
- The Adam learning-rate sweep.
- Profiling.
