"""Curvature spectra of the Un-EM-BSDE loss on 1D log-space Black-Scholes.

Question: is the curvature matrix of Seo et al.'s product loss indefinite, and
if so how badly? For each network state and each stack size we form

  H_unem   full Hessian of the sample Un-EM loss  0.5*mean(main*sub)
  G_unem   its "Gauss-Newton" part  0.5/N * sum_n (a_n b_n^T + b_n a_n^T),
           a_n = grad main_n, b_n = grad sub_n  (indefinite per sample)
  G_pop    the population Gauss-Newton matrix  1/N * sum_n g_n g_n^T with
           g_n = grad of the block mean over a very large stack (~ grad r)
  H_em, G_em  the same for the plain EM loss (G_em = J^T J is PSD)

and report eigenvalue counts / extremes. Small network, float64, CPU.

Usage:  python spectrum_unem.py [--stacks 1,5,50] [--adam_steps 1000] ...
"""
from argparse import ArgumentParser
from typing import Callable, Dict, List, Tuple

import torch
from torch import Tensor, nn, zeros
from torch.func import functional_call, grad, jacrev, vmap

from rla_pinns.black_scholes_logS_equation import (
    MATURITY, RATE, SIGMA, bs_call_price, bs_call_payoff,
)
from rla_pinns.bsde_loss import (
    bsde_loss_unem, sample_paths, sample_unem_noise, unem_components,
)

torch.set_default_dtype(torch.float64)


def hessian(f: Callable, chunk_size: int) -> Callable:
    """Reverse-over-reverse Hessian, chunked to bound memory."""
    return jacrev(jacrev(f), chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# Small MLP with a flat-parameter functional interface
# ---------------------------------------------------------------------------

def make_net(width: int, depth: int) -> nn.Module:
    layers: List[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers += [nn.Linear(width, 1)]
    return nn.Sequential(*layers)


class FlatParams:
    """Flatten/unflatten a module's parameters to one vector."""

    def __init__(self, net: nn.Module):
        self.net = net
        self.names = [n for n, _ in net.named_parameters()]
        self.shapes = [p.shape for _, p in net.named_parameters()]
        self.sizes = [p.numel() for _, p in net.named_parameters()]

    def flat(self) -> Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.net.parameters()])

    def unflat(self, theta: Tensor) -> Dict[str, Tensor]:
        out, i = {}, 0
        for n, shape, size in zip(self.names, self.shapes, self.sizes):
            out[n] = theta[i:i + size].view(shape)
            i += size
        return out

    def model_fn(self, theta: Tensor) -> Callable[[Tensor], Tensor]:
        params = self.unflat(theta)
        return lambda X: functional_call(self.net, params, (X,))

    def value_and_dx(self, theta: Tensor, X_flat: Tensor) -> Tuple[Tensor, Tensor]:
        """u and du/dx at [N, 2] points, differentiable in theta via torch.func."""
        params = self.unflat(theta)

        def u_single(x: Tensor) -> Tensor:
            return functional_call(self.net, params, (x.unsqueeze(0),))[0, 0]

        u = vmap(u_single)(X_flat)
        ux = vmap(grad(u_single))(X_flat)[:, 1]
        return u, ux


# ---------------------------------------------------------------------------
# Per-sample residual functions of theta (all torch.func-differentiable)
# ---------------------------------------------------------------------------

def unem_blocks(fp: FlatParams, theta: Tensor, X: Tensor, eta: Tensor,
                main_stack: int, sub_stack: int, p: int) -> Tuple[Tensor, Tensor]:
    n_paths, n_pts, _ = X.shape
    u, ux = fp.value_and_dx(theta, X.reshape(-1, 2))
    Y = u.view(n_paths, n_pts)[:, :-1]
    Z = ux.view(n_paths, n_pts)[:, :-1]
    main, sub = unem_components(fp.model_fn(theta), X, Y, Z, eta, main_stack, sub_stack, p)
    return main.reshape(-1), sub.reshape(-1)


def em_residual(fp: FlatParams, theta: Tensor, X: Tensor, dW: Tensor) -> Tensor:
    """EM one-step error / dt, [n_paths*steps]. Equals main block with stack 1."""
    n_paths, n_pts, _ = X.shape
    steps = n_pts - 1
    dt = MATURITY / steps
    u, ux = fp.value_and_dx(theta, X.reshape(-1, 2))
    Y = u.view(n_paths, n_pts)
    Z = ux.view(n_paths, n_pts)
    predicted = RATE * Y[:, :-1] * dt + SIGMA * Z[:, :-1] * dW
    claimed = Y[:, 1:] - Y[:, :-1]
    return ((claimed - predicted) / dt).reshape(-1)


# ---------------------------------------------------------------------------
# Spectrum bookkeeping
# ---------------------------------------------------------------------------

def spectrum_row(name: str, M: Tensor) -> Dict:
    M = 0.5 * (M + M.T)
    ev = torch.linalg.eigvalsh(M)
    lmax = ev.max().item()
    lmin = ev.min().item()
    tol = 1e-10 * max(abs(lmax), abs(lmin), 1e-300)
    neg = ev[ev < -tol]
    pos = ev[ev > tol]
    return dict(
        matrix=name,
        dim=M.shape[0],
        n_neg=int(neg.numel()),
        n_pos=int(pos.numel()),
        lam_min=lmin,
        lam_max=lmax,
        neg_over_max=abs(lmin) / lmax if lmax > 0 else float("inf"),
        neg_mass=(neg.abs().sum() / ev.abs().sum()).item(),
    )


def print_rows(rows: List[Dict]) -> None:
    hdr = (f"{'state':<11}{'stack':>6}  {'matrix':<8}{'n_neg':>7}{'n_pos':>7}"
           f"{'lam_min':>13}{'lam_max':>13}{'|min|/max':>11}{'neg_mass':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['state']:<11}{str(r['stack']):>6}  {r['matrix']:<8}{r['n_neg']:>7}"
              f"{r['n_pos']:>7}{r['lam_min']:>13.3e}{r['lam_max']:>13.3e}"
              f"{r['neg_over_max']:>11.3e}{r['neg_mass']:>10.3f}")


# ---------------------------------------------------------------------------
# Network states: init, Adam on Un-EM loss, supervised fit to analytic solution
# ---------------------------------------------------------------------------

def ic_loss(net: nn.Module, n: int) -> Tensor:
    X0 = sample_paths(n, steps=1)[0][:, -1, :]  # tau = 0 endpoints
    return 0.5 * ((net(X0) - bs_call_payoff(X0)) ** 2).mean()


def train_adam_unem(net: nn.Module, steps: int, n_paths: int, path_steps: int,
                    main_stack: int, sub_stack: int, p: int, lr: float) -> None:
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for i in range(steps):
        opt.zero_grad()
        X, dW = sample_paths(n_paths, steps=path_steps)
        loss = bsde_loss_unem(net, X, dW, main_stack, sub_stack, p) + ic_loss(net, n_paths)
        loss.backward()
        opt.step()
        if i % max(steps // 5, 1) == 0 or i == steps - 1:
            print(f"    adam step {i:5d}  loss {loss.item(): .3e}")


def train_supervised(net: nn.Module, steps: int, n_paths: int, path_steps: int,
                     lr: float) -> None:
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for i in range(steps):
        opt.zero_grad()
        X = sample_paths(n_paths, steps=path_steps)[0].reshape(-1, 2)
        loss = ((net(X) - bs_call_price(X)) ** 2).mean()
        loss.backward()
        opt.step()
        if i % max(steps // 5, 1) == 0 or i == steps - 1:
            print(f"    fit step {i:5d}  mse {loss.item(): .3e}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=2, help="number of hidden layers")
    ap.add_argument("--n_paths", type=int, default=10)
    ap.add_argument("--steps", type=int, default=20, help="time steps per path")
    ap.add_argument("--stacks", type=str, default="1,5,25",
                    help="main_stack = sub_stack values to test")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--K_pop", type=int, default=4000,
                    help="stack size used to approximate the population GN")
    ap.add_argument("--pop_chunk", type=int, default=200,
                    help="noises per chunk when building the population GN")
    ap.add_argument("--adam_steps", type=int, default=1000)
    ap.add_argument("--fit_steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hess_chunk", type=int, default=32,
                    help="rows of the Hessian computed at once")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=str, default="", help="optional output CSV path")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    stacks = [int(s) for s in args.stacks.split(",")]

    # Fixed evaluation sample shared by every state and matrix.
    X, dW = sample_paths(args.n_paths, steps=args.steps)
    N = args.n_paths * args.steps
    K_max = max(max(stacks) * args.p, 1)
    eta_all = sample_unem_noise(dW, K_max)          # slice per stack
    args.K_pop -= args.K_pop % args.pop_chunk
    eta_pop = sample_unem_noise(dW, args.K_pop)

    def eta_for(stack: int) -> Tensor:
        K = stack + (args.p - 1) * stack
        return eta_all[..., :K]

    net = make_net(args.width, args.depth)
    fp = FlatParams(net)
    print(f"network 2-{'-'.join([str(args.width)] * args.depth)}-1, "
          f"{fp.flat().numel()} parameters; N = {N} path-steps")

    states: List[Tuple[str, Tensor]] = [("init", fp.flat().clone())]

    print("\n[state: adam_unem] training with Un-EM loss + IC loss")
    train_adam_unem(net, args.adam_steps, args.n_paths, args.steps,
                    stacks[len(stacks) // 2], stacks[len(stacks) // 2], args.p, args.lr)
    states.append(("adam_unem", fp.flat().clone()))

    print("\n[state: fit_exact] supervised fit to the analytic solution")
    with torch.no_grad():
        for p_, p0 in zip(net.parameters(), fp.unflat(states[0][1]).values()):
            p_.copy_(p0)
    train_supervised(net, args.fit_steps, args.n_paths, args.steps, args.lr)
    states.append(("fit_exact", fp.flat().clone()))

    rows: List[Dict] = []
    for state, theta in states:
        print(f"\n=== state {state} ===")
        theta = theta.clone().requires_grad_(False)

        # EM baseline: full Hessian and PSD Gauss-Newton J^T J.
        em_fn = lambda th: em_residual(fp, th, X, dW)
        J = jacrev(em_fn)(theta)                                        # [N, P]
        loss_em = lambda th: 0.5 * (em_fn(th) ** 2).mean()
        H_em = hessian(loss_em, args.hess_chunk)(theta)
        G_em = J.T @ J / N
        for name, M in (("H_em", H_em), ("G_em", G_em)):
            rows.append(dict(state=state, stack="-", **spectrum_row(name, M)))

        # Population GN of the unbiased residual: one huge block, processed in
        # noise chunks (the block mean and its Jacobian are linear in chunks).
        Jp = zeros(N, theta.numel())
        r_pop = zeros(N)
        n_chunks = args.K_pop // args.pop_chunk
        for c in range(n_chunks):
            eta_c = eta_pop[..., c * args.pop_chunk:(c + 1) * args.pop_chunk]
            pop_fn = lambda th, e=eta_c: unem_blocks(fp, th, X, e, args.pop_chunk, 0, 1)[0]
            Jp += jacrev(pop_fn)(theta) / n_chunks
            with torch.no_grad():
                r_pop += pop_fn(theta) / n_chunks
        G_pop = Jp.T @ Jp / N
        rows.append(dict(state=state, stack=args.K_pop, **spectrum_row("G_pop", G_pop)))
        print(f"  mean |r_pop| = {r_pop.abs().mean().item():.3e}, "
              f"0.5*mean(r_pop^2) = {0.5 * (r_pop ** 2).mean().item():.3e}")

        # Un-EM at each stack size: full Hessian and cross-product GN.
        for stack in stacks:
            eta = eta_for(stack)
            blocks_fn = lambda th, eta=eta, s=stack: unem_blocks(fp, th, X, eta, s, s, args.p)
            loss_fn = lambda th, f=blocks_fn: 0.5 * (f(th)[0] * f(th)[1]).mean()
            with torch.no_grad():
                main, sub = blocks_fn(theta)
            Ja = jacrev(lambda th, f=blocks_fn: f(th)[0])(theta)       # grad main
            Jb = jacrev(lambda th, f=blocks_fn: f(th)[1])(theta)       # grad sub
            G_unem = 0.5 * (Ja.T @ Jb + Jb.T @ Ja) / N
            H_unem = hessian(loss_fn, args.hess_chunk)(theta)
            # |a - b| / |a| measures gradient noise relative to signal: the GN
            # matrix is PSD exactly when a = b for every sample.
            print(f"  stack {stack:3d}: sample loss = {0.5 * (main * sub).mean().item(): .3e}"
                  f"  |G_unem - G_pop|_F/|G_pop|_F = "
                  f"{((G_unem - G_pop).norm() / G_pop.norm()).item():.3f}"
                  f"  |Ja - Jb|_F/|Ja|_F = {((Ja - Jb).norm() / Ja.norm()).item():.3f}"
                  f"  |Ja - Jp|_F/|Jp|_F = {((Ja - Jp).norm() / Jp.norm()).item():.3f}")
            for name, M in (("H_unem", H_unem), ("G_unem", G_unem)):
                rows.append(dict(state=state, stack=stack, **spectrum_row(name, M)))

    print("\n")
    print_rows(rows)
    print("\nG_em, G_pop are PSD by construction (n_neg should be 0 up to rounding).\n"
          "G_unem = 0.5/N sum(a b^T + b a^T): n_neg counts its negative directions;\n"
          "|min|/max is the Levenberg-Marquardt damping (relative to lam_max) needed to\n"
          "make it positive definite. neg_mass = sum|neg eigs| / sum|all eigs|.")

    if args.csv:
        import csv
        from pathlib import Path
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
