import torch
from torch import nn
from torch.func import functional_call, hessian, jacrev, vmap
from einops import einsum
 
from rla_pinns.black_scholes_equation import (
    SIGMA,
    evaluate_interior_loss,
    interior_points,
)
 
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
 
 
def verify():
    # small model, same shape as the benchmark uses
    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )
 
    # a handful of interior points [t, S]
    X = interior_points(8)            # shape (8, 2)
    y = torch.zeros(8, 1)
 
    # --- Reference: the residual from the EQUATION FILE (known-correct) ---
    # evaluate_interior_loss returns (loss, residual, None); residual is the
    # per-point operator value V_t + 0.5*sigma^2*S^2*V_SS.
    _, residual_reference, _ = evaluate_interior_loss(model, X, y)
    residual_reference = residual_reference.squeeze(-1).detach()   # (8,)
 
    # --- Our operator: the SAME function the Gramian will differentiate ---
    param_names = [n for n, _ in model.named_parameters()]
 
    def f(x, *params):
        variable = dict(zip(param_names, params))
        return functional_call(model, variable, x).squeeze()
 
    def black_scholes_pde_operator(x, *params):
        hess_f = hessian(f, argnums=0)
        jacobian_f = jacrev(f, argnums=0)
        hess = hess_f(x, *params)[1:][:, 1:]
        V_SS = einsum(hess, "d d ->")
        V_t = jacobian_f(x, *params)[0]
        S = x[1]
        return V_t + 0.5 * SIGMA**2 * S**2 * V_SS
 
    # evaluate our operator at each point (vmap over the batch)
    params = [model.get_parameter(n) for n in param_names]
    batch_size = X.shape[0]
    params_expanded = []
    for p in params:
        keep = p.ndim * [-1]
        params_expanded.append(p.unsqueeze(0).expand(batch_size, *keep))
 
    operator_ours = vmap(black_scholes_pde_operator)(X, *params_expanded).detach()  # (8,)
 
    # --- Compare ---
    diff = (residual_reference - operator_ours).abs()
    print("reference residual :", residual_reference)
    print("our operator       :", operator_ours)
    print("abs difference     :", diff)
    print(f"\nmax abs difference : {diff.max().item():.3e}")
    if diff.max().item() < 1e-10:
        print("PASS — operator matches the equation file's residual. Safe to use.")
    else:
        print("FAIL — operator does NOT match. There is a bug; do not trust ENGD.")
 
 
if __name__ == "__main__":
    verify()