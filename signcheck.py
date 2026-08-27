import torch
from rla_pinns import black_scholes_logS_equation as bs

X = bs.interior_points(500).double()
X[:, 0] = X[:, 0].clamp(min=0.05)
X.requires_grad_(True)
u = bs.bs_call_price(X)
g = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
u_tau, u_x = g[:, [0]], g[:, [1]]
u_xx = torch.autograd.grad(u_x.sum(), X, create_graph=True)[0][:, [1]]

s2 = bs.SIGMA ** 2
r = bs.RATE
resid = u_tau + (0.5 * s2 - r) * u_x - 0.5 * s2 * u_xx + r * u
print("residual of analytic solution:", resid.abs().mean().item())
print("RATE =", r, " _MU_CONST =", bs._MU_CONST)
