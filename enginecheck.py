import torch
from torch.nn import Module
from rla_pinns import black_scholes_logS_equation as bs

class Analytic(Module):
    def forward(self, X):
        squeeze = X.dim() == 1
        if squeeze:
            X = X.unsqueeze(0)
        out = bs.bs_call_price(X)
        return out.squeeze(0) if squeeze else out

X = bs.interior_points(300).double()
X[:, 0] = X[:, 0].clamp(min=0.05)
y = torch.zeros(X.shape[0], 1, dtype=X.dtype)

loss, residual, _ = bs.evaluate_interior_loss(Analytic(), X, y)
print("engine residual (mean abs):", residual.abs().mean().item())
print("engine loss:", loss.item())
