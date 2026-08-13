import torch
import matplotlib.pyplot as plt

X_MIN, X_MAX = -3.0, 3.0
SIGMA, T_MAX = 0.2, 1.0
STEPS = 50
N = 3000
X0 = 0.0
torch.manual_seed(0)

def path_points(n_paths):
    dt = T_MAX / STEPS
    xs = torch.zeros(n_paths, STEPS + 1)
    xs[:, 0] = X0
    for k in range(STEPS):
        z = torch.randn(n_paths)
        xs[:, k+1] = xs[:, k] + (0.0 - 0.5*SIGMA**2)*dt + SIGMA*(dt**0.5)*z
    taus = torch.linspace(0, T_MAX, STEPS + 1)
    return torch.stack([taus.repeat(n_paths), xs.flatten()], dim=1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

uni = torch.stack([torch.rand(N)*T_MAX,
                   torch.rand(N)*(X_MAX-X_MIN)+X_MIN], dim=1)
axes[0].scatter(uni[:,0], uni[:,1], s=2, alpha=0.35, color="tab:orange")
axes[0].set_title("Uniform sampling")
axes[0].set_xlabel(r"$\tau$"); axes[0].set_ylabel(r"$x$")

pts = path_points(500)
pts = pts[torch.randperm(pts.shape[0])[:N]]
axes[1].scatter(pts[:,0], pts[:,1], s=2, alpha=0.35, color="tab:blue")
axes[1].set_title(r"Path sampling ($x_0=0$)")
axes[1].set_xlabel(r"$\tau$")

for ax in axes:
    ax.set_xlim(0, T_MAX); ax.set_ylim(X_MIN, X_MAX)
    ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("uniform_vs_cone.png", dpi=200)
print("saved uniform_vs_cone.png")
