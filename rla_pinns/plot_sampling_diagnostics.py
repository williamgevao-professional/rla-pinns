import torch
import matplotlib.pyplot as plt

X_MIN, X_MAX = -3.0, 3.0
SIGMA, T_MAX = 0.2, 1.0
STEPS = 50
N = 3000
MUS = [-0.5, -0.2, 0.0, 0.2, 0.5]
torch.manual_seed(0)

def sample_paths(mu, n_paths, keep_full_paths=False):
    # mirrors path_interior_points: x0 ~ U(X_MIN,X_MAX), log-space GBM, reject exits
    dt = T_MAX / STEPS
    x0 = torch.rand(n_paths) * (X_MAX - X_MIN) + X_MIN
    xs = torch.zeros(n_paths, STEPS + 1)
    xs[:, 0] = x0
    for k in range(STEPS):
        z = torch.randn(n_paths)
        xs[:, k+1] = xs[:, k] + (mu - 0.5*SIGMA**2)*dt + SIGMA*(dt**0.5)*z
    alive = ((xs >= X_MIN) & (xs <= X_MAX)).all(dim=1)  # reject any path that exits
    xs = xs[alive]
    taus = torch.linspace(0, T_MAX, STEPS + 1)
    if keep_full_paths:
        return taus, xs
    # flatten to (tau, x) points like training does
    tau_flat = taus.repeat(xs.shape[0])
    x_flat = xs.flatten()
    return torch.stack([tau_flat, x_flat], dim=1)

# Figure 1: spaghetti paths per mu
fig, axes = plt.subplots(1, len(MUS), figsize=(4*len(MUS), 4), sharey=True)
for ax, mu in zip(axes, MUS):
    taus, xs = sample_paths(mu, 60, keep_full_paths=True)
    for i in range(min(40, xs.shape[0])):
        ax.plot(taus, xs[i], lw=0.7, alpha=0.6)
    ax.set_title(f"mu={mu}  (survived {xs.shape[0]}/60)")
    ax.set_xlabel("tau"); ax.axhline(0.0, color='k', ls='--', lw=0.5)  # strike
    ax.set_ylim(X_MIN, X_MAX)
axes[0].set_ylabel("x (log-moneyness)")
fig.tight_layout()
fig.savefig("paths_by_mu.png", dpi=150)
print("saved paths_by_mu.png")

# Figure 2: density, uniform vs path for each mu
fig, axes = plt.subplots(1, len(MUS)+1, figsize=(4*(len(MUS)+1), 4), sharey=True)
uni = torch.stack([torch.rand(N)*T_MAX, torch.rand(N)*(X_MAX-X_MIN)+X_MIN], dim=1)
axes[0].hist2d(uni[:,0], uni[:,1], bins=40, cmap="viridis")
axes[0].set_title("uniform"); axes[0].set_ylabel("x")
for ax, mu in zip(axes[1:], MUS):
    pts = sample_paths(mu, 500)
    idx = torch.randperm(pts.shape[0])[:N]  # subsample to N for fair density
    pts = pts[idx]
    ax.hist2d(pts[:,0], pts[:,1], bins=40, cmap="viridis")
    ax.set_title(f"path mu={mu}")
    ax.set_xlabel("tau")
fig.tight_layout()
fig.savefig("density_uniform_vs_path.png", dpi=150)
print("saved density_uniform_vs_path.png")
