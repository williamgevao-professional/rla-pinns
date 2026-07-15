from torch import Tensor, randn, exp, log, zeros, cat, full
from math import sqrt

def sample_gbm_paths(
    n_paths: int,
    n_steps: int,
    mu: float,
    sigma: float,
    S0: float,
    T: float,
    X_MIN: float,
    X_MAX: float,
) -> Tensor:
    
    dt = T / n_steps
    x0 = log(Tensor([S0])) # start in log-space

    
    Z = randn(n_paths, n_steps) # matrix of random shocks for path x step y for example
    
    increments = (mu - 0.5 * sigma**2) * dt + sigma * sqrt(dt) * Z # log-price increments: (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z

    # cumulative sum gives the log-price at each step
    x = x0 + cat([zeros(n_paths, 1), increments.cumsum(dim=1)], dim=1)  # keep running sum of price increments along timesteps, glue a column of zeros at the front for t = 0 an dshift to the starting price

    # reject paths that ever leave the domain
    inside = ((x >= X_MIN) & (x <= X_MAX)).all(dim=1)
    x = x[inside] # keep only surviving paths

    # build the time axis
    t = Tensor([i * dt for i in range(n_steps + 1)]) # build the time axis
    tau = T - t  
    tau = tau.unsqueeze(0).expand(x.shape[0], -1) # add a 0th dimension row to tensor, then replicates it downwards to match number of paths

    # flatten into collocation points
    return cat([tau.reshape(-1, 1), x.reshape(-1, 1)], dim=1) # flatten both grids into single columns and concatenate along 1st dimension to get [N, 2]