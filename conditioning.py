import torch
from torch import nn
from torch.func import jacrev, vmap
import numpy as np
from torch.nn import MSELoss
from curvlinops import GGNLinearOperator

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

N = 200
from rla_pinns.black_scholes_equation import SIGMA

# Define the range of time horizons you want to test
time_horizons = [0.5, 1.0, 2.0, 4.0, 8.0]


class ResidualModel(nn.Module):
    def __init__(self, net, T): # Added T here to pass the loop's horizon parameter
        super().__init__()
        self.net = net
        self.T = T  # Keep track of the current maximum time horizon

    def forward(self, X_normalized): # Changed to match the clone variable name below
        X = X_normalized.clone()
        X[:, 0] = X[:, 0] * self.T  # Time column now spans [0, T]
        
        # x is a 1D tensor of size [2] -> [t, S]
        def net_single(x):
            return self.net(x)  # Returns a 1D tensor of size [1] -> [V]
        

        # Define functions for first and second derivatives using jacrev
        # First derivative: returns a [1, 2] Jacobian matrix
        # [0, 0] is dV/dt, [0, 1] is dV/dS
        first_deriv = jacrev(net_single)
        
        # Second derivative (Hessian-like): returns a [1, 2, 2] tensor
        # [0, 1, 1] is d^2V / dS^2
        second_deriv = jacrev(jacrev(net_single))

        # This computes the derivatives for all points in parallel
        grads_1st = vmap(first_deriv)(X)  # Shape: [Batch, 1, 2]
        grads_2nd = vmap(second_deriv)(X) # Shape: [Batch, 1, 2, 2]

        # 4. Extract the exact terms needed for the Black-Scholes PDE
        V_t  = grads_1st[:, 0, 0] # dV/dt (all batches, output dim 0, input dim 0)
        V_SS = grads_2nd[:, 0, 1, 1] # d^2V/dS^2 (all batches, output dim 0, input dim 1, input dim 1)

        # Extract S from input tensor
        S = X[:, 1] 

        # Compute and return the PDE residual
        residual = V_t + 0.5 * (SIGMA**2) * (S**2) * V_SS
        return residual.unsqueeze(-1)  # Changes shape from [200] to [200, 1]


print(f"{'Time Horizon (T)':<18}{'Max Eigenvalue':<18}{'Min Eigenvalue':<18}{'Condition Number':<18}")
print("-" * 72)

for T in time_horizons:
    # 1. Reset the model weights to the exact same seed each time 
    # to isolate the effect of T from random initialization variations
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 1, bias=False), # bias can't affect the residual as a constant disappears under differentiation
    )
    
    # 2. Instantiate the residual model with the current T
    residual_model = ResidualModel(model, T=T) # Added T=T to pass current horizon step
    params = [p for p in residual_model.parameters()]
    n_params = sum(p.numel() for p in params)

    # 3. Sample static normalized data points
    X_data = torch.rand(N, 2) # 200 random (t, S) points
    y_data = torch.zeros(N, 1) # 200 zeros for the PDE residual at those points
    data = [(X_data, y_data)] # curvlinops expects data as list of (x, y) tuples
    
    loss_function = MSELoss(reduction="mean") # average the loss over the batch

    # 4. Build GGN and materialize via torch.eye
    GGN = GGNLinearOperator(residual_model, loss_function, params, data)
    identity_matrix = torch.eye(n_params, dtype=torch.float64)
    GGN_dense = GGN @ identity_matrix

    # 5. Compute the spectrum
    # Using eigh because the materialized GGN is symmetric positive semi-definite
    eigenvalues = torch.linalg.eigh(GGN_dense)[0]
    
    lambda_max = eigenvalues[-1].item()
    lambda_min = eigenvalues[0].item()
    
    # Avoid division by zero if minimum eigenvalue is perfectly zero
    cond_num = (lambda_max / lambda_min) if lambda_min > 1e-14 else float('inf')

    print(f"{T:<18.2f}{lambda_max:<18.3e}{lambda_min:<18.3e}{cond_num:<18.3e}")