from math import pi, sqrt
from torch import Tensor, tensor, cat, float64, manual_seed, no_grad, rand, zeros, exp, log
from torch.nn import Linear, Sequential, Tanh
from torch.optim import Adam
from jet import jet

manual_seed(42)
DTYPE = float64

 
f = Sequential(Linear(2, 64), Tanh(), Linear(64, 1)).to(DTYPE) # the network - takes (tau, x), outputs V

jet_f = jet(f, (zeros(2, dtype=DTYPE),)) # jet_f takes in f and returns taylor coefficients - zeros(2) just makes sure jet can understand the shape needed is a length 2 tensor - takes it as a tuple


def bs_pde_operator(z: Tensor, sigma: Tensor) -> Tensor:

    f0, _, d2_dxx = jet_f((z, tensor([0.0, 0.0], dtype=DTYPE), tensor([0.0, 1.0], dtype=DTYPE))) # obtain V and dV/dxx
    
    _, d_dtau = jet_f(z, tensor([1.0,0.0], dtype=DTYPE)) # obtain dV/dτ
    
    _, d_dx = jet_f(z, tensor([0.0,1.0], dtype=DTYPE)) # obtain dV/dx
    
    return d_dtau - 0.5 * sigma**2 * d2_dxx + 0.5 * sigma**2 * d_dx
    