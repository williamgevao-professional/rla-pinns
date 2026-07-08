"""Animate PINN delta (dV/dS) converging onto analytic N(d1) over training."""
import glob, os, re
import numpy as np, torch
from torch import nn
import matplotlib.pyplot as plt, matplotlib.animation as animation
from scipy.stats import norm

CHECKPOINT_DIR = "checkpoints_rngd_fine"
GLOB = "black-scholes-logS_1d_call_payoff_mlp-tanh-64_*_step*.pt"
TAUS = [1.0, 0.5, 0.25]
S_MAX = 3.0
FPS = 6
DTYPE = torch.float64

from rla_pinns.black_scholes_logS_equation import SIGMA, STRIKE

def build(): return nn.Sequential(nn.Linear(2,64),nn.Tanh(),nn.Linear(64,1)).to(DTYPE)

def load(p):
    c=torch.load(p,map_location="cpu"); m=build(); m.load_state_dict(c["model"]); m.eval()
    return m, c.get("step",0)

def pinn_delta(m,tau,S):
    x=np.log(S)
    X=torch.tensor(np.stack([np.full_like(x,tau),x],1),dtype=DTYPE,requires_grad=True)
    V=m(X); dV=torch.autograd.grad(V.sum(),X)[0][:,1].detach().numpy()
    return dV/S

def analytic(tau,S):
    if tau<=0: return (S>STRIKE).astype(float)
    d1=(np.log(S/STRIKE)+0.5*SIGMA**2*tau)/(SIGMA*np.sqrt(tau))
    return norm.cdf(d1)

paths=sorted(glob.glob(os.path.join(CHECKPOINT_DIR,GLOB)),
             key=lambda p:int(re.search(r"step(\d+)",p).group(1)))
if not paths: raise FileNotFoundError(f"no checkpoints in {CHECKPOINT_DIR}")
S=np.linspace(0.3,S_MAX,300)
d_true=[analytic(t,S) for t in TAUS]
print(f"{len(paths)} frames")
nets=[[pinn_delta(load(p)[0],t,S) for t in TAUS] for p in paths]
steps=[int(re.search(r"step(\d+)",p).group(1)) for p in paths]

fig,axes=plt.subplots(1,len(TAUS),figsize=(5*len(TAUS),4.5))
def draw(f):
    for ax,i,tau in zip(axes,range(len(TAUS)),TAUS):
        ax.clear()
        ax.plot(S,d_true[i],"k--",lw=2.5,label="analytic N(d1)")
        ax.plot(S,nets[f][i],"C0-",lw=2,label="PINN dV/dS")
        ax.axvline(STRIKE,color="grey",ls=":",lw=1)
        ax.set_xlabel("stock price S"); ax.set_ylabel("delta")
        ax.set_title(f"tau={tau}"); ax.set_ylim(-0.05,1.05)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"PINN delta vs analytic — step {steps[f]}",fontsize=13)
anim=animation.FuncAnimation(fig,draw,frames=len(paths),interval=1000/FPS)
anim.save("delta_learning.gif",writer=animation.PillowWriter(fps=FPS))
print("wrote delta_learning.gif")