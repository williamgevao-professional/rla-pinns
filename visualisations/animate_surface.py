"""Animate PINN learning the BS surface. 2D curve morph + heatmap panels."""
import glob, os, re
import numpy as np, torch
from torch import nn
import matplotlib.pyplot as plt, matplotlib.animation as animation

CHECKPOINT_DIR = "checkpoints_rngd_fine"
GLOB = "black-scholes-logS_1d_call_payoff_mlp-tanh-64_*_step*.pt"
S_MAX = 3.0                    # <-- x-axis cap (was ~20)
CURVE_TAU = 0.9               # time slice for the morphing curve
HEATMAP_TAUS = [1.0, 0.5, 0.0]  # slices for the static heatmaps
FPS = 6
DTYPE = torch.float64
GRID_N = 60

from rla_pinns.black_scholes_logS_equation import SIGMA, STRIKE, MATURITY, bs_call_price

def build(): return nn.Sequential(nn.Linear(2,64),nn.Tanh(),nn.Linear(64,1)).to(DTYPE)
def load(p):
    c=torch.load(p,map_location="cpu"); m=build(); m.load_state_dict(c["model"]); m.eval()
    return m, c.get("step",0)

def ev(fn,tau,S):
    x=np.log(S)
    X=torch.tensor(np.stack([np.full_like(x,tau),x],1),dtype=DTYPE)
    with torch.no_grad(): return fn(X).numpy().ravel()

paths=sorted(glob.glob(os.path.join(CHECKPOINT_DIR,GLOB)),
             key=lambda p:int(re.search(r"step(\d+)",p).group(1)))
if not paths: raise FileNotFoundError(f"no checkpoints in {CHECKPOINT_DIR}")
steps=[int(re.search(r"step(\d+)",p).group(1)) for p in paths]
print(f"{len(paths)} frames")

# ---- ANIMATION: morphing curve at CURVE_TAU (capped x-axis) ----
S=np.linspace(0.3,S_MAX,200)
true_curve=ev(bs_call_price,CURVE_TAU,S)
vmax=true_curve.max()*1.1
nets=[ev(load(p)[0],CURVE_TAU,S) for p in paths]

fig,ax=plt.subplots(figsize=(7,5))
def draw(f):
    ax.clear()
    ax.plot(S,true_curve,"k--",lw=2.5,label="analytic")
    ax.plot(S,nets[f],"C0-",lw=2.5,label="network")
    ax.plot(S,np.clip(S-STRIKE,0,None),color="grey",ls=":",lw=1.2,label="payoff")
    ax.axvline(STRIKE,color="grey",ls=":",lw=0.8)
    ax.set_xlabel("stock price S"); ax.set_ylabel("option value V")
    ax.set_ylim(-0.5,vmax); ax.set_title(f"value curve (t={MATURITY-CURVE_TAU:.2f})")
    ax.legend(fontsize=8,loc="upper left"); ax.grid(alpha=0.3)
    fig.suptitle(f"PINN learning Black-Scholes — step {steps[f]}",fontsize=13)
anim=animation.FuncAnimation(fig,draw,frames=len(paths),interval=1000/FPS)
anim.save("surface_learning.gif",writer=animation.PillowWriter(fps=FPS))
plt.close(fig)
print("wrote surface_learning.gif")

# ---- ANIMATED 3-PANEL: network -> analytic + relative error ----
taus=np.linspace(0.0,MATURITY,GRID_N); Sg=np.linspace(0.3,S_MAX,GRID_N)
TAU,SS=np.meshgrid(taus,Sg,indexing="ij")
Xh=torch.tensor(np.stack([TAU.ravel(),np.log(SS.ravel())],1),dtype=DTYPE)
true_surf=bs_call_price(Xh).detach().numpy().reshape(GRID_N,GRID_N)
t_axis=MATURITY-TAU; vval=true_surf.max(); EPS=1e-3
nets=[]
for p in paths:
    with torch.no_grad(): nets.append(load(p)[0](Xh).numpy().reshape(GRID_N,GRID_N))
relerr=lambda n: np.clip(np.abs(n-true_surf)/(np.abs(true_surf)+EPS),0,1)
vrel=max(relerr(n).max() for n in nets)
figh,(a0,a1,a2)=plt.subplots(1,3,figsize=(16,4.5))
def drawh(f):
    for a in (a0,a1,a2): a.clear()
    im0=a0.pcolormesh(t_axis,SS,nets[f],cmap="viridis",shading="auto",vmin=0,vmax=vval)
    a0.set_title(f"network - step {steps[f]}")
    a1.pcolormesh(t_axis,SS,true_surf,cmap="viridis",shading="auto",vmin=0,vmax=vval)
    a1.set_title("analytic")
    a2.pcolormesh(t_axis,SS,relerr(nets[f]),cmap="magma",shading="auto",vmin=0,vmax=vrel)
    a2.set_title("relative error")
    for a in (a0,a1,a2): a.set_xlabel("time t"); a.set_ylabel("stock price S")
    return [im0]
drawh(0); figh.tight_layout()
animh=animation.FuncAnimation(figh,drawh,frames=len(paths),interval=1000/FPS)
animh.save("surface_heatmaps.gif",writer=animation.PillowWriter(fps=FPS))
print("wrote surface_heatmaps.gif")