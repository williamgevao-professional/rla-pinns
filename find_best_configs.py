"""Best config per optimizer from the 3000s sweeps (filters out 300s runs)."""
import wandb

ENTITY = "williamgevao202-concordia-university"
PROJECT = "rla-pinns-rla_pinns"
MIN_RUNTIME = 2000        # seconds; keeps 3000s runs, drops 300s ones

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

best = {}   # optimizer -> (l2, config, runtime)
for r in runs:
    if r.state != "finished":
        continue
    runtime = r.config.get("num_seconds")==3000
    if r.config.get("num_seconds") != 3000:
        continue
    runtime = r.summary.get("_runtime", 0)
    opt = r.config.get("optimizer")
    l2 = r.summary.get("l2_error")
    if opt is None or l2 is None:
        continue
    if opt not in best or l2 < best[opt][0]:
        best[opt] = (l2, dict(r.config), runtime)

for opt, (l2, cfg, rt) in sorted(best.items()):
    hp = {k: v for k, v in cfg.items()
          if opt.lower() in k.lower() or k in ("lr","damping")}
    print(f"{opt:12s} l2={l2:.4e} runtime={rt:.0f}s  {hp}")