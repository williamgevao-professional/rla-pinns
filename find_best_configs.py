"""Best config per optimizer from the 3000s runs."""
import wandb

ENTITY = "williamgevao202-concordia-university"
PROJECT = "rla-pinns-rla_pinns"

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

best = {}
for r in runs:
    if r.state != "finished":
        continue
    if r.config.get("num_seconds") != 3000:
        continue
    opt = r.config.get("optimizer")
    l2 = r.summary.get("l2_error")
    if opt is None or l2 is None:
        continue
    l2 = float(l2)
    if opt not in best or l2 < best[opt][0]:
        best[opt] = (l2, dict(r.config))

for opt, (l2, cfg) in sorted(best.items()):
    hp = {k: v for k, v in cfg.items()
          if opt.lower() in k.lower() or k in ("lr", "damping")}
    print(f"{opt:12s} l2={l2:.4e}  {hp}")