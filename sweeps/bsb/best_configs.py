"""Best 100-d BSB sweep configuration per optimizer, as launch flags.

Reads the bsb-100d-sweeps wandb project, keeps finished 3000 s runs, picks the
lowest final rl2_error_path per optimizer, and prints (a) a summary table and
(b) the optimizer flags to paste into launch_bsb.sh.

Run on the Mac:  python sweeps/bsb/best_configs.py [--metric rl2_error_path] [--top 3]
"""
import math
from argparse import ArgumentParser

import wandb

ENTITY = "williamgevao202-concordia-university"
PROJECT = "bsb-100d-sweeps"
SKIP = {"lr"}  # RNGD/KFAC lr is a line-search tuple, not a flag value


def main() -> None:
    ap = ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metric", default="rl2_error_path")
    ap.add_argument("--top", type=int, default=3, help="show the top-k per optimizer")
    args = ap.parse_args()

    api = wandb.Api(timeout=120)
    per_opt = {}
    for r in api.runs(f"{ENTITY}/{PROJECT}"):
        c, s = r.config, r.summary
        opt = c.get("optimizer")
        try:
            val = float(s.get(args.metric))
        except (TypeError, ValueError):
            continue
        if opt is None or math.isnan(val) or r.state != "finished":
            continue
        per_opt.setdefault(opt, []).append((val, r.name, r.id, c))

    for opt in sorted(per_opt):
        runs = sorted(per_opt[opt], key=lambda x: x[0])
        print(f"\n=== {opt}: {len(runs)} finished runs")
        for val, name, rid, c in runs[: args.top]:
            hp = {k[len(opt) + 1:]: v for k, v in c.items()
                  if k.startswith(f"{opt}_") and k[len(opt) + 1:] not in SKIP | {"equation"}}
            print(f"  {args.metric}={val:.3e}  {name} ({rid})  {hp}")
        best = runs[0][3]
        flags = " ".join(f"--{k} {v}" for k, v in best.items()
                         if k.startswith(f"{opt}_") and k[len(opt) + 1:] not in SKIP | {"equation"}
                         and not isinstance(v, (list, dict)))
        print(f"  launch_bsb.sh flags: --optimizer {opt} {flags}")


if __name__ == "__main__":
    main()
