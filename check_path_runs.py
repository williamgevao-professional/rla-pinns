"""Pull yesterday's path-vs-uniform + mu-sweep runs, tabulate final L2."""
import wandb
from datetime import datetime, timedelta, timezone

ENTITY = "williamgevao202-concordia-university"
PROJECT = "deep-hedging-compare"

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

# keep only recent runs (last ~2 days) so we get yesterday's batch
cutoff = datetime.now(timezone.utc) - timedelta(days=2)

rows = []
for r in runs:
    try:
        created = datetime.fromisoformat(r.created_at.replace("Z", "+00:00"))
    except Exception:
        created = None
    if created and created < cutoff:
        continue
    rows.append({
        "name": r.name,
        "id": r.id,
        "state": r.state,
        "sampling": r.config.get("interior_sampling", "?"),
        "batch_freq": r.config.get("batch_frequency", "?"),
        "num_sec": r.config.get("num_seconds", "?"),
        "runtime": r.summary.get("_runtime", 0),
        "l2": r.summary.get("l2_error", None),
        "loss": r.summary.get("loss", None),
    })

# sort by name so the mu sweep / sampling arms group together
rows.sort(key=lambda d: str(d["name"]))

print(f"{'name':<24}{'sampling':<10}{'freq':<6}{'runtime':<9}{'l2_error':<14}{'state'}")
print("-"*80)
for d in rows:
    l2 = f"{d['l2']:.4e}" if isinstance(d['l2'], (int, float)) else str(d['l2'])
    rt = f"{d['runtime']:.0f}s" if isinstance(d['runtime'], (int, float)) else "?"
    print(f"{str(d['name']):<24}{str(d['sampling']):<10}{str(d['batch_freq']):<6}{rt:<9}{l2:<14}{d['state']}")

print(f"\n{len(rows)} runs in the last 2 days")