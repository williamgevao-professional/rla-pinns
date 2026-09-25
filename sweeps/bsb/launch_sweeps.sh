#!/bin/bash
# Register the 100-d BSB hyperparameter sweeps on wandb and submit SLURM agents.
#
# Each sweep (sweeps/bsb/{sgd,adam,kfac,hf,rngd}.yaml) is random search with
# run_cap 16, 3000 s per run, optimising rl2_error_path, in wandb project
# bsb-100d-sweeps. Each agent job runs AGENT_COUNT runs back to back (3 x ~52 min
# fits the 3 h job limit); AGENTS agents per sweep run in parallel.
#
# Usage, on Vulcan from ~/scratch/rla-pinns after `git pull` (wandb must be logged in):
#   bash sweeps/bsb/launch_sweeps.sh              # all five optimizers
#   bash sweeps/bsb/launch_sweeps.sh kfac hf      # a subset
#   AGENTS=3 bash sweeps/bsb/launch_sweeps.sh     # fewer parallel agents (default 6)
#   DRY_RUN=1 bash sweeps/bsb/launch_sweeps.sh    # print, don't register or submit
#
# Sweep ids are appended to sweeps/bsb/sweep_ids.txt so agents can be re-added later:
#   sbatch --export=ALL,SWEEP_ID=<id>,AGENT_COUNT=3 sweep_submit.sh
set -euo pipefail

AGENTS="${AGENTS:-6}"
AGENT_COUNT="${AGENT_COUNT:-3}"
OPTS="${*:-sgd adam kfac hf rngd}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

for opt in $OPTS; do
    yaml="sweeps/bsb/$opt.yaml"
    [[ -f "$yaml" ]] || { echo "no such sweep config: $yaml" >&2; exit 1; }
    echo "== $opt"
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "  wandb sweep $yaml  ->  $AGENTS x sbatch --export=ALL,SWEEP_ID=<id>,AGENT_COUNT=$AGENT_COUNT sweep_submit.sh"
        continue
    fi
    # `wandb sweep` prints "Created sweep with ID: xxxx" and the full path on stderr
    out="$(wandb sweep "$yaml" 2>&1)"
    echo "$out" | sed 's/^/  /'
    sweep_id="$(echo "$out" | sed -n 's/.*wandb agent \([^ ]*\).*/\1/p' | tail -1)"
    [[ -n "$sweep_id" ]] || { echo "could not parse sweep id" >&2; exit 1; }
    echo "$(date -u +%FT%TZ) $opt $sweep_id" >> sweeps/bsb/sweep_ids.txt
    for i in $(seq "$AGENTS"); do
        sbatch --job-name="sw_${opt}_$i" --gres=gpu:l40s:1 --time=03:00:00 \
            --export=ALL,SWEEP_ID="$sweep_id",AGENT_COUNT="$AGENT_COUNT" sweep_submit.sh
    done
done
