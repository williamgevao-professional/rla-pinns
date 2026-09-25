#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --account=aip-fdangel

module load python/3.11
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/scratch/wgevao/rla-pinns

# AGENT_COUNT bounds runs per job so a job never dies mid-run at the time limit
wandb agent ${AGENT_COUNT:+--count $AGENT_COUNT} $SWEEP_ID
