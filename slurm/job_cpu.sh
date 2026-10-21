#!/bin/bash
#SBATCH --job-name=duo-cpu
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=aip-evanesce
#SBATCH --qos=high             # the author's standing rule: every job runs at QOS high (sbatch --qos=normal overrides it)

# Like slurm/job.sh (runs any repo script inside a SLURM job, submit from the repo root) but WITHOUT a GPU, for
# steps that never run a model: ranking stored tuning trials, rendering tables from the cache, setup checks.
#
#   sbatch slurm/job_cpu.sh scripts/setup_check.py
#   sbatch slurm/job_cpu.sh scripts/03_tab_adaptation_modes.py --report_only
#   sbatch -t 00:30:00 -c 4 --mem=16G slurm/job_cpu.sh scripts/02_tune_tta.py --model vit_b_16 --report_only
#
# Do NOT use it for a step that adapts models: on the CPU that runs at about 1 image/s, so a full run would
# take months. Those go through slurm/job.sh. Options given to sbatch override the #SBATCH lines above.
#
# STAGE = space-separated dataset keys of `paths:` in cfgs/global.yaml, extracted to this job's node-local
# $SLURM_TMPDIR/data before the script starts, exactly as in job.sh.

module load python/3.11.5
module load cuda/12.6
module load gcc arrow/22.0.0

source ~/py38/bin/activate

export PYTHONPATH=$PYTHONPATH:.
export TORCH_HOME=${TORCH_HOME:-/scratch/alxstaub/torch_cache}
export PYTHONUNBUFFERED=1

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: sbatch slurm/job_cpu.sh <script.py> [args...]" >&2
    exit 2
fi

echo "job $SLURM_JOB_ID on $(hostname) (no GPU), SLURM_TMPDIR=${SLURM_TMPDIR:-<unset>}"
if [ -n "${STAGE:-}" ]; then
    python -c 'from src.utils.datasets.staging import main; main()' $STAGE
fi

exec python "$@"
