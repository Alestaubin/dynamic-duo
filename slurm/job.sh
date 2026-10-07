#!/bin/bash
#SBATCH --job-name=duo
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=aip-evanesce

# The one generic launcher: runs any repo script inside a SLURM job. Submit from the repo root.
#
#   sbatch slurm/job.sh scripts/tab_accuracy.py --duo vitb16_rn50 --seeds 0 1 2
#   STAGE="imagenet_a imagenet_r imagenet_v2 imagenet_sketch" sbatch slurm/job.sh scripts/setup_check.py --probe 200
#   sbatch --array=0-8 -J ccc slurm/job.sh scripts/tab_continual_duo.py ...   # scripts read $SLURM_ARRAY_TASK_ID
#   sbatch --gres=gpu:0 -t 00:30:00 -J check slurm/job.sh scripts/setup_check.py   # sbatch options override the #SBATCH lines
#
# STAGE = space-separated dataset keys of `paths:` in cfgs/global.yaml (imagenet_a, imagenet_r, ...). Archives
# among them are extracted to this job's node-local $SLURM_TMPDIR/data before the script starts (never into
# the shared folder, which has a file-count quota); scripts also stage lazily on first use, so STAGE is only
# an up-front convenience that keeps the extraction time out of the script's own timings.

module load python/3.11.5
module load cuda/12.6
module load gcc arrow/22.0.0

source ~/py38/bin/activate

export PYTHONPATH=$PYTHONPATH:.
export TORCH_HOME=${TORCH_HOME:-/scratch/alxstaub/torch_cache}
export PYTHONUNBUFFERED=1

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: sbatch slurm/job.sh <script.py> [args...]" >&2
    exit 2
fi

echo "job $SLURM_JOB_ID on $(hostname), SLURM_TMPDIR=${SLURM_TMPDIR:-<unset>}"
if [ -n "${STAGE:-}" ]; then
    python -c 'from src.utils.datasets.staging import main; main()' $STAGE
fi

exec python "$@"
