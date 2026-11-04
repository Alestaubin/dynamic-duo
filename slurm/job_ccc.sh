#!/bin/bash
#SBATCH --job-name=ccc
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --account=aip-evanesce
#SBATCH --qos=high             # the author's standing rule: every job runs at QOS high

# CPU launcher for the CCC generator (scripts/ccc_generate.py): its own environment (~/ccc_env: webdataset, Wand, numba, scikit-image, torch on top of the
# opencv module), not ~/py38. Submit from the repo root:
#   sbatch -t 01:00:00 slurm/job_ccc.sh scripts/ccc_generate.py --timing --n 40
#   sbatch --array=0-79 -t 12:00:00 slurm/job_ccc.sh scripts/ccc_generate.py --difficulty medium --speed 1000 --seed 43 --images 2000000 --workers 80

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    exec >> "logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}.out" \
         2>> "logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}.err"
fi

module load StdEnv/2023 python/3.11.5 opencv/4.11.0
source ~/ccc_env/bin/activate

export PYTHONPATH=$PYTHONPATH:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

set -euo pipefail
echo "job $SLURM_JOB_ID${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID} on $(hostname) (CPU), SLURM_TMPDIR=${SLURM_TMPDIR:-<unset>}"
python "$@"
