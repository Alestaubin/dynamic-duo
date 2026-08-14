``` bash
salloc --gres=gpu:1 --gres=gpu:l40s:1 --mem=64G -c 16 --time=03:00:00 --account=aip-evanesce

module load python/3.11.5
module load cuda/12.6
module load gcc arrow/22.0.0

source ~/py38/bin/activate

export PYTHONPATH=$PYTHONPATH:.
export TORCH_HOME=/scratch/alxstaub/torch_cache
```
