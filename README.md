
## Getting started

### Setting Up Your Environment
First, create a new python environment:
```bash
module load python/3.11 cuda/12.2

# create venv
python -m venv ~/dduos
source ~/dduos/bin/activate

pip install -r requirements.txt
pip install --no-index torch
```

Create a new folder `data/` and add ImageNet-C and ImageNet.

slurm:
```bash
salloc --account=aip-evanesce --gpus=1 --cpus-per-task=8 --mem=32G --time=03:00:00
source ~/dduos/bin/activate
export PYTHONPATH=$PYTHONPATH:/project/aip-evanesce/alxstaub/dynamic-duo

python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode no_adapt \
    --proxy_kind prototype \
    --steps 1 \
    --seed 0 \
    --num_samples 50000 \
    --calibration_mode proxy_anchor_coca \
    --csv_path results/proxy_anchor_coca_proto.csv \
    --proxy_cache cache/vit_res_50k.pt
    --wandb

# proxy-anchor COCA with calibrated anchor selection:
python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode no_adapt \
    --calibration_mode proxy_anchor_coca \
    --proxy_kind prototype \
    --proxy_cache resnet50_vitb16 \
    --calib_map resnet50_vitb16_dev \
    --calibrated_selection \
    --num_samples 5000

python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode no_adapt \
    --calibration_mode proxy_anchor_coca \
    --proxy_kind nuclear_norm \
    --proto_metric cosine \
    --calib_map resnet50_vitb16_nuclear_norm \
    --calibrated_selection \
    --seed 0 \
    --num_samples 1000 \
    --csv_path out/temp

python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode no_adapt \
    --calibration_mode proxy_anchor_coca \
    --proxy_kind nuclear_norm \
    --seed 0 --num_samples 50000 --csv_path out/temp --fixed_ts_config checkpoints/default

```