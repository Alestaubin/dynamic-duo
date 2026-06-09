## Setting Up Your Environment

```bash
python -m venv dynamicduos
source dynamicduos/bin/activate
pip install -r requirements.txt
```

```bash
export PYTHONPATH=$PYTHONPATH:~/dynamic-duo
```

```bash
# See what's already running on the GPU(s) before you start
nvidia-smi

tmux new -s duo
source /scratch0/alxstaub/ddenv/bin/activate
# inside the session, run your python command
# detach with Ctrl-b then d; reattach later with: tmux attach -t duo
export PYTHONPATH=$PYTHONPATH:~/dynamic-duo

# Pin your job to a specific GPU if there are several
CUDA_VISIBLE_DEVICES=0 python scripts/fit_fixed_ts.py --config cfgs/dynamic_duo_config.yaml --out checkpoints/fixed_ts/clean_norm --clean_only
```