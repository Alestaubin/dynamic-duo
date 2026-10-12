# Dynamic duos

Test-time adaptation (TTA) with a **duo**: two pretrained ImageNet models of different size, such as ViT-B/16 (large) and ResNet-50 (small). At test time both models adapt to corrupted images with Tent (they minimize the entropy of their own predictions by updating their normalization layers), and their logits are combined into one prediction.

The simplest combination is **Fixed TS** (temperature scaling): divide each model's logits by a fixed temperature and average them, `(z_L / T_L + z_S / T_S) / 2`. The paper's method replaces it with a gate that, batch by batch, gives more weight to the model that currently looks more reliable. Reliability comes from a label-free score called the *proxy* (by default the nuclear norm of the softmax matrix). Everything is evaluated on ImageNet-C.

This repo holds the experiments of the paper: **one script per table or figure**. `plan.md` has the status of each script and the design decisions behind them.

## Setup

```bash
module load python/3.11 cuda/12.2
python -m venv ~/dduos && source ~/dduos/bin/activate
pip install -r requirements.txt
pip install --no-index torch
```

`slurm/job.sh` activates `~/py38`; if your environment lives elsewhere, change that line.

Dataset locations, output folders and the protocol constants (severity 5, batch size 64, seeds 0 1 2, the corruption lists) are all in `cfgs/global.yaml`. `python scripts/setup_check.py` checks that the data is where the config says. Run everything from the repo root.

## Layout

| Folder | What is in it |
|---|---|
| `scripts/` | One numbered script per result (`01_...`, `02_...`), in order of appearance in the paper. Older scripts without a number are being retired. |
| `src/experiments/` | The engine every script shares: it runs the models, caches their logits, scores combinations of them, and builds the tables. |
| `src/tta/`, `src/calibrators/`, `src/reliability/` | TTA methods (Tent, ...), the ways of combining the two models' logits, and the proxies behind the gate. |
| `src/utils/` | Config loading, one class per dataset, LaTeX table writer, ... |
| `cfgs/global.yaml` | Shared settings: dataset paths, output folders, protocol constants. |
| `cfgs/duos/` | The model pairs (default `vitb16_rn50`). |
| `cfgs/tuned/` | Tuned Tent settings per model, written by `scripts/03_tune_tta.py` and read by every other script. |
| `slurm/job.sh` | The one launcher for running any script as a SLURM job. |
| `out/paper/` | The final tables and figures (tracked by git). |
| `out/results/<script>/` | Each script's CSV files (git-ignored). |
| `cache/` | The cached model outputs, about 3 GB per full run (git-ignored). |
| `checkpoints/` | Fitted temperatures and other fitted values (git-ignored). |

## How the scripts work

**Adapt once, combine many times.** Adapting the models is the expensive part, and it does not depend on how the two models are combined afterwards. So a script adapts the models once per (TTA method, duo, stream, seed) and saves both models' logits in `cache/`. Every way of combining them is then a cheap replay on those saved logits, with no model and no dataset needed.

**The cache can't go stale.** Its name includes a fingerprint of everything that changes the logits: the models, their learning rates, the TTA method, the seed, the batch size. Change a learning rate and the old cache is simply not found, so you never read outdated logits by accident.

**Resumable.** Each ImageNet-C corruption is saved as soon as it finishes. If a job is killed or runs out of walltime, run the same command again: streams already saved are skipped and only the missing ones run. (`03_tune_tta.py` works the same way with one file per trial.)

**Shardable.** The work of a script is a list of independent jobs, for example one per seed. `--shard i/N` runs only every N-th job. `--shard auto` reads its position from the SLURM job array, so one command line splits the work over many GPUs:

```bash
sbatch --array=0-2 slurm/job.sh scripts/01_tab_temperatures.py --seeds 0 1 2 --shard auto
```

Array indices must be `0..N-1`. A shard only fills the cache and exits. When you then run the script without `--shard`, it finds everything in the cache and builds the tables. Anything still missing is computed at that point, unless you pass `--report_only`: then it fails and lists what is missing.

**Flags every script takes:**

| Flag | Meaning |
|---|---|
| `--duo NAME` | Which model pair, `cfgs/duos/NAME.yaml`. |
| `--tta METHOD` | TTA method of the adapting models (default `tent`). |
| `--seeds 0 1 2` | Stream-order seeds. Tables show mean ± std over them. |
| `--num_samples N` | Use only N images per stream. **For smoke tests**: results are written to `out/results/`, never to `out/paper/`. |
| `--report_only` | Never run a model: build the tables from the cache, or fail and list what is missing. |
| `--shard i/N` or `auto` | Run only every N-th job (see above). |
| `--only TEXT [...]` | Run only the jobs whose label contains TEXT, such as `seed1`. |
| `--out DIR`, `--global FILE`, `--device cuda/cpu` | Output folder, alternative global config, device. |

**Running a script as a job.** `slurm/job.sh` takes the script and its arguments. It loads the modules, checks that the GPU is usable, and runs the script. Output goes to `logs/<jobid>_<name>.out` and `.err`. Options you give to `sbatch` override the defaults at the top of the file:

```bash
sbatch slurm/job.sh scripts/01_tab_temperatures.py                 # 1 GPU, default walltime
sbatch -t 05:00:00 slurm/job.sh scripts/01_tab_temperatures.py     # longer walltime
sbatch --gres=gpu:0 -t 00:30:00 slurm/job.sh scripts/setup_check.py # a job without a GPU
```

On this cluster the login node kills processes above 5 GB of memory, so run anything with real models as a job.

## Scripts so far

| Script | What it does |
|---|---|
| `01_tab_temperatures.py` | Which temperatures Fixed TS should use, and how much better temperature scaling could possibly be (below). |
| `02_tab_adaptation_modes.py` | Joint, large anchor, small anchor and independent adaptation of the duo, scored with the same Fixed TS. |
| `03_tune_tta.py --model NAME` | Tunes Tent's optimizer settings for one model on the 4 held-out corruptions, and writes `cfgs/tuned/`. |
| `setup_check.py` | Checks the dataset paths and the config. |

Later scripts (`08` onwards, the main results) are planned in `plan.md`.

## Script 01: temperature study

`01_tab_temperatures.py` produces one table. It settles which temperatures T_L and T_S Fixed TS uses, and gives upper bounds on what any temperature-based method could reach.

**Streams.** One row per stream: clean validation images, the 4 *held-out* corruptions (speckle noise, Gaussian blur, spatter, saturate; used only for tuning, never for results), and the 15 test corruptions at severity 5. The last row averages the 15 test corruptions.

**Columns**, with the duo being `(z_L / T_L + z_S / T_S) / 2`:

| Column | Meaning |
|---|---|
| `T_L*`, `T_S*` | The temperatures that minimize the duo's NLL on that stream (this uses the labels). |
| Large, Small | Each model's own accuracy. |
| Clean TS | Accuracy with temperatures fitted on clean data. |
| Held-out TS | Accuracy with temperatures fitted on the 4 held-out corruptions. This is the paper's Fixed TS. |
| Corr.-wise | Oracle: the temperatures `T_L*`, `T_S*` fitted on the stream itself. |
| Batch-wise, `B=16`, `B=128` | Oracle: temperatures refitted on every consecutive batch of B samples (`--batch_sizes` sets the sizes). |
| Per-sample | Oracle: temperatures fitted on each single sample. The ceiling. |

The oracles use the labels and are fitted on the same data they are scored on, so they are upper bounds and not methods.

**Running it.**

```bash
sbatch slurm/job.sh scripts/01_tab_temperatures.py                       # Tent-adapted members, seed 0
sbatch slurm/job.sh scripts/01_tab_temperatures.py --mode no_adapt       # frozen members instead
sbatch slurm/job.sh scripts/01_tab_temperatures.py --report_only         # rebuild the table from the cache
```

The first command adapts both models over the 20 streams and caches their logits. Expect roughly 1 to 1.5 hours per seed on one GPU. Run the table step as a job too: the fits go over all 20 streams' logits and are slow on a CPU login node. The default is seed 0 only. `--seeds 0 1 2` adds the mean ± std over seeds, and makes the work three jobs (one per seed), which `--shard auto` can spread over a job array (see above).

**Output.**
- `out/paper/tab_temperatures__<tag>.tex`: the table.
- `out/results/tab_temperatures/<tag>.csv`: the accuracy and NLL of every column.
- `<tag>` is the TTA method (`tent`), or `frozen` with `--mode no_adapt`, with `_trainslice` added if `--clean_source train_slice`.

## Checking the code

There is no pytest. Every module has a self-test, run from the repo root:

```bash
python -m src.experiments.runner       # the engine, end to end on tiny models
python -m src.experiments.tuning       # the tuner
python -m src.utils.config             # config loading
```
