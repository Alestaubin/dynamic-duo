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
| `cfgs/tuned/` | Tuned Tent settings per model, written by `scripts/02_tune_tta.py` and read by every other script. |
| `slurm/job.sh` | The one launcher for running any script as a SLURM job. |
| `out/paper/` | The final tables and figures (tracked by git). |
| `out/results/<script>/` | Each script's CSV files (git-ignored). |
| `cache/` | The cached model outputs, about 3 GB per full run (git-ignored). |
| `checkpoints/` | Fitted temperatures and other fitted values (git-ignored). |

## How the scripts work

**Adapt once, combine many times.** Adapting the models is the expensive part, and it does not depend on how the two models are combined afterwards. So a script adapts the models once per (TTA method, duo, stream, seed) and saves both models' logits in `cache/`. Every way of combining them is then a cheap replay on those saved logits, with no model and no dataset needed.

**The cache can't go stale.** Its name includes a fingerprint of everything that changes the logits: the models, their learning rates, the TTA method, the seed, the batch size. Change a learning rate and the old cache is simply not found, so you never read outdated logits by accident.

**Resumable.** Each ImageNet-C corruption is saved as soon as it finishes. If a job is killed or runs out of walltime, run the same command again: streams already saved are skipped and only the missing ones run. (`02_tune_tta.py` works the same way with one file per trial.)

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
| `01_tab_temperatures.py` | Which temperatures Fixed TS should use, and how much better temperature scaling could possibly be. |
| `02_tune_tta.py --model NAME` | Tunes Tent's optimizer settings for one model on the 4 held-out corruptions, and writes `cfgs/tuned/`. |
| `03_tab_adaptation_modes.py` | Six ways of adapting the duo (joint, anchors, independent), all scored with the same Fixed TS. |
| `setup_check.py` | Checks the dataset paths and the config. |

Later scripts (`08` onwards, the main results) are planned in `plan.md`.

**Run order.** Run `02_tune_tta.py` first, once for each model of the duo (for the default duo: `--model vit_b_16` and `--model resnet50`). Its result in `cfgs/tuned/` changes the learning rates that every other script uses, and the learning rates are part of the cache fingerprint. Logits cached before tuning are therefore not found any more, and `01` and `03` start from the tuned settings. After that, run `01` and `03` in any order.

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

## Script 02: tune Tent

`02_tune_tta.py` finds good optimizer settings for Tent, one model at a time. Tent's only hyperparameters are the optimizer's, and the best ones differ between models, so they are tuned on the 4 held-out corruptions. Those are never test corruptions, so tuning on them does not leak into the results.

**What one trial is.** One optimizer setting: Adam with a learning rate, or SGD with a learning rate and a momentum. The model is adapted over each of the 4 held-out streams in turn (severity 5, batch size 64, all 50,000 images), and reset to its original weights before each stream. The trial's score is the mean accuracy over the 4 streams. A frozen reference (no updates, test-batch statistics) is run as well, and the winner has to beat it.

The default grid is Adam with learning rates 5e-5 to 1e-3, and SGD with learning rates 1e-4 to 1e-2 and momentum 0 or 0.9. That is 15 trials plus the reference. `--optimizers`, `--adam_lrs`, `--adam_betas`, `--sgd_lrs`, `--sgd_momentums` and `--wd` change it. If the best learning rate is the smallest or largest one tried, the script warns that the real optimum may lie outside the grid.

The model is any name known to `src/utils/model.py` (`resnet50`, `vit_b_16`, `convnext_tiny`, ...). The type of normalization layer that Tent adapts (BN, LN or GN) is read from the model; `--norm` overrides it.

**Running it.**

```bash
sbatch slurm/job.sh scripts/02_tune_tta.py --model resnet50                              # the whole grid in one job
sbatch --array=0-15 slurm/job.sh scripts/02_tune_tta.py --model vit_b_16 --shard auto   # one trial per array task
sbatch slurm/job.sh scripts/02_tune_tta.py --model vit_b_16 --report_only                # rank the stored trials, write the result
```

Each trial runs 4 × 50,000 images, so for the large models use the array. A finished trial is saved on its own: if a job dies, run the same command again and only the missing trials run. When all 16 are saved, a run without `--shard` ranks them. Do not use `--num_samples` to make tuning cheaper: a learning rate that wins over a few updates is usually too large for the 780 updates of a full stream. (It is fine for a smoke test, and a capped run never writes the result.)

**Output.**
- `cfgs/tuned/tent__<model>.yaml`: the winning optimizer settings, plus its score and the frozen model's score. Every other script reads it: for that model and `--tta tent`, it replaces the optimizer settings in `cfgs/duos/*.yaml`. Delete the file to go back to the defaults in the duo file. The script writes it only if the winner beats the frozen model, and only from a full run.
- `out/results/tune_tta/<model>__tent/ranking.csv`: every trial, best first, with its accuracy on each stream.
- `out/results/tune_tta/<model>__tent/trials/`: one JSON file per trial (this is what makes the run resumable).

## Script 03: ways of adapting the duo

`03_tab_adaptation_modes.py` asks: when both models adapt, **what should they adapt on?** It runs six ways of adapting, plus a frozen reference, and scores every one with the same Fixed TS combination `(z_L / T_L + z_S / T_S) / 2`. The rows then differ only in how the models' weights were adapted.

The *duo entropy* is the entropy of that combined prediction. In every name below, the model named is the one that adapts.

| Mode | Who adapts | On which loss |
|---|---|---|
| Joint | Both models | The duo entropy, backpropagated through both models together. |
| Large anchor | The large model only | The duo entropy. The small model is frozen, and its logits are a constant in the loss. |
| Small anchor | The small model only | The duo entropy. The large model is frozen. |
| Independent large | The large model only | Its own entropy (plain Tent). The small model is not adapted, but still contributes to the prediction. |
| Independent small | The small model only | Its own entropy. |
| Independent duo | Both models | Each on its own entropy, with no signal between them. The paper's standard setting. |
| No adaptation | Neither | The reference row: frozen weights, test-batch statistics. |

**The temperatures.** The duo entropy depends on T_L and T_S, so they must exist before the run. All modes share one pair, the paper's Fixed TS: it is fitted once on the 4 held-out corruptions, using models adapted independently. It is the same fit as the "Held-out TS" column of script 01, and it reuses 01's cache. The pair is saved in `checkpoints/fixed_ts/heldout_...`. `--temps_checkpoint DIR` uses another saved pair instead.

**Running it.** The work is one job per (mode, seed): 7 modes × 3 seeds = 21 jobs, which is why the array has indices 0 to 20. Fit the temperatures once first, so that the array tasks do not all fit them at the same time:

```bash
sbatch slurm/job.sh scripts/03_tab_adaptation_modes.py --temps_only                  # once: fit and save T_L, T_S
sbatch --array=0-20 slurm/job.sh scripts/03_tab_adaptation_modes.py --shard auto     # one job per (mode, seed)
sbatch slurm/job.sh scripts/03_tab_adaptation_modes.py --report_only                 # build the tables from the cache
```

`--modes joint indep_duo ...` runs only some modes (then the array has `modes × seeds` tasks), and `--seeds` changes the seeds. The mode names are `joint`, `large_anchor`, `small_anchor`, `indep_large`, `indep_small`, `indep_duo` and `frozen`. The independent-duo and no-adaptation runs are the same runs as in script 01, so they are read from the cache when the settings match. As a rough estimate, the 21 runs take about 9 GPU-hours on an H100.

**Output.**
- `out/paper/tab_adaptation_modes__<tag>.tex`: accuracy per corruption, one row per mode, grouped as in the ImageNet-C paper.
- `out/paper/tab_adaptation_modes_summary__<tag>.tex`: one row per mode with each model's accuracy under that mode, then the duo's accuracy, ECE and NLL, averaged over the 15 corruptions.
- `out/results/tab_adaptation_modes/<tag>.csv`: the numbers behind both tables.
- `<tag>` is the TTA method (`tent`), followed by the duo's name if it is not the default duo.

## Checking the code

There is no pytest. Every module has a self-test, run from the repo root:

```bash
python -m src.experiments.runner             # the engine, end to end on tiny models
python -m src.experiments.tuning             # the tuner (script 02)
python -m src.experiments.adaptation_modes   # the adaptation modes (script 03)
python -m src.utils.config                   # config loading
```
