# Dynamic duos

Test-time adaptation (TTA) with a **duo**: two pretrained ImageNet models of different size (default: ViT-B/16 and ResNet-50) that both adapt to corrupted images with Tent. Their logits are combined into one prediction, either with fixed temperatures (**Fixed TS**) or with the paper's gate, which gives more weight to whichever model currently looks more reliable (a label-free *proxy*). Everything is evaluated on ImageNet-C.

One script per table or figure. `plan.md` has the status of each script and the design decisions; `docs/script_reference.md` has the long notes on each script.

## Setup

```bash
module load python/3.11 cuda/12.2
python -m venv ~/py38 && source ~/py38/bin/activate     # slurm/job.sh activates ~/py38; edit that line to use another environment
pip install -r requirements.txt
pip install --no-index torch
python scripts/setup_check.py                           # checks that the data is where cfgs/global.yaml says
```

Run everything from the repo root. Dataset paths, output folders and the protocol constants (severity 5, batch size 64, seeds 0 1 2) are in `cfgs/global.yaml`.

## Reproduce the paper tables

The default duo is `vitb16_rn50v1`: torchvision's ViT-B/16 and the *original* (V1) ResNet-50 weights. `--bn` makes a frozen ResNet use the test batch's statistics, as in the paper. A step that starts an `--array` must finish before the next step. Tables are written to `out/paper/`.

**1. Table: optimal temperatures and oracles** (frozen members; needs nothing else)

```bash
sbatch slurm/job.sh scripts/01_tab_temperatures.py --mode no_adapt --batch_sizes 16 64 128 --bn
```

Writes `tab_temperatures__frozen_bn.tex`: per stream (clean, the 4 held-out corruptions, the 15 test corruptions) the NLL-optimal temperatures, the two members, Fixed TS fitted on clean data and on the held-out corruptions, and three oracles (per-corruption, per-batch and per-sample temperatures).

**2. Tune Tent's learning rates** (once; the result is already stored in `cfgs/duo_lrs/`, so skip this to reuse it)

```bash
sbatch --array=0-15 slurm/job.sh scripts/02_tune_tta.py --model vit_b_16 --shard auto
sbatch --array=0-15 slurm/job.sh scripts/02_tune_tta.py --model resnet50_v1 --bn --shard auto
# when both arrays are done (reads the saved trials; no GPU, no grid flags):
sbatch slurm/job_cpu.sh scripts/02_tune_tta.py --model vit_b_16 --rank_cache
sbatch slurm/job_cpu.sh scripts/02_tune_tta.py --model resnet50_v1 --bn --rank_cache
```

`--rank_cache` ranks every trial stored for the model, whatever grid it came from (results made with other settings are listed and left out). It only prints the ranking and writes `ranking.csv`; add `--record` to record the winner in `cfgs/duo_lrs/vitb16_rn50v1__indep.yaml`. The Tent rates in the repo are two by-hand choices (ViT 5e-5, ResNet 2.5e-4), so `--record` replaces them (it prints what it replaced).

**3. Tune the learning rates of the Joint mode** (both members adapt on the duo's entropy; its rates differ from step 2's)

```bash
sbatch slurm/job.sh scripts/02b_tune_joint_lr.py --bn --temps_only                       # once: fit T_L, T_S
sbatch -t 00:40:00 --array=0-24 slurm/job.sh scripts/02b_tune_joint_lr.py --bn --shard auto   # one task per (lr_large, lr_small) pair
sbatch slurm/job_cpu.sh scripts/02b_tune_joint_lr.py --bn --rank_cache --record         # ranks every cached Joint run, records the best pair in cfgs/duo_lrs/<duo>__joint.yaml
```

The array tasks must all get the same grid flags (the default grid is 5 x 5 pairs, hence `--array=0-24`); the ranking needs none, and takes the temperatures from the cached runs. Without `--record` it only prints the ranking.

**4. Table: ways of adapting the duo** (joint, one member adapting, independent; all scored with the same Fixed TS). Needs steps 2 and 3.

```bash
sbatch slurm/job.sh scripts/03_tab_adaptation_modes.py --bn --temps_only            # once: fit and save T_L, T_S
sbatch --array=0-20 slurm/job.sh scripts/03_tab_adaptation_modes.py --bn --shard auto    # 7 modes x 3 seeds
sbatch slurm/job.sh scripts/03_tab_adaptation_modes.py --bn --report_only           # builds the tables from the cache
```

Writes `tab_adaptation_modes__tent_bn.tex` (accuracy per corruption) and `tab_adaptation_modes_summary__tent_bn.tex` (accuracy, ECE, NLL per mode).

**The gate itself** (proxy, beta and proxy-batch size, Joint DD) is selected by `04_cache_logits.py` to `10_select_joint_dd_beta.py`, run in numerical order; each reads what the one before stored, and each script's docstring has its commands. They write the selection in `results/selection_section45.json`, not a table yet. The tables that use it (main accuracy, beta, proxies, continual streams, ablations) are not written yet: see `plan.md`.

## Where things are

| Path | What |
|---|---|
| `scripts/` | The numbered scripts, in the order they are run. |
| `src/` | The code they share: `experiments/` (the engine), `tta/`, `calibrators/`, `reliability/` (the gate and its proxies), `reset/`, `utils/`. |
| `cfgs/global.yaml`, `cfgs/duos/`, `cfgs/duo_lrs/` | Shared settings; which two models a duo is; **the learning rates** of every duo, per regime (`indep`: members adapt on their own entropy; `joint`: on the duo's). |
| `out/paper/`, `out/results/` | The tables and figures (tracked); each script's CSV files (git-ignored). |
| `docs/`, `plan.md` | Design notes; status and decisions. |

## Tests

There is no pytest: every module has a self-test, for example `python -m src.experiments.runner`, `python -m src.utils.config`, `python -m src.reset.live`.
