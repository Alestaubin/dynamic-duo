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

Writes `tab_temperatures__frozen_bn__<duo>.tex` (`--duo <name>`; default `vitb16_rn50v1`): per stream (clean, the 4 held-out corruptions, the 15 test corruptions) the NLL-optimal temperatures, the two members, Fixed TS fitted on clean data and on the held-out corruptions, and three oracles (per-corruption, per-batch and per-sample temperatures).

**2. Tune Tent's learning rates: each model alone, and the independent duo** (once; the result is stored in `cfgs/duo_lrs/`, so skip this to reuse it)

Rules (author, 2026-10-22): only the 4 held-out corruptions are used (no test leakage); every score uses the Fixed TS fitted on the **frozen** models, never refitted per method or pair (`sbatch slurm/job.sh scripts/03_tab_adaptation_modes.py --bn --temps_only` fits it once; the table of step 1 reports it); SGD with momentum 0.9, the rate only, on a sqrt-2 grid that is extended until the optimum is **interior**; the criterion is the whole-stream accuracy, and a candidate whose accuracy on the last 20% of each stream is below the frozen model's is ruled out (drift); the frozen model is on every table; the finalists are confirmed on 3 stream-order seeds.

```bash
# the members never see each other, so a run at (a_i, b_i) stores the large member at a_i and the small one at b_i: N runs give N individual curves and N x N pairs
for i in 0..N-1: sbatch slurm/job.sh scripts/04_cache_logits.py --bn --streams heldout --lr_large a_i --lr_small b_i --seeds 0
sbatch slurm/job.sh scripts/04_cache_logits.py --bn --streams heldout --mode no_adapt --seeds 0        # the frozen members
sbatch slurm/job_cpu.sh scripts/02c_tune_indep_pair.py --bn --lrs_large a_0 .. a_N-1 --lrs_small b_0 .. b_N-1 --seeds 0        # stage A: the full grid
sbatch slurm/job_cpu.sh scripts/02c_tune_indep_pair.py --bn --lrs_large ... --lrs_small ... --seeds 0 1 2 --record              # stage B: neighbourhood on seeds 1, 2 (it lists the runs it needs), then record
```

`02c` writes each model's best rate to `cfgs/duo_lrs/<duo>__single.yaml` (the regime of one model adapting alone: script 03's Independent large / small) and the best pair to `<duo>__indep.yaml` (the independent duo: Independent both, Vanilla DD); `--record` is refused while an optimum sits on the edge of the grid. `scripts/02_tune_tta.py` (trial-based, one model at a time) is the older tuner; its results are superseded for Tent.

**3. Tune the learning rates of the Joint mode** (both members adapt on the duo's entropy; the grid starts at step 2's single rates and increases)

```bash
sbatch --array=0-47 slurm/job.sh scripts/02b_tune_joint_lr.py --bn --lrs_large a_0 .. --lrs_small b_0 .. --seeds 0 --shard auto --only joint/   # one task per pair
sbatch slurm/job_cpu.sh scripts/02b_tune_joint_lr.py --bn --lrs_large ... --lrs_small ... --seeds 0 1 2 --report_only                       # ranks, records the best pair in <duo>__joint.yaml
```

The array tasks must all get the same grid flags; the temperatures are the frozen fit; every point reports its late-stream accuracy and the ones that end below the frozen duo are ruled out. The ranking prints a warning when the best pair sits on the edge of the grid (extend it; finished points are reused).

**3b. Tune the learning rate of Large adapt and of Small adapt** (ONE member adapts on the duo's entropy while the other is frozen: its own regime, tuned for that mode alone, NOT the joint pair's rate; the grid starts at step 2's single rate and goes both ways)

```bash
sbatch --exclude=kn117 -t 00:45:00 --array=0-7 slurm/job.sh scripts/02d_tune_large_adapt_lr.py --bn --shard auto      # Large adapt: one task per rate
sbatch --exclude=kn117 -t 00:30:00 --array=0-7 slurm/job.sh scripts/02e_tune_small_adapt_lr.py --bn --shard auto      # Small adapt
python scripts/02d_tune_large_adapt_lr.py --bn --report_only                       # ranks (add --lrs ... to refine, --seeds 0 1 2 for the finalists)
python scripts/02d_tune_large_adapt_lr.py --bn --lrs ... --seeds 0 1 2 --report_only --record   # records the winner in <duo>__large_duo.yaml
```

Same rules as steps 2 and 3 (held-out streams, frozen Fixed TS, an interior optimum, a late-stream guard against the frozen duo); `--record` is explicit and refused while a warning is open. Script 03 reads these rates for its Large adapt / Small adapt modes and raises `NotTuned` until they are recorded.

**4. Table: ways of adapting the duo** (joint, one member adapting, independent; all scored with the same Fixed TS). Needs steps 2, 3 and 3b.

```bash
sbatch slurm/job.sh scripts/03_tab_adaptation_modes.py --bn --temps_only            # once: fit and save T_L, T_S
sbatch --array=0-20 slurm/job.sh scripts/03_tab_adaptation_modes.py --bn --shard auto    # 7 modes x 3 seeds
sbatch slurm/job.sh scripts/03_tab_adaptation_modes.py --bn --report_only           # builds the tables from the cache
```

Writes `tab_adaptation_modes__tent__<duo>_bn.tex` (accuracy per corruption) and `tab_adaptation_modes_summary__tent__<duo>_bn.tex` (accuracy, ECE, NLL per mode). Every table and figure is tagged with its models and the time it was generated (the first comment line of a `.tex` file, the footer and metadata of a figure; `src/utils/provenance.py`), and every file name carries the duo, so two duos never share a file.

**Other duos.** The same steps 1 to 4 run for any duo of `cfgs/duos/` with `--duo <name>`: besides the main `vitb16_rn50v1` (ViT-B/16 + ResNet-50), `vitb16_rn34` (ViT-B/16 + ResNet-34) and `convnextb_deits` (ConvNeXt-B + DeiT-S, no BatchNorm layer, so `--bn` changes nothing). A new duo needs `cfgs/duos/<name>.yaml` and its three `cfgs/duo_lrs/<name>__{single,indep,joint}.yaml` files, which `src/utils/duo_lrs.py` can create (`init_single`, `init_indep`, `init_joint`); steps 2 and 3 then fill in the rates, and step 3b creates `<name>__large_duo.yaml` and `<name>__small_duo.yaml` itself. Use `--exclude=kn117` on GPU jobs.

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
