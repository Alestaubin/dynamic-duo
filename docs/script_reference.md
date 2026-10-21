# Script reference

The per-script notes of the former README.md (moved here on 2026-10-21 so that the README only says how to reproduce the tables). Parts may be out of date: the docstring at the top of each script is the reference, `plan.md` has the status and the decisions.

## About the project

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
| `scripts/` | Numbered scripts (`01_...`, `02_...`), in the order they are run for the experiments of the paper; `04` to `10` are the shared pipeline (cache, tune, select), `11` onwards the paper's tables and figures. Older scripts without a number are being retired. |
| `src/experiments/` | The engine every script shares: it runs the models, caches their logits, scores combinations of them, and builds the tables. |
| `src/tta/`, `src/calibrators/`, `src/reliability/` | TTA methods (Tent, ...), the ways of combining the two models' logits, and the proxies behind the gate. |
| `src/utils/` | Config loading, one class per dataset, LaTeX table writer, ... |
| `cfgs/global.yaml` | Shared settings: dataset paths, output folders, protocol constants. |
| `cfgs/duos/` | The model pairs: which two models, and which norm layers Tent adapts. Nothing else (default `vitb16_rn50v1`, the main duo: ViT-B/16 + ResNet-50 V1). |
| `cfgs/duo_lrs/` | **The one place the learning rates live**: `<duo>__indep.yaml` (each member on its own entropy: Vanilla DD; the tuned settings per TTA method and member, with the default a member runs with when nothing is tuned) and `<duo>__joint.yaml` (both members on the duo's entropy: Joint DD; the pairs of `02b`). Written by `02`, `02b` and `05` (a joint pair can also be copied in by hand with `python -m src.utils.duo_lrs --import_joint`), read by every other script (`python -m src.utils.duo_lrs --show <duo> indep`). |
| `slurm/job.sh`, `slurm/job_cpu.sh` | The launchers for running a script as a SLURM job: with one GPU, and without a GPU (for steps that run no model). |
| `out/paper/` | The final tables and figures (tracked by git). |
| `out/results/<script>/` | Each script's CSV files (git-ignored). |
| `paths.cache` (`/scratch/alxstaub/dynamic-duos/cache`) | The cached model outputs, about 3 GB per full run (outside the repo, on scratch: set in `cfgs/global.yaml`). |
| `checkpoints/` | Fitted temperatures and other fitted values (git-ignored). |

## Models and weights

The model name decides which pretrained weights are loaded (`src/utils/model.py`), and it is part of every cache key, so a different set of weights always gets a different name. Several names look alike:

| Name | Weights |
|---|---|
| `resnet50` | torchvision **V2** (80.9% clean). Trained with label smoothing and mixup, so its outputs are very under-confident and Tent needs a much larger learning rate. |
| `resnet50_v1` | torchvision **V1**, the original weights (76.1% clean) that Tent, RobustBench and COCA use. |
| `vit_b_16` | torchvision ViT-B/16 (ImageNet-1k only). |
| `vit_b_16_augreg` | timm `vit_base_patch16_224.augreg_in21k_ft_in1k` (ImageNet-21k pretrained). What a bare `vit_base_patch16_224` gave in timm 0.6.x, as used by SAR, EATA and COCA. |
| `vit_b_16_augreg2` | timm `...augreg2_in21k_ft_in1k`. What a bare `vit_base_patch16_224` gives in timm 0.9 and later, including the timm pinned here. |
| `vit_b_16_sar` | The ViT of SAR (`timm==0.6.11`, so the `augreg_in21k_ft_in1k` weights of `vit_b_16_augreg`), with SAR's input pipeline: ImageNet mean/std, although these weights were trained with 0.5/0.5. The mismatch is in SAR's published numbers (frozen on the 15 ImageNet-C corruptions at severity 5: 29.9% in SAR's paper, 31.4% here, but 51.5% with the weights' own normalization), so it is kept to stay comparable with SAR, EATA and DeYO. |

The duos `vitb16aug_rn50v1` and `vitb16aug2_rn50v1` pair each timm ViT with `resnet50_v1`, to compare with COCA ("When Small Guides Large"; its ViT is AugReg2 fed ImageNet mean/std). `vitb16sar_rn50v1` pairs `vit_b_16_sar` with `resnet50_v1` and SAR's optimizer settings: the duo to use to compare with SAR, EATA and DeYO. Weights are downloaded on first use and then cached (`$TORCH_HOME` for torchvision, `~/.cache/huggingface` for timm). Loading a new model once on the login node first is a good habit, so that a job array does not start many downloads at once; Hugging Face is reachable from the compute nodes, but I have not tested torchvision's download server from there.

**Input pipeline.** ImageNet-C and CCC images are already 224x224 centre crops, so they go to the models without the models' own Resize(256) (`ShiftDataset.input_transform`, `native_224` in `src/utils/model.py`), as in the Tent, EATA and SAR code. Resizing them zooms in by 14% and costs Tent on ResNet-50 about 7 points of ImageNet-C accuracy (it reproduced COCA's published numbers only without the resize; frozen `resnet50_v1` with source BatchNorm statistics scores 18.2% on the 15 corruptions here, 18.0% in SAR's paper). Clean ImageNet images and the natural shifts keep the models' own transform. This is part of every cache key of an ImageNet-C or CCC stream (`INPUTS_TAG` in `src/experiments/specs.py`): caches and temperature checkpoints made before it are simply not found, and the trials stored by `02_tune_tta.py` are keyed by it too. The learning-rate entries in `cfgs/duo_lrs/` carry no such key, so entries written before it were tuned on zoomed images: rerun `02_tune_tta.py` for a model before trusting its file.

## How the scripts work

**Adapt once, combine many times.** Adapting the models is the expensive part, and it does not depend on how the two models are combined afterwards. So a script adapts the models once per (TTA method, duo, stream, seed) and saves both models' logits in `cache/`. Every way of combining them is then a cheap replay on those saved logits, with no model and no dataset needed.

**The cache can't go stale.** Its name includes a fingerprint of everything that changes the logits: the models, their learning rates, the TTA method, the seed, the batch size. Change a learning rate and the old cache is simply not found, so you never read outdated logits by accident. A member that does not adapt (a frozen run, or the frozen side of `large_indep`/`small_indep`) has no learning rate in the fingerprint, since its optimizer is never built: retuning does not orphan frozen logits.

**Resumable.** Each ImageNet-C corruption is saved as soon as it finishes. If a job is killed or runs out of walltime, run the same command again: streams already saved are skipped and only the missing ones run. (`02_tune_tta.py` works the same way with one file per trial.)

**Shardable.** The work of a script is a list of independent jobs, for example one per seed. `--shard i/N` runs only every N-th job. `--shard auto` reads its position from the SLURM job array, so one command line splits the work over many GPUs:

```bash
sbatch --array=0-2 slurm/job.sh scripts/01_tab_temperatures.py --seeds 0 1 2 --shard auto
```

Array indices must be `0..N-1`. A shard only fills the cache and exits. When you then run the script without `--shard`, it finds everything in the cache and builds the tables. Anything still missing is computed at that point, unless you pass `--report_only`: then it fails and lists what is missing.

**Flags the scripts share** (which script takes which, and the defaults, are in [Arguments of each script](#arguments-of-each-script) below):

| Flag | Meaning |
|---|---|
| `--duo NAME` | Which model pair, `cfgs/duos/NAME.yaml`. |
| `--tta METHOD` | TTA method of the adapting models (default `tent`). `--tta_kwargs JSON` gives it hyperparameters, e.g. `'{"d_margin": 0.1}'`. |
| `--seeds 0 1 2` | Stream-order seeds. Tables show mean ± std over them. |
| `--bn` | Frozen models that use BatchNorm (such as ResNet-50) run on the **test batch's** statistics. Without it they run in the usual `eval()` mode on the **source** statistics. Only frozen runs are affected (see below). |
| `--num_samples N` | Use only N images per stream. **For smoke tests**: results are written to `out/results/`, never to `out/paper/`. |
| `--report_only` | Never run a model: build the tables from the cache, or fail and list what is missing. |
| `--shard i/N` or `auto` | Run only every N-th job (see above). |
| `--only TEXT [...]` | Run only the jobs whose label contains TEXT, such as `seed1`. |
| `--out DIR`, `--global FILE`, `--device cuda/cpu` | Output folder, alternative global config, device. |

**Frozen models and `--bn`.** A model that adapts with Tent always uses the test batch's statistics in its BatchNorm layers, so `--bn` does not touch it. It matters only for a model that is *not* adapted:
- without `--bn`, a frozen BatchNorm model is put in the usual `eval()` mode and uses the statistics it was trained with (the source statistics);
- with `--bn`, it uses the statistics of each test batch instead. This usually helps on corrupted images, so it is the stricter baseline.

LayerNorm models (ViT, DeiT, Swin) have no running statistics, so for them `--bn` changes nothing. The flag only has an effect where something is frozen: the frozen runs of script 01, the frozen reference of script 02, and in script 03 the frozen reference rows and the model that is not adapted in the Large / Small adapt and Independent large / small modes. Every output file made with `--bn` gets a `_bn` in its name, in every mode, so the two settings never overwrite each other. All frozen runs and caches made before this flag existed used the test batch's statistics: they are the `--bn` runs, and `--bn` reuses them.

**Running a script as a job.** `slurm/job.sh` takes the script and its arguments. It loads the modules, checks that the GPU is usable, and runs the script. Output goes to `logs/<jobid>_<name>.out` and `.err`. Options you give to `sbatch` override the defaults at the top of the file:

```bash
sbatch slurm/job.sh scripts/01_tab_temperatures.py                 # 1 GPU, default walltime
sbatch -t 05:00:00 slurm/job.sh scripts/01_tab_temperatures.py     # longer walltime
sbatch slurm/job_cpu.sh scripts/setup_check.py                      # a job without a GPU
```

`slurm/job_cpu.sh` is the same launcher without the GPU request. Use it only for steps that run no model, such as checking the setup or ranking saved results. Never use it for a step that adapts models: on the CPU that runs at about 1 image per second.

On this cluster the login node kills processes above 5 GB of memory, so run anything with real models as a job.

## Scripts so far

| Script | What it does |
|---|---|
| `01_tab_temperatures.py` | Which temperatures Fixed TS should use, and how much better temperature scaling could possibly be. |
| `02_tune_tta.py --model NAME` | Tunes Tent's optimizer settings for one model on the 4 held-out corruptions, and records the winner in `cfgs/duo_lrs/` (the indep file of every duo that uses the model). |
| `02b_tune_joint_lr.py` | Tunes the two members' learning rates **together** for the Joint mode (both adapt on the duo entropy) on the 4 held-out corruptions: a grid over pairs, scored by the duo's accuracy. Records the winning pair in `cfgs/duo_lrs/<duo>__joint.yaml` itself (a full ranking run only); prints the script-03 command that evaluates the winner. |
| `03_tab_adaptation_modes.py` | Six ways of adapting the duo (joint, one model adapting, independent), all scored with the same Fixed TS. |
| `04_cache_logits.py` | Adapts a duo's two members (they never see each other) and stores their logits **per member**, under a key made of that member's own settings, in `cache/logits/`. `--grid` runs the learning-rate sweep (run i uses the i-th value of each model's grid), `--streams` picks held-out / test / clean / the clean training slice / a natural shift. Any large member can be paired with any small one afterwards. Produces no table. |
| `05_select_member_lr.py` | Reads the sweep that `04` stored for ETA, EATA or SAR and picks each member's learning rate on the held-out streams (the counterpart of `02` for the other algorithms); `--write_tuned` records the winners in `cfgs/duo_lrs/`. |
| `06_run_coca.py` | COCA through its official code, run live (its temperature is refitted every batch, so it cannot be replayed from logits); the same learning-rate grid, one pair of learning rates per run. |
| `07_eval_duo_grid.py` | Reads a stored pair and evaluates Vanilla Dynamic Duos offline: every proxy x beta x proxy-batch size, the members alone, the fixed combinations, the oracles and the proxy diagnostics, from the cached logits alone (no model). |
| `08_select_proxy.py` | Chooses the proxy, beta and proxy-batch size of Vanilla Dynamic Duos on the held-out streams, per base algorithm, and writes `results/selection_section45.json`. |
| `09_run_joint_dd.py` | Live Joint Dynamic Duos runs on the held-out streams (the gate sits inside the adaptation loop), one run per (proxy, b_t, beta). |
| `10_select_joint_dd_beta.py` | Picks Joint DD's beta from the runs of `09` and records it in the selection JSON. |
| `setup_check.py` | Checks the dataset paths and the config. Not numbered: run it first. |

Later scripts (`11` onwards, the paper's tables and figures) are planned in `plan.md`.

**Run order.** `setup_check.py`; then `02_tune_tta.py` once for each model of the duo (for the default duo: `--model vit_b_16` and `--model resnet50_v1`), because its result in `cfgs/duo_lrs/` changes the learning rates every other script uses, and the learning rates are part of the cache fingerprint (logits cached before tuning are not found any more); then `01`, `03`, `04`, ... in numerical order. `04` to `06` and `09` run models, so launch them as jobs (never on the CPU); `07`, `08` and `10` only read saved results. The design is in `docs/design_section45.md`; launch arrays only after checking `plan.md`, Phase 3.

```bash
python scripts/04_cache_logits.py --duo vitb16_rn50v1 --streams heldout clean_train_slice --grid --seeds 0 --shard auto   # as a SLURM array
python scripts/05_select_member_lr.py --duo vitb16_rn50v1 --tta sar --seeds 0 --write_tuned                                 # offline, per algorithm
python scripts/07_eval_duo_grid.py --duo vitb16_rn50v1 --seeds 0 --grid_large 3 --grid_small 2                              # offline, one pair
python scripts/08_select_proxy.py --duo vitb16_rn50v1 --algos tent eta --seed 0                                             # writes the selection JSON
```

## Script 01: temperature study

`01_tab_temperatures.py` produces one table. It settles which temperatures T_L and T_S Fixed TS uses, and gives upper bounds on what any temperature-based method could reach.

**Streams.** One row per stream: clean validation images, the 4 *held-out* corruptions (speckle noise, Gaussian blur, spatter, saturate; used only for tuning, never for results), and the 15 test corruptions at severity 5. The last row averages the 15 test corruptions.

**Columns**, with the duo being `(z_L / T_L + z_S / T_S) / 2`:

| Column | Meaning |
|---|---|
| `T_L*`, `T_S*` | The temperatures that minimize the duo's NLL on that stream (this uses the labels). |
| `T_L*/T_S*` | Their ratio. The Average row is the mean of the 15 per-stream ratios, not the ratio of the two averaged columns. |
| Large, Small | Each model's own accuracy. |
| Clean TS | Accuracy with temperatures fitted on clean data. |
| Held-out TS | Accuracy with temperatures fitted on the 4 held-out corruptions. This is the paper's Fixed TS. |
| Corr.-wise | Oracle: the temperatures `T_L*`, `T_S*` fitted on the stream itself. |
| Batch-wise, `B=128`, `B=16` | Oracle: temperatures refitted on every consecutive batch of B samples (`--batch_sizes` sets the sizes; the columns run from the largest B to the smallest). |
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
- `<tag>` is the TTA method (`tent`), or `frozen` with `--mode no_adapt`. `_bn` is added with `--bn` (in every mode), `_trainslice` with `--clean_source train_slice`, and `__<duo>` if the duo is not the default one. For example `tab_temperatures__frozen_bn__vitb16_rn50v1.tex`.

## Script 02: tune Tent

`02_tune_tta.py` finds good optimizer settings for Tent, one model at a time. Tent's only hyperparameters are the optimizer's, and the best ones differ between models, so they are tuned on the 4 held-out corruptions. Those are never test corruptions, so tuning on them does not leak into the results.

**What one trial is.** One optimizer setting: Adam with a learning rate, or SGD with a learning rate and a momentum. The model is adapted over each of the 4 held-out streams in turn (severity 5, batch size 64, all 50,000 images), and reset to its original weights before each stream. The trial's score is the mean accuracy over the 4 streams. A frozen reference (no updates) is run as well, and the winner has to beat it. For a BatchNorm model such as ResNet-50 the frozen reference is in `eval()` mode on the source statistics, or on the test batch's statistics with `--bn` (a stricter bar). The trials themselves always use batch statistics, so `--bn` changes nothing else, and the trials are reused either way.

The default grid is Adam with learning rates 1e-5 to 1e-3, and SGD with learning rates 1e-4 to 1e-2 and momentum 0 or 0.9. That is 15 trials plus the reference. `--optimizers`, `--adam_lrs`, `--adam_betas`, `--sgd_lrs`, `--sgd_momentums` and `--wd` change it. If the best learning rate is the smallest or largest one tried, the script warns that the real optimum may lie outside the grid.

The model is any name known to `src/utils/model.py` (`resnet50`, `vit_b_16`, `convnext_tiny`, ...). The type of normalization layer that Tent adapts (BN, LN or GN) is read from the model; `--norm` overrides it.

**Running it.**

```bash
sbatch slurm/job.sh scripts/02_tune_tta.py --model resnet50                              # the whole grid in one job
sbatch --array=0-15 slurm/job.sh scripts/02_tune_tta.py --model vit_b_16 --shard auto   # one trial per array task
python scripts/02_tune_tta.py --model vit_b_16 --report_only                             # rank the stored trials, write the result
```

Each trial runs 4 × 50,000 images, so for the large models use the array. A finished trial is saved on its own: if a job dies, run the same command again and only the missing trials run. When all 16 are saved, a run without `--shard` ranks them. That last step only reads the saved trials, so it needs no GPU and runs in seconds on the login node (as in the third command above; it loads the model once to read its norm type, so set `TORCH_HOME` as `slurm/job.sh` does, or pass `--norm LN` / `--norm BN` to skip that). Do not use `--num_samples` to make tuning cheaper: a learning rate that wins over a few updates is usually too large for the 780 updates of a full stream. (It is fine for a smoke test, and a capped run never writes the result.)

**Output.**
- `cfgs/duo_lrs/<duo>__indep.yaml`, `methods.tent.<large|small>` of every duo that uses the model: the winning optimizer settings, plus its score and the frozen model's score. Every other script reads it (for that model and `--tta tent`; a member with no tuned entry runs with the file's `defaults`). The script records it only if the winner beats the frozen model, and only from a full run.
- `out/results/tune_tta/<model>__tent/ranking.csv`: every trial, best first, with its accuracy on each stream.
- `out/results/tune_tta/<model>__tent/trials/`: one JSON file per trial (this is what makes the run resumable).

## Script 02b: tune the learning rates of the Joint mode

`02b_tune_joint_lr.py` asks: when both models adapt on the duo entropy (the Joint mode of script 03), **which pair of learning rates is best?** Script 02 tunes one model alone, on its own entropy. The duo entropy is a steadier signal (a model that starts to collapse is held up by its partner), so Joint may tolerate larger rates, and the two rates interact, so the grid is over **pairs** `(lr_large, lr_small)`.

**The grid.** By default 1, 2, 4, 8 and 16 times each model's *current* learning rate (the tuned one in `cfgs/duo_lrs/<duo>__indep.yaml`, else its default): 25 pairs. Only the learning rate changes; momentum and the rest of each `OPTIM` stay. `--mults` changes the multiples, `--lrs_large` / `--lrs_small` give explicit rates. Finished points are reused, so extending a grid only runs the new pairs.

**What one run is.** One Joint run over the 4 held-out streams (severity 5, all 50,000 images each), both models reset to source weights before every stream. The score is the mean over the 4 streams of the accuracy of the static duo `(z_L / T_L + z_S / T_S) / 2`, the same prediction script 03's Joint row scores. The worst stream and each model's own accuracy are reported next to it. Two references run as well: the frozen duo, and Independent both at the current rates (already cached: the temperatures' fit needs it). The report then says what the duo signal buys at *equal* rates (the Joint run at the current rates against Independent both) and what *larger* rates buy on top (the best pair against that run).

**The temperatures** T_L, T_S are fixed for the whole grid, because a pair per point would change the objective being compared. They are script 03's Fixed TS (fitted on these 4 corruptions on independently adapted models at the current rates), or `--temps_checkpoint DIR`.

**Divergence.** A learning rate so large that the logits become non-finite cannot be cached. Such a run is recorded as *diverged* (a `diverged.json` next to its cache), counts as finished, and ranks last.

**Running it.** One task per grid point; the references fall onto the first tasks. About 17 minutes per point on an L40S (a held-out-only run at Tent's speed, from the rate in `plan.md`; not measured for this script), so ask for a tight limit:

```bash
sbatch slurm/job.sh scripts/02b_tune_joint_lr.py --duo vitb16_rn50v1 --bn --temps_only          # once: fit and save T_L, T_S
sbatch -t 00:40:00 --array=0-24 slurm/job.sh scripts/02b_tune_joint_lr.py --duo vitb16_rn50v1 --bn --shard auto
sbatch slurm/job_cpu.sh scripts/02b_tune_joint_lr.py --duo vitb16_rn50v1 --bn --report_only     # rank (no model runs)
```

All tasks and the ranking run must be given the same grid flags. `--bn` only affects the frozen reference.

**Output**, in `out/results/tune_joint_lr/<duo>__<tta>[_bn]/` (a capped `--num_samples` run has its own folder):
- `ranking.csv`: one row per run (references and diverged points included), the ranked grid points first; accuracies are fractions.
- `results.json`: the same rows plus the temperatures, optimizers, grid and the winner.
- The printout: the grid of duo accuracies and of worst-stream accuracies, the top of the ranking, the comparisons above, and a warning when a model's best rate is the smallest or largest tried (the optimum may lie outside the grid).

**The best pair is recorded automatically.** When a full ranking run (not a capped `--num_samples` one) has written `results.json`, it records the winning pair in `cfgs/duo_lrs/<duo>__joint.yaml` (the joint regime), replacing the pair already there for that TTA method and any `pending:` note, and prints what it replaced. From then on script 03 and the reset scripts read the pair from that file. The file always holds the winner of the *last* ranking run, so a report with a smaller grid than before can record a different winner; pass the same grid flags as the tasks. `python -m src.utils.duo_lrs --import_joint <results.json>` does the same by hand.

**It does not touch the indep file of `cfgs/duo_lrs/`**: those rates also set the independent modes, and these rates belong to the Joint mode alone. The last line printed is the command that scores the recorded pair on the 15 test corruptions: script 03 with `--modes joint --temps_checkpoint <the same temperatures>` (no rates needed; `--joint_lr_large X --joint_lr_small Y` would give them by hand).

## Script 03: ways of adapting the duo

`03_tab_adaptation_modes.py` asks: when both models adapt, **what should they adapt on?** It runs six ways of adapting, plus a frozen reference, and scores every one with the same Fixed TS combination `(z_L / T_L + z_S / T_S) / 2`. The rows then differ only in how the models' weights were adapted.

The *duo entropy* is the entropy of that combined prediction. In every name below, the model named is the one that adapts.

| Mode | Who adapts | On which loss |
|---|---|---|
| Joint | Both models | The duo entropy, backpropagated through both models together. |
| Large adapt | The large model only | The duo entropy. The small model is frozen, and its logits are a constant in the loss. |
| Small adapt | The small model only | The duo entropy. The large model is frozen. |
| Independent large | The large model only | Its own entropy (plain Tent). The small model is not adapted, but still contributes to the prediction. |
| Independent small | The small model only | Its own entropy. |
| Independent both | Both models | Each on its own entropy, with no signal between them. The paper's standard setting. |
| No adaptation | Neither | The reference row: frozen weights (BatchNorm on the source statistics, or on the test batch's with `--bn`). |

**The temperatures.** The duo entropy depends on T_L and T_S, so they must exist before the run. All modes share one pair, the paper's Fixed TS: it is fitted once on the 4 held-out corruptions, using models adapted independently. It is the same fit as the "Held-out TS" column of script 01, and it reuses 01's cache. The pair is saved in `checkpoints/fixed_ts/heldout_...`. `--temps_checkpoint DIR` uses another saved pair instead.

**The learning rates.** The two kinds of mode adapt on different signals, and each runs with its own rates (`cfgs/duo_lrs/<duo>__<regime>.yaml`):
- The modes that adapt on a model's **own entropy** (Independent large / small / both) use the *indep* regime's rates, tuned one model at a time by script 02 (or 05).
- The modes that adapt on the **duo entropy** (Joint, Large adapt, Small adapt) use the *joint* regime's pair, tuned together by script 02b. Large / Small adapt use the rate of the model that adapts.
- A duo whose joint pair is not tuned yet makes those three modes an **error** (`NotTuned`), never a silent run at the indep rates. `--joint_lr_large LR --joint_lr_small LR` run them with rates given by hand (both, to skip the file); `--lr_large` / `--lr_small` do the same for the own-entropy modes.
- Both tables' captions list both sets of rates and say which modes use which. The temperatures stay one pair for every mode (fitted on the indep regime's members, as before).

**Running it.** The work is one job per (mode, seed): 7 modes × 3 seeds = 21 jobs, which is why the array has indices 0 to 20. Fit the temperatures once first, so that the array tasks do not all fit them at the same time:

```bash
sbatch slurm/job.sh scripts/03_tab_adaptation_modes.py --temps_only                  # once: fit and save T_L, T_S
sbatch --array=0-20 slurm/job.sh scripts/03_tab_adaptation_modes.py --shard auto     # one job per (mode, seed)
sbatch slurm/job.sh scripts/03_tab_adaptation_modes.py --report_only                 # build the tables from the cache
```

`--modes joint indep_duo ...` runs only some modes (then the array has `modes × seeds` tasks), and `--seeds` changes the seeds. The mode names are `joint`, `large_anchor`, `small_anchor`, `indep_large`, `indep_small`, `indep_duo` and `frozen`. The independent-both and no-adaptation runs are the same runs as in script 01, so they are read from the cache when the settings match. As a rough estimate, the 21 runs take about 9 GPU-hours on an H100.

Every run also visits the clean validation images (the 50,000 images ImageNet-C is rendered from), like one more corruption. A cache from before this existed only computes that one stream per run, since the corruption caches keep their keys.

**Output.**
- `out/paper/tab_adaptation_modes__<tag>.tex`: accuracy per corruption, grouped as in the ImageNet-C paper, with no standard deviations in the cells (the caption gives their maximum). Four sections, each named by what is scored and the entropy the adapting models minimize, with one row per choice of which models adapt:
  - *Frozen*: Large, Small, Duo (the No adaptation run).
  - *Single model, own entropy (Tent)*: Large adapts, Small adapts (the large model of the Independent large run, the small model of the Independent small run, each scored without the duo).
  - *Duo, own entropy*: Large adapts, Small adapts, Both adapt (Independent) (the duo of the Independent large / small / both runs).
  - *Duo, duo entropy*: Large adapts, Small adapts, Both adapt (Joint) (the duo of the Large adapt / Small adapt / Joint runs).

  A *Clean* column holds the accuracy on the clean validation images; it is not part of *Avg.*, which is the mean over the 15 corruptions.
- `out/paper/tab_adaptation_modes_summary__<tag>.tex`: one row per mode with each model's accuracy under that mode, then the duo's accuracy, ECE and NLL, averaged over the 15 corruptions.
- `out/results/tab_adaptation_modes/<tag>.csv`: the numbers behind both tables (clean stream included, as segment `clean`).
- Both tables' captions list the hyperparameters of the adapting models: the batch size and each model's optimizer with all its parameters, read from the duo config as `cfgs/duo_lrs/` gives it (the values the cached logits were made with). When the two regimes' optimizers differ, the caption lists both and says that the models adapting on their own entropy use the first and those adapting on the duo entropy the second.
- `<tag>` is the TTA method (`tent`), followed by the duo's name if it is not the default duo, and `_bn` with `--bn`.

## Arguments of each script

`python scripts/<name>.py --help` always shows the live list. Giving a script a flag it does not take is an error (for example `--duo` for script 02, which works on one model).

**Who takes what.** The meaning of the shared flags is in the table under [How the scripts work](#how-the-scripts-work).

| Flag | 01 | 02 | 03 | `setup_check` |
|---|---|---|---|---|
| `--duo NAME` | default `vitb16_rn50v1` | no (it has `--model`) | default `vitb16_rn50v1` | no |
| `--tta METHOD` | default `tent` | `tent` is the only choice | default `tent` | no |
| `--tta_kwargs JSON` | yes | no | yes | no |
| `--seeds` | default `0` | no (it has `--seed`, default 0) | default `0 1 2` | no |
| `--num_samples N` | yes | yes (and the tuned file is not written) | yes | no |
| `--bn` | yes | yes (the frozen reference) | yes | yes (with `--probe`) |
| `--report_only` | yes | yes (ranks the stored trials) | yes | no |
| `--shard`, `--only` | one job per seed, labelled `seed0` | one job per trial, labelled like `Adam_lr0.001_b0.9` or `frozen` | one job per (mode, seed), labelled like `joint/seed0` | no |
| `--out DIR` | CSV folder, default `out/results/tab_temperatures` | trials and ranking, default `out/results/tune_tta/<model>__tent` | CSV folder, default `out/results/tab_adaptation_modes` | no |
| `--global FILE` | yes | yes | yes | yes |
| `--device` | yes | yes | yes | no |

For 01 and 03 the tables go to `out/paper/` (`paths.paper`), not to `--out`, except in a capped run (`--num_samples`), which writes them to `--out`. `--global FILE` replaces `cfgs/global.yaml` for that run (the environment variable `DUO_GLOBAL_CONFIG` does the same for a whole shell).

### Script 01: `01_tab_temperatures.py`

| Argument | Meaning | Default |
|---|---|---|
| `--mode {both_indep,no_adapt}` | `both_indep`: the two models adapt independently with `--tta`. `no_adapt`: frozen models (BatchNorm on the source statistics, or on the test batch's with `--bn`). | `both_indep` |
| `--clean_source {val,train_slice}` | What defines the "clean" row: the 50,000 validation images, or the held-out clean slice of the training set. | `val` |
| `--batch_sizes B [B ...]` | Batch sizes of the batch-wise oracle, one table column each (largest first, whatever order you give). | `16 128` |
| `--steps N` | Gradient steps of the per-sample oracle. | `200` |

### Script 02: `02_tune_tta.py`

A trial is one combination of optimizer settings: Adam is learning rate × beta, SGD is learning rate × momentum. A trial that is already saved is reused, so enlarging a grid only runs the new combinations.

| Argument | Meaning | Default |
|---|---|---|
| `--model NAME` | **Required.** A model name from [Models and weights](#models-and-weights), such as `resnet50_v1` or `vit_b_16`. | none |
| `--norm {BN,LN,GN}` | The normalization layer that Tent adapts. | read from the model |
| `--optimizers {Adam,SGD} [...]` | Which optimizer families to try. | both |
| `--adam_lrs LR [...]` | Adam learning rates. | `1e-5 5e-5 1e-4 5e-4 1e-3` |
| `--adam_betas B [...]` | Adam beta1 (beta2 is fixed at 0.999). | `0.9` |
| `--sgd_lrs LR [...]` | SGD learning rates. | `1e-4 5e-4 1e-3 5e-3 1e-2` |
| `--sgd_momentums M [...]` | SGD momentums. | `0 0.9` |
| `--wd W` | Weight decay, for both optimizers. | `0` |
| `--seed N` | Seed of the stream order. | `0` |

Two cautions. All shards of an array, and the final ranking run, must be given the **same grid flags**: the list of trials comes from the flags, and a trial outside the list is neither run nor ranked. And `--report_only` writes the tuned file from the best trial *in the grid you pass*, so a report with a smaller grid can overwrite it with a different winner.

### Script 02b: `02b_tune_joint_lr.py`

Takes the shared flags `--duo`, `--tta`, `--tta_kwargs`, `--seeds` (default `0`), `--num_samples`, `--bn`, `--report_only`, `--shard`, `--only` (labels like `joint/L5e-05_S0.00025/seed0`, `frozen/seed0`), `--out`, `--global` and `--device`.

| Argument | Meaning | Default |
|---|---|---|
| `--mults X [...]` | The grid of each model: these multiples of its current learning rate. | `1 2 4 8 16` |
| `--lrs_large LR [...]` | Explicit learning rates of the large model (replaces `--mults` for it). | from `--mults` |
| `--lrs_small LR [...]` | The same for the small model. | from `--mults` |
| `--temps_checkpoint DIR` | Use the temperatures saved in a `JointFixedTS` checkpoint instead of fitting them on the held-out corruptions. | fit on the held-out corruptions |
| `--temps_only` | Fit and save the temperatures, then stop. Run it once before a `--shard` array. | off |

The number of array tasks is the number of grid points times the number of seeds (the two references wrap around onto the first tasks).

### Script 03: `03_tab_adaptation_modes.py`

| Argument | Meaning | Default |
|---|---|---|
| `--modes MODE [...]` | Which modes to run: `joint`, `large_anchor`, `small_anchor`, `indep_large`, `indep_small`, `indep_duo`, `frozen`. | all seven |
| `--temps_checkpoint DIR` | Use the temperatures saved in a `JointFixedTS` checkpoint (a folder with a `config.json`) instead of fitting them on the held-out corruptions. | fit on the held-out corruptions |
| `--temps_only` | Fit and save the temperatures, then stop. Run it once before a `--shard` array. | off |
| `--lr_large LR`, `--lr_small LR` | Replace the learning rate of that model in the modes that adapt on its **own** entropy (Independent large / small / both); the rest of its optimizer stays. The rates are part of every cache key, and other own-entropy rates get their own temperatures. | the indep regime's rates |
| `--joint_lr_large LR`, `--joint_lr_small LR` | The same for the modes that adapt on the **duo** entropy (Joint, Large adapt, Small adapt). With both given, the joint file is not read. | the joint regime's pair (an error while it is pending) |

With any of these four flags the output files get a `_lrL<large>_S<small>` and/or `_jlrL<large>_S<small>` tag and a time stamp, so they never replace the tuned run's files.

The number of array tasks is the number of modes times the number of seeds.

### `setup_check.py`

| Argument | Meaning | Default |
|---|---|---|
| `--full` | Count every ImageNet-C corruption's 50,000 files, instead of only checking the folder structure. | off |
| `--probe N` | Also score a pretrained ResNet-50 on N images from each dataset, a check that the labels are right. It needs the archives extracted, so run it as a job. | off |
| `--bn` | With `--probe`: run the ResNet-50 on the test batch's BatchNorm statistics instead of `eval()` mode. | off |
| `--strict` | Count a dataset path that is not set yet as a failure, not as pending. | off |
| `--skip-train-slice` | Skip building the list of clean training-slice files (it lists 1.28M files the first time). | off |
| `--global FILE` | Use another global config. | `cfgs/global.yaml` |

## Checking the code

There is no pytest. Every module has a self-test, run from the repo root:

```bash
python -m src.experiments.runner             # the engine, end to end on tiny models
python -m src.experiments.tuning             # the tuner (script 02)
python -m src.experiments.adaptation_modes   # the adaptation modes (script 03)
python -m src.utils.config                   # config loading
```

## Reproducing Paper Tables 
*Optimal Temperatures and Oracles table:*
```bash
sbatch slurm/job.sh scripts/01_tab_temperatures.py --duo vitb16_rn50v1 --mode no_adapt --batch_sizes 16 64 128 --bn
```
This runs the frozen Vit-B + ResNet50 duo on all 15 test corruptions of ImageNet-C, as well as the 4 heldout corruptions and the clean ImageNet validation set. Two baselines (Fixed TS fitted on clean and Fixed TS fitted on the 4 heldout shifts) are compared to the individual members as well as 3 temperature oracles (Corruption-wise temperatures, batch-wise temperatures, and per-sample temperatures). With the `--bn` flag, ResNet50 uses test batch statistics.

*Tent HP tuning:*
```bash
sbatch --array=0-15 slurm/job.sh scripts/02_tune_tta.py --model vit_b_16 --shard auto
sbatch --array=0-15 slurm/job.sh scripts/02_tune_tta.py --model resnet50_v1 --bn --shard auto
# once done, run these
sbatch slurm/job_cpu.sh scripts/02_tune_tta.py --model vit_b_16 --report_only
sbatch slurm/job_cpu.sh scripts/02_tune_tta.py --model resnet50_v1 --bn --report_only
```

*Adaptation modes:*
This script uses the hyperparameters found above
```bash
sbatch --array=0-20 slurm/job.sh scripts/03_tab_adaptation_modes.py --duo vitb16_rn50v1 --shard auto --bn
```