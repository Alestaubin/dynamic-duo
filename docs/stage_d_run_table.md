# Stage D: runs and GPU-hour estimate (for the author's approval)

Written for the author, 2026-10-18. Per the brief, nothing in Stage D is submitted until this table is approved: no array, no
smoke job. Everything below is an estimate from runs that already happened, with each assumption marked.

## 1. Scope, after the author's answers

Base algorithms: **Tent, ETA, EATA, SAR** (recovery off, as a duo base algorithm), plus **COCA** as a duo method with its own
run. CoTTA, RMT, ROID and DeYO are deferred. Main pair `vitb16_rn50v1` (ViT-B/16 + ResNet-50 V1), seed 0, the 4 held-out
corruptions only (selection never touches a test stream), SGD with momentum 0.9 and the brief's grids:

| Member | Grid |
|---|---|
| ViT-B/16 | 1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3 |
| ResNet-50 V1 | 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3 |

Because the members adapt independently, **run i uses the i-th value of each model's grid**: 5 runs per algorithm, not 25.
Each member's learning rate is then chosen on its own standalone mean held-out accuracy (the criterion used for Tent), with the
worst held-out corruption reported next to it, and the best large and best small member are paired offline. COCA couples the
members, so its 5 runs cannot be composed; its learning rate is chosen per run.

## 2. The run table

One run = both members over 4 streams x 50,000 images = 200,000 images, on one L40S. One array per algorithm, 5 tasks,
`scripts/cache_logits.py --streams heldout --grid --shard auto` (COCA through its own driver).

| Algorithm | Runs | Minutes per run | GPU-hours | Basis |
|---|---|---|---|---|
| Tent, grid cached | 5 | 17.1 | 1.43 | measured speed |
| ETA | 5 | 17.1 | 1.43 | same step as Tent: assumed |
| EATA (Fisher from 2000 clean source images) | 5 | 17.9 | 1.49 | 1.05 x Tent: assumed |
| SAR, recovery off | 5 | 32.7 | 2.73 | 2 x Tent (two forward and backward passes per member): the plan's figure, **not measured** |
| COCA | 5 | 18.7 | 1.56 | 1.1 x Tent: assumed, see 3 |
| **Tuning runs** | **25** | | **8.6** | |
| Clean training slice (10,000 images) for the chosen configuration of each algorithm: temperatures and source statistics | 5 | 2.3 | 0.2 | |
| Smoke jobs before the arrays (one per algorithm, 2,000 images per stream, single GPU job each): throughput, parity, resume | 5 | about 8 | 0.7 | |
| **Stage D total** | **35 jobs** | | **9.5** | |

Every task fits well inside `slurm/job.sh`'s default 3 h (the longest, SAR, is about 33 minutes); I would pass `-t 01:00:00`
so that the jobs backfill sooner. Memory: the script 03 tasks peaked at 19, 27 and 32 GB (MaxRSS) of the 32 GB they requested, but the new path holds one stream's logits at a time and the smoke job measured a **peak RSS of 4.9 GB** and 6.5 to 11.5 GB of GPU memory (section 6), so the default 32 GB is ample. Storage: 5 runs x 4
streams x 200 MB = 4 GB per algorithm, 20 GB for the four cached ones, plus 20 MB per clean slice.

Contingencies, **not** in the total:
- **Grid edge.** The finished Tent tuning chose ViT-B/16 lr 1e-4, the lowest point of its grid (the ResNet-50's 2.5e-4 is interior).
  If the same happens for ETA, EATA or SAR, one more point (5e-5) for the ViT costs one more run: +0.29 h for Tent-class
  algorithms, +0.55 h for SAR. The brief's grids are settled, so this is only done if you say so.
- **Tent as a full grid.** If you only need Tent's logits at the configuration `cfgs/tuned/` already holds (ViT 1e-4, ResNet
  2.5e-4, both momentum 0.9), one run suffices: 0.29 h instead of 1.43 h, saving 1.14 h and bringing the total to 8.3 h.
  I listed the grid so that all four algorithms are selected by the same procedure; say which you prefer (decision A).

For orientation only (it is Stage F, with its own approval): the test runs of these four algorithms on the main pair are about
3.1 h (Tent), 3.1 h (ETA), 3.3 h (EATA) and 6.2 h (SAR), i.e. **about 16 L40S-hours** for 3 seeds x 16 streams each.

## 3. Where the numbers come from

- **Speed, measured.** The script 03 array (job 6413974, L40S nodes, `vitb16_rn50v1`, 20 streams = 1,000,000 images per task): the
  six tasks in which both members adapt took 85, 79, 78, 77, 77 and 77 minutes (median 77.6). I use 78 minutes per million
  images, i.e. 15.6 minutes per 200,000, plus 1.5 minutes per run for model loading, dataset listing and writing. That
  agrees with the 202 images/s of the plan's earlier measurement.
- **ETA and EATA** change the loss, not the number of passes; EATA adds a penalty on the parameters' drift, which I took as +5%.
- **SAR** takes two forward and backward passes per member per step, so I took twice Tent's time, as the plan's Compute table
  does. It is the one multiplier that matters (it is a third of the total) and the first thing the smoke job measures.
- **COCA** (read in `third_party/COCA/COCA-maincode/COCAV4.py`): one forward pass per model, then 10 steps of AdamW on a scalar
  temperature (64 x 1000 logits, negligible), then one joint backward through both models: about one Tent step. The brief says
  3 to 5 steps to fit the temperature; the official code uses **10** (decision C). COCA feeds the ViT its own transform,
  separately from the ResNet's.
- **Not in the estimate:** queue time (the arrays of 2026-10-18 waited 35 to 45 minutes before starting), reruns after failures, ImageNet-C-bar
  and LAION-C, CoTTA/RMT/ROID/DeYO, the Stage E live Joint DD runs.

## 4. Work before any GPU job (no GPU needed)

1. **ETA/EATA.** Diff the repo's `eata` method against `third_party/EATA/eata.py` and the benchmark's `methods/eata.py`. (Done:
   the repo's `eata` differs from upstream ETA, see section 6, so ETA and EATA both run on the port, with the Fisher term for EATA:
   weight 2000, 2000 clean images from the training slice, margins 0.4 ln K and 0.05.)
2. **SAR.** Port with SAM (rho 0.05), margin 0.4 ln K, a recovery switch (0.2 on a decay-0.9 average; off here, on for the
   standalone row), scoring the first pass.
3. **The member adapter** with [0, 1] inputs and a `Normalize` wrapper per member (decision 10), and the parity check: a ported
   Tent must agree with `TentMethod` on a smoke stream to fp tolerance.
4. **An offline selection script**: per member, macro held-out accuracy and the worst held-out corruption from the stored
   logits, writing the chosen learning rates.
5. **A COCA driver** (see decision B). It is the only item that depends on an answer from you.

## 5. Decisions I need

| | Decision | My proposal |
|---|---|---|
| A | Tent: cache the grid (1.43 h) or only the configuration `cfgs/tuned/` holds (0.29 h)? | The grid, for one procedure across algorithms. Your call: the single run saves 1.14 h. |
| B | COCA's repository has **no LICENSE file**, and the repo's rule is to port with the upstream license header. Copy code into `src/`, or run the checkout in place without copying? | Run it in place from `third_party/COCA` (a thin driver, nothing copied), and ask the authors if anything is to be redistributed. |
| C | COCA fits the temperature with **10** steps per batch in the official code, the brief says 3 to 5. | 10, the official code, since the brief says to start from the published settings. |
| D | ViT grid edge contingency (5e-5). | Only if a winner lands on 1e-4, and then with your OK. |
| E | Logs: `slurm/job.sh` writes `logs/%j_%x`, which gives one file per array, so the tasks of an array share a log (the four failed tasks of 6413974 left no separable error). `logs/%A_%a_%x` gives one file per task. | Change the two `#SBATCH --output/--error` lines. It is the one existing file I would touch, so only with your OK. |

When you approve, the order is: items 1 to 5 above, then the five smoke jobs (the SAR measurement may move the total), then the five arrays, one algorithm at a time so that a problem shows up on the first.

## 6. Update after the answers and the smoke job (2026-10-19)

**Decisions applied.** A: Tent is one run at the settings `cfgs/tuned/` holds (ViT 1e-4, ResNet 2.5e-4, momentum 0.9). B: COCA runs in
place from the checkout, nothing copied. C: COCA's 10 temperature steps. D: a 5e-5 point for the ViT if a winner lands on 1e-4.
E: `slurm/job.sh` gives every array task its own log files.

**What the work found** (details in `plan.md`, Stage D as built): the repo's older `eata` is not upstream ETA (it steps the optimizer on a
fully filtered batch, so momentum keeps moving the weights), so **ETA runs on the port**; SAR's skip-the-top-layers rule had to be mapped
onto torchvision's ViT names; COCA's released code never assigns `tryT` and `halfcombine`.

**Measured in one GPU job (6424163, L40S, 9 min 49 s),** ViT-B/16 lr 5e-4 + ResNet-50 V1 lr 2.5e-4, steady state on the second of two
5000-image streams (so each rate includes a per-stream file listing: conservative):

| | Setup | Images/s | Peak RSS | Peak GPU | Per run of 200,000 images (with the 1.5 min overhead) | The table assumed |
|---|---|---|---|---|---|---|
| Tent | 2.8 s | 171 | 3.8 GB | 11.5 GB | 21.0 min | 17.1 |
| ETA | 1.7 s | 187 | 4.6 GB | 6.5 GB | 19.4 min | 17.1 |
| EATA | 14.2 s (Fisher) | 169 | 4.7 GB | 6.5 GB | 21.4 min | 17.9 |
| SAR (recovery off) | 1.7 s | 97.5 | 4.7 GB | 6.5 GB | 35.7 min | 32.7 |
| COCA | 1.8 s | 190 | 4.7 GB | 11.5 GB | 19.1 min | 18.7 |

SAR is 1.75 x Tent, not the 2 x assumed. The rates here are 12 to 21% under the 216 images/s of the script 03 array, because they include
the per-stream listing of 50,000 file names, which is amortized over a 50,000-image stream; I keep these as the conservative figures.

**Revised budget** (conservative), with Tent as one run: ETA 1.61 h + EATA 1.79 h + SAR 2.98 h + COCA 1.59 h + Tent 0.35 h = **8.3 L40S-hours
for the tuning runs**, plus 0.2 h for the five clean-slice runs and the smoke job already spent (0.16 h): **8.7 h in all**, against 8.3 h
approved. Each task takes 19 to 36 minutes. Contingency (D): +0.3 h per Tent-class algorithm, +0.55 h for SAR, only where a ViT winner
lands on 1e-4.

**Parity and the CLIs on real models.** Tent through the new [0, 1] path equals the existing path (largest logit difference 1.5e-5,
identical accuracy on 2000 images). `cache_logits.py --tta sar --grid`, `select_member_lr.py` and `--report_only` ran end to end.
COCA's temperature rose from 1.1 to 2.1 over a stream, and on the first 5000 speckle-noise images COCA scored 57.1% against 55.1% for the
large member alone (Tent-like settings, one learning-rate pair: a functional check, not a result).

**Launched.** Only the ETA array (job 6424535, tasks 0 to 4 = grid points 0 to 4), as the first of the five so that a problem shows
early. Waiting for your explicit go: the Tent single run, EATA, SAR and COCA (about 6.7 of the 8.3 hours).
