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
`scripts/04_cache_logits.py --streams heldout --grid --shard auto` (COCA through its own driver).

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
identical accuracy on 2000 images). `04_cache_logits.py --tta sar --grid`, `05_select_member_lr.py` and `--report_only` ran end to end.
COCA's temperature rose from 1.1 to 2.1 over a stream, and on the first 5000 speckle-noise images COCA scored 57.1% against 55.1% for the
large member alone (Tent-like settings, one learning-rate pair: a functional check, not a result).

**Launched.** The ETA array (job 6424535, tasks 0 to 4 = grid points 0 to 4) first, so that a problem would show early; it finished cleanly
(17.6 to 18.0 minutes per task, every checksum verified). On your instruction to launch the rest if all was good, 2026-10-19: job 6425539
(Tent, one run), 6425540 (EATA), 6425541 (SAR), 6425542 (COCA), 16 tasks, about 6.7 of the 8.3 hours.

## 7. Results of the tuning runs (2026-10-19)

All 16 tasks finished with exit 0; every stored file's sha256 verifies. Standalone accuracy of each member on the four held-out streams,
macro mean (worst stream), %, seed 0, SGD momentum 0.9, run i = i-th value of each model's grid:

| ViT-B/16 lr | 1e-4 | 2.5e-4 | 5e-4 | 1e-3 | 2.5e-3 |
|---|---|---|---|---|---|
| ETA | 57.5 (42.8) | 59.6 (45.5) | **60.3 (45.9)** | 59.5 (43.3) | 54.5 (30.4) |
| EATA | 57.3 (42.7) | 59.5 (45.5) | **60.3 (46.0)** | 60.0 (44.8) | 57.1 (38.3) |
| SAR | 52.1 (34.8) | 54.3 (39.0) | 56.1 (41.8) | 57.9 (44.2) | **59.4 (45.8)** (top of the grid) |
| Tent (the `02` runs) | **50.9 (35.0)** (bottom of the grid) | 46.2 | 45.1 | 44.5 | 43.9 |
| COCA, its ViT member | 56.5 (40.7) | 58.8 (43.8) | 60.0 (45.3) | **60.5 (45.6)** | 56.0 (28.4) |

| ResNet-50 V1 lr | 5e-5 | 1e-4 | 2.5e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|
| ETA | 45.2 (23.7) | 47.4 (26.6) | **48.5 (26.4)** | 48.3 (24.5) | 41.8 (2.3) |
| EATA | 45.0 (23.3) | 47.1 (26.5) | 48.8 (28.2) | **48.9 (26.4)** | 47.7 (22.4) |
| SAR | 39.7 (17.1) | 42.0 (19.6) | 44.5 (21.1) | 46.5 (24.1) | **46.5 (20.6)** (top of the grid) |
| Tent (the `02` runs) | 39.6 | 41.8 | 44.5 (21.8) | **45.6** | 44.7 |
| COCA, its ResNet member | 43.8 (22.3) | 46.5 (26.6) | 49.6 (31.0) | **51.3 (33.5)** | 50.4 (27.9) |

(Tent's ViT and ResNet grids in the `02` ranking are the momentum-0.9 rows. The frozen models score 48.8 (ViT) and 34.8 (ResNet).)

COCA's prediction is the combination of its two members, so its grid is read by pair (run i = ViT grid i with ResNet grid i), macro mean
(worst stream) of the combined prediction: 58.7 (41.8), 60.5 (44.7), 61.6 (46.1), **62.1 (46.5)**, 57.5 (29.3), with NLL 1.91, 1.81, 1.77, 1.79,
2.57. The best is run 3 (ViT 1e-3, ResNet 5e-4), interior, and there the combination beats COCA's own ViT member by 1.6 points and its ResNet
by 10.8. COCA's temperature ends a stream at about 2.1 to 2.6, and near 6.5 to 6.9 on Gaussian blur.

**Cross-checks.** The stored Tent run (ViT 1e-4, ResNet 2.5e-4) reproduces the `02` tuner's per-stream accuracies exactly. ETA and EATA are
within 0.1 point of each other on the ViT, and EATA's ResNet gains 0.3 over ETA (its 2.5e-4 and 5e-4 are a tie, 48.82 against 48.85).

**What needs your decision.**
1. **SAR's winners are at the top of the grid** for both members, and the ViT is still rising by about 1.5 points per step. The brief fixes the
   grids, and decision D covered only the lowest edge. One more run, ViT 5e-3 with ResNet 2e-3 (both next points at the grid's own spacing),
   costs about 0.6 L40S-hours and would show whether SAR peaks beyond. The ResNet is nearly flat from 5e-4 to 1e-3 (worst stream falls), so its
   grid is probably adequate. Without it, SAR is reported at ViT 2.5e-3 and ResNet 1e-3 with the edge noted.
2. **Tent's ResNet learning rate.** The `02` ranking's winner is 5e-4 (45.58), but `cfgs/tuned/tent__resnet50_v1.yaml` says 2.5e-4 (44.52 on
   these runs), with the 5e-4 trial's score still in it, and its header says not to edit it by hand, so it looks like a deliberate edit to
   Tent's published setting. The single Tent run I launched followed the file, as instructed. If `02`'s winner is meant, the ResNet member at
   5e-4 is already being produced by the extra run below, so nothing more is needed but your choice.
3. **COCA's published pair** (ViT 1e-3, ResNet 2.5e-4) is not a grid point, because the runs pair index i of both grids. One extra run (0.3 h) puts
   the paper's own setting next to the best grid point (62.1).

**Launched under approved decisions.** Clean-slice runs for ETA (6426793) and EATA (6426794) at their chosen pairs, and for Tent the extra ViT
point of decision D (lr 5e-5, with the ResNet at 5e-4: job 6426795), after which the Tent ViT lr is chosen between 5e-5 and 1e-4.
Tuned files written for ETA and EATA (interior winners, no ambiguity). Not yet written: SAR (decision 1), Tent (decision 2 and the 5e-5 point).

## 8. Answers to section 7 and the follow-up runs (2026-10-19)

1 and 2: yes, both extra runs were added: SAR at ViT 5e-3 with ResNet 2e-3 (job 6440009) and COCA at its published pair, ViT 1e-3 with
ResNet 2.5e-4 (job 6440010). 3: Tent's ResNet lr 2.5e-4 is deliberate (the best lr reported in the Tent paper), so `cfgs/tuned/tent__resnet50_v1.yaml`
stays as it is; its `macro_accuracy` (0.45581) and `frozen_macro_accuracy` fields describe the 5e-4 trial of the `02` ranking, not the 2.5e-4 setting
(44.52 on these streams).

**Tent's extra ViT point (decision D), job 6426795:** ViT lr 5e-5 scores 51.18 macro with 32.2 on its worst stream (Gaussian blur), against 50.94 and
35.0 at the `02` choice 1e-4. By the stated criterion (best macro) 5e-5 wins, by 0.24 points that one seed cannot separate from noise, at the price of
the worst stream losing 2.8 points, and the curve is flat across the two (the best is at the lowest point tried again). The pair for the Tent
runs of Stages E and F is therefore either (ViT 1e-4, ResNet 2.5e-4), the file as it stands, or (ViT 5e-5, ResNet 2.5e-4), and both are
already stored member by member, so either needs only a clean-slice run for its ViT member.

## 9. The SAR extension and COCA's published pair (2026-10-19)

Both jobs finished with exit 0 and verify (SAR extension 34 min, COCA 17 min).

**SAR, the two points beyond the grid (run: ViT 5e-3 with ResNet 2e-3), macro mean (worst stream), %:**

| ViT-B/16 lr | 1e-4 | 2.5e-4 | 5e-4 | 1e-3 | 2.5e-3 | **5e-3** |
|---|---|---|---|---|---|---|
| SAR | 52.1 (34.8) | 54.3 (39.0) | 56.1 (41.8) | 57.9 (44.2) | 59.4 (45.8) | **59.9 (45.9)** |

| ResNet-50 V1 lr | 5e-5 | 1e-4 | 2.5e-4 | 5e-4 | 1e-3 | 2e-3 |
|---|---|---|---|---|---|---|
| SAR | 39.7 (17.1) | 42.0 (19.6) | 44.5 (21.1) | 46.5 (24.1) | **46.5 (20.6)** | 42.7 (3.0) |

The ResNet now peaks inside the grid: 2e-3 collapses (a worst stream of 3%), 1e-3 is its best, and 5e-4 is a tie (46.49 against 46.53). The ViT is
still best at the highest point tried, but the gain from 2.5e-3 to 5e-3 is 0.5 point against 1.5 to 2 for each earlier step, so the curve is
flattening: another point would add little. I stopped there, as the author approved one extra point. SAR is chosen at ViT 5e-3 and ResNet
1e-3 (tuned files written), and the report should say that the ViT's optimum may lie slightly beyond 5e-3. Its clean-slice run is job 6444625.

**COCA, the published pair (ViT 1e-3, ResNet 2.5e-4):** combined 61.88 (worst stream 46.35), NLL 1.815, ECE 0.118, with its ViT member at 60.41
and its ResNet member at 49.51; against the best grid pair (ViT 1e-3, ResNet 5e-4) at 62.11 (46.53), NLL 1.794, ECE 0.116, members 60.54 and
51.29. The two are within 0.23 point on the combination, so the published setting is essentially as good as the best grid pair, which the
protocol (best held-out macro) keeps.

**Learning rates chosen for the main pair (ViT-B/16, ResNet-50 V1), SGD momentum 0.9:** Tent 1e-4 (or 5e-5, see section 8) and 2.5e-4 (the paper's
value, by the author); ETA 5e-4 and 2.5e-4; EATA 5e-4 and 5e-4; SAR 5e-3 and 1e-3; COCA 1e-3 and 5e-4 as a pair.
