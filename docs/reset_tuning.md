# Tuning the reset rule `own_fast_vs_slow`

The rule resets a member to its source weights when its own proxy (the normalized nuclear norm of its predictions, one value per batch of 256
images) has dropped: a **fast** moving average of the proxy's logit falls more than `delta` noise units below a **slow** one. This page is the
protocol that chooses learning rates and detector settings, for any duo and for either **signal**, with one script:
`python scripts/tune_reset.py --duo <duo> [--signal independent|joint]`. The selection rules are code, in `src/reset/tuning.py`; this page says what
they do and why.

**The signal** is the loss the members adapt on, and it changes the best learning rates, so it is tuned separately (own folders, rates and files):

| signal | the members adapt on | the gate | the members |
|---|---|---|---|
| `independent` (Vanilla DD, the default) | each its own entropy | outside the loop, scores the duo | decoupled: each is tuned alone |
| `joint` (Joint DD) | the entropy of the gated duo prediction | **inside the loss**, its beta changes what the members learn | coupled: tuned together, on the duo's accuracy |

## Fixed, and tuned

| fixed | value | note |
|---|---|---|
| adaptation batch size | 64 | `cfgs/global.yaml` |
| proxy, proxy batch | normalized nuclear norm, 256 images | `nnn_bt256` in every run name |
| rule | `own_fast_vs_slow`: one detector per member, on its own proxy | the other rules were screened and dropped |
| temperatures | T_L, T_S of the frozen-model fit on the held-out shifts | main duo: 0.651 / 1.0097 |
| gate (independent) | beta 0.5 | evaluation only: the members and their proxy values do not depend on beta |

| tuned | independent signal | joint signal |
|---|---|---|
| learning rates | each member's own, on sqrt-2 rungs anchored at its episodic rate (main duo: ViT 7e-5, ResNet 5e-4) | the PAIR, on the same rungs anchored at the joint pair (cfgs/duo_lrs/&lt;duo&gt;__joint.yaml; main duo 3.2e-3 / 4e-3) |
| detector: speeds (alpha_fast, alpha_slow), warm-up W0 (cooldown = W0) | per member, from a few candidates of the offline screen | per member, the same |
| detector: threshold delta | per member, sqrt-2 rungs around the candidate's value | per member, the same |
| gate beta | -- | 0, 0.25, 0.5, 0.75, 1 (extended upward if the best is 1) |

The detector's time constants are in **proxy batches** (256 images), so they are tied to the proxy batch size above.

## Data: the held-out stream, and ONE rotation of it

ImageNet-C, severity 5, the four shifts that are in no test stream: speckle noise, Gaussian blur, spatter, saturate (50,000 images each).
The stream is these four **4 times**, 16 visits and 800,000 images. Round `r` opens on shift `(k + r) mod 4`, and rotation `k` is the stream
that opens with shift `k` (Sp = speckle, Bl = blur, Spa = spatter, Sat = saturate):

| rotation | round 0 | round 1 | round 2 | round 3 |
|---|---|---|---|---|
| **0 (the search runs here)** | Sp Bl Spa Sat | Bl Spa Sat Sp | Spa Sat Sp Bl | Sat Sp Bl Spa |
| 1 | Bl Spa Sat Sp | Spa Sat Sp Bl | Sat Sp Bl Spa | Sp Bl Spa Sat |
| 2 | Spa Sat Sp Bl | Sat Sp Bl Spa | Sp Bl Spa Sat | Bl Spa Sat Sp |
| 3 | Sat Sp Bl Spa | Sp Bl Spa Sat | Bl Spa Sat Sp | Spa Sat Sp Bl |

Nothing is ever chosen on the 15 test shifts. The search uses rotation 0 only (`config.json`: `tuning_rotations`, or `--rotations 0 2` to pool two). One stream-order
seed (0) is used; the rotations hold the same images in another order, so they are not independent seeds, and the uncertainty used below (a paired standard
error over visits) does not include seed-to-seed variation.

**Why one rotation.** On the main duo's stored rate ladders, rotation 0 alone chose the same rung as the pooled rotations 0 and 2 in 5 of 6 cases (the miss
cost 0.22 points on the other rotation), and the four rotations of a chosen configuration differ by under 0.1 points on the duo. **Its limit:** with 16 visits
instead of 32 the "within one paired standard error" tie is wider, so a close call between detector candidates can go the other way. On the main duo's ViT the
shared setting at 5.6e-4 (54.81) and the candidate (0.1, 0.01, 50, 1.4) at 2.8e-4 (54.42) are tied on rotation 0 alone (gap 0.39, one SE 0.39) and the candidate
wins on fewer resets, while pooled over rotations 0 and 2 the gap (0.33) exceeds the SE (0.24) and the shared setting wins. The main duo's study began under the
pooled protocol, so its `config.json` pins `tuning_rotations: [0, 2]`.

## What is scored

**Independent:** the member's own whole-stream accuracy on the tuning rotation. A member's trajectory depends only on its own rate and its own reset times,
so **one run measures both members**, whatever the partner's rate or setting is; each member is tuned alone and a run pairs one request of each member.
**Joint:** the duo's whole-stream accuracy (the in-loop gate's prediction). Both members train on it, so a member's trajectory depends on its partner's rate and
resets: nothing is measured per member, and a cell is a whole configuration.

## The search, independent signal (per member)

1. **Candidates.** The shared setting (the offline screen's choice for both members; the main duo: alpha 0.05 / 0.005, W0 20, delta 0.7) and up to
   two settings with other speeds, from the per-member offline screen (`src/reset/screens.py`: each setting is replayed over stored proxy sequences of
   collapsing and healthy streams; the candidates are the two settings with the shortest detection delay at one false alarm per 1,000 batches whose
   speed pairs differ from each other and from the shared one).
2. **A learning-rate ladder for each candidate** (rules below). The shared setting's ladder comes first (start: 0.5x to 4x the anchor for the large
   member, 0.25x to 2x for the small one); the candidates' ladders start at every second rung from 4 below to 2 above its chosen rung.
3. **The best candidate** at its chosen rate; ties go to the fewest resets, then to the shared setting.
4. **Its threshold delta**, on the sqrt-2 grid at that rate (the same rules); then, if delta moved, the rate one rung either side at the new delta;
   repeat until neither moves (at most 3 rounds).

The rules of one ladder (`plan_1d`):

1. Start with every second rung (ViT 3.5e-5 .. 2.8e-4, ResNet 1.25e-4 .. 1e-3) on the tuning rotation.
2. Run the argmax's two sqrt-2 neighbours, recompute the argmax, until it has both (a local maximum: "interior").
3. (Only with two tuning rotations: run the argmax and its neighbours on the second.)
4. Choose: the best, then everything within **one paired standard error** of it (per-visit differences), then the fewest resets per 1,000 batches,
   then the lowest rate.
5. The choice must not be at an end of what was tried: if it is, run the rung beyond (at most two x2 extensions per side; a choice that stays
   at the capped end is reported as a plateau, not chased).
6. A configuration in which the member is dead (four windows of 5,120 images below 5%) is excluded.

## The search, joint signal (`plan_joint`)

A cell is (rate rungs of the large and small member, gate beta, the large member's detector setting, the small member's); every cell is scored by the duo's
accuracy, and a cell with a dead member is excluded. **One detector per member** (different architectures collapse differently), so the search is in blocks, each
by the rules above with the others held fixed, repeated as a pass until nothing moves (at most 2 passes, `max_passes`):

1. **Rates**: the pair of rungs (`plan_pair`). First pass: the diagonal of the two X2 ladders at gate beta 0.5 and the shared setting for both members;
   then the argmax needs its whole neighbourhood tried (one member's rung up or down, and both together); the best three go on; the chosen cell needs its own
   neighbourhood. Later passes start from the previous pair.
2. **Gate beta**: every value of the grid at the chosen rates; if the best is the sharpest tried (1), try 2, then 4. Ties go to the **lowest beta**: beta 0 is the
   fixed-TS duo (no gate), so a choice of 0 means the gate does not help the joint loss at these rates, which is reported, not extended. The gate is part of the loss, so every beta
   is its own run (it cannot be replayed offline).
3. **Large member's detector**, then **the small member's**: its candidate settings (the shared one and the screen's) at the chosen rates and gate, the best, then
   its threshold on the sqrt-2 grid (ties: fewest resets, then the shared setting).

On a synthetic surface with a known optimum the search finds it in about 14 sequential waves and 40 runs (each wave about 70 minutes: roughly 16 hours of
wall-clock, about 45 GPU-hours). The joint signal has its own continual-rate ladder, calibration streams and screens (stages 2 to 4 below, on joint runs, the gate
inside the loss at beta 0.5), because the joint loss collapses differently.

## Report only

The chosen configuration is run once on the tuning rotation (the independent signal's search never ran that exact pair; the joint search did), with the
random-matched control (each member reset as often as the rule did, at random steps) is a baseline, not a tuning step: the final test runs it, on the test stream. Other rotations are not
run unless asked (`confirm_rotations`). The final test, once, on the 15 test shifts in five orderings, compares with the no-reset baseline and the schedules by
paired differences over the shifts.

## The command, for any duo and signal

```bash
python scripts/tune_reset.py --duo convnextb_deits                    # what is done and the next step; nothing is launched
python scripts/tune_reset.py --duo convnextb_deits --submit           # do it: submit the next wave of jobs, or run the next CPU step
python scripts/tune_reset.py --duo convnextb_deits --signal joint     # the same for the joint signal
python scripts/tune_reset.py --duo convnextb_deits --submit --chain   # ... and run itself: see below
```

Call it again after each wave finishes (about 70 minutes), or add `--chain` (below). It walks the stages in order and stops at the first that is not done
(`src/reset/pipeline.py`):

1. **prerequisites**: the duo's frozen-model temperatures (one job, once per duo).
2. **continual**: the continual-rate ladder (no reset; both members on the held-out stream x 4 rounds, stopped once both are dead): each member's *conservative*
   rate (stable) and *stress* rate (a complete collapse). Independent: each member alone; joint: the two ladders run together (the i-th rungs as a pair).
3. **calibration**: the calibration streams, stress S (both members collapse) and healthy H, 10 runs.
4. **screens**: the offline replay of the detector (CPU, inline): the shared setting and each member's two candidates.
5. **tune**: the search above, in waves of live runs (`member__rates.json` / `joint__choice.json`).
6. **confirm**: the chosen configuration run once on the tuning rotation (`member__confirm.json` / `joint__confirm.json`).

**`--chain` runs the tuning by itself.** With `--submit --chain`, a wave is followed by one small CPU job (`rt_next_<duo>[__joint]`, 2 h limit, QOS high) that waits
for the wave (`--dependency=afterok`, all its jobs) and calls this script again with the same arguments: it submits the next wave, or runs a CPU stage inline, and
chains again, until every stage is done. It stops, with nothing left in the queue, when (a) a stage is blocked and needs a decision (the log of the last chain job,
`logs/<id>_rt_next_<duo>.out`, says which: a ladder that does not bracket, a member that never collapses, no final choice), (b) a job of the wave failed or ran out
of time (`afterok` is not met and the chain job is cancelled, never retried), or (c) a chain call would submit exactly the wave of the last one (the jobs ran and changed
nothing). Then fix the cause and call `--submit --chain` again. A chain job already queued is not duplicated; `scancel` it to stop the loop. The 12 jobs of a wave
run in parallel, so a wave costs about its slowest job (continual 1:35 h, calibration 2:15 h, live 1:50 h) plus the queue wait.

A job is submitted once: one already queued under its name is skipped, so calling twice is safe. Everything duo-specific is derived (the rate grid from
the duo's episodic rates for the signal, the stress and conservative rates from stage 2, the shared setting from stage 4). To override, pass a flag or edit
`config.json` in the duo's tune folder:

```bash
python scripts/tune_reset.py --duo X --stress LR_L LR_S        # stage 3's stress pair, if stage 2's rates do not collapse both members
python scripts/tune_reset.py --duo X --stress_up 1             # ... or one rung higher up the ladder (stage 4 says when a member never collapses)
python scripts/tune_reset.py --duo X --mults 0.06 0.09 0.125 ...   # stage 2's ladder, when its report says it does not bracket the frontier
python scripts/tune_reset.py --duo X --signal joint --betas 0 0.5 1    # the joint signal's gate grid
python scripts/tune_reset.py --duo X --rotations 0 2           # pool two held-out rotations in the search
python scripts/tune_reset.py --duo X --baseline fixed1000      # a comparison method that has only a rate to tune (independent only): none, fixed500, fixed1000, own_fast_vs_slow@<delta>
```

What the script does not do: the duo's yaml (`cfgs/duos/<duo>.yaml`) and its episodic rates for the signal (`cfgs/duo_lrs/<duo>__indep.yaml`, `scripts/02c`;
`<duo>__joint.yaml`, `scripts/02b`) must exist. Slurm runs the same script for one unit of work (`--job <spec>`, the specs are in
`<out>/results/reset_tune/[<duo>/]jobs/`; `src/reset/workers.py`). State and results are under `out/results/`: `reset_tune/<duo>[__joint]/` (the main duo's independent
files predate the folders and sit in `reset_tune/`), `reset_live/<duo>__tent_bn[_joint]/` (the live runs), `tune_continual_lr/`, `gate_stress/`, `screen_resets_stress/`.

## Dropped

`own_cusum`, `gap_fast_vs_slow`, `own_level`, `both_fast_vs_slow`, the staggered schedule and the oracle trigger were screened or run against this rule
and dropped; their code is in git at the tag `pre-reset-cleanup`, and their runs and tuning states are in `out/_archive/` (a manifest lists each
item and where to move it back). The report `docs/reset_screening_report.tex` describes them.

## Limits

The stress rates that stage 2 names are the ones that collapsed a member on the continual stream; if they do not collapse it on the calibration streams
the screen has no delay to measure and says so (stage 4 stops with the member's name): raise them with `--stress` or `--stress_up`. The main duo's own stress
pair was set by trial above stage 2's rates, and its `config.json` records that, with its explicit continual ladder and the start rungs its earlier
candidates ran at.

The comparison methods (no reset, fixed schedules) are tuned for the independent signal only; the joint signal's would need a pair search each. The calibration
streams (10 runs: stress on four rotations and two seeds, healthy on two rotations) are not reduced by the one-rotation search: the screen needs the collapse events.

What was checked. Independent signal, main duo, against the old scripts: the same requested runs, the same ladder tables, the same stored screens and continual
report, and run files with identical arrays, gate records and metadata for a fixed schedule and the screened rule. Another duo (convnextb_deits): every stage on
synthetic data (`python -m src.reset.pipeline`: derived rate grid, calibration rates and jobs, the screens, the first search wave) and real-node smoke runs, on
256-image streams, of the continual, calibration and live job kinds. Joint signal: the live loop with the gate inside the loss against `run_stream`'s joint path
(same predictions, proxy batches and reset steps), the search to a known optimum on a synthetic surface, through the pipeline, with its controls, and a real-node
smoke run (see the plan.md entry). Not exercised: the temperatures job, and any full-length wave of a new duo or of the joint signal.
