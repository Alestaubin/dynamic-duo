# Tuning the reset rule `own_fast_vs_slow`

The rule resets a member to its source weights when its own proxy (the normalized nuclear norm of its predictions, one value per batch of 256
images) has dropped: a **fast** moving average of the proxy's logit falls more than `delta` noise units below a **slow** one. This page is the
protocol that chooses each member's learning rate and detector setting. The selection rules are code, in `src/reset/tuning.py`; this page says
what they do and why.

## Fixed, and tuned

| fixed | value | note |
|---|---|---|
| adaptation batch size | 64 | `cfgs/global.yaml` |
| proxy, proxy batch | normalized nuclear norm, 256 images | `nnn_bt256` in every run name |
| rule | `own_fast_vs_slow`: one detector per member, on its own proxy | the other rules were screened and dropped |
| gate | beta 0.5, fixed temperatures T_L 0.651 / T_S 1.0097 (frozen-model fit) | evaluation only: the members and their proxy values do not depend on beta |

| tuned, **per member** | grid |
|---|---|
| learning rate | sqrt-2 rungs, ViT anchor 7e-5, ResNet anchor 5e-4 |
| detector: averaging speeds (alpha_fast, alpha_slow) and warm-up W0 (cooldown K = W0) | a few candidate settings from the offline screen (below) |
| detector: threshold delta | sqrt-2 rungs around the candidate's value |

The detector's time constants are in **proxy batches** (256 images), so they are tied to the proxy batch size above.

## Data: the held-out stream

ImageNet-C, severity 5, the four shifts that are in no test stream: speckle noise, Gaussian blur, spatter, saturate (50,000 images each).
The stream is these four **4 times**, 16 visits and 800,000 images. Round `r` opens on shift `(k + r) mod 4`, and rotation `k` is the stream
that opens with shift `k` (Sp = speckle, Bl = blur, Spa = spatter, Sat = saturate):

| rotation | round 0 | round 1 | round 2 | round 3 | used for |
|---|---|---|---|---|---|
| 0 | Sp Bl Spa Sat | Bl Spa Sat Sp | Spa Sat Sp Bl | Sat Sp Bl Spa | tuning |
| 1 | Bl Spa Sat Sp | Spa Sat Sp Bl | Sat Sp Bl Spa | Sp Bl Spa Sat | validation |
| 2 | Spa Sat Sp Bl | Sat Sp Bl Spa | Sp Bl Spa Sat | Bl Spa Sat Sp | tuning |
| 3 | Sat Sp Bl Spa | Sp Bl Spa Sat | Bl Spa Sat Sp | Spa Sat Sp Bl | validation |

Nothing is ever chosen on the 15 test shifts. One stream-order seed (0) is used: the four rotations hold the same images in another order, so
they are not independent seeds, and the uncertainty used below (a paired standard error over visits) does not include seed-to-seed variation.

## What is scored

The member's **own** whole-stream accuracy, averaged over the tuning rotations 0 and 2 (32 visits pooled). Under `own_fast_vs_slow`, the
no-reset baseline and the fixed schedules, a member's trajectory depends only on its own rate and its own reset times, so **one run measures both
members**, whatever the partner's rate or setting is. Each member is therefore tuned alone, and a run pairs one request of each member.

## The search, per member

1. **Candidates.** The shared setting (alpha 0.05 / 0.005, W0 20, delta 0.7) and up to two settings with other speeds, from the offline screen
   (`scripts/44_screen_member_detector.py`: each setting is replayed over stored proxy sequences of collapsing and healthy streams; the candidates
   are the two settings with the shortest detection delay at one false alarm per 1,000 batches whose speed pairs differ from each other and from the
   shared one).
2. **A learning-rate ladder for each candidate** (rules below).
3. **The best candidate** by pooled accuracy at its chosen rate; ties go to the fewest resets, then to the shared setting.
4. **Its threshold delta**, on the sqrt-2 grid at that rate (the same rules); then, if delta moved, the rate one rung either side at the new delta;
   repeat until neither moves (at most 3 rounds).

The rules of one ladder (`plan_1d`):

1. Start with every second rung (ViT 3.5e-5 .. 2.8e-4, ResNet 1.25e-4 .. 1e-3) on rotation 0.
2. Run the argmax's two sqrt-2 neighbours, recompute the argmax, until it has both (a local maximum: "interior").
3. Run the argmax and its neighbours on rotation 2.
4. Choose by the mean of rotations 0 and 2: the best, then everything within **one paired standard error** of it (per-visit differences), then the
   fewest resets per 1,000 batches, then the lowest rate.
5. The choice must not be at an end of what was tried: if it is, run the rung beyond (at most two x2 extensions per side; a choice that stays
   at the capped end is reported as a plateau, not chased).
6. A configuration in which the member is dead (four windows of 5,120 images below 5%) is excluded.

## Report only

The chosen configuration is run on all four rotations; rotations 1 and 3 are reported and never used for a choice. Then the random-matched
control (each member reset as often as the rule did, at random steps; `random~<cond>~d<draw>`), and the final test, once, on the 15 test shifts in
five orderings, compared with the no-reset baseline and the schedules by paired differences over the shifts.

## Commands

```bash
python scripts/40_tune_reset.py member                  # status of the per-member search and the next wave of runs (add --submit to launch it)
python scripts/40_tune_reset.py confirm --cond <cond> --rates LR_L LR_S    # rotations 0-3 for the chosen configuration
python scripts/40_tune_reset.py rates --cond fixed1000  # a comparison baseline: its two rate ladders (also fixed500, own_fast_vs_slow@0.7)
python scripts/38_run_reset_live.py ...                 # the runs themselves (one condition, one rotation, one pair of rates)
```

A wave takes about 70 minutes; repeat `member` after it finishes. State is written to `out/results/reset_tune/`, runs to
`out/results/reset_live/<duo>__tent_bn/` (file names carry condition, stream, rotation, rates, gate and seed).

## Dropped

`own_cusum`, `gap_fast_vs_slow`, `own_level`, `both_fast_vs_slow`, the staggered schedule and the oracle trigger were screened or run against this rule
and dropped; their code is in git at the tag `pre-reset-cleanup`, and their runs and tuning states are in `out/_archive/` (a manifest lists each
item and where to move it back). The report `docs/reset_screening_report.tex` describes them.
