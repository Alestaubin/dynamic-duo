# Learning rates: one sheet

**This file is generated** (`python -m src.utils.rates_sheet`) from the files that hold the numbers; do not edit it by hand, and rerun it after any tuning, `--record` / `--import_*`, or experiment with explicit rates. The numbers live in `cfgs/duo_lrs/*.yaml` (episodic), `out/results/tune_continual_lr/*/results.json` (continual) and the gate-stress runs; scripts read them from there only.

All adapting members use Tent unless the TTA column says otherwise, SGD with the momentum shown, batch size 64. A *setting* is the stream a rate was tuned on (episodic: each shift from source weights; continual: one long stream, no reset) and the *regime* says which entropy drives the members.

## 1. Reading the columns

**Regimes**

| regime | what adapts | on which entropy |
|---|---|---|
| single | each model alone | its own entropy (the single-model Tent baseline) |
| indep | both members, never coupled | each on its OWN entropy (Independent / Vanilla DD; engine mode both_indep) |
| joint | both members | the entropy of the duo's COMBINED prediction (Joint DD; both_duo) |
| large_duo | the large member only | the duo's entropy, the small member frozen (Large adapt; large_duo) |
| small_duo | the small member only | the duo's entropy, the large member frozen (Small adapt; small_duo) |

**Tuning methods**

| code | script | what it did |
|---|---|---|
| 02c | `scripts/02c_tune_indep_pair.py` | episodic: each model alone (single) and the independent pair, scored on the 4 held-out shifts (speckle noise, gaussian blur, spatter, saturate; severity 5; 50k images each, each from source weights); whole-stream accuracy with a late-stream guard; sqrt-2 grid extended until interior; 3 seeds; temperatures fixed from the frozen models |
| 02b | `scripts/02b_tune_joint_lr.py` | episodic: the PAIR on the duo's entropy, from the single rates upward (x2 ladder, sqrt-2 refinement), scored by the static duo on the 4 held-out shifts; late-stream guard; finalists on 3 seeds |
| 02d | `scripts/02d_tune_large_adapt_lr.py` | episodic: the large member alone on the duo's entropy (partner frozen), 4 held-out shifts |
| 02e | `scripts/02e_tune_small_adapt_lr.py` | episodic: the small member alone on the duo's entropy (partner frozen), 4 held-out shifts |
| 02 | `scripts/02_tune_tta.py` | episodic: one model alone (Tent), 4 held-out shifts, seed 0 (the older single-model tuner) |
| 05 | `scripts/05_select_member_lr.py` | episodic: each member alone for ETA / EATA / SAR from the stored sweep, macro accuracy on the 4 held-out shifts, seed 0 |
| default | `pre-tuning default` | nothing tuned: the OPTIM block the duo had before the tuning scripts (never used for results) |
| 04b | `scripts/04b_tune_continual_lr.py` | CONTINUAL: the 4 held-out shifts x 4 rounds (each round rotated), 800k images, no reset; stability rule (late gain >= -2 points per visit against the frozen member); conservative = one rung below the stable frontier, stress = lowest rung that dies completely |
| trial | `by trial` | picked by hand for one experiment, not by a tuning rule |

## 2. Episodic rates (cfgs/duo_lrs): every duo, regime and TTA method

### vitb16_rn50v1  (the main duo)

large = vit_b_16, small = resnet50_v1. A `-` is a member with no entry for that method (not tuned in this duo).

| setting (regime) | TTA | large LR | small LR | optimizer | tuned by | score (macro acc., %; L / S = per member) | frozen | notes |
|---|---|---|---|---|---|---|---|---|
| single | tent | 7e-5 | 5e-4 | SGD m0.9 | 02c | L 51.8 / S 45.7 | L 48.8 / S 34.8 | seeds 0,1,2; each model tuned ALONE |
| single | (untuned default) | 1e-4 | 2.5e-4 | L: Adam b0.9, S: SGD m0.9 | default |  |  |  |
| indep | tent | 7e-5 | 5e-4 | SGD m0.9 | 02c | 56.6 (duo) |  | seeds 0,1,2; tuned as a PAIR for this duo (static-duo accuracy) |
| indep | eata | 5e-4 | 5e-4 | SGD m0.9 | 05 | L 60.3 / S 48.8 |  | seed 0 |
| indep | eta | 5e-4 | 2.5e-4 | SGD m0.9 | 05 | L 60.3 / S 48.5 |  | seed 0 |
| indep | sar | 5e-3 | 1e-3 | SGD m0.9 | 05 | L 59.9 / S 46.5 |  | seed 0 |
| indep | (untuned default) | 1e-4 | 2.5e-4 | L: Adam b0.9, S: SGD m0.9 | default |  |  |  |
| joint | tent | 3.2e-3 | 4e-3 | SGD m0.9 | 02b | 62.0 (duo) |  |  |
| large_duo | tent | 1.12e-3 | (frozen) | SGD m0.9 | 02d | 59.9 (duo) | 52.1 | seeds 0,1,2; the large member adapts on the duo entropy, the other member is frozen |
| small_duo | tent | (frozen) | 1.4e-3 | SGD m0.9 | 02e | 55.3 (duo) | 52.1 | seeds 0,1,2; the small member adapts on the duo entropy, the other member is frozen |

### convnextb_deits

large = convnext_base, small = deit_small. A `-` is a member with no entry for that method (not tuned in this duo).

| setting (regime) | TTA | large LR | small LR | optimizer | tuned by | score (macro acc., %; L / S = per member) | frozen | notes |
|---|---|---|---|---|---|---|---|---|
| single | tent | 1.25e-5 | 8e-4 | SGD m0.9 | 02c | L 54.7 / S 56.8 | L 49.6 / S 47.8 | seeds 0,1,2; each model tuned ALONE |
| single | (untuned default) | 1e-4 | 1e-4 | SGD m0.9 | default |  |  |  |
| indep | tent | 1.25e-5 | 8e-4 | SGD m0.9 | 02c | 60.9 (duo) |  | seeds 0,1,2; tuned as a PAIR for this duo (static-duo accuracy) |
| indep | (untuned default) | 1e-4 | 1e-4 | SGD m0.9 | default |  |  |  |
| joint | tent | 1.4e-4 | 1.1e-3 | SGD m0.9 | 02b | 62.3 (duo) |  |  |
| large_duo | tent | 2.8e-4 | (frozen) | SGD m0.9 | 02d | 59.3 (duo) | 52.7 | seeds 0,1,2; the large member adapts on the duo entropy, the other member is frozen |
| small_duo | tent | (frozen) | 1.6e-3 | SGD m0.9 | 02e | 57.6 (duo) | 52.7 | seeds 0,1,2; the small member adapts on the duo entropy, the other member is frozen |

### vitb16_rn34

large = vit_b_16, small = resnet34. A `-` is a member with no entry for that method (not tuned in this duo).

| setting (regime) | TTA | large LR | small LR | optimizer | tuned by | score (macro acc., %; L / S = per member) | frozen | notes |
|---|---|---|---|---|---|---|---|---|
| single | tent | 7e-5 | 7e-4 | SGD m0.9 | 02c | L 51.8 / S 43.7 | L 48.8 / S 32.7 | seeds 0,1,2; each model tuned ALONE |
| single | (untuned default) | 7e-5 | 5e-4 | SGD m0.9 | default |  |  |  |
| indep | tent | 7e-5 | 1e-3 | SGD m0.9 | 02c | 55.9 (duo) |  | seeds 0,1,2; tuned as a PAIR for this duo (static-duo accuracy) |
| indep | (untuned default) | 7e-5 | 5e-4 | SGD m0.9 | default |  |  |  |
| joint | tent | 1.6e-3 | 2.8e-3 | SGD m0.9 | 02b | 61.1 (duo) |  |  |
| large_duo | tent | 7.9e-4 | (frozen) | SGD m0.9 | 02d | 59.1 (duo) | 51.7 | seeds 0,1,2; the large member adapts on the duo entropy, the other member is frozen |
| small_duo | tent | (frozen) | 2.8e-3 | SGD m0.9 | 02e | 54.5 (duo) | 51.7 | seeds 0,1,2; the small member adapts on the duo entropy, the other member is frozen |

### vitb16_rn50

large = vit_b_16, small = resnet50. A `-` is a member with no entry for that method (not tuned in this duo).

| setting (regime) | TTA | large LR | small LR | optimizer | tuned by | score (macro acc., %; L / S = per member) | frozen | notes |
|---|---|---|---|---|---|---|---|---|
| indep | tent | 1e-4 | 1e-3 | L: SGD m0.9, S: Adam b0.9 | 02 | L 50.9 / S 45.7 | L 48.8 / S 38.0 | stale; seed 0 |
| indep | eata | 5e-4 | - | SGD m0.9 | 05 | L 60.3 |  | seed 0 |
| indep | eta | 5e-4 | - | SGD m0.9 | 05 | L 60.3 |  | seed 0 |
| indep | sar | 5e-3 | - | SGD m0.9 | 05 | L 59.9 |  | seed 0 |
| indep | (untuned default) | 1e-4 | 2.5e-4 | Adam b0.9 | default |  |  |  |

### vitb16_vits16

large = vit_b_16, small = vit_s_16. A `-` is a member with no entry for that method (not tuned in this duo).

| setting (regime) | TTA | large LR | small LR | optimizer | tuned by | score (macro acc., %; L / S = per member) | frozen | notes |
|---|---|---|---|---|---|---|---|---|
| indep | tent | 1e-4 | - | SGD m0.9 | 02 | L 50.9 | L 48.8 | seed 0 |
| indep | eata | 5e-4 | - | SGD m0.9 | 05 | L 60.3 |  | seed 0 |
| indep | eta | 5e-4 | - | SGD m0.9 | 05 | L 60.3 |  | seed 0 |
| indep | sar | 5e-3 | - | SGD m0.9 | 05 | L 59.9 |  | seed 0 |
| indep | (untuned default) | 1e-4 | 1e-4 | Adam b0.9 | default |  |  |  |

### vitb16aug2_rn50v1

large = vit_b_16_augreg2, small = resnet50_v1. A `-` is a member with no entry for that method (not tuned in this duo).

| setting (regime) | TTA | large LR | small LR | optimizer | tuned by | score (macro acc., %; L / S = per member) | frozen | notes |
|---|---|---|---|---|---|---|---|---|
| indep | tent | - | 2.5e-4 | SGD m0.9 | 02 | S 45.6 | S 34.8 | seed 0; frozen BN on batch stats |
| indep | eata | - | 5e-4 | SGD m0.9 | 05 | S 48.8 |  | seed 0 |
| indep | eta | - | 2.5e-4 | SGD m0.9 | 05 | S 48.5 |  | seed 0 |
| indep | sar | - | 1e-3 | SGD m0.9 | 05 | S 46.5 |  | seed 0 |
| indep | (untuned default) | 1e-4 | 2.5e-4 | L: Adam b0.9, S: SGD m0.9 | default |  |  |  |

### vitb16aug_rn50v1

large = vit_b_16_augreg, small = resnet50_v1. A `-` is a member with no entry for that method (not tuned in this duo).

| setting (regime) | TTA | large LR | small LR | optimizer | tuned by | score (macro acc., %; L / S = per member) | frozen | notes |
|---|---|---|---|---|---|---|---|---|
| indep | tent | - | 2.5e-4 | SGD m0.9 | 02 | S 45.6 | S 34.8 | seed 0; frozen BN on batch stats |
| indep | eata | - | 5e-4 | SGD m0.9 | 05 | S 48.8 |  | seed 0 |
| indep | eta | - | 2.5e-4 | SGD m0.9 | 05 | S 48.5 |  | seed 0 |
| indep | sar | - | 1e-3 | SGD m0.9 | 05 | S 46.5 |  | seed 0 |
| indep | (untuned default) | 1e-4 | 2.5e-4 | L: Adam b0.9, S: SGD m0.9 | default |  |  |  |

### vitb16sar_rn50v1

large = vit_b_16_sar, small = resnet50_v1. A `-` is a member with no entry for that method (not tuned in this duo).

| setting (regime) | TTA | large LR | small LR | optimizer | tuned by | score (macro acc., %; L / S = per member) | frozen | notes |
|---|---|---|---|---|---|---|---|---|
| indep | tent | - | 2.5e-4 | SGD m0.9 | 02 | S 45.6 | S 34.8 | seed 0; frozen BN on batch stats |
| indep | eata | - | 5e-4 | SGD m0.9 | 05 | S 48.8 |  | seed 0 |
| indep | eta | - | 2.5e-4 | SGD m0.9 | 05 | S 48.5 |  | seed 0 |
| indep | sar | - | 1e-3 | SGD m0.9 | 05 | S 46.5 |  | seed 0 |
| indep | (untuned default) | 1e-3 | 2.5e-4 | SGD m0.9 | default |  |  |  |

### vitl16_rn50

large = vit_l_16, small = resnet50. A `-` is a member with no entry for that method (not tuned in this duo).

| setting (regime) | TTA | large LR | small LR | optimizer | tuned by | score (macro acc., %; L / S = per member) | frozen | notes |
|---|---|---|---|---|---|---|---|---|
| indep | tent | - | 1e-3 | Adam b0.9 | 02 | S 45.7 | S 38.0 | stale; seed 0; frozen BN on batch stats |
| indep | (untuned default) | 1e-4 | 2.5e-4 | Adam b0.9 | default |  |  |  |

## 3. Continual rates (04b, held-out stream x 4 rounds)

### vitb16_rn50v1  (continual, 04b, seeds 0)

Stream: the 4 held-out shifts x 4 rounds (800000 images), round r rotated by r. Stable = every visit ends within 2 points of the frozen member (last 20% of the visit); dead = below 5% for 4 windows of 5120; conservative = 1 rung below the frontier. Warnings: 0.

| member | episodic rate (anchor) | ladder | frontier (largest stable) | **conservative** | **stress** (lowest dead) | mean acc. at conservative / at frontier | frozen |
|---|---|---|---|---|---|---|---|
| large | 7e-5 | 2.2e-6 .. 2.8e-4 (15 rungs) | 1.75e-5 | 1.26e-5 | 3.5e-5 | 51.6 / 52.0 | 48.8 |
| small | 5e-4 | 1.6e-5 .. 2e-3 (15 rungs) | 3.1e-5 | 2.2e-5 | 2.5e-4 | 41.2 / 41.5 | 34.9 |

Rungs below the original ladder were added (the ResNet had no stable rung down to 6.25e-5): the ResNet 1.6e-5 .. 4.4e-5, paired with throwaway ViT rungs 2.2e-6 .. 6.2e-6 (a run pairs the i-th rungs of the two ladders).

## 4. Rates actually run in the gate-stress experiments (04c)

Experiments A and B use the 04b conservative / stress rates; B1 and B2 use ViT rates above the 04b stress rate, picked by trial because the 04b stress rate (3.5e-5) does not collapse the ViT on the test stream. S, H and T are the runs the reset rules are screened on offline (`scripts/37b_screen_resets_stress.py`): S = both members at a rate chosen to collapse them (ViT 1.4e-4, ResNet 5e-4; ViT above its 04b stress rate by trial, ResNet above it by trial), held-out stream, calibration; H = the 04b conservative rates, held-out stream, healthy time for the false-alarm budget; T = the S rates on the test stream, evaluation only. Every run here is Vanilla DD (both members on their own entropy, the gate outside the loop).

| duo | stream | experiment | gate (beta, b_t) | ViT LR (how found) | ResNet LR (how found) | seed | summary file |
|---|---|---|---|---|---|---|---|
| vitb16_rn50v1 | heldout x4 rounds | A | 0.5, 256 | 1.26e-5 (04b conservative) | 2.5e-4 (04b stress) | 0 | summary_heldout.json |
| vitb16_rn50v1 | heldout x4 rounds | B | 0.5, 256 | 3.5e-5 (04b stress) | 2.2e-5 (04b conservative) | 0 | summary_heldout.json |
| vitb16_rn50v1 | heldout x4 rounds | H | 2, 256 | 1.26e-5 (04b conservative) | 2.2e-5 (04b conservative) | 0 (rotations 0,2) | 2 runs, no summary (screening runs, scripts/37b_screen_resets_stress.py) |
| vitb16_rn50v1 | heldout x4 rounds | S | 2, 256 | 1.4e-4 (TRIAL: above the 04b stress rate) | 5e-4 (TRIAL: above the 04b stress rate) | 0,1 (rotations 0,1,2,3) | 8 runs, no summary (screening runs, scripts/37b_screen_resets_stress.py) |
| vitb16_rn50v1 | heldout x4 rounds | live resets (12 conditions: both_fast_vs_slow, boundary, fixed1000, fixed2000, fixed250, fixed500, gap_fast_vs_slow, none, oracle, own_cusum, own_fast_vs_slow, own_level) | 2, 256 | 1.4e-4 (TRIAL: above the 04b stress rate) | 5e-4 (TRIAL: above the 04b stress rate) | 0 (rotations 0,2) | scripts/38_run_reset_live.py, out/results/reset_live/ |
| vitb16_rn50v1 | test | A | 2, 256 | 1.26e-5 (04b conservative) | 2.5e-4 (04b stress) | 0 | summary__nnn_bt256_beta2.json |
| vitb16_rn50v1 | test | B1 | 1, 256 | 1.4e-4 (TRIAL: above the 04b stress rate) | 2.2e-5 (04b conservative) | 0 | summary__nnn_bt256_beta1.json |
| vitb16_rn50v1 | test | B1 | 2, 256 | 1.4e-4 (TRIAL: above the 04b stress rate) | 2.2e-5 (04b conservative) | 0 | summary__nnn_bt256_beta2.json |
| vitb16_rn50v1 | test | B2 | 2, 256 | 2.8e-4 (TRIAL: above the 04b stress rate) | 2.2e-5 (04b conservative) | 0 | summary__nnn_bt256_beta2.json |
| vitb16_rn50v1 | test | T | 2, 256 | 1.4e-4 (TRIAL: above the 04b stress rate) | 5e-4 (TRIAL: above the 04b stress rate) | 0,1,2 | 3 runs, no summary (screening runs, scripts/37b_screen_resets_stress.py) |
| vitb16_rn50v1 | test | live resets (9 conditions: both_fast_vs_slow, boundary, fixed1000, gap_fast_vs_slow, none, oracle, own_cusum, own_fast_vs_slow, own_level) | 2, 256 | 1.4e-4 (TRIAL: above the 04b stress rate) | 5e-4 (TRIAL: above the 04b stress rate) | 0,1,2 | scripts/38_run_reset_live.py, out/results/reset_live/ |

## 5. Planned, not yet run

- **Rates for a resetting scheme (Vanilla DD): not tuned yet.** The 04b rates are the largest rates that never collapse WITHOUT a reset, so they are expected to be too low once collapses are caught: on the held-out shifts a model that starts every shift from source weights (the idealised reset at every boundary) scores 45.7 for the ResNet at its episodic 5e-4 against 41.2 at the 04b conservative rate; for the ViT 51.8 against 51.6. They must be tuned WITH the reset rule running, on the held-out stream x 4 rounds (live runs, whole-stream accuracy with the resets counted), separately per rule family and for the oracle trigger.
- The B1 setting (ViT 1.4e-4 (trial), ResNet 2.2e-5 (04b conservative), test stream, beta 2) stays as the qualitative stress case for the reset plots; it is not a tuned setting.

_Generated 2026-10-26 20:24 -0400 by src/utils/rates_sheet.py._
