# DD final sprint (Nov 4 – Nov 16, 2026)

> **Order of business (author, Nov 4 evening), replaces the tiers of Section 6 where they differ:**
> 1. Finish the continual ImageNet-C tables, both duos (Independent and Joint DD).
> 2. ETA with both duo types (main duo).
> 3. Establish whether Joint DD is the main method, and why.
> 4. Proxy study: which proxies we tried, why, and why the nuclear norm (`nnn`).
> 5. Reset-rule study: which rules we tried, why, and why own fast vs slow average.
> 6. Standalone baselines on continual ImageNet-C (ASR; RDumb done). SAR as a member method comes after these.

Plan for the last experiments before the submission on **Monday Nov 16**.

> **Nov 4: CCC dropped (author).** Every CCC item below (D4 stream, the CCC column of Section 3, C3/C5 and the CCC part of C6, CCC launches in Section 7, cut items 2 and 4) is void. The CCC data (`/project/6104790/alxstaub/datasets/ccc`, 40 GB, 80 shards) and code (`scripts/ccc_generate.py`, `src/reset/checkpoint.py`, `src/utils/datasets/ccc.py`) stay, unused. The freed GPU time (~130 GPU-h) goes first to Joint ETA (Tier 3).
Duos: **ViT-B/16 + ResNet-50 V1** (`vitb16_rn50v1`, the main duo) and **ViT-B/16 + ConvNeXt-T** (`vitb16_convnextt`). BatchNorm on test-batch statistics (`--bn`).

## 1. Decisions taken

| # | Decision | Source |
|---|---|---|
| D1 | Two duo types in every experiment: **Independent DD** (each member adapts on its own loss, the gate combines them outside the loop; called "Vanilla DD" until now, renamed) and **Joint DD** (both members adapt on the gated duo output, the gate is inside the loss). | author, Nov 3 |
| D2 | The gate is on in every experiment: **beta = 0.25**, proxy `nnn`, logit link, log pool, no filter, proxy batch 256. Fixed in advance from the episodic and held-out results (Section 2), not tuned on any test stream. The paper says that held-out continual tuning picked beta 0 for the main duo, and that 0.25 is tied with it there. | author, Nov 3 |
| D3 | Configurations: frozen; reset at every shift (continual ImageNet-C only); no reset at the accuracy-tuned rates (the "conservative" rates are dropped everywhere); reset every 500 and every 1,000 steps; per-member own fast/slow rule; random resets matched to that rule. The run matrix is in Section 3. | author, Nov 3 |
| D4 | Stream: continual ImageNet-C only (the 15 test corruptions, 5 orders: standard + 4 of CoTTA's). **CCC dropped** (author, Nov 4); the generated 2M-image CCC-Medium data and the CCC code stay in place, unused. | author, Nov 3; CCC dropped Nov 4 |
| D5 | TTA methods: Tent on both duos; **ETA on the main duo, tuned from scratch in our setting exactly as Tent was** (episodic member rates, the Joint pair at beta 0.25, then the continual/reset tuning); **RDumb** as the only external baseline for now. DeYO and ROID are out of scope. **Nov 4 update (author): member methods = Tent, ETA and SAR; standalone baselines = ASR (ICLR 2026, the main one) and RDumb.** | author, Nov 3; Nov 4 |
| D6 | Every hyperparameter is chosen on the 4 held-out shifts only (speckle, gaussian blur, spatter, saturate), never on a test stream or CCC. All rates are tuned for the config they run in. | standing protocol (plan.md) |
| D7 | ConvNeXt-T duo: criterion **`worst_shift`** in every tuning and test (consistent with its episodic table); the main duo keeps `mean`. The paper says why (ConvNeXt-T collapses on test shifts next to its mean-optimal rates). | author, Nov 3 |
| D8 | The paper explains the choice of beta 0.25 from the earlier experiments (Section 2.1, "Rationale"). | author, Nov 3 |

## 2. What earlier experiments establish

### 2.1 Gate configuration

| Evidence (mean accuracy, %) | Numbers | Source |
|---|---|---|
| Episodic, main duo | Independent: fixed TS 52.2, gate b0.25 52.5, b0.5 52.5. Joint: fixed TS 59.1, gate b0.25 58.9 (tie), b0.5 57.6 (worse in every seed) | `out/paper/tab_adaptation_modes_avg__tent__vitb16_rn50v1_vs_vitb16_convnextt_gate.tex` |
| Episodic, ConvNeXt-T duo | Independent: fixed TS 43.2, gate b0.25 45.8, b0.5 46.0. Joint: fixed TS 38.6, gate b0.25 44.1, b0.5 45.2 (retuned with the gate: 44.7 / 44.5) | same |
| Held-out, main duo, Joint, episodic, pair retuned per beta (02f) | beta 0 / 0.25 / 0.5 / 0.75 / 1: 61.98 / 61.30 / 60.60 / 60.28 / 59.88 | `docs/tables_convnextt/joint_gate_pair_by_beta.tex` |
| Held-out, main duo, Joint, continual with resets | beta 0 / 0.25 / 0.5 / 0.75 / 1: 59.75 / 59.67 (tied) / 58.80 / 58.48 / 57.46 | `out/results/reset_tune/vitb16_rn50v1__joint/joint__choice.json` |
| Held-out, main duo, Independent, continual with resets: gate minus fixed TS | beta 0.25 +0.02 ± 0.02; 0.5 −0.05; 1 −0.28; 2 −0.75; 4 −1.36; 8 −1.90. In the last 1,280 images before a ViT reset: +2.0 / +3.3 / +4.1 / +3.2 / +0.8 / −1.6 | `out/results/reset_final/gate_routing_heldout.json` |
| Continual test, ConvNeXt-B + DeiT-S (weak duo): gate (b0.5) vs fixed TS | rule 40.6 vs 14.6; no reset 32.7 vs 26.8; every 500 46.2 vs 41.3; episodic 48.5 vs 43.3 | `docs/tables_convnextb_deits/final_test_main.tex` |
| Proxy, held-out, collapsing pair (ViT 2e-4, ResNet 2e-3), proxy batch 128 | nnn +6.4 over fixed TS, 0.09 from the oracle weight; msp loses to fixed TS (score backwards under collapse, Spearman −0.61); the links differ only by a rescaling of beta (within 0.1 at their own best beta) | `out/results/ablate_gate_link/*summary.csv` |

**Rationale for beta 0.25 (to go into the paper, in substance).**

1. *What beta trades off.* A steeper gate (larger beta) moves weight to the member the proxy scores higher. That pays when one member is collapsing: in the last 1,280 images before a ViT reset the gate gains +2.0 (beta 0.25), +3.3 (0.5) and +4.1 (1) over fixed TS on the held-out streams. It costs when both members are healthy, because the proxy gap is then mostly noise: over the whole held-out stream the gate goes from +0.02 (0.25) to −0.28 (1) and −1.90 (8). Every held-out study of the main duo shows accuracy falling monotonically with beta: episodic Joint (61.98 → 59.88 from beta 0 to 1), continual Joint with resets (59.75 → 57.46), continual Independent with resets.
2. *Why not beta 0 (fixed TS).* Fixed TS has no protection when a member fails. On the weak duos the gate is worth a lot: ConvNeXt-T, episodic, +2.6 (Independent) and +5.5 (Joint) at beta 0.25; ConvNeXt-B + DeiT-S, continual, +26 points for the reset rule (14.6 → 40.6) and +5 to +6 in every other configuration. Held-out tuning on the main duo picks beta 0 because that duo, at tuned rates, rarely has a failing member; it cannot see the failures the gate is there for.
3. *Why 0.25.* It is the largest beta that is never measurably worse than fixed TS on the strong duo: tied in the episodic Joint test (58.9 vs 59.1, within seed noise), tied in the continual held-out Joint study (59.67 vs 59.75) and in the continual held-out Independent ladder (+0.02 ± 0.02); beta 0.5 already loses 1.5 points in the episodic Joint test. On the weak duo it keeps most of what a steeper gate gains (Independent 45.8 vs 46.0 at 0.5; Joint 44.7 vs 44.5).
4. *How it was chosen.* Once, from these episodic and held-out results, before any continual or CCC test run; never retuned per duo, per stream or per config. Every Joint rate is tuned *at* beta 0.25, because Joint rates tuned without the gate drift once it is on (Section 2.2).
5. *Proxy and the rest.* nnn, because a collapsed member stays unconfident under nnn while msp scores it confidently wrong (Spearman −0.61); the link only rescales beta; proxy batches of 128 and 256 agree.

### 2.2 Other facts that save time

| Fact | Consequence |
|---|---|
| Joint rates must be retuned at the beta they run with: on the main duo the beta-0 pair drifts at every beta > 0 (53–56% vs 59.9–61.3% retuned, held-out). On the ConvNeXt-T duo retuning made no difference. | every Joint DD config is tuned at beta 0.25 |
| Independent DD rates and detectors are tuned on each member's own accuracy (decoupled) | Independent configs need no retuning for beta 0.25 |
| In Independent DD the members never interact | the member columns of the Independent runs **are** the single-model results (single model + Tent + that reset method); no extra runs |
| The stored main-duo test runs (`reset_live/vitb16_rn50v1__tent_bn/live-*__order*`) hold per-sample correctness for `ts`, `large`, `small` and the beta-0.5 gate only, not the logits | they are rerun with beta 0.25 as the main gate (beta 0.5 as an extra gate) |
| Temperatures come from the frozen models; `nnn` needs no fitted statistics | reused for ETA and CCC, no setup |
| Throughput (L40S): live run + frozen copies ~140 img/s, frozen ~526 img/s | continual ImageNet-C ≈ 1.5 h per run; CCC 7.5M ≈ 15 h (frozen ≈ 4 h) |
| Under queue load only short jobs start | CCC runs are checkpointed and resumable (C3) |
| The random control copies the rule run's reset counts | it waits for its rule run (+15 h on CCC): rule runs go in first |

### 2.3 Known table issue (not fixed)

`docs/tables_per_member/final_test_main.tex`: the row "Joint DD, reset at every shift (episodic)" shows 59.1, which is Joint **fixed TS**; the caption calls it "03b's Joint DD at beta 0.5" (55.5 at the old pair, 57.6 retuned). To fix when the tables are regenerated.

## 3. Run matrix

The same matrix for each duo and each TTA method. Every run scores, from the same members: the gate at beta 0.25 (the main gate), the fixed-TS combination (`ts`), each member alone, and, for Independent DD only, extra gates (beta 0.5 for continuity with the existing tables).

| Config | Independent DD | Joint DD | Continual ImageNet-C (5 orders) | CCC-Medium |
|---|---|---|---|---|
| Frozen | one run, shared by both types (gate + fixed TS) | (same run) | 1 run (order-independent) | 1 run |
| Reset at every shift | **existing** episodic runs, beta 0.25 rows | **existing** (02f pair at beta 0.25) | no new runs | — (no shifts) |
| No reset, accuracy-tuned rates | ✓ | ✓ | 5 + 5 | 1 + 1 |
| Every 500 steps | ✓ | ✓ | 5 + 5 | 1 + 1 |
| Every 1,000 steps | ✓ | ✓ | 5 + 5 | 1 + 1 |
| Own fast/slow, per-member settings | ✓ | ✓ | 5 + 5 | 1 + 1 |
| Random, matched to the rule | ✓ | ✓ | 5 + 5 | 1 + 1 |
| **Runs per duo** | | | **51** | **11** |

"Fixed TS" as a *config* (members adapted without the gate) is run for Independent DD only; it is the `ts` column of the same run. For Joint DD, the `ts` column is the static combination of members adapted *with* the gate inside the loss; the "Joint, fixed TS" reference row on continual ImageNet-C is the existing beta-0 rule run (main duo, 56.9).

External baseline: **RDumb** (ETA + reset of weights and optimizer every 1,000 steps), each member of both duos as a single model; continual ImageNet-C (5 orders) and CCC. In our pipeline it is the Independent `fixed1000` condition with ETA, so its member columns are the single models and its duo columns come for free. Two rows:
- **RDumb as published** (rate 2.5e-4, e_margin 0.4 ln 1000, d_margin 0.05, no tuning): needs only C1, so it can run before ETA's tuning is done.
- **RDumb tuned in our setting**: identical to the ETA "every 1,000 steps" row of the matrix (rate tuned on the held-out shifts), so no extra runs; it appears once ETA's tuning is frozen.

## 4. Tuning status

| Duo | Independent DD | Joint DD at beta 0.25 |
|---|---|---|
| ViT-B/16 + ResNet-50 | **done** (`reset_tune/final_configs.json`, criterion mean) | rule: exists at beta 0 only (not converged) → retune at 0.25: `tune_reset.py --duo vitb16_rn50v1 --signal joint --betas 0.25 --submit --chain`. Baselines (no reset, every 500, every 1,000): need C2 |
| ViT-B/16 + ConvNeXt-T | **done**, frozen Nov 3, under both criteria (`reset_tune/vitb16_convnextt[__worst_shift]/final_configs.json`) | nothing yet: full joint study at beta 0.25 + baselines (C2) |
| ETA (main duo) | (1) episodic member rates (02c with `--tta`, the native ETA of C4); (2) continual/reset tuning (`tune_reset.py --tta eta`, after C1) | (1) Joint pair at beta 0.25 (02f `--betas 0.25`); (2) joint reset tuning at 0.25 (after C2, C4); Tier 3 |

**ETA is tuned in our setting from scratch.** The recorded `eta` rates (`cfgs/duo_lrs/vitb16_rn50v1__indep.yaml`, 5e-4 / 2.5e-4) came from the ported implementation, which cannot run with the joint loss. To use one implementation for both types, ETA runs through the native method (`src/tta/methods/eata.py`, ETA's loss without EATA's Fisher term) and every rate is retuned with it, in the same stages as Tent. The ported rates serve only as the anchor of the first grid and as a parity check (C4).

## 5. Code needed (Wed Nov 4, with GPU smoke tests)

| # | Change | Why | Size |
|---|---|---|---|
| C1 | Pass `--tta` through `src/reset/` (`layout.TTA = "tent"` is hard-coded; `workers.py`, `pipeline.py`) | ETA, RDumb | small |
| C2 | Joint-signal baselines: `tune_reset.py --baseline none / fixed500 / fixed1000 --signal joint` (refused today: "the comparison methods are independent-only") | Joint DD rows | ~half a day |
| C3 | `--stream ccc` in `workers.live` + checkpoint/resume every K batches (member weights, optimizer states, gate buffer and last weight, detector state, trigger state, reader offset) | CCC within queue limits | ~half a day |
| C4 | ETA through the native `src/tta/methods/eata.py` in the `src/experiments` engine and the episodic tuners (02c, 02f): the ported `eta` refuses joint modes (`algorithms.make_runner`). Parity check native vs ported on one episodic stream (same rate); per-member reset of its running-prob state; RDumb's hyperparameters reachable through `--tta_kwargs` | ETA, Joint ETA, RDumb | ~half a day |
| C5 | `scripts/ccc_generate.py` (Slurm CPU array around `third_party/CCC/generate.py`), separate venv (`webdataset`, `Wand`, `opencv`, `numba`), `paths.ccc` in `cfgs/global.yaml` | CCC data | small |
| C6 | Script 42 report: beta-0.25 main gate, Independent and Joint blocks per duo, CCC column, member columns labelled as single models, the 2.3 fix | the tables | small |

## 6. Priorities

| Tier | Content | GPU-h |
|---|---|---|
| **1 (must)** | Tent, both duos, continual ImageNet-C, full matrix, incl. the Joint tuning at beta 0.25; RDumb as published, both duos | ~150 (IN-C tests) + ~185 (Joint tuning) + ~25 (RDumb) ≈ **360** (CCC dropped Nov 4: −85) |
| **2** | ETA, main duo, Independent DD: episodic tuning, reset tuning, continual ImageNet-C tests (gives the tuned-RDumb row) | ~30 + ~60 + ~35 ≈ **125** |
| **3** | ETA, main duo, Joint DD: Joint pair at beta 0.25, joint reset tuning, tests | ~145 |

Tiers 1 + 2 keep ~4 GPUs busy from Nov 4 to Nov 12. CCC at 2M images is ~4 h per live run (frozen ~1 h); generation is CPU only, its cost is measured on Nov 4.

## 7. Schedule

| Date | What | Needs |
|---|---|---|
| **Wed Nov 4** | **CCC:** venv, time one 25k-image shard (time, size; the first 80 shards = 2M images are generated first). Launch CCC-Medium (1000, 43) onto `/project`. **Code:** C1–C5, smoke tests. **GPU, no code needed:** (a) Independent DD test runs on continual ImageNet-C at beta 0.25, **both duos** (ConvNeXt-T under `worst_shift`; frozen + 5 configs × 5 orders each; controls after their rule); (b) Joint rule at beta 0.25: main duo retune, ConvNeXt-T study start (`--criterion worst_shift`). | — |
| **Thu Nov 5 – Sat Nov 7** | Joint baselines at beta 0.25, both duos (after C2). RDumb as published on continual ImageNet-C, both duos (after C1, C4). ETA: episodic member rates (Nov 5–6), then Independent reset tuning (Nov 6–). | C1, C2, C4 |
| **Sun Nov 8** | **Gate 1: freeze the Tent Joint configs of both duos** (not converged → keep the last choice, say so). Launch the Joint DD test runs on continual ImageNet-C, both duos. Launch Tent CCC (2M), both duos (11 runs each, rules first) and RDumb on CCC. | C3, the first 80 CCC shards |
| **Mon Nov 9 – Tue Nov 10** | **Freeze Independent ETA Tue Nov 10** → its continual ImageNet-C and CCC test runs. Joint ETA: pair at beta 0.25, then reset tuning (Tier 3). | — |
| **Wed Nov 11** | **Gate 2: freeze Joint ETA or drop it.** Tent CCC rule runs done ~Nov 8–9, controls ~Nov 9–10. | — |
| **Thu Nov 12** | **Last launch.** All reports and tables generated (C6). | — |
| **Fri Nov 13 – Mon Nov 16** | No new runs except failed-job reruns. Tables, the CCC accuracy-over-time figure (members, duo, gate weight, reset marks), writing. | — |

## 8. Risks and what to cut (in this order)

1. Tier 3 (ETA Joint DD).
2. ETA on CCC.
3. ETA Independent DD: keep RDumb as published (no tuning) even if ETA's own tuning is dropped.
4. CCC length: below 2M if generation is slower than expected (same configs; the paper states the length); ConvNeXt-T CCC before the main duo's.
5. Queue congestion: CCC chunked (C3); tuning chains are short jobs. Exclude kn117, kn101, kn104 on every GPU job; QOS high.
6. Joint DD at beta 0.25 below Joint fixed TS (possible on the main duo): a result, reported against the existing beta-0 reference row.

## 9. Answered (Nov 3)

1. Beta 0.25 for everything: **yes**, with the rationale of Section 2.1 in the paper (D8).
2. ConvNeXt-T criterion: **`worst_shift`** (D7).
3. ETA: **tuned in our setting, as Tent was** (D5); through the native implementation so that Joint DD is possible (Section 4).
4. CCC: **shortened** (D4: 2M images, 3M if time allows).
5. External baselines: **RDumb only, for now** (Section 3).

No open questions block Wed Nov 4.

## 10. Status log

### Nov 3, 23:00 – Nov 4, 00:30 (night of day 0)

**Cancelled:** 28 pending ConvNeXt-T continual test runs submitted by another session (jobs 7237105–7237132): `mean`-criterion configs, beta 0.5, conservative rates included, so they contradict D2, D3 and D7. Two of that batch had finished (no reset, std + order 0, `mean`, beta 0.5); kept, not part of the sprint.

**Submitted (GPU):**

| What | Jobs | Notes |
|---|---|---|
| Independent DD test runs, continual ImageNet-C, beta 0.25, main duo | 7241668–7241692 | `out/results/reset_tune/sprint_configs.json` (no reset 1.75e-5 / 3.1e-5, every 500, every 1,000, per-member own fast/slow, random control); `scripts/42_final_test.py submit --configs ... --beta 0.25` |
| Same, ConvNeXt-T duo, `worst_shift` | 7241693–7241717 | `out/results/reset_tune/vitb16_convnextt__worst_shift/sprint_configs.json` |
| Joint DD, main duo, beta 0.25: the rule's search | 7241831–7241834, chain 7241835 | `tune_reset.py --signal joint --fixed_beta 0.25 --submit --chain`; study `reset_tune/vitb16_rn50v1__joint__beta0.25/` |
| Joint DD baselines, main duo, beta 0.25 | none 7241839–41 (chain 7241842), fixed500 7241844–47 (7241848), fixed1000 7241849–52 (7241853) | `--baseline none|fixed500|fixed1000` |
| Joint DD, ConvNeXt-T, beta 0.25, `worst_shift`: continual ladder | 7241863–7241874, chain 7241875 | new joint study from scratch; `reset_tune/vitb16_convnextt__joint__beta0.25__worst_shift/` |
| Joint DD baselines, ConvNeXt-T, fixed500 / fixed1000 | 7241877–80 (7241881), 7241884–87 (7241888) | `none` waits for the ladder (it starts there) |
| ETA parity check (C4) | 7242421 | `out/_smoke/sprint_20261104/eta_parity.py`: ported `eta` vs native `eta_native`, spatter, 10k images, same rates |

**Submitted (CPU):** CCC timing per corruption (7242038); CCC-Medium shard 0 of 80 (7242039).

**Code (C1–C5 done, self-tests pass; nothing committed):**
- `--fixed_beta B` (joint signal): `layout.Paths.fixed_beta` (study `<duo>__joint__beta<B>[__<criterion>]`, runs/screens shared), `tuning.plan_joint(fixed_beta=)` (every cell at B, no beta block), joint comparison methods (`pipeline.stage_baseline_rates_joint`: a pair search on the duo's accuracy at B; `none` starts at the joint continual ladder's conservative pair and the two rungs above it, the schedules on the X2 diagonal like the rule).
- `--tta` through the reset pipeline (`Paths.tta`, job specs, workers, `tune_reset.py --tta`, `42_final_test.py --tta`). Tent's names, specs and job hashes are unchanged (checked against the queued jobs).
- Native ETA: `eta_native` (= the `eata` TTAMethod, ETA's loss without the Fisher term) in the engine (`algorithms.ALGO_DEFAULTS`, `NATIVE`: MemberRunner, joint modes allowed); a member reset now also clears that member's (and the duo signal's) running average of predictions.
- CCC: `scripts/ccc_generate.py` + `slurm/job_ccc.sh` (venv `~/ccc_env` on `StdEnv/2023 python/3.11.5 opencv/4.11.0`), output `/project/6104790/alxstaub/datasets/ccc` (= `paths.ccc`), accuracy matrix downloaded there. Live runs on a stream `ccc-<difficulty>-<speed>-<seed>-n<images>` with checkpoint/resume (`src/reset/checkpoint.py`, `run_live(checkpoint=, deadline=)`, CCC loader `start=`); `42_final_test.py --stream ccc-medium-1000-43-n2000000` submits each run as a chain of 3 resuming pieces (1:50 h each, stop at 95 min) plus a frozen run.

**Decisions taken while building (for the author to check):**
1. *Calibration beta of the fixed-beta joint studies.* The continual ladder and the calibration streams (which only supply the detector screen's candidates and the start of the no-reset search) stay at the default beta 0.5; every decision (rates, detectors, thresholds, the baselines' pairs) is made in cells at beta 0.25. Reason: the continual-run file names carry no beta, so a beta-0.25 ladder would overwrite the main duo's existing one.
2. *CCC image draws.* Upstream `generate.py` seeds NumPy/torch once per worker, so with N parallel workers the first N shards would hold the same images. The wrapper reseeds them from (seed, shard, step) for every step; the corruption walk (Python `random`) is upstream's, unchanged. So our CCC-Medium has upstream's corruption sequence but not upstream's exact images (upstream's are not reproducible across worker counts either).
3. *RDumb "as published"* uses rate 2.5e-4 for every member, the value published for ResNet-50; for the ViT and ConvNeXt-T it is an extrapolation, which is why the tuned row exists.
4. *ETA's name.* The native ETA is `eta_native` everywhere (runs, caches, `cfgs/duo_lrs`), never `eta` / `eata` (the ported ETA / EATA).

**Next:** read the CCC timing and shard 0 → size and launch the 80-task array; GPU smoke of a resumed CCC run on real models (needs shard 0); if parity passes: ETA episodic member grid (02c, `--tta eta_native`) and RDumb as published (both duos, continual ImageNet-C, 5 orders); ConvNeXt-T joint `--baseline none` once its ladder is done.

### Nov 4, 01:00 – 01:30

- **ETA parity (C4) PASSED** (job 7242421; spatter, 10k images, ViT 5e-4 / ResNet 2.5e-4): ViT predictions identical (59.64 / 59.64), ResNet 98.3% identical (49.27 ported vs 49.28 native); first-batch logits equal. The native `eta_native` is ETA.
- **CCC timing** (job 7242038) failed on upstream code written for older libraries; fixed by compatibility shims in `scripts/ccc_generate.py`, each an exact equivalent: `skimage.filters.gaussian(multichannel=True)` → `channel_axis=-1`; NumPy 2's removed binary `np.fromstring` → `np.frombuffer`; removed aliases (`np.float_` → `float64`, ...). All 15 corruptions run (login-node test, 2 images): mean 67 ms/image per corruption, so a 25k shard is at most ~0.9 h. A 50-image test shard reads back through `src/utils/datasets/ccc.py` (224x224 RGB, label = class index of its source file). ImageNet val is read in place (no per-task copy).
- **CCC-Medium generation launched:** worker 0 = job 7242039 (running), workers 1–79 = array 7245213 (`-t 03:00:00`).
- **ETA episodic member grid (stage A, seed 0)** launched: 9 diagonal runs of `04_cache_logits.py --tta eta_native --streams heldout` (ViT 1.25e-4 .. 2e-3, ResNet 6.25e-5 .. 1e-3, sqrt-2 grid centred on the ported ETA's 5e-4 / 2.5e-4): jobs 7245219–7245227. Note: `02c`'s printed commands omit `--tta` (they would run Tent): add it by hand.
- **RDumb as published** (`out/results/reset_tune/rdumb_published_configs.json`: ETA, fixed1000, 2.5e-4 / 2.5e-4, Independent DD, beta 0.25), continual ImageNet-C, 5 orders: main duo 7245238–7245242, ConvNeXt-T 7245243–7245247; runs in `reset_live/<duo>__eta_native_bn/`.
- **Queue:** the 32 pending Independent DD test runs were given `nice=500` so that the CCC and parity jobs could start first (CPU jobs share the l40s partitions here).
- **Check of the first Independent DD test runs at beta 0.25** (11 done, ~65 min each): their member and fixed-TS predictions are bit-identical to the stored beta-0.5 runs (members do not depend on the gate), so only the gate column changed. Gate at 0.25 vs 0.5: schedules equal (±0.02); no reset +0.1..+0.3 for 0.25 (e.g. order 2: 49.67 vs 49.40); both within 0.1–0.2 of fixed TS.

### Nov 4, 11:00 (the session restarted overnight; the watches had stopped)

- **Bad node kn029**: no usable CUDA device although a GPU was allocated; 4 joint runs of the main duo's rule search failed in seconds (7251446–7251449), which stopped that search's chain. kn029 added to `jobs.EXCLUDE` (and to the memory note); the search was resubmitted (7261634–7261637, chain 7261638). The other six joint searches had kept running (their chains are alive).
- **ConvNeXt-T joint no-reset search** started (its ladder is done): 7261641–7261642 (one submission timed out on Slurm's side; the chain 7261643 asks for it again).
- **CCC**: 71 of 80 shards done; 9 workers hit the 1-hour limit (shards 1, 3, 7, 8, 9, 11, 25, 26, 27: the slow corruptions). Their partial tars deleted (upstream would delete one and skip a walk step, shifting the shard) and the 9 resubmitted with 3 h: array 7261644. A shard takes 11–60+ min, ~490 MB (2M images ≈ 39 GB).
- **CCC reader fixed** (`src/utils/datasets/ccc.py`): it indexed EVERY shard in the folder, so it crashed on a shard still being written (the first smoke run, 7245638, failed that way in its joint ETA half) and refused any folder with a gap. It now indexes only the contiguous shards a prefix needs, in order (self-test extended).
- **CCC resume smoke, first half** (Independent DD, Tent, per-member rule, 25k images on real models): stopped by the deadline at 11,712 images, checkpoint 2.4 MB, resumed: predictions and resets identical to the uninterrupted run, but the gate's proxy batches NOT identical. Rerun with a second uninterrupted run as a noise reference and the first difference printed (job 7261687): GPU SVD noise in the nnn proxy, or a state the checkpoint misses.
- **Independent DD test runs at beta 0.25**: main duo 25/25 done; ConvNeXt-T 23/25 (the last two random controls running). **RDumb as published**: 10/10 done (both duos, 5 orders).
- **ETA episodic grid, seed 0 (81 pairs)**: best Independent pair ViT 5e-4 / ResNet 3.54e-4 (duo 62.24, late 63.32, worst stream 47.17; frozen duo 52.05); single optima ViT 7.07e-4, ResNet 2.5e-4; both interior. Confirmation on seeds 1 and 2 (`--confirm_around 2`, 12 member runs): 7261755–7261766. Then `02c ... --seeds 0 1 2 --confirm_around 2 --record` and the ETA reset tuning (`tune_reset.py --tta eta_native`).
- **CCC resume smoke (job 7261772), result:** the checkpoint/resume is correct; the remaining differences are GPU non-determinism, present in a plain repeat too.
  - Independent DD, Tent, per-member rule (25k CCC images, stop at 11,712): predictions and resets identical to the uninterrupted run; proxy values differ by ≤ 2.0e-6, and two uninterrupted runs differ by ≤ 1.3e-6 already at proxy batch 0 (the SVD of the nnn proxy on the GPU is not bit-reproducible).
  - Joint DD, ETA, every 500 (stop at 11,840): the resumed run differs from the uninterrupted one by ≤ 0.027 in proxy values and in some predictions, and a REPEAT of the uninterrupted run differs from it by ≤ 0.026: with the gate inside the loss, the 1e-6 proxy noise moves the gate weight, the gradients and the trajectory. Joint DD runs on the GPU are not bit-reproducible, resumed or not (the CPU self-test, deterministic, shows the resume itself is exact for joint ETA with resets). Consequence for the paper: seed-to-seed (order-to-order) variation of Joint DD includes this numerical noise.
- **CCC "len" fix:** `CCC.__len__` returned the length of the indexed stream (which read every shard, and failed on the regenerating shard 3: job 7261687); it now returns the nominal length (7.5M) and the loader checks the prefix.
- **beta-0.25 reference rows** (frozen; reset at every shift, Independent and Joint DD) for script 42: `03b --beta 0.25` jobs 7261883 (main duo) and 7261884 (ConvNeXt-T, worst_shift, Joint DD from `run_joint_dd_gated__worst_shift`).

### Nov 4, afternoon: CCC dropped

- **The author dropped CCC.** Nothing CCC was queued (generation had finished: 80/80 shards; the CCC test runs were never submitted), so no job was cancelled. Data (40 GB on /project) and code left in place, unused. The queue holds only the Joint DD searches at beta 0.25 (both duos), unaffected.
- Effect on the plan: Tier 1 ≈ 360 GPU-h, Tier 2 ≈ 125; the Sun Nov 8 and Tue Nov 10 launches are continual ImageNet-C only; the Nov 11 "CCC rule/control runs" item is gone. Freed time → Joint ETA (Tier 3) is no longer the first cut.

### Nov 4, evening: ASR and SAR added (author)

- **Standalone baseline: ASR** (Lim et al., ICLR 2026, `github.com/YonseiML/asr`, commit 2957180, no license file), cloned to `third_party/ASR` and pinned in `third_party/repos.yaml`. Its continual ImageNet-C setting (`cfgs/cin_c/eta/eta<seed>.yaml`) is ours: severity 5, 15 corruptions, 50k images each, ETA (EATA with FISHER_ALPHA 0) on SGD 2.5e-4, momentum 0.9, BS 64, ASR defaults M0 0.15, P0 5, A0 0.5, R0 0.5, R1 20. To do: run it on our 5 orders and our inputs (native 224), each member of both duos as a single model, published settings; save logits so the gate and fixed TS give an "ASR duo" row for free (as for RDumb).
- **Member method: SAR.** Only the ported SAR exists (`src/tta/ported/sar.py`, a PairRunner: Independent, episodic only); the reset pipeline runs native methods only. To do: a native `sar` method (SAM's two passes; for Joint DD the second pass reuses the gate weight of the first, so the gate's buffer sees each batch once; SAR's own model recovery off, as in the duo base setting, since our resets replace it), parity with the ported SAR, then the same tuning as ETA (anchor: the recorded ported SAR rates in `cfgs/duo_lrs/vitb16_rn50v1__indep.yaml`).

### Nov 4, night: order of business; item 1 and 2 status

- **Item 1, Independent DD tables done for both duos** (beta 0.25): main duo `docs/tables_sprint/vitb16_rn50v1/`; ConvNeXt-T (worst_shift) `docs/tables_sprint/vitb16_convnextt/` (numbers in `out/results/reset_final_sprint/vitb16_convnextt__worst_shift/`). ConvNeXt-T: rule 51.4 vs no reset 46.1 (+5.27, CI [2.93, 7.78]), vs random 45.2 (+6.18, CI [3.13, 9.58]), every 500 45.7, every 1,000 46.1, frozen 41.0, episodic 45.8.
- **Item 1, Joint DD at beta 0.25:** fixed500 done for both duos (main duo 2.26e-3 / 5.66e-3, held-out duo 60.4; ConvNeXt-T 1.75e-5 / 9.9e-5, 55.6). Rule, none and fixed1000 searches still walking for both duos (all their requested runs queued, chains alive). Then freeze → Joint test runs.
- **Item 2, ETA:** Independent pair confirmed on seeds 0–2 and recorded: ViT 5e-4 / ResNet 3.54e-4 (duo 62.31 ± 0.12, interior; single rates 5e-4 / 2.5e-4). Independent reset tuning launched: `tune_reset.py --duo vitb16_rn50v1 --tta eta_native --submit --chain` (12 continual-ladder runs 7270509–7270520, chain 7270521).
- **02f now takes `--tta`** (Tent folders, job names and anchors unchanged; a method without a recorded joint pair anchors at its Independent pair). Joint ETA pair at beta 0.25 launched: `02f --duo vitb16_rn50v1 --bn --tta eta_native --betas 0.25 --start_rungs -2 1 4 7 --submit --chain` (start diagonal widened because Tent's joint pair is 8–46x its Independent one): 7270605–7270608, chain 7270609. Then `--record`, then `tune_reset.py --signal joint --fixed_beta 0.25 --tta eta_native`.

### Nov 4, night: item 3, is Joint DD the main method? (analysis of existing runs, no new jobs)

- **Episodic test (15 shifts, 3 seeds), Tent, beta 0.25:** main duo Joint DD 58.85 vs Independent DD 52.47 (+6.4 in every seed: 6.45 / 6.20 / 6.49; better on 15/15 shifts). ConvNeXt-T duo (worst_shift) Joint DD 44.69 vs Independent DD 45.79 (-1.1 in every seed; 7/15 shifts), the mean hides huge swings: fog +49.5, frost +7.9, contrast -46.5, zoom -12.1, motion -9.2. Without the gate: +6.9 / -4.6.
- **Held-out:** episodic, main duo 62.0 vs 56.6; ConvNeXt-T 55.7-55.9 vs 55.6 (held-out never showed the ConvNeXt-T risk). Continual with resets, main duo, gate: Joint (beta 0.25, rot 0) none 59.3 / every 500 60.4 / every 1,000 59.7 / rule 59.5 (search unfinished) vs Independent none 55.1 (conservative pair) / 56.2 / 55.9 / rule 58.0. Continual test, main duo: Joint rule (beta 0, Oct 30) 56.9 vs Independent rule 54.1 (beta 0.25); Joint beta-0.25 test runs pending.
- **Mechanism: the joint signal lets the members adapt at much higher rates.** Tuned rates: main duo ViT 3.2e-3 joint vs 7e-5 independent (46x), ResNet 4e-3 vs 5e-4 (8x); ConvNeXt-T 5e-5 vs 3.1e-6 (16x) while its ViT drops to 6.2e-6 (near frozen: 40.9 vs frozen 40.4). On their own entropy those rates collapse (ViT spatter above 7e-5; ConvNeXt-T worst stream 2% at 3.5e-5, unstable at 5e-5). The partner in the duo target damps the self-reinforcing loop: even a FROZEN ResNet lets the ViT run at 1.12e-3 (Large adapt: ViT 54.9 vs 42.9 independent). The gain lands in the members: the main duo's Joint ViT alone (56.5) beats the Independent duo (52.5).
- **When it fails:** the anchor only works where the partner is decent. ConvNeXt-T duo, contrast: the ViT is weak there (frozen 18%), ConvNeXt-T at its 16x rate collapses (2.4%, vs 50.9 independent); the nnn scores of both are low (0.05 vs 0.06-0.11), the gate weight stays ~0.48, and the log pool follows the collapsed member (duo 2.9 while the ViT holds 17.7). On fog the opposite: independently both collapse (ViT 14.4, ConvNeXt-T 2.9), jointly both hold (52.4 / 65.7).
- **Missing for a clean claim:** (a) the Joint beta-0.25 continual test runs, both duos (item 1); (b) Joint vs Independent with ETA (item 2); (c) a test-time control "Independent at the joint rates" (main duo episodic: ViT 3.2e-3 and 1.12e-3 on its own entropy, 1 seed, ~2 GPU-h) to show the gain is the joint signal, not the rate.

### Nov 4, 23:40: Joint DD test runs (beta 0.25), frozen configs only

- **Queue stall:** since 19:30 none of the 13 tuning jobs (1:50 h) has started (Reason=Priority; our fair-share fell to 0.03); the 40-min 02f jobs still get through.
- **02f fix:** a one-value `--betas` grid is now a FIXED beta, never extended to 2 / 4 (`JG.plan`, self-test 3b). The Joint ETA search had started beta-2 cells (cancelled 7278243; the stored beta-2 runs are ignored). Joint ETA search done: ViT 2.5e-4 / ResNet 1.42e-3, held-out 62.64 (Independent ETA 62.31: +0.3, vs Tent's +5.4); confirmation (seeds 0-2) 7279257–7279266, chain 7279267; then `--record`.
- **Joint test runs submitted for the 4 frozen configs** (author: go), `nice=500` so the tuning goes first: main duo `joint_no_reset_tuned` (2e-4 / 1.41e-3), `joint_fixed500` (2.26e-3 / 5.66e-3), `joint_fixed1000` (8e-4 / 4e-3); ConvNeXt-T (worst_shift) `joint_fixed500` (1.75e-5 / 9.9e-5); 5 orders each = 20 runs, jobs 7279358–7279378. Config files (separate from the Independent ones): `out/results/reset_tune/sprint_joint_configs.json`, `out/results/reset_tune/vitb16_convnextt__worst_shift/sprint_joint_configs.json`. Add the rule + control (and ConvNeXt-T none / fixed1000) when frozen, then merge into sprint_configs.json for the report.

### Nov 5: proxy study (item 4, why nnn) submitted

- New: `src/reset/proxy_study.py` (self-test) + `scripts/49_proxy_study.py plan|submit|run|report`; `gate_stress.run_experiment` takes an optional `gate=` and `frozen=False` (defaults unchanged). The held-out calibration streams (Independent DD, Tent; stress S rot 0-3 + healthy H rot 0, 2; seed 0) are re-run with a `PanelGate`: the live gate is still nnn (same runs), and every proxy batch also logs nnn, msp (AC-MC), neg_entropy, atc, atc_negent, energy, dispersity, asr_concentration, marginal_entropy, mutual_info, agree_duo. ATC/energy source statistics: frozen members on the clean train slice (`source_stats.json`). The original frozen runs are reused.
- The report replays the `own_fast_vs_slow` screen per proxy, re-selecting the detector grid for each proxy and each orientation (drop / rise, the better one reported), at 0.5 / 1 / 2 false alarms per 1,000 batches; plus Spearman with acc − frozen and the response at a collapse onset. Outputs `out/results/proxy_study/<duo>/`, `docs/tables_sprint/proxy_study.tex`.
- Jobs 7281392–7281403 (12: 6 per duo, `ps_*`). When they are done: `python scripts/49_proxy_study.py report`.

### Nov 5, 15:00

- **Bad node kn159:** from 12:30 every job it got died at launch (exit 0:53, no log): all 20 Joint test runs (7279358–7279378), 9 joint-search runs, 1 calibration run of the proxy study. Added to `EXCLUDE` (src/reset/jobs.py, joint_gate_tuning.py) and the memory note. Chains resubmitted (ETA Independent; main-duo joint rule; ConvNeXt-T joint rule / none / fixed1000). Joint test runs resubmitted: 7295356–7295376, nice 500.
- **Joint ETA rates recorded** (`cfgs/duo_lrs/vitb16_rn50v1__joint.yaml`, eta_native): ViT 2.5e-4 / ResNet 1.42e-3; confirmation seeds 0-2: at beta 0.25 62.63 (at beta 0: 62.86); Independent ETA 62.31. **Joint ETA reset tuning launched:** `tune_reset.py --duo vitb16_rn50v1 --signal joint --fixed_beta 0.25 --tta eta_native --submit --chain` (7295349–7295352 + chain).
- Search progress: main-duo joint rule best 59.54 (ViT 1.13e-3 / ResNet 1.41e-3, detector (0.1, 0.01, 20, 1.5)), still searching; ConvNeXt-T joint fixed1000 best 55.78 (4.37e-6 / 7e-5), still searching.

### Nov 5, ~16:00: proxy study report (item 4)
- All 12 `ps_*` runs done; `python scripts/49_proxy_study.py report` run → `docs/tables_sprint/proxy_study.tex`, trace PNGs in `out/results/proxy_study/<duo>/`.
- Budget 1 false alarm / 1,000 proxy batches, detector re-selected per proxy and orientation, seed 0 only. Delay (proxy batches, misses penalised): main duo dispersity 43, marginal entropy 54, **nnn 87 (3rd)**, mutual info 89, AC-MC 95, neg. entropy 96; ConvNeXt-T duo **nnn 37.7 (1st)**, dispersity 37.9, mutual info 38.6, ATC 40.3 (a tie among the top 3). Miss rates are high everywhere (0.38-0.78).
- What it supports: the confidence proxies (AC-MC, neg. entropy, ATC, energy) only detect collapse in the FLIPPED orientation (they rise as a member collapses; in the natural orientation delay ≈ 350 = never), so as a gate score they would favour the collapsing member. The batch-diversity proxies (nnn, dispersity, marginal entropy, MI) fall at a collapse onset (70-83% of onsets) and correlate positively with acc − frozen. What it does NOT support: nnn over dispersity; on the main duo dispersity detects about twice as fast at a similar false-alarm rate.
- Open: 16 / 45 degraded stretches from one seed; a seed-1 rerun (`--seeds 0 1`) would tighten it. Whether to swap the detector to dispersity is the author's call (the rule was tuned on nnn).

### Nov 5, evening: proxy study re-framed as correlation with accuracy (author's call)
- The proxy section comes before the reset rule, so its main evidence is now the **correlation with true accuracy** on the Independent DD held-out streams (both members adapting; healthy H + collapsing S), not the detection delay (that screen moves to the reset section). Tuned-rate streams: NOT used (author).
- `src/reset/proxy_study.py`: `correlations()` (within-stream Spearman, mean per regime: gap = (s_L - s_S) vs (acc_L - acc_S), and each member's own; streams where nnn's gap rho >= the proxy's) + `correlation_latex()`; self-test extended. `scripts/49_proxy_study.py report` writes `docs/tables_sprint/proxy_correlation.tex` and `<duo>/correlation.csv`. Healthy streams now at every rotation (`HEALTHY_ROTATIONS = 0..3`; rotations 1, 3 make their own frozen run). Third duo `convnextb_deits` in the defaults. `submit` now passes `--qos=high` (it did not before) and takes `--nice`.
- **Placeholder table** rendered from the current streams (2 H + 4 S per duo, seed 0). Main duo: nnn best gap rho in both regimes (H 0.67, S 0.97); MI 0.64 / 0.96; dispersity 0.24 healthy. ConvNeXt-T: nnn, MI, dispersity tied (H 0.83-0.86, S 0.93-0.94). Confidence proxies: per-member rho 0.83-0.96 on healthy streams but NEGATIVE on collapsing ones (gap -0.34..-0.72).
- Jobs 7298611-7298622 (12, nice 500): H rot 1, 3 for both duos; convnextb_deits S rot 0-3 + H rot 0-3. When done: `python scripts/49_proxy_study.py report` (the table's caption counts the streams itself).
