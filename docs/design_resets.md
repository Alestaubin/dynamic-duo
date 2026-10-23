# Design note: reset rules for Dynamic Duos (Phase 4, stage R-A)

> **Scope note, 2026-10-23.** The author narrowed the experiments to the continual stream only (the 15 ImageNet-C corruptions concatenated); the recurring streams below are no longer run, and the lengths and hours of section 4 are superseded by the run list in plan.md, Phase 4.

2026-10-20. Written for the author, before any code of stages R-B to R-E. Nothing was run on a GPU, no job was submitted, no CCC data was generated, and no existing file was changed except `plan.md` (Phase 4 rewritten to the brief).

Everything below was read from the code as it is today (paths are relative to the repo root). Section 0 lists what needs your decision; sections 1 to 5 are the four items of the brief.

## 0. Decisions needed, and facts that differ from the brief

| # | Point | What I propose |
|---|---|---|
| D1 | **`results/selection_section45.json` does not exist yet.** Phase 3's Stage E is still open (`08_select_proxy.py` has not written it; `09`/`10` not launched). The brief reads proxy, b_t, beta and c from it. | Stage R-B needs none of it. R-C needs the proxy, b_t, beta (for the gated-duo columns) and the runner-up proxy. The selection code has **no centering constant**, so c = 0 throughout. |
| D2 | **eps of s_t.** The brief clips r to [1e-6, 1 - 1e-6] before the logit; the live gate (`JointProxyWeighted`, `dd_eval.gate_weights`) clips to [1e-3, 1 - 1e-3] (author's decision 2 of Phase 3), which bounds s_t at +-6.9 instead of +-13.8. | The **gate keeps 1e-3** (it is the selected Vanilla DD). The **detectors use the brief's 1e-6**. R-C stores the raw r_t, so either can be applied offline and the effect of eps can be reported. |
| D3 | **There is no live Vanilla DD runner.** Vanilla DD is an offline replay of cached logits (`runner.replay`, `dd_eval`); the only live gates are `evaluate_live` (CCC, no per-batch hook) and `joint_dd.run_joint_dd` (episodic: it calls `runner.reset()` and `gate.set_corruption` at every stream). | A new live loop in `src/reset/live.py` serves both (section 1). It reuses `MemberRunner`, `PureProxyGate`, `RecordingCalibrator` and the stream loaders unchanged. |
| D4 | **Order of the 15 corruptions.** The brief says "a fixed order". `cfgs/protocols.yaml` has CoTTA's 10 orders (the repo's `continual` protocol uses one); `cfgs/global.yaml` lists the 15 grouped by family. | **CoTTA order 0** for both the recurring and the continual stream, so round 0 of the recurring stream *is* the continual stream (a free consistency check). Tuning stream: the order of `tuning_corruptions` (speckle, gaussian blur, spatter, saturate). |
| D5 | **Recurring rounds: same image order or reshuffled?** With the same loader seed every round is bit-identical data (period 750,000 images). | **Reshuffle**: round r uses loader seed `seed + 1000 r` (round 0 uses `seed`, so it equals the existing episodic/continual order). Say if you want exact repetition instead. |
| D6 | **Stream lengths for screening** (section 3.4): full recurring/CCC streams are 7.5M images, about 17 L40S-hours each. | Screen on **2M images** of each CCC stream, **R = 3** rounds (2.25M) of the test recurring stream, **R = 5** (1M) of the tuning stream, and the full continual stream (0.75M). About 23 L40S-hours in total against 76 for the full lengths. |
| D7 | **CCC variant.** 27 streams exist (3 difficulties x 3 speeds x 3 seeds). | One speed and seed per difficulty: **speed 1000, seed 43** (index 0 of `generate.py`'s table). |
| D8 | **How to reset one member.** `TTAMethod` has only `reset()` (all sides); the private `_reset_side(side)` does exactly what is needed. Phase 3's list already has "public `reset_side(side)`" as a Phase 4 item. | `src/reset/actions.py::reset_member(runner, side)` calls `runner.tta._reset_side(side)` (`MemberRunner`) or `runner.members[side].reset()` (`PairRunner`). **No existing file is edited.** If you prefer a public `TTAMethod.reset_side`, it is a 4-line additive change. |
| D9 | **Environment.** The brief names the virtualenv `ddenv`; it does not exist here. `slurm/job.sh` activates `~/py38`. CLAUDE.md says `~/dduos` (also absent). `~/py38` has no `webdataset`, `Wand`, `cv2`, `numba`, which `third_party/CCC/generate.py` imports (all four exist as Alliance wheels; the opencv one is a stub for the cluster module). | I assume `~/py38` is the environment. CCC generation needs those packages: a separate venv, or your OK to install into `~/py38`. I installed nothing. |
| D10 | **Scale of Stage R-E.** The brief runs 6 conditions (no reset, fixed schedule, oracle, 3 rules) + Joint DD on 5 long streams x 3 seeds. At full length that is about 850 L40S-hours (section 4.5). | Decide the live lengths after R-D, when the collapse times are known. |
| D11 | **Two regimes (answered 2026-10-20).** Continual runs use both indep (Vanilla DD) and joint (Joint DD) duos, with different learning rates, kept in `cfgs/duo_lrs/` (section 6). | Every run names its regime; the rates are read from the regime's file; a pending joint entry is an error. |

## 1. Where the reset hooks go

### 1.1 What exists

| Piece | File | What matters for resets |
|---|---|---|
| Members, one batch at a time | `src/experiments/members.py::MemberRunner` (`step_tensors`, `iter_batches`) | `iter_batches` never resets (its docstring says the caller decides). `reset()` calls `tta.reset()` = all adapting sides. Returns the **pre-update** logits of each step. |
| Tent | `src/tta/methods/tent.py::TentMethod` | Per-side state already: `_optimizers[side]`, `_snapshots[side]` (a deepcopy of the model and optimizer state dicts taken at `setup`, before any update). `_reset_side(side)` = `load_model_and_optimizer(...)`. |
| Ported members (ETA, EATA, SAR) | `src/experiments/algorithms.py::PairRunner` | `members[side].reset()` is per member already. Not needed for the Tent task. |
| Vanilla DD gate | `src/experiments/joint_dd.py::PureProxyGate` (a `JointProxyWeighted`) | Fills a proxy bucket of b_t samples across adaptation batches; on each flush appends a record `{r_l, r_s, x_l, x_s, w_l, ...}` to `gate.records`. The members never see it. |
| Joint DD | `joint_dd.run_joint_dd`, `RecordingCalibrator` | The gate is the `loop_calibrator` of a `both_duo` `MemberRunner`; its weight is a constant of each step's loss. |
| Streams | `protocols.build_segments`, `ShiftDataset.loader` | `Segment(label, dataset, reset_before)`; loaders are seeded (same seed, same batch sequence). `reset_before=False` for every segment after the first in the continual protocol; a CCC stream is one segment. |

### 1.2 The loop (`src/reset/live.py`, one loop for Vanilla DD and Joint DD)

```
for part in stream:                                  # StreamPart(segment, loader seed, round, position)
    gate.mark_segment(part.label)                    # label only: filters, buffer and weight kept (no reset, no boundary signal)
    for x_l, x_s, y in loader(part):                 # PairCollate: both members' inputs, one JPEG decode
        z_l, z_s = runner.step_tensors(x_l, x_s, ds)        # adapts (pre-update logits)
        if frozen: fz_l, fz_s = frozen.step_tensors(x_l, x_s, ds)    # evaluation only (R-C, oracle trigger)
        Vanilla DD: z_duo = gate.calibrate(z_l, z_s)        # no grad, outside the loop
        Joint DD:   z_duo = rec.last                        # the gate already ran inside step_tensors
        for record in gate.records[seen:]:                  # proxy batches completed by THIS adaptation batch
            for side, rule, stat in controller.update(record.r_l, record.r_s):
                reset_member(runner, side)                  # before the NEXT batch
                log(position, side, rule, stat)
        metrics / summaries update
```

* **When a decision lands.** A proxy batch flushes inside `calibrate`, so its decision is known at the end of the adaptation batch that completed it, and the reset applies before the next batch. For b_t a multiple of the adaptation batch (64; the evaluated grid is 64 to 2048) the bucket ends exactly on a batch boundary inside a segment, so there is no lag at all. Across a segment boundary the 16-sample tail batch of each 50,000-image stream shifts the alignment (50,000 = 781 x 64 + 16): the bucket then completes mid-batch, the rest of that batch is scored by the pre-reset member, and the next bucket's partial buffer holds those few pre-reset samples. This is the gate's existing rule (`dd_eval.plan_buckets`), costs at most one batch of lag, and I will log it rather than special-case it. It is the same on the CCC stream only at its very end (7.5M = 117,187 x 64 + 32).
* **Buckets cross corruption boundaries** in these streams: the gate is not told where a shift happens (`mark_segment`, not `set_corruption`). R-C records the segment id of each bucket's last sample and the share of its samples that come from that segment.
* **The gate is not reset when a member is.** With the filter `none` the gate's state is its bucket buffer and the last weight, both functions of the past logits; after a reset the weight is stale for one bucket, which is the honest behaviour of the method. (`JointProxyWeighted._pending`, the double-assimilation guard of `calibrate_with_grad` -> `calibrate`, is untouched: a reset never calls the calibrator.)
* **Joint DD.** Same loop with `mode="both_duo"` and the gate as loop calibrator, as `run_joint_dd` builds it, but without its per-stream `runner.reset()` / `gate.set_corruption`. `reset_member(runner, "large")` while the duo loss is joint is well defined: it restores one member's norm parameters and optimizer state; the other member's gradient path is unchanged.
* **The controller never sees labels or frozen logits.** `ResetController.update(r_l, r_s)` takes the two raw proxy values of the bucket. The oracle trigger (R-E) is a separate object that takes the frozen copies' accuracies and is labelled as a ceiling in every output.

### 1.3 The frozen copies (evaluation only)

A second `MemberRunner(mode="no_adapt", bn=True, models=load_models(...))`: separately loaded weights (so nothing is shared with the adapting pair), ViT in `eval()`, ResNet in `configure_model_frozen` (BatchNorm on the **test batch's** statistics: the brief's setting and the `--bn` convention of decision 14, so the frozen and adapted ResNet see the same 64-image batches). It receives the **same preprocessed tensors** as the adapting pair (`step_tensors` accepts them), so it costs two forward passes and no extra decoding; about +0.5 GB of GPU memory. Its outputs go only to the summaries. A test (R-B) asserts that the adapting pair's logits are bit-identical with and without the frozen copies attached.

## 2. Restoring one member without touching the other

`TentMethod._reset_side(side)` is `model.load_state_dict(snapshot, strict=True)` and `optimizer.load_state_dict(snapshot)` on that side's model and optimizer only:

* **Parameters.** The snapshot is a deepcopy taken in `setup()` before the first update, i.e. the source weights. `load_state_dict` copies values in place, so the parameters equal the source bit for bit; `requires_grad` flags are not touched. Only the norm affine parameters change during Tent, but the whole state dict is restored. BatchNorm has `running_mean = None` under Tent (`configure_model`), so there is no running statistic to restore.
* **Optimizer.** The tuned settings are SGD with momentum 0.9 (`cfgs/duo_lrs/vitb16_rn50v1__indep.yaml`, formerly `cfgs/tuned/tent__*.yaml`). The snapshot of the optimizer is the *empty* state (SGD creates its momentum buffer lazily at the first step), so loading it deletes every buffer, and the first update after a reset is exactly a fresh run's first update. R-B test 2 checks both "state empty or all-zero" and, stronger, that the next step after a reset is bitwise the first step of a fresh member. (Adam would be the same: its `exp_avg` / `exp_avg_sq` / `step` appear at the first step.)
* **Independence.** The two sides own different `nn.Module`s and different optimizers; `_reset_side` touches nothing else. Method-level state shared between sides (EATA's moving-average probabilities, SAR's EMA) is not a Tent concern; `PairRunner` members keep their own, so the same `reset_member` dispatch works for ETA/EATA/SAR later. Tests 4 and 5 of R-B are the proof for Tent.
* **Cost.** A restore copies about 100M floats on the GPU (milliseconds); the snapshot lives on the device (about 0.4 GB).
* **What is not part of "source".** The DataLoader order, the gate's buffer and the detectors' state: the last is the controller's job (restart + cooldown), the first two are deliberately untouched.

## 3. Streams

All streams use the existing `ShiftDataset` loaders (ImageNet-C and CCC images are 224x224 and bypass the models' Resize, `native224`). Batch 64 (so 50,000 images = 781 batches of 64 + one of 16). Proxy batches below are at b_t = 128, the gate default, as an example; the selection will fix b_t.

### 3.1 Recurring ImageNet-C

`recurring(corruptions, rounds=R, severity=5, seed)`: the corruption list in a fixed order, repeated R times, one loader per (round, corruption) with seed `seed + 1000 r` (D5), the members reset **once**, at the start. Round 0 is the continual stream. Implementation reuses `ImageNetC.from_config`; the stream cursor knows its absolute position, so a checkpoint can resume mid-segment (the permutation is drawn from the seeded generator and sliced; a self-test asserts that resumed batches equal the uninterrupted ones, and that round 0 equals `build_segments(RunSpec(protocol="continual", ...))` batch for batch).

| Stream | Images | Adaptation steps | Proxy batches (b_t = 128) |
|---|---|---|---|
| Tuning, recurring (4 held-out), one round | 200,000 | 3,128 | 1,563 |
| Tuning, R = 10 (brief's default) | 2,000,000 | 31,280 | 15,625 |
| Tuning, R = 5 (screening) | 1,000,000 | 15,640 | 7,813 |
| Test, recurring (15), one round = continual | 750,000 | 11,730 | 5,860 |
| Test, recurring, R = 10 (brief's default) | 7,500,000 | 117,300 | 58,594 |
| Test, recurring, R = 3 (screening) | 2,250,000 | 35,190 | 17,578 |

(`plan.md` decision 9 had proposed about 1M images of random-order cycles for tuning; R = 5 of the fixed order is the same length. The brief asks for a fixed order.)

### 3.2 Continual ImageNet-C

`recurring(..., rounds=1)` with CoTTA order 0: one pass, 750,000 images, 11,730 steps. This is `RunSpec(protocol="continual")`; reusing it means the new no-reset run can be compared with any existing continual cache.

### 3.3 CCC Easy, Medium, Hard

Generated locally with `third_party/CCC/generate.py` (the RDumb repository's generator; the hosted stream is gone). `--baseline 40 / 20 / 0` give Easy / Medium / Hard; `--processind` selects speed and seed (`processind % 9`: index 0 = speed 1000, seed 43; workers of one stream use `processind = 9 j`, `--totalprocesses = 9 w`). What the code does, read from `generate.py`:

* A **walk** over a 21 x 21 grid of fractional severities of two corruptions (n1, n2), found from the published accuracy matrix (`ccc_accuracy_matrix.pickle`, downloaded once from a Tuebingen server) so that the baseline accuracy stays at the target; when a walk ends, a new random pair starts. Each walk step is `speed` images. Pools: Hard = gaussian / shot / impulse noise + contrast (4); Medium = 13 (the 15 test corruptions minus brightness, jpeg); Easy = 14 (adds jpeg). None of the 4 tuning corruptions is in any pool.
* Each step draws `speed` random validation images (`os.walk` of the 50,000-image directory, shuffle, take `speed`), applies `RandomHorizontalFlip` + `RandomResizedCrop(224)`, the two corruptions, and saves a JPEG at quality 85 into 25,000-image webdataset shards (26,000 at speed 2000).
* **Length:** 7,500,000 images per stream = 300 shards (289 at speed 2000), 117,188 steps, 58,594 proxy batches at b_t = 128. 7,500 walk steps at speed 1000.
* The stream is ordered (`CCC.loader` refuses `shuffle`), has no corruption labels in the batches, and supports a truncated read (`num_samples`) natively, but only the shards up to the truncation must exist.

**Cost of generating CCC.** *Unmeasured.* I could not benchmark it: `~/py38` lacks the generator's dependencies (D9) and I did not install anything. What can be said from the code and from ImageNet-C's own files:

* **Storage** (JPEG-85 with `optimize`, as in ImageNet-C). Mean size of a severity-5 ImageNet-C file (200 sampled): gaussian noise 35.8 KB, frost and snow 14.8, elastic 12.2, pixelate 7.5, fog 7.2, glass blur 6.1, contrast 2.0. CCC mixes two corruptions at severities 0 to 5, so about 20 to 30 KB per image for Hard (noise-dominated), 10 to 15 KB for Medium and Easy: **roughly 220, 90 and 65 GB per full stream, about 375 GB for the three** (range 250 to 500). For the 2M-image screening length: about 100 GB for the three. Files: 300 per stream, so the inode quota is not a concern. `/scratch` has 854 GiB free and `/project` 940 GiB (but `/project` is at 81% of its file quota and shared); I would put CCC on `/scratch` next to the logit cache and check its purge policy.
* **CPU time** (the generator is CPU-bound; one decode of a full-size validation JPEG, a crop, two corruptions, one JPEG encode per image). The corruptions differ by 10x: noise, contrast and pixelate are cheap, `glass_blur` (numba), `zoom_blur` (scipy), `motion_blur` and `snow` (ImageMagick through Wand) are not. My guess is 0.05 s per image for the Hard pool and 0.2 s for Medium and Easy (off by 2 to 3x either way): **about 100 CPU-hours for Hard and 400 each for Medium and Easy** per full stream, i.e. about 900 CPU-hours (110 8-core node-hours) for the three; **about 250 CPU-hours (30 node-hours) for 2M images each**. Workers split a stream by shard, so a 16-task array finishes the truncated set in about 2 hours of wall-clock once running.
* **Other costs.** The validation images are re-read at random 150 times over a stream (`os.walk` of 50,000 files per walk step, 7,500 steps at speed 1000): staging the validation set to `$SLURM_TMPDIR` per task is worth doing (6.4 GB). The accuracy-matrix pickle (one-off download) and the frost textures (`generate.py` downloads them from GitHub; `third_party/robustness` has them) need internet or a local copy: do these once on the login node. A worker count changes which random images fill a step (numpy RNG consumption differs between workers) but not the walk (the `random` module is only used for the walk), so the number of workers must be recorded with the stream.
* **A thin wrapper is needed for truncation** (`scripts/ccc_generate.py`, already planned for Phase 8): `GenerateDataset` has `cutoff = 7500000` as a local constant. The wrapper executes the upstream source with that constant replaced by `--max_images`, so no upstream file is edited, and a toy test asserts that the truncated shards equal the first shards of the unmodified generator.
* **First step if you agree:** one shard (25 walk steps of 1,000 images) per pool, measured in a short CPU job, gives the real CPU-hours before any array is launched.

### 3.4 Truncated length for screening

What screening measures (R-D): delay from a collapse onset to the first alarm, misses, and false alarms outside degraded stretches. Without resets each member collapses **once** and stays collapsed, so the informative part of a stream is the time before the collapse plus a margin after it; the long collapsed tail adds only cost (and false alarms cannot occur there by definition). I do not know the collapse times of these two members (the existing runs are 50,000-image episodes; RDumb reports Tent collapsing on CCC well before the end of the stream, but I have not checked by how much, and our learning rates differ).

So: **2M images per CCC stream** (31,250 steps, 4.6 h), **R = 3 of the test recurring stream** (2.25M), **R = 5 of the tuning stream** (1M), the **full continual stream** (0.75M). R-C runs the tuning stream first (it is the cheapest: 2.3 h) and reports when each member collapses; if the collapse is late, the other streams are extended by resuming their checkpoints (the runs are resumable, section 4.3), not restarted.

## 4. Compute and storage for the R-C runs

### 4.1 Throughput

Measured on an L40S (plan.md, "Compute and storage"; Stage D): both members adapting with Tent, 171 to 202 images/s end to end; both members frozen, forward only, 471 images/s. The frozen copies reuse the adapting pair's batches, so their cost is at most the second figure: 1/202 + 1/471 = 7.1 ms to 1/171 + 1/471 = 8.0 ms per image, 125 to 141 images/s. The proxy library on the gate's buckets (14 proxies x 2 members; timed on CPU for a 128-sample bucket: `nnn` 41 ms, `dispersity` 8 ms, `cot` 4 ms, the other eleven under 1 ms each, about 56 ms per member) adds at most 0.11 s per bucket against about 1.1 s to process a 128-image bucket: **about 10%, less on the GPU**, to be measured in the R-C smoke test. **Planning rate: 120 images/s** (fp32, as in the rest of Phase 3; TF32 would be +38% but breaks bitwise comparability with the existing caches, decision 11).

### 4.2 Run list (all Vanilla DD, Tent, seed 0, both members adapting independently; ViT lr 5e-5, ResNet lr 2.5e-4, SGD momentum 0.9, the tuned files)

| Run | Stream | Images | Steps | L40S-hours at 120 img/s |
|---|---|---|---|---|
| T1 | Tuning, recurring, R = 5 | 1.0M | 15,640 | 2.3 |
| X1 | Test, recurring, R = 3 | 2.25M | 35,190 | 5.2 |
| X2 | Test, continual, order 0 | 0.75M | 11,730 | 1.7 |
| C1, C2, C3 | CCC Easy, Medium, Hard (speed 1000, seed 43), first 2M | 2.0M each | 31,250 each | 4.6 each, 13.9 |
| | **Screening set** | 10.0M | | **23.1** |
| | *Full lengths instead: T R = 10, X R = 10, CCC 7.5M each* | *32.75M* | | *75.8* (4.6 + 17.4 + 1.7 + 3 x 17.4) |

Each run is one sequential job (a no-reset stream cannot be sharded in time, unlike the episodic runs); runs are independent of each other, so the six run in parallel.

### 4.3 Jobs: resumable, and which partition

Killarney L40S partitions: `b1` 3 h (168 nodes), `b2` 12 h (126), `b3` 24 h (84), `b4` 3 days. The queue is congested today and only short jobs start promptly (memory note on queue waits), and a 5-hour job does not fit `b1`. So each run **checkpoints every ~10 minutes** (both members' adapted state and momentum, the stream position, the gate's buffer and last weight: under 2 MB, since only the norm layers adapt; the summaries are appended to the run's file in chunks) and is resumable by resubmitting. Two ways to run it, your choice: **(a)** one job per run with a limit near the estimate (T1 2.3 h -> `-t 03:00`, `b1`; X1/C1-C3 4.6 to 5.2 h -> `-t 06:30`, `b2`); **(b)** chained 3-hour jobs (`--dependency=afterok`), which stay in the biggest partition but pay a queue wait per link. I recommend (a) for T1, X2 (`b1`) and (b) only if the `b2` wait turns out to be long.

### 4.4 Storage

Per-sample correctness is one byte (a bit per series: adapted L, adapted S, frozen L, frozen S, gated duo, fixed held-out TS duo, and one per extra gate configuration), so **7.5 MB for a 7.5M-image stream** and every later bucket size is a sum over it (no information lost, no dependence on b_t). Proxy scores need the logits, so they are stored per bucket: 14 proxies x 2 members x float32 plus metadata (segment id of the last sample, share of the bucket in it, stream position) = about 150 bytes per bucket, at b_t = 128 **8.8 MB per 7.5M-image stream** (the b_t of the selection and of the runner-up proxy, if different, are the only ones computed; this is where the compute overhead comes from). Gate records (r, w) are the same rows. In all about **16 MB per full 7.5M-stream, about 22 MB for the six screening runs (10.0M images; 71 MB at full lengths), as 6 files** (one `.npz` per run) plus 6 small checkpoint files. Full logits (both members, centred fp16, the format of `logit_store.py`): 3.0 GB for the continual stream (750,000 x 1000 x 2 B x 2) and 0.8 GB for one round of the tuning stream, which are worth keeping (they allow any gate to be replayed offline on the exact no-reset trajectory); 30 GB for a 7.5M stream, which I would not keep. CCC shards: section 3.3.

### 4.5 The live runs of R-E, for scale

Without the frozen copies a Tent step is 1/202 s per image, except the oracle trigger which needs them. Per condition and seed the five long streams are 7.5M x 4 + 0.75M = 30.75M images, **42 L40S-hours**; seven conditions (no reset [reusable from R-C for seed 0], fixed schedule, oracle, three rules, Joint DD with the best rule) x 3 seeds is **about 850 L40S-hours at full length** (the oracle's frozen copies add a little), about 260 at the screening lengths of section 3.4 (9.0M test images per condition and seed, 12.4 hours). This is the reason to fix the lengths after R-D, and the Joint DD part (which needs the joint-tuned learning rates of `02b_tune_joint_lr.py`, not Tent's independent ones) is a separate approval.

## 5. What R-B will build, and how the checks map

* `src/reset/detectors.py`: `Level`, `ImmVsEma`, `FastVsSlow`, `Slope`, `Cusum`, `RunningStd`, a `Combined` (OR) wrapper; `update`, `restart`, `state_dict` / `load_state_dict`; the synthetic-sequence tests of the brief in the module's `__main__` (no pytest here).
* `src/reset/controller.py`: per-member rule, per-member cooldown, the partner `gap` restart, the reset log; a pure function of the proxy stream, shared **unchanged** by the live loop and the offline replay (so the replay of R-D is the live rule minus the trajectory change, by construction).
* `src/reset/actions.py`: `reset_member`.
* Equivalence tests on tiny models (the repo's `_testing` helpers) on CPU, bitwise: 1 (never-firing rule = no-reset run), 2 (source parameters exactly, optimizer state empty, next step = a fresh first step), 3 (**key test**: forced reset of both members at every boundary of a continual stream = the episodic run, same seed; the loader seed is the same in both because `iter_batches` takes one seed for every segment), 4 (one member's reset leaves the other's parameters and optimizer state unchanged), 5 (both members fire on one proxy batch: each ends as after its own reset alone, in either order). An extra GPU smoke check: round 0 of the new loop on a real tuning stream reproduces the first stream of the existing Tent logit cache.

## 6. Addendum (2026-10-20, after the author's answer): two regimes, two sets of learning rates

The continual setting is tested on the two main kinds of duo: **indep** (`both_indep`: each member on its own entropy; Vanilla DD, the gate outside the loop) and **joint** (`both_duo`: both members on the entropy of the combined prediction; Joint DD, the gate inside). They use **different learning rates**, so every run states its regime and reads its rates from one file per duo and regime, never from `cfgs/tuned/` by default:

* `cfgs/duo_lrs/<duo>__indep.yaml` and `cfgs/duo_lrs/<duo>__joint.yaml`, for every duo of `cfgs/duos/` (14 files), are **the only files that hold learning rates** (since 2026-10-21 `cfgs/duos/` names the models only and `cfgs/tuned/` is gone). Each lists, per TTA method that was tuned, both members' complete OPTIM block with its provenance. `src/utils/duo_lrs.py` reads them, through `load_duo(duo, g, tta=, regime=)` and `Context.create(..., regime=)` (the OPTIM block is part of every cache key, so it is complete and was compared with what `cfgs/tuned/` gave before the move).
* **indep** files are written by the tuning scripts (`02`, `05`: the winner is recorded in every duo that uses the model) and also hold the `defaults` a member runs with when nothing is tuned for a method. For `vitb16_rn50v1`: Tent ViT 5e-5 / ResNet 2.5e-4, ETA 5e-4 / 2.5e-4, EATA 5e-4 / 5e-4, SAR 5e-3 / 1e-3 (all SGD, momentum 0.9). A method appears only if both members have a tuned file; the reason is written under `not_listed` otherwise (e.g. `vitb16_rn50`: its ResNet-50 is the V2 model, tuned on the old pipeline, so nothing is listed for it).
* **joint** files hold the pairs of `scripts/02b_tune_joint_lr.py`. Today **`vitb16_rn50v1` / Tent is `pending`** (the 02b run is not finished): asking for it raises `NotTuned`, it never falls back to the indep rates. When 02b is done: `python -m src.utils.duo_lrs --import_joint out/results/tune_joint_lr/vitb16_rn50v1__tent_bn/results.json`. Only Tent has a joint mode (the ported ETA, EATA and SAR run their members independently), so only Tent can ever be listed there.
* **A caveat to decide on:** 02b tunes the pair for the *static* duo (fixed held-out temperatures, `(z_L/T_L + z_S/T_S)/2`), the Joint mode of script 03. Joint DD puts the gate in the loop (the weight varies per proxy batch), so these pairs are a good starting point for it, not a tuned optimum. I use them as they are unless you say otherwise.

**Consequence for the runs of section 4.** Each stream of R-C is run **once per regime**: the Vanilla DD runs with the indep rates (Tent: ViT 5e-5, ResNet 2.5e-4: available), the Joint DD runs with the joint rates (pending 02b; they also need Joint DD's gate: proxy, b_t, beta, from the Stage E selection and `10_select_joint_dd_beta.py`). The screening set is therefore 23.1 L40S-hours per regime, **46.2 for both** (a joint step costs about the same as an independent one: both forward passes, one backward through both members). R-C starts with the indep regime and the tuning stream; the joint runs follow when their rates and gate exist. A reset rule is screened on the proxy stream of its own regime (the members of a joint duo hold each other up, so their proxies behave differently); the matched-budget choice of R-D is therefore made per regime.

## 7. R-B: what was built, and what it showed (2026-10-20)

`src/reset/{detectors,controller,actions,live}.py`, every module with its `__main__` self-test; `python -m src.reset.<module>`.

* **Detectors.** The five procedures as streaming objects (`update`, `restart`, `state_dict`/`load_state_dict`; JSON round trip tested). Choices the brief left open: the scale used to judge x_t is the running std *before* x_t enters it; `level` in noise units is (x_t - warm-up mean) / scale < tau; `slope` has no alarm before W values exist (effective warm-up max(W0, W)); CUSUM keeps its state after an alarm until restarted.
* **Sign of c.** The brief writes g^(m) = s^(m) - s^(m') - c for both members. With one c that is not antisymmetric (a constant offset between the members cannot be removed from both gaps at once), so c is the offset of s_L - s_S, the way the gate uses it: g^(large) = s_L - s_S - c and g^(small) = s_S - s_L + c. With c = 0 (today's selection) the two readings coincide. Tested both ways.
* **Partner-gap restart.** In the closed-loop test (the large member collapses, its rule resets it, it recovers; the small member never changes) the partner's gap detector with the restart never fires, and CUSUM (reference 0) and `level` (reference = the healthy warm-up mean) never fire without it either. `imm_vs_ema`, `fast_vs_slow` and `slope` **do fire without the restart, but only on raw (unstandardized) gaps**: with the default standardization the running std grows 2 to 3 times within a few batches of the jump and absorbs most of the spurious drop (0 flags with or without the restart in the test). So the restart is decisive for unstandardized rules and cheap insurance for standardized ones; it also resets the standardizer, which would otherwise stay inflated for about 100 batches (alpha_var = 0.01).
* **Equivalence tests** (tiny models on the CPU, bitwise, both `both_indep` and `both_duo`): (1) a rule that never fires reproduces the no-reset run, and the loop itself equals `MemberRunner.iter_batches`; (2) after a reset the parameters equal the source and the optimizer state is empty, and the next step is a fresh run's first step; (3) **forced resets of both members at every boundary of a continual stream reproduce the episodic run**, members and duo prediction, for Vanilla DD and for Joint DD with the gate inside the loop (the gate is re-initialised at the boundaries too, as the episodic run does); resetting one member only, or none, does not reproduce it; (4) resetting one member leaves the other's parameters, optimizer state and gradients untouched; (5) both members reset on one proxy batch end as after their own reset alone, in either order. Also: a controller's resets are exactly the scripted resets at the same steps.
* **Not covered yet:** a real-model GPU check (round 0 of the new loop against the first stream of the existing Tent logit cache) and the `PairRunner` path (ETA/EATA/SAR have no loader in `run_stream`); both belong to the R-C smoke test.

## 8. Not done, on purpose

No code, no job, no CCC generation; no existing file edited beyond `plan.md`. The estimates of section 4 use measured rates from earlier phases, not a measurement of this loop; the CCC generation cost (3.3) is a guess from the code and is marked as one.
