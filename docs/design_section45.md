# Design note, Stage A: episodic Dynamic Duos experiments (paper Section 4.5)

Written for the author, 2026-10-18. This is Stage A of the brief in `plan.md` (Phase 3): reading and design only.
Nothing was implemented, nothing was submitted to SLURM, no existing file was changed except `plan.md`.

**What I read.** `src/calibrators/joint_proxy_weighted.py`; in `src/experiments/`: `gates`, `evaluate`, `runner`, `members`,
`cache`, `specs`, `protocols`, `tuning`; `src/tta/methods/*`; `scripts/02_tune_tta.py`; `src/utils/datasets/*`;
`src/utils/model.py`; `slurm/job.sh`; `src/reliability/proxies/*`; and the pinned upstream checkouts in `third_party/`
(`test-time-adaptation`, `EATA`, `SAR`, `cotta`). Line numbers below are for the working tree of 2026-10-18.

**What I could not check.** The paper text (so I cannot say which gate semantics it describes), COCA's code (not in the
repo), GPU run-to-run determinism, and the free space of `$SLURM_TMPDIR` on a compute node.

## 0. The five answers in short, and what I need from you

| Question | Short answer | Section |
|---|---|---|
| 1. Gate semantics | Neither option. The live gate is a hybrid: the weight of proxy batch t is applied only to the slice that completes the batch; the earlier slices of the same batch keep the weight of batch t-1. | 1 |
| 2. Which logits are scored | Tent, ETA, EATA, SAR, DeYO: the first forward pass, before the update (as wanted). CoTTA returns the EMA teacher's logits, RMT the sum of student and teacher logits, ROID logits multiplied in place by a class prior (about 1/K per class). These three need a decision. | 2 |
| 3. Where to hook caching | `ensure_members` / `_store_segment` in `src/experiments/runner.py` is already the right place. A new per-member store plus a per-member algorithm adapter, and nothing existing changes. | 3 |
| 4. How NNN is computed | Softmax of the raw logits, no temperature. Normalization `nuc / sqrt(n * min(n, K))`, where n is the proxy batch size. Kept as it is. | 4 |
| 5. Which base algorithms exist | In the repo: Tent and ETA (`eata`, without the Fisher term). In the pinned upstream benchmark: ETA, EATA, SAR, CoTTA, RMT, ROID and DeYO. Not available anywhere in the repo: COCA. | 5 |

Decisions I need before Stage B (details in section 7). Items 1 to 3 of the Phase 3 list in `plan.md` are still open too.

1. **Gate semantics:** keep the live hybrid as the definition of Vanilla DD?
2. **Clip eps:** the live gate uses 1e-3, the brief says 1e-6.
3. **Temperature in the proxies:** NNN stays raw, the brief wants the others on softmax(z / T_m).
4. **Scored logits of CoTTA, RMT and ROID.**
5. **Temperatures per base algorithm:** refit them, or reuse Tent's?
6. **Storage design:** a per-member store with a shared key, centered fp16 plus the row maximum, and where it lives.
7. **Copy to persistent storage after each stream**, not only at job end.
8. **Learning-rate grid re-centring:** what "same spacing" means (CoTTA and RMT publish 1e-2).
9. **RMT needs labelled source data** (a 50,000-sample warm-up and source replay).
10. **Input convention for upstream methods:** [0, 1] pixels with normalization inside the model.
11. **Random augmentations:** one generator per member.
12. **COCA:** where does its code come from?

## 1. Gate semantics (question 1)

### 1.1 What the live code does

`JointProxyWeighted._forward` (`src/calibrators/joint_proxy_weighted.py:453-506`) receives one adaptation batch of
`BS = 64` samples (the last batch of a 50,000-image stream has 16) and walks over it in slices:

```
take  = min(b_t - buffered, samples left in this batch)          # a slice never crosses a batch or a bucket boundary
buffer += this slice of (z_L, z_S)
if buffered >= b_t or the stream is exhausted:                    # the bucket is full
    r_m = proxy(bucket logits of member m)                        # _flush_bucket -> _gate
    w   = sigmoid(beta * (to_logit(r_L) - to_logit(r_S)))
    cached_w = w ; clear the buffer
output for this slice = w_cached * z_L / T_L + (1 - w_cached) * z_S / T_S          # log pool
```

The slice is combined **after** the flush, so the slice that completes a bucket gets the weight computed from that bucket,
including the slice itself. Slices before it get whatever weight was cached: the previous bucket's. At the start of every
stream (`set_corruption`, lines 299-321) the buffer is cleared and the cached weight is reset to
`sigmoid(beta * (prior_L - prior_S))`, which is 0.5 because `build_calibrator` passes `to_logit(0.5) = 0` for both priors
(`src/experiments/gates.py:106`). A stream whose length is not a multiple of `b_t` ends with a smaller last bucket
(forced flush, line 494-499); its NNN uses the smaller n. `calibrate_with_grad` / `calibrate` caches a result
(`_pending`) so that a batch is not assimilated twice; it only matters when the calibrator is inside the loop, which
Vanilla DD never does, so the offline evaluator does not need it.

### 1.2 What that means for the proxy batch sizes (BS = 64)

| b_t | adaptation batches per bucket | share of samples scored with their own bucket's weight | the rest |
|---|---|---|---|
| 16, 32 | a batch holds 4 or 2 buckets | 100% | none |
| 64 | 1 | 100% | none |
| 128 | 2 | 50% | previous bucket's weight |
| 256 | 4 | 25% | previous bucket's weight |
| 512 | 8 | 12.5% | previous bucket's weight |
| 1024 | 16 | 6.25% | previous bucket's weight |
| 2048 | 32 | 3.1% | previous bucket's weight |

During the first bucket of a stream every slice before the first flush uses w = 0.5. So b_t = 16 and 32 **are** supported by the
live gate (no floor at one batch), and the brief's grid of b_t up to 2048 mostly measures a lagging weight, which is the
noise-versus-reaction trade-off the b_t ablation is about.

### 1.3 Against the two options of the brief

- Option 1 ("computed from batch t's own predictions and applied to those same samples, which requires buffering") would
  have to delay the first half of every bucket's predictions until the bucket is full. An online TTA run cannot do that
  (each batch needs its prediction, and the adapting members' next step does not wait). It is only possible offline.
- Option 2 ("computed from batch t - 1, applied to batch t") is a strict lag of one bucket.
- The live gate is **option 2 for all but the last adaptation batch of the bucket, and option 1 for that last batch.**
  It reduces to option 1 when b_t <= BS and has no buffering delay at all.

Every number produced with the gate so far (the 51.3% of the old paper tables, and the Phase 2 parity runs against the old engine) comes from this behavior, and the
offline evaluator must reproduce it exactly (equivalence test 2). I suggest keeping it as the **definition** of Vanilla DD.
If you want the other two as ablations, they are one switch in a vectorised evaluator: `gate_semantics` in
`{live, lagged, own}` where `lagged` is a strict one-bucket lag and `own` applies the bucket's weight to all its samples
(not causal, an upper bound). I will only build them if you ask. **Decision 1.**
The paper text must describe whichever one is meant; I could not check that.

### 1.4 Where the brief's formulas and the live gate differ

| Point | Brief | Live code | Effect |
|---|---|---|---|
| Clip before the logit | `clip(r, 1e-6, 1 - 1e-6)` | `eps = 1e-3` twice: the predicted accuracy is clipped to `[eps, 1 - eps]`, then `to_logit(a, eps)` clips again (`_gate`, lines 352-358; `build_calibrator` does not pass eps) | Matters only for proxies that reach 0 or 1 (`atc`, `doc`, `agree_duo` can give exact 0 or 1): the log-odds saturates at +-6.9 instead of +-13.8. NNN is far from the edges. **Decision 2.** |
| Calibration map | none | `identity` (a = r) | Same. |
| Temporal filter | none | `none` by default (`gate.filter: none` in `cfgs/global.yaml`) | Same. |
| Pool | log pool (`w_L z_L/T_L + w_S z_S/T_S`) | `log` by default | Same. |
| Initial weight | not stated | 0.5 | The first bucket of every stream. |

With the filter `none` and the map `identity`, a bucket's weight is a pure function of that bucket's logits. That is what
makes the offline evaluator cheap:

1. **Proxy scores** `r_L`, `r_S` per bucket, once per (proxy, b_t) (and per temperature set for the proxies that use T_m).
   They do not depend on beta.
2. **Weights** for every beta: `w = sigmoid(beta * (s_L - s_S))`, one vectorised line per beta.
3. **Combination** of each slice with the right weight (the rule of 1.1), and the metrics, streaming (`StreamMetrics`
   in `src/experiments/evaluate.py` never holds an N x K matrix).

The live `JointProxyWeighted` run through `OnlineEval` stays as the **reference implementation for the tests**, not for the
546-cell grid (13 proxies x 7 betas x 6 b_t per run). Equivalence test 2 compares the two on a smoke stream; the runner
self-test already does a replay-versus-live comparison for the members and the gate on tiny models (`runner.py`, step 4),
and `gates.py` already checks that beta = 0 reproduces the fixed combination (test 1).

## 2. Which logits are scored (question 2)

Our own path (`MemberRunner.step_tensors`, `src/experiments/members.py:113-132`): both members run forward with the graph
alive, the adapting ones update, and the **pre-update logits** are returned and cached. This is what the brief wants for Tent.
`TentMethod` and `EataMethod` (ETA without the Fisher term) follow it.

For the upstream methods I read what `forward_and_adapt` returns (`third_party/test-time-adaptation/classification/methods/`):

| Method | What the step returns | Passes per step | Cache it as |
|---|---|---|---|
| Tent, ETA, EATA | `outputs` of the one forward pass, before the update (EATA skips the update if every sample was filtered) | 1 | as returned |
| SAR | `outputs` of the **first** pass; the second pass (at the perturbed weights) is only for the loss. With recovery on, `self.reset()` happens after the logits were taken | 2 | the first pass, as returned |
| DeYO | `outputs` of the pass on the original images; the augmented pass (patch shuffle) runs under `no_grad` for the filter only | 2 | as returned |
| CoTTA | `outputs_ema`: the **EMA teacher's** output, averaged over 32 augmentations when the anchor's mean confidence is below `AP`, else one teacher pass; computed before this step's update | 3 to 34 | the teacher's output (this is what CoTTA scores) |
| RMT | `outputs_test + outputs_ema`: the **sum** of student and teacher logits, about twice the scale | about 4, plus a labelled source batch | the sum, as returned (and see decision 9) |
| ROID | `outputs` after `outputs *= smoothed_prior`, an **in-place multiplication of the logits by a probability vector** that sums to 1, so about 1/K = 1/1000 per class | 1 to 2 | see below |

Two consequences.

- **Scale.** The temperatures of the brief (held-out TS 0.77 / 1.22) were fitted on Tent members. The scored logits of RMT
  (about 2x), CoTTA (a teacher, smoother) and ROID (about 1000x smaller if cached as returned) are on other scales, so a
  fixed combination or a softmax-based proxy on them is not comparable with Tent's. By decision 12 of the plan the Fixed TS
  is fitted on the logits of the regime in use, which suggests a held-out TS **per base algorithm**. **Decision 5.**
- **ROID.** The prior correction is a deterministic function of the batch's own logits (`prior = mean softmax` over the
  batch, then a smoothed vector), so it can be applied offline. I propose to cache ROID's **uncorrected** student logits and
  apply the correction when a standalone row is scored; the duo then uses uncorrected logits whose scale means something,
  and one cache serves both. The cost is that "exactly what is returned" holds for the standalone row only after the offline
  step. **Decision 4.** For CoTTA and RMT I propose to cache what the method returns (the teacher's logits; the sum).

The `bn` flag does not interact: all of these methods force batch statistics in BatchNorm, like Tent here.

## 3. Where to hook caching, and the storage (question 3)

### 3.1 Today's flow

`evaluate` -> `ensure_members` -> `_store_segment` -> `MemberRunner.iter_batches` -> `MemberCache.write`
(`src/experiments/runner.py:97-152`, `cache.py`), then `replay` feeds the cached logits in the original batch size to the
gates. One run (a `RunSpec` fingerprint) has one directory with one `.pt` per stream holding **both** members, centered
fp16 (`cache.py:32-37`), plus `manifest.json` (spec, batch size, git commit, timings). A full 20-stream run is 21 files and
3.8 GiB (`cache/members/` today: 21 files, 3.8 GiB).

### 3.2 Minimal changes

1. **A per-member algorithm adapter**, `reset()` and `step(x) -> scored_logits` as in the brief, with two implementations:
   `LossMethodAlgorithm` wrapping the existing `TTAMethod`s (Tent, ETA) so that nothing about their caches changes, and a
   port of each upstream method (Stage D). Vanilla DD is two such members that never see each other.
2. **`_runner(spec, ctx)`** returns a pair runner built from two adapters instead of `MemberRunner` when the algorithm is
   not Tent/ETA. `ensure_members`, `_store_segment` and `replay` are unchanged except for the store they call.
3. **A new store** (3.4) next to `MemberCache`, which stays readable, so the frozen `vitb16_rn50v1` cache keeps working.
4. The joint modes (`both_duo` and friends) stay on `MemberRunner`; they are not composable (the duo loss couples the members).

The upstream methods are ported, not imported: the benchmark needs `yacs` (not installed in `~/py38`), takes a `cfg` object,
and `methods/base.py` imports `models.model`, which pulls in `robustbench`. The repo's own rule (`third_party/repos.yaml`) is
to port with the license header and a file-and-commit reference anyway.

### 3.3 Input convention (the double-normalization pitfall)

The upstream benchmark feeds the methods **[0, 1] pixels** (`ToTensor` only in its loaders) and puts the normalization **inside
the model** (`normalize_model` / `ImageNormalizer`, `models/model.py:63,101`). Our pipeline normalizes in the transform
(`native_224` keeps ToTensor and Normalize) and the models are bare. Mixing the two normalizes twice, which is the pitfall
you mentioned. It is worse than a wrong scale for the methods with pixel-space augmentations (CoTTA, RMT and ROID use
`get_tta_transforms`, which **starts with `Clip(0, 1)`**, then colour jitter, padding, an affine warp, ...): a normalized tensor
is clipped to [0, 1] on entry (checked: torchvision's colour jitter maps a tensor in [-2, 2] into [0, 1]), which destroys it.

Proposal for the new algorithms: a ToTensor-only transform, and a thin wrapper `Sequential(Normalize(mean, std), model)` per
member, built from the model's own mean and std, i.e. the upstream convention. Tent stays on the existing path. As a
check on the wrapper, an upstream-port Tent on the new path must agree with our `TentMethod` on a smoke stream to fp
tolerance (not bitwise: the normalization moves inside the forward). **Decision 10.**

Assertions to add in the loaders (the two pitfalls of the brief):
- **Normalized exactly once:** count the normalization operations over (transform + model): `transforms.Normalize` in the
  transform plus `ImageNormalizer` or a `Normalize` module in the model must be exactly 1, and the first batch must satisfy
  `x.min() >= 0` for the [0, 1] convention.
- **Not single-class batches:** the loader used to adapt must have `shuffle=True`, and every batch written to the cache must
  have more than 0.8 * BS distinct labels (a shuffled batch of 64 from 1000 classes has about 62). `WnidFolder` lists
  samples class by class, 50 per class, so an unshuffled batch has one or two classes. `make_loader` defaults to
  `shuffle=True` but nothing enforces it today.

### 3.4 The store

**Constraints measured today** (`diskusage_report`, 2026-10-18):

| Filesystem | Space | Files |
|---|---|---|
| `/project` (shared by the group) | 1069 GiB of 2000 GiB | **8,272K of 10M (83%)** |
| `/scratch` (user) | 27 GiB of 1000 GiB | 288 of 5000K |
| `/home` (user) | 14 GiB of 50 GiB | 245K of 500K |

The file count on `/project` is the tight one, and it is shared. Bytes are not a problem on either. `h5py` is not installed in
`~/py38`, so the format is `.npy`. I did not check the purge policy of `/scratch`; that decides where long-lived caches go.

**Layout proposal.** Key a member's logits by a **member key**, not by the pair:

```
<cache>/logits/<model>/<algorithm>/<member_key>/<stream>.npy      # (N, K) float16, centered
<cache>/logits/<model>/<algorithm>/<member_key>/<stream>.meta.npz # labels, sample order (dataset indices), batch boundaries,
                                                                  #   per-row maximum (float32), stream name
<cache>/runs/<run_id>.json                                        # manifest: the pair (two member keys), config, git hash,
                                                                  #   seed, sha256 of every file
```

`member_key` = hash of (the member's model entry including its OPTIM, the algorithm and its kwargs, the protocol and stream
list, the seed, the batch size, `INPUTS_TAG`, and `--bn` if the member is frozen). In `*_indep` modes a member's logits depend on
nothing else, so a run is a **pair of member keys**, and composing the best L of one run with the best S of another is a
manifest edit, with no copy. It also stops two runs from storing the same member twice (ResNet-50 V1 would be shared between
the main duo and any other pair that uses it). The brief's "one file per stream per model" is kept; there is one small
`.meta.npz` beside each, which I think is worth the extra files (4 per stream and pair instead of 2).

**Format.** Centered fp16 (as today) **plus the float32 row maximum**, so the raw logits come back as `centered + rowmax`.
Reason: `energy` (mean logsumexp(z)) is not invariant to a per-row constant, and `cache.py`'s own docstring says that anything
needing raw logits must not use the centered cache (line 18). Storing the maximum costs 200 KB per stream and member.
Uncentered fp16 would also work, but loses accuracy near the top logits (fp16 spacing is 0.0156 between 16 and 32).
**Decision 6.**

**Size, rough.** 100 MB per stream and member (50,000 x 1000 x 2 bytes), so 200 MB per stream for the pair. A 20-stream run is
2.0 GB per member (the brief's figure).
- Tuning, held-out streams only: 4 streams, 0.8 GB per run for both members; 5 runs per algorithm (one per grid point) is 4 GB.
- Test runs: 15 corruptions plus clean is 16 streams, 3.2 GB per seed for both members, 9.6 GB for 3 seeds.
- Natural shifts (about 98k images in total): 0.4 GB per run for both members, 1.2 GB per algorithm for 3 seeds.
- So about 15 GB per base algorithm on the main pair; 8 of them come to about 110 GB. Tent and ETA on two extra pairs (test runs
  only) add about 40 GB, and the natural shifts of those another 5 GB or so.
- **Roughly 160 GB in total**, before ImageNet-C-bar and LAION-C (sizes unknown) and before the saving from sharing members
  between pairs. That is about 8% of `/project`'s space, in about 3,500 files (two per member and stream).

These are estimates from the per-stream size; Stage B will report measured totals. The live runs (Joint DD, COCA, SAR with
recovery on, standalone baselines) cache nothing.

**Staging and resume.** The brief asks to write in `$SLURM_TMPDIR` and copy at job end. A job that is killed or runs out of its
3 h walltime (the default of `slurm/job.sh`) loses every stream still on the node. The current cache is built to survive that
(a finished stream is on disk, a rerun computes only the rest). Proposal: write each stream to `$SLURM_TMPDIR`, **copy it to
persistent storage as soon as the stream is complete**, and verify the checksum before marking it done; a rerun skips verified
streams. The free space of `$SLURM_TMPDIR` has to be measured in the first smoke job. **Decision 7.**

### 3.5 Composition and its limits

Two runs with the same seed produce the same batches: `make_loader` draws the subset and the shuffle from generators seeded
with `seed` (`src/utils/datasets/base.py:55-73`), independent of the model and of `num_workers`. The store keeps the **sample
order** (dataset indices) per stream, and composition checks that both members' index arrays are identical before pairing.
Two caveats:

- **GPU determinism.** Equivalence test 4 should not demand bit equality: the same member run twice on a GPU can differ in
  the last bits. I propose measuring that noise once in Stage B (the same configuration twice) and setting the tolerance of
  test 4 from it.
- **Random augmentations.** CoTTA, ROID, RMT and DeYO draw augmentations from the global RNG (`torch.rand`, `numpy.random`,
  `torch.randperm`). In a run that holds both members, L and S would consume the same stream, so a member's logits would
  depend on the other's history and composition would not reproduce the paired run. Fix: one `torch.Generator` per member,
  seeded from (seed, member). That needs a small patch to the ported code. **Decision 11.**

## 4. How NNN is computed (question 4)

`src/reliability/proxies/nuclear_norm.py:17-22`:

```python
p = torch.softmax(logits, dim=1)                 # raw logits, no temperature
n, c = p.shape
nuc = torch.linalg.matrix_norm(p, ord="nuc")
return float(nuc / (n * min(n, c)) ** 0.5)       # n = the proxy batch size, c = K after masking
```

It is called on the bucket's raw logits (`_proxy_scores`, `joint_proxy_weighted.py:335-344`); the temperatures enter only
the combination (`_combine`). The normalization is the brief's. For ImageNet-A and ImageNet-R the evaluator masks the
logits to the 200 classes before the gate (`OnlineEval.update` applies `apply_class_mask`), so K = 200 follows without a
change, including `min(n, c)`. I keep NNN as it is.

The brief says "unless noted, p = softmax(z / T_m)" and notes only that `nnn` is "as currently implemented", so `nnn` is the
single proxy on the raw softmax. Two consequences: the proxy scores of the others have to be computed **per temperature set**
(clean TS and held-out TS, per base algorithm if decision 5 goes that way), whereas NNN's do not; and the comparison of NNN
against the others mixes a tempered and a raw softmax. If you want to separate those, an extra `nnn_T` row (NNN on the
tempered softmax) is cheap. **Decision 3.**

## 5. Which base algorithms exist (question 5)

Repo today (`src/tta/methods/`): `tent` and `eata` only. Our `eata` is ETA (no Fisher regularizer). The files that
`third_party/repos.yaml` lists under `used_by` (`sar.py`, `cotta.py`, `rdumb.py`) do not exist yet.

Pinned upstream, `third_party/test-time-adaptation` at the commit of `repos.yaml` (MIT, Döbler et al.; RMT and ROID are
Marsden et al.): `tent`, `eata` (ETA with `FISHER_ALPHA: 0`), `sar`, `cotta`, `rmt`, `roid` and **`deyo`**, plus others we do
not need. Official repositories pinned: `tent`, `EATA`, `SAR`, `cotta`. **Not present: COCA** (and the official DeYO
repository; the benchmark's `deyo.py` is what we have).

Official ImageNet-C configurations in the benchmark (`classification/cfgs/imagenet_c/*.yaml`), all with `ARCH: resnet50`
(the benchmark's torchvision loader defaults to the `IMAGENET1K_V1` weights), SGD with momentum 0.9, batch size 64:

| Method | Learning rate (ResNet-50) | Method hyperparameters in the configs and defaults | Matches the brief? |
|---|---|---|---|
| Tent | 2.5e-4 | | n/a (tuned here) |
| ETA | 2.5e-4 | `FISHER_ALPHA 0`, `D_MARGIN 0.05`, `MARGIN_E0 0.4` x ln K | yes |
| EATA | 2.5e-4 | `FISHER_ALPHA 2000`, `D_MARGIN 0.05`, `MARGIN_E0 0.4`, `SOURCE.NUM_SAMPLES 2000` | yes |
| SAR | 2.5e-4 (the code forces **1e-3 for any `vit_` / `swin_`** arch) | `MARGIN_E0 0.4` x ln K, `RESET_CONSTANT_EM 0.2`, SAM `rho 0.05` (`third_party/SAR/sam.py`) | yes (decay 0.9 not re-checked) |
| CoTTA | **1e-2** | `RST 0.001` (the global default is 0.01), `AP 0.1` (default 0.92), teacher momentum 0.999, 32 augmentations | restore probability yes; lr is outside the brief's grid |
| DeYO | 2.5e-4 | `PLPD 0.2`, `MARGIN 0.5` x ln K, `AUG_TYPE patch`, `PATCH_LEN 4`, both reweightings on; skips the top layers when collecting parameters (`layer4` for ResNet, blocks 9 to 11 for ViT) | "official ImageNet configuration" |
| RMT | **1e-2** | losses all 1.0, **`NUM_SAMPLES_WARM_UP 50000`, source replay each step** | lr is outside the grid; needs source data |
| ROID | 2.5e-4 | weighting, prior correction, consistency on; `MOMENTUM_SRC 0.99`, `MOMENTUM_PROBS 0.9`, temperature 1/3 | "official ImageNet configuration" |
| COCA | 2.5e-4 (ResNet-50), 1e-3 (ViT), 3 to 5 steps for tau, tau starts at 1 (the brief) | | code not in the repo |

Things in that table to settle:

- **Learning-rate grids.** The brief's ResNet-50 grid {5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3} is the published 2.5e-4 times
  {0.2, 0.4, 1, 2, 4}; the ViT-B/16 grid {1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3} is SAR's 1e-3 times {0.1, 0.25, 0.5, 1, 2.5}. "Re-centre on
  the published value with the same spacing" needs one reading. I propose: **the same multipliers around the published
  value** (CoTTA and RMT on ResNet-50: {2e-3, 4e-3, 1e-2, 2e-2, 4e-2}). The benchmark publishes no ViT-B/16 configuration for
  any method except SAR's 1e-3, so every other ViT grid stays as it is. **Decision 8.**
- **RMT is not source-free.** Its configuration warms the teacher on 50,000 labelled source samples and replays source
  batches at every step. The settled decision gives a 10k held-out slice for source statistics. Either RMT gets a larger
  training-image slice (and what that does to a duo comparison against source-free methods has to be said in the paper), or
  `LAMBDA_CE_SRC = 0` (the config documents it as the source-free variant), which is not the official ImageNet
  configuration. **Decision 9.**
- **EATA's Fisher** uses 2000 clean source images: they come from the same held-out training slice, which `ImageNetTrainSlice` (10 per
  class, seed 0) already provides.
- **COCA.** The brief says to ask if the official code is not in the repo, so: should `ycarobot/COCA` (named in decision 15 of
  the plan) be pinned in `third_party/repos.yaml`? The repo's `joint_coca.py` only reimplements its per-batch temperature
  calibrator, not the coupled objective. **Decision 12.**

## 6. Other findings that matter for Stages B to D

- **`~/py38`** has torch 2.7.1, timm 1.0.24, einops, scipy, scikit-learn, robustbench 1.1 and wandb; it has **no yacs and no
  h5py**. DeYO needs einops, which is there.
- **`slurm/job.sh`** requests one GPU of the default type (L40S), 8 CPUs, 32 GB and 3 hours, and activates `~/py38` (the brief
  says `ddenv`, which does not exist). ImageNet-C is read in place from `data/ImageNet-C` (a symlink to
  `/datasets/imagenet-c`); only the natural-shift archives are staged to `$SLURM_TMPDIR`. The brief's "extracted from tarballs
  in each job" does not describe this setup.
- **Cost of the extra passes.** From the Compute table of the plan (L40S, both members adapting, fp32): 202 images/s for Tent;
  SAR about 2x, CoTTA about 3 to 4x (CoTTA with 32 augmentations when the anchor is unsure is more). The Stage D run table
  will use these.
- **Existing proxies for Stage C.** `nuclear_norm` is the brief's `nnn`. `atc` exists with both score kinds (`maxconf`,
  `neg_entropy`) and a threshold fit. `cot` exists but is **stochastic**: it draws b_t class labels from the fitted source
  marginal with `torch.multinomial` (`cot_score` in `cot.py`), so an exact offline-equals-live equivalence needs a seeded generator
  per bucket; the brief says "uniform label prior", which the 10-per-class training slice is exactly. The rest of the
  table is new. The repo's `Proxy` classes (`score(logits, features, labels)`, `fit_source`) do not match the brief's pure
  function `proxy(logits_batch, T, source_stats)`, and `agree_duo` cannot be a function of one member's batch. I will keep
  the registry and put the pure functions underneath it.

## 7. Decisions I need

| # | Decision | My proposal |
|---|---|---|
| 1 | Definition of Vanilla DD's gate | The live hybrid (1.1 and 1.3). Optional ablations `lagged` and `own`, only if you ask. |
| 2 | Clip eps | Make it a parameter; use 1e-3 to match the live gate in the equivalence tests; you say which one the paper uses and whether the live default changes. |
| 3 | Temperature in the proxies | Brief as written: NNN raw, the others on softmax(z / T_m), scored per temperature set. Optionally an extra `nnn_T` row. |
| 4 | Scored logits of CoTTA, RMT, ROID | CoTTA: the teacher's. RMT: the sum. ROID: cache the uncorrected logits, apply the prior correction offline for the standalone row. |
| 5 | Temperatures per base algorithm | Refit the held-out TS per algorithm (plan decision 12). Also settles the open question of 0.77 / 1.22 against the 0.651 / 1.010 on disk. |
| 6 | Store | Per-member keys; `.npy` centered fp16 + float32 row maximum + a `.meta.npz`; the location (`/project` is at 83% of its file count; the purge policy of `/scratch` is not checked). |
| 7 | Copy-out | After each stream, with a checksum, not only at job end. |
| 8 | Grid re-centring | Same multipliers around the published value; CoTTA and RMT on ResNet-50 at {2e-3, 4e-3, 1e-2, 2e-2, 4e-2}; other ViT grids unchanged. |
| 9 | RMT's source data | Larger training-image slice for RMT only, stated in the paper; or the source-free variant. Your call. |
| 10 | Input convention of upstream methods | [0, 1] pixels with a `Normalize` wrapper per member; Tent stays on the existing path; a parity check ties the two. |
| 11 | Randomness | One generator per member, seeded from (seed, member), patched into the ported code. |
| 12 | COCA | Pin `ycarobot/COCA` in `third_party/repos.yaml`, or give me the code. |

Still open in `plan.md` and blocking Stage B: the held-out TS values (item 1), clean TS from the training slice (item 2) and
momentum 0.9 only for the tuned settings (item 3).

## 8. What Stage B would do, once you answer

1. The adapter and the store (3.2 and 3.4), the asserts of 3.3, and a manifest writer with checksums.
2. A smoke test on SLURM only (a few hundred images, no array): measure `$SLURM_TMPDIR` space, GPU run-to-run noise, and write
   one 4-stream run of Tent through the new store; compose it from two runs and compare with a direct run (test 4).
3. Report the measured cache sizes and the quota arithmetic of 3.4, then **STOP** after Stage C as the brief says.

## 9. The author's answers (2026-10-18) and how they are applied

| # | Answer | How it is applied |
|---|---|---|
| 1 | Keep the live gate as the definition; no ablations for the other two semantics | `gate_semantics` is not built. The evaluator implements the rule of 1.1 only. |
| 2 | eps = 1e-3 | Used in the evaluator (a parameter whose default is 1e-3, so equivalence test 2 compares like with like). |
| 3 | "raw" | Read as: **every proxy uses the raw softmax**, no temperature, like NNN. The `T=None` argument of the brief's signature stays, so a tempered variant remains possible. Proxy scores then do not depend on the temperature set. If only NNN was meant to stay raw, say so. |
| 4 | Tent, ETA/EATA and SAR first; CoTTA, RMT and ROID later | Stage D ports ETA/EATA and SAR only. DeYO was not on the list, so it is deferred too. ROID's cache question (decision 4) and CoTTA's/RMT's scored logits wait. |
| 5 | Refit the held-out TS for every base algorithm | The evaluator fits (T_L, T_S) on the pooled held-out streams of each (algorithm, pair); clean TS is fitted per algorithm too. |
| 6 | Per-member keys; `.npy` centered fp16 + float32 row maximum + `.meta.npz` | Implemented in Stage B. The store lives under `paths.cache`, which was moved to `/scratch/alxstaub/dynamic-duos/cache` on 2026-10-18 (see plan.md, Stage B). The purge policy of `/scratch` was not found. |
| 7 | Copy out after each stream, with a checksum | Implemented: each stream is staged in `$SLURM_TMPDIR`, copied, verified by sha256, and only then recorded. |
| 8 | Same multipliers around the published value | Recorded for Stage D; CoTTA/RMT grids are deferred with those methods. |
| 9 | Skip RMT for now | Out of scope. |
| 10 | [0, 1] pixels and a `Normalize` wrapper per member for the new algorithms; Tent stays on the existing path; a parity check ties the two | SAR and EATA (Stage D) use the wrapper; the parity check (a ported Tent against our `TentMethod`) comes with them. |
| 11 | One generator per member, seeded from (seed, member) | Applied to the ported code in Stage D (SAR and EATA have no random augmentation, so it only matters once DeYO/CoTTA/ROID come). |
| 12 | https://github.com/ycarobot/COCA | Pinned in `third_party/repos.yaml` at `e13f75f` and checked out. The repository has no LICENSE file. |

## 10. Stages B and C as built (2026-10-18)

Where the plan's Stage B and Stage C sections hold the details and the numbers, this is what changed against the design above.

- **Check on batches (3.3).** The test on single-class batches is on the MEAN over the stream's full batches (at least half as many distinct labels as samples), not on every batch: a 200-class batch of 64 holds about 54 distinct labels, so the 0.8 per batch I proposed would reject healthy streams. A loader that does not shuffle is caught by the order check as well.
- **Members of one pair run together (3.2).** `ensure_logits` always runs both members, even when only one is missing, because they share the DataLoader; only the missing member's files are written.
- **Source statistics.** They come from the member's own logits on the stream `clean_train_slice` (the held-out clean slice of the training images, 10 per class) under the same algorithm and hyperparameters, which also fits the clean TS. That makes the clean TS, the ATC thresholds, DoC's constants and the energy normalization one coherent set per (algorithm, pair).
- **Decision 3 read as "all proxies raw".** `T=None` everywhere; the argument stays in the signatures.
- **Not built, as decided:** the `lagged` and `own` gate semantics; CoTTA, RMT, ROID and DeYO; any Stage D code. Joint DD (`joint_dd.py`) supports the 1000-way streams only.
- **Still to measure.** GPU run-to-run noise over a full 50,000-image stream (so far over 3 update steps), which sets the tolerance of the composition test for Stage F.
