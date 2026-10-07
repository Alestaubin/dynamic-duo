# Plan: one script per paper result

Goal: every result in the paper's Experiments section has its own short script in `scripts/` that outputs its table or figure. All shared logic lives in `src/`. Past-experiment scripts are removed. Shared information (dataset paths etc.) lives in one global config, and each dataset has its own class in `src/utils/`.

Status key: `[x]` done, `[ ]` open, `[?]` needs the author's decision.

## What the repo shows (drives the design)

- **Adaptation is decoupled from the gate.** The paper adapts both members independently (`both_indep`), so the calibrator never feeds back into adaptation. One live run per (TTA method, duo, stream, seed) can cache both members' logits, and every gate variant (beta, b_t, proxy, pooling, filter, oracle) is a cheap replay. `slurm/run_paper_tables.sh` already works this way.
- **Reset policies break the replay for continual runs.** A reset changes the member logits, so each reset policy needs its own live run. The gate is then replayed on top of it.
- **CCC (7.5M-image streams) cannot use the current loop.** `run_duo` concatenates all N x K probabilities in memory. CCC needs streaming metrics and compact per-proxy-batch logs.
- **Loader and reset gaps.**
  - `load_imagenetC` shuffles a `ConcatDataset`, which would mix corruptions in a continual stream. The new `ImageNetC` class gives one loader per corruption, so the continual protocol is the same loop as episodic minus the resets.
  - `DynamicDuo.reset()` resets all sides, but resets must be per member.
- **Paper-vs-code mismatches.**
  - Temperatures, ATC thresholds and proxy source statistics are fit on `VAL_DIR` (`scripts/fit_fixed_ts.py`, `src/reliability/setup.py:52`). The paper requires a held-out train slice. Every existing `checkpoints/fixed_ts/*`, `checkpoints/proxy_stats/*` and `data/calibration_maps/*` file is therefore stale.
  - The main calib config uses `ema alpha=0.9999` where the paper says "no filter".
  - `zoom_blur` is now in EVAL, which makes 15 corruptions. The paper text says "fourteen", so the reported averages change on rerun.
  - The existing `eata` method has no Fisher term, which makes it ETA. True EATA is only needed as a single-model baseline.
  - CLAUDE.md says SAR is blocked by the stateful calibrator. That does not apply under `*_indep`, where the calibrator is outside the adaptation loop. SAR only needs a closure-style re-forward hook in `TTAMethod`.

- **Read from the upstream repos in `third_party/` (not from memory).**
  - CCC is generated locally from ImageNet *val*; the hosted stream is gone. Each stream is 7.5M JPEG-85 images in 25,000-image webdataset shards. Speeds are 1000/2000/5000 and seeds 43/44/45. `--baseline` 0/20/40 gives Hard/Medium/Easy. Generation needs `webdataset`, `wand` (ImageMagick), `opencv` and `numba`, and downloads an accuracy-matrix pickle from a Tuebingen server.
  - The CCC corruption lists match the paper text: Hard is gaussian/shot/impulse noise plus contrast, brightness is in no level, and none of the 4 tuning corruptions appears in any level.
  - RDumb is ETA with a reset to the source weights *and optimizer state* every 1000 steps (batch 64, SGD lr 2.5e-4, momentum 0.9, e_margin 0.4 ln K, d_margin 0.05).
  - The benchmark repo `mariodoebler/test-time-adaptation` has Tent, ETA/EATA, SAR, CoTTA, RMT, ROID and an RDumb config, and treats ImageNet-C and CCC images as already 224x224 (no resize).
  - CoTTA's 10 orders exist as `cotta/imagenet/cfgs/10orders/` (identical to the CIFAR ones): ten permutations of our 15 corruptions.
  - ImageNet-V2 folders are class *indices*; torchvision `ImageFolder` would mislabel them (tested in the dataset self-test).

## Target layout

```
cfgs/global.yaml            # shared: dataset roots, output roots, seeds, protocol constants, corruption lists
cfgs/protocols.yaml         # GENERATED from upstream: CoTTA's 10 orders, CCC speeds/seeds/corruption lists
cfgs/duos/*.yaml            # only LARGE/SMALL (name, norm, optim) for the 3 paper pairs
cfgs/tuned/*.yaml           # written by tuning scripts, read by everything else
src/utils/config.py         # load_global(), load_duo(name)
src/utils/datasets/         # one class per dataset (+ class_lists/, generated from upstream)
third_party/                # pinned upstream repos: repos.yaml, fetch.py, extract_assets.py (clones git-ignored)
src/reset/                  # reset policies, detectors, targets, diagnostics
src/experiments/            # run/cache/replay engine shared by all scripts
src/utils/tables.py, figures.py, results.py
scripts/                    # one script per paper result, nothing else
out/paper/                  # tab_*.tex, fig_*.pdf, plus the CSV each was built from (tracked)
out/*                       # everything else under out/ is git-ignored
```

## Phase 0: Safety net and inventory  (status: done)

- [x] Tag `pre-paper-cleanup` (annotated, at `4847a93`). It restores every tracked script, cfg and src file. Local only, not pushed.
- [x] Back up what the tag cannot restore: `archive/pre-paper-cleanup/ignored_files.tar.gz` (32 MB) holds `slurm/*.sh` (30 files), `CLAUDE.md`, `checkpoints/`, `data/calibration_maps/` and `data/proxy_weight_cfg/`. All of these are git-ignored (`*.sh`, `CLAUDE.md`, `checkpoints`, `*.pt`), so deleting or editing them without this backup would be irreversible. `archive/` is itself git-ignored.
- [x] `.gitignore`:
  - `out/*` plus `!out/paper/`: experiment outputs are regenerable and untracked, but the final tables and figures in `out/paper/` stay tracked.
  - `datasets` anchored to `/datasets`. The unanchored rule would have silently git-ignored the planned `src/utils/datasets/` package. No nested directory was newly un-ignored by this change.
  - `/archive/` added.
- [x] Inventory (below).
- [x] Untracked `out/` (`git rm -r --cached out/`: 148 files staged for deletion, nothing deleted on disk). Not committed. Add `out/paper/` with `git add` when the first final table exists.
- [x] `.gitignore` now tracks `slurm/*.sh` (`!slurm/*.sh`).
- [x] Cleaned `slurm/`: 28 past-experiment launchers deleted (all are in the archive tarball). Kept `run_paper_tables.sh` (the Phase 5 parity reference) and `test_tta_refactor.sh` (goes with `test_tta_parity.py`). A single generic launcher, `slurm/job.sh`, replaces them (added in Phase 1 because staging runs at job start).
- Not part of the tag: uncommitted `src/tta/methods/eata.py` and the `src/tta/methods/__init__.py` edit. Nothing planned for deletion depends on them.

### Inventory (what each existing file becomes)

All of `scripts/` and `cfgs/` is tracked and unmodified, so the tag restores it.

| File | Fate |
|---|---|
| `scripts/plot_run_diagnostics.py` | Extract the replay engine (`_replay_compare_configs_from_cache`) and the run orchestration, then delete. `write_accuracy_latex_table` and `plot_per_corruption_gate_weight` live in `src/utils/diagnostics_plots.py` and move to `tables.py` / `figures.py` in Phase 2 |
| `scripts/compare_calibrators.py` | Extract the run_cfg to calibrator construction, then delete |
| `scripts/sweep_proxies.py` | Extract `bal_sel_acc`, `gap_bias`, `gap_corr`, then delete |
| `scripts/sweep_tent_hp.py` | Becomes `tune_tta.py` |
| `scripts/fit_fixed_ts.py` | Becomes `fit_temperatures.py` |
| `scripts/_cli.py` | Fold into `src/experiments/cli.py` (keep `tta_tag`), then delete |
| `scripts/run_dynamic_duo.py`, `scripts/run_tent.py` | Delete once the protocol layer and single-model runner exist (Phases 2 and 3) |
| `scripts/test_tta_parity.py` | Delete after the Phase 5 parity gate replaces it |
| `scripts/calibrate_gate_oracle.py`, `diagnose_ema_filter.py`, `eval_per_corruption_ts.py`, `plot_optimal_w_sanity.py`, `proxy_vs_optimal_temperature.py`, `screen_duo_candidates.py`, `test_proxies.py`, `test.py` | Delete (past experiments) |
| `cfgs/dynamic_duo_config_vitb_resnet.yaml` | Becomes `cfgs/duos/vitb16_rn50.yaml` |
| other `cfgs/dynamic_duo_config*.yaml`, `cfgs/test.yaml` | Delete (pairs not in the paper), after the two new pairs are written |
| `cfgs/compare_runs/paper_*.json` | Port each sweep into a spec builder in Phase 6, then delete the directory |
| `cfgs/calib_configs/*.json` | `nuclear_norm_identity_pbs128.json` is the default gate and `fixed_ts_default.json` the fixed-TS baseline. Port, then delete |
| `slurm/*.sh` | 28 deleted (backed up). `run_paper_tables.sh` and `test_tta_refactor.sh` stay until Phase 5 / Phase 9. `slurm/README.md` keeps the `salloc` recipes |
| `optimal_w_sanity.png`, `results/*.csv` (tracked) | Delete (past experiments) |
| `report/experiments.tex` | Keep |

Library code that looks dead from the paper's point of view, to be removed only after Phase 8 and only with the author's OK: `joint_coca.py`, `calibrators/temp/`, `joint_optimal_w_oracle.py`, the `oracle_ts` / `coca` / `optimal_w_oracle` modes in `dynamic_duo.py`, proxies `prototype` / `ac_mc` / `cot` / `agreement`, calibration maps `linear` / `platt` / `beta` / `isotonic`, filters `kalman` / `running_mean`, `csv_to_latex.py`.

## Phase 1: Global config and dataset classes  (status: done, waiting on your dataset imports)

- [x] `third_party/`: 11 upstream repos pinned in `repos.yaml` (url, commit, license, what we use it for). `python third_party/fetch.py` re-creates the checkouts at the pinned commits (tested, including the sparse ones). `python third_party/extract_assets.py` regenerates `cfgs/protocols.yaml` and `src/utils/datasets/class_lists/` from them. Clones are git-ignored. Rule from now on: TTA methods and benchmark code are ported from these checkouts, with the upstream license header and file + commit in the docstring.
- [x] `cfgs/global.yaml`: dataset roots, output roots, runtime, protocol constants (severity 5, BS 64, seeds 0/1/2, 15 ECE bins), default gate, source-slice settings, corruption lists. `imagenet_train`, `imagenet_val` and `imagenet_c` point at the existing `data/` links. `imagenet_r`, `imagenet_a`, `imagenet_sketch`, `imagenet_v2` and `ccc` are `null` placeholders with their expected layout documented next to them. `$DUO_GLOBAL_CONFIG` overrides the file.
- [x] `src/utils/config.py`: `load_global()`, `load_duo(name)` (returns the legacy keys `TEST_DIR`/`VAL_DIR`/`BS`/`WORKERS`/`EVAL`/`CALIBRATOR` so the current `src/tta` engine keeps working). Validation enforces the splits: tuning corruptions are in no test stream and no CCC level, CCC corruptions are test corruptions, every CoTTA order is a permutation of the tests. Self-test: `python -m src.utils.config`.
- [x] `cfgs/duos/{vitb16_rn50,vitl16_rn50,vitb16_vits16}.yaml` (model choices only).
- [x] `src/utils/datasets/`: `ImageNetVal`, `ImageNetTrainSlice`, `ImageNetC`, `ImageNetR`, `ImageNetA`, `ImageNetSketch`, `ImageNetV2`, `CCC`, with `get_dataset(name)`. One loader contract (batches of `(list[PIL], LongTensor)`, no preprocessing; images converted to RGB). Self-tests: `python -m src.utils.datasets` (fake on-disk copies of every layout; `ImageNetC` reproduces the old `load_imagenetC` batch order exactly; a missing or extra class folder is an error; the ImageNet-V2 label pitfall; the 200-class masks; staging) and `python -m src.utils.datasets.ccc` (order-preserving multi-worker reads across shard boundaries, partial-stream and missing-shard detection).
- [x] **Archives and staging** (`src/utils/datasets/staging.py`), following the cluster notes. The natural-shift datasets are archives in `/project/6104790/shared/datasets/` (file-count quota), so they are never extracted there: `stage_dataset(key)` / `stage_archive(path)` extract with `tar -xf` / `unzip -q` into `$SLURM_TMPDIR/data` once per job (marker keyed on archive path/size/mtime; extraction into a `.partial_*` dir renamed into place; file lock) and return the dataset root. Outside a job there is no default staging dir (`StagingUnavailable`); `$DUO_STAGING_DIR` overrides it for local tests. A path that is a directory is read in place, and any path in `global.yaml` can be a list of candidates (first existing wins): ImageNet-R lists `imagenet-r.tar` then the extracted directory, ImageNet-V2 lists `.tar` then `.tar.gz`.
- [x] **Count and class assertions** for the four natural-shift sets, from the real data: ImageNet-A 7,500 images / 200 classes, ImageNet-R 30,000 / 200, ImageNet-V2 10,000 / 1000 (10 per class), ImageNet-Sketch 50,889 / 1000. Checked every time the dataset is built, and by `check()` straight from the archive listing (no extraction, so it works on a login node). The class folders must equal the upstream lists exactly (A and R from hendrycks' `eval.py`, Sketch and V2 from the canonical 1000-wnid list, which equals the sorted Sketch folders); an extra or missing class folder is an error.
- [x] **Masked vs full objective** (your TTA note): `protocol.objective_logits_200class: masked|full` in `global.yaml` (default `masked`, which is what the reference benchmark does: it wraps the model in a masking layer). `ShiftDataset.objective_logits(logits, mode)` implements it; accuracy and ECE always use `apply_class_mask`. Phase 2 plumbs it through the protocol layer so every compared method gets the same setting; the proxies, gate and temperatures see the masked 200-way logits in both modes (decision 10).
- [x] `scripts/setup_check.py` (`--full`, `--probe N`, `--strict`, `--skip-train-slice`). Archives are verified from their member lists, so the check runs on a login node. `--probe` needs the extracted data and reports SKIPPED outside a job.
- [x] **Verified on real data in a SLURM job** (6019035: CPU-only, `STAGE="imagenet_a imagenet_r imagenet_v2 imagenet_sketch" sbatch slurm/job.sh scripts/setup_check.py --probe 200`, 3 min). Staging to `$SLURM_TMPDIR/data` (which is `/tmp` on Killarney) took 4s for A, 7s for V2 and 87s for Sketch; all four count assertions pass on the staged files; ImageNet-R was read in place. Probe, frozen ResNet-50 on 200 random images per set: val 80.5%, IN-C fog 39.0%, R 40.5%, A 13.5% (masked 200-way), Sketch 31.5%, V2 73.5%. Every set is far above chance, so the label mappings are right (a scrambled mapping, e.g. ImageFolder on V2, gives about chance). 200 images means roughly +-3 points of noise; these are sanity checks, not results.
- [x] Built the clean train-slice list (10 images/class, seed 0) at `cache/train_slice_k10_s0.tsv`.
- [x] You: imported ImageNet-R/A/Sketch/V2. Paths set in `global.yaml`; counts verified (above).
- [ ] Optional, once: `cd /project/6104790/shared/datasets && tar -cf imagenet-r.tar imagenet-r`. Until it exists, ImageNet-R is read in place from its 30,000-file directory (slow on shared storage); with the tar it is staged like the others. Not done by me: it writes into the shared folder.
- [ ] You, later: generate the CCC streams (Phase 8 below) and set `paths.ccc`.
- Left for later phases: the `vit_l_16` and `vit_s_16` model loaders (Phase 3; `setup_check` reports those two duos as pending), removal of `TEST_DIR`/`VAL_DIR` from the legacy code paths, and deleting `src/utils/data.py` (Phase 9).

## Phase 2: Engine (src/, no experiments yet)

- [ ] Protocol layer: `episodic`, `continual(order)`, `ccc`, `natural`. Reset at boundaries only for episodic.
- [ ] Streaming accumulators (accuracy, ECE from confidence and correctness only) and a compact per-proxy-batch log: member and duo accuracy, r_l, r_s, confidence, dispersity, w_l, reset events.
- [ ] Member-stream cache (fp16 logits) keyed by duo, TTA method plus hparam hash, protocol, stream and seed.
- [ ] `replay_calibrators(...)` extracted from `plot_run_diagnostics.py`. Same code runs online for CCC, with several calibrators fed the same batch.
- [ ] `tables.py` extracted from `write_accuracy_latex_table`. It keys columns by corruption name, never by position (the likely cause of the Frozen-row column-order issue), and supports mean+-std and percent formatting.
- [ ] Every script writes a CSV first and renders tables and figures only from that CSV.
- [ ] `src/experiments/cli.py` with the common script contract: `--duo --tta --seeds --report_only --only/--shard i/N`.

## Phase 3: New components

- [ ] Proxies (each with a `__main__` self-test): mean normalized negative entropy, MSP, `confidence_fro` (||P||_F), `dispersity` (||P||_* / ||P||_F), all mapped to [0,1]. Add ordering accuracy and Spearman rho_S, reusing `gap_corr`.
- [ ] Oracle selection: hard per-proxy-batch pick of the member with the higher batch accuracy, as a replay and online.
- [ ] TTA methods, ported from the `third_party/` checkouts (`EATA`, `SAR`, `CCC/models/rdumb.py`, `test-time-adaptation/classification/methods/`): alias `eata` as `eta`, add true `EATA` with the Fisher term, add `SAR` (recovery off) with a re-forward closure hook, optional `RMT`. First diff our existing `eata.py` against `EATA/eata.py` and `CCC/models/eata.py`.
- [ ] Single-model runner for the reference rows (Tent, ETA, CoTTA, EATA, ROID, RDumb), ported from `cotta`, `test-time-adaptation` (ROID, RMT) and `CCC/models/rdumb.py`. RDumb is ETA plus a reset of model and optimizer every 1000 steps.
- [ ] Models: `vit_l_16` (torchvision), ViT-S/16 loader. `timm` is in requirements.txt, but confirm it imports in the venv.
- [ ] Public `reset_side(side)`.

## Phase 4: Reset framework (`src/reset/`, same registry idiom as proxies and filters)

- [ ] Statistics: per-member NNN, gap, collapse signature (high confidence with low dispersity), gap plus signature.
- [ ] Detectors: threshold, derivative, CUSUM. Refractory period tau_min.
- [ ] Targets: source, last healthy snapshot, shrink toward source.
- [ ] Policies: none, periodic simultaneous, periodic staggered, random at a matched rate, entropy-EMA, proxy (ours), oracle (same rule on true batch accuracy).
- [ ] Diagnostics: collapse onset against frozen per-batch accuracy on the identical stream, detection delay, false resets, resets per 1M samples, event-aligned accuracy and reset-cost area.
- [ ] Self-tests on synthetic collapse streams.

## Phase 5: Setup scripts, then tuning

- [ ] `setup_check`, `tab_models`, then `fit_temperatures`, `tune_tta`, `tune_gate`, `tune_reset`. Everything else reads `cfgs/tuned/`.
- [ ] Throughput benchmark on a CCC stream. It decides the Phase 8 grid.
- [ ] Parity gate: rerun the old `paper_vitb_resnet_both_indep` configuration through the new pipeline and compare to the existing CSVs. Do not overwrite the old `checkpoints/fixed_ts/default` until this passes (a copy is in the archive).

## Phase 6: Episodic IN-C results

`tab_accuracy`, `tab_beta`, `fig_gate_weights`, `tab_proxies`, `tab_abl_pbs`, `tab_abl_pooling`, `tab_abl_filter` (IN-C columns), `tab_overhead`.

## Phase 7: Generality and natural shifts

`tab_generality`, `fig_divergence`.

## Phase 8: Continual, CCC, reset ablations, stress tests

`tab_continual_refs`, `tab_continual_duo`, `tab_continual`, `fig_ccc_stream`, `fig_reset_cost`, `tab_stress`, the four reset ablations, `tab_ablations` (CCC-M column), CCC runs of `tab_abl_pbs` / `tab_abl_pooling` / `tab_abl_filter`.

CCC prerequisites (from `third_party/CCC`):
- [ ] `scripts/ccc_generate.py`: a thin SLURM-array wrapper around `generate.py` (needs the packages listed at the end of `requirements.txt`, ImageMagick, one-off internet for the accuracy-matrix pickle, and the frost textures taken from `third_party/robustness` instead of downloaded). The streams are 27 (3 difficulties x 3 speeds x 3 seeds) x 7.5M images.
- [ ] Storage and generation cost are unmeasured. Rough guess: a 224x224 JPEG-85 is 15-30 KB, so 110-225 GB per stream and 3-6 TB for all 27; generation is CPU-bound (glass blur is slow). Measure on one shard first (Phase 5), then decide the grid.
- [ ] Label-correlated streams for `tab_stress` use `sort_by_dirichlet` from `test-time-adaptation/classification/datasets/data_loading.py`.
- [ ] The reset-threshold tuning stream cannot be a CCC stream: CCC's walk needs an accuracy matrix that exists only for its own corruptions. Plan: cycles of the 4 tuning corruptions in random order (see decisions).

## Phase 9: Cleanup and consistency

- [ ] Delete the old scripts, cfgs and slurm launchers per the inventory. `slurm/job.sh` (any script + args, arrays via `sbatch --array`) already exists; no per-script launchers. Update CLAUDE.md (git-ignored, backed up) and README.
- [ ] `check_consistency.py` asserts that shared cells agree across tables. The beta=0.5, b_t=128, log-pool, no-filter cell must be identical in the main, beta, b_t, pooling and filter tables. This resolves the 51.1 vs 51.3, 48.8 vs 49.2 and dagger mismatches by construction.
- [ ] Switch the main gate config from `ema alpha=0.9999` to `none` and confirm numbers are unchanged.

## Script map (one script per result, in `scripts/`)

Every script takes `--duo --tta --seeds --report_only --only/--shard i/N`. Each builds its specs, runs or loads cached results, writes a CSV, and renders `out/paper/<label>.tex` or `.pdf`. Logic stays in `src/`.

| Script | Result |
|---|---|
| `setup_check.py` | Dataset paths, disjoint tuning, test and CCC corruptions, temperature data not from val |
| `tab_models.py` | Params, FLOPs, small-to-large FLOPs ratio and clean accuracy per pair |
| `tune_tta.py` | Tent/ETA/SAR hparams per model on the 4 held-out corruptions |
| `fit_temperatures.py` | T_L, T_S per duo on the train slice |
| `tune_gate.py` | beta and b_t per proxy |
| `tune_reset.py` | Reset thresholds, tau_min, collapse-signature thresholds |
| `tab_accuracy.py` | Main table: members, fixed TS, ours, oracle row, ECE column, std, % of oracle gap captured |
| `tab_beta.py` + `fig_gate_weights.py` | beta sweep per corruption in %; w_l figure for brightness and fog |
| `tab_proxies.py` | `tab:proxies` with Avg, Collapse (fog, snow, glass blur, elastic), Order, rho_S |
| `tab_generality.py` | `tab:generality`: 3 TTA x 3 pairs x {IN-C, R, A, Sketch, V2}, fixed TS / ours |
| `fig_divergence.py` | Gain vs member divergence, one point per corruption x TTA x pair |
| `tab_overhead.py` | Proxy and gate wall-clock overhead, forward and backward pass counts relative to the large model |
| `tab_continual_refs.py` | Frozen and single-model rows (CoTTA-orders and CCC columns) |
| `tab_continual_duo.py` | Gated-duo reset-policy rows plus diagnostics |
| `tab_continual.py` | Assembles `tab:continual` from the two CSVs above |
| `fig_ccc_stream.py` | One CCC-M stream: no-reset duo vs RDumb duo vs ours, with member accuracies and reset ticks |
| `tab_stress.py` | Joint collapse (same-family pair) and label-correlated streams |
| `fig_reset_cost.py` | Event-aligned reset dip, plus the table splitting the gain into warm partner, proxy timing and frequency |
| `tab_abl_pbs.py` | b_t sweep 16-2048 on IN-C and CCC, per-corruption deltas |
| `tab_abl_pooling.py` | Linear vs log pooling, per-corruption accuracy and ECE, rho_P and rho_S |
| `tab_abl_filter.py` | None vs EMA alpha=0.05 and 0.01 on IN-C and CCC-M |
| `tab_abl_reset_stat.py`, `tab_abl_detector.py`, `tab_abl_refractory.py`, `tab_abl_target.py` | Reset ablations on one CCC-M stream |
| `tab_ablations.py` | Assembles `tab:ablations` |
| `check_consistency.py` | Cross-table agreement |

## Compute and storage

Rough numbers assuming about 800 img/s for a two-model adaptation step. They are unmeasured, and the Phase 5 benchmark replaces them.

| Item | Estimate |
|---|---|
| Episodic IN-C, 27 runs (3 TTA x 3 pairs x 3 seeds) | about 7 GPU-hours |
| Natural shifts | about 1 GPU-hour |
| CoTTA-orders protocol, about 13 live rows x 10 orders | about 35 GPU-hours (CoTTA 3-4x slower) |
| **CCC, full grid** (27 streams x about 14 rows x 7.5M) | **about 10^3 GPU-hours** |
| CCC reset ablations (about 12 runs on one stream) | about 30 GPU-hours |
| IN-C logit caches | about 3 GB per run, about 80 GB total |
| CCC shards on disk (27 streams) | about 3-6 TB (a guess, see Phase 8) |

CCC is about 25 times everything else combined, and CPU corruption generation may be the real bottleneck. Option: run all policies of one stream in one process on a shared batch to amortise data generation.

## Decisions (defaults apply unless changed)

1. CCC grid: scripts support the full grid, but develop and report on a reduced one (one speed, three seeds) until throughput, storage and generation cost are measured.
2. ViT-S/16 source: timm `vit_small_patch16_224` (the paper says ViT-S/16), not the DeiT-S that `model.py` already loads.
3. CoTTA and ROID: port from the official repos (`cotta`, `test-time-adaptation`) rather than vendoring a whole benchmark repo.
4. CoTTA 10 orders: resolved. They come from the official repo (`cotta/imagenet/cfgs/10orders`) and are in `cfgs/protocols.yaml`.
5. Corruption count: report 15 and update the "fourteen" wording and averages in the paper text.
6. Old scripts: tag plus delete, with the git-ignored files backed up in `archive/`.
7. **Resize on already-224 images (new).** ImageNet-C and CCC images are already 224x224 centre crops, and the benchmark repo evaluates them with no resize. Our pipeline applies each model's own `preprocess` (e.g. ViT-B/16: resize 256, centre crop 224), which zooms them by about 14%. This changes absolute accuracies against published numbers. Default: keep the current behaviour for IN-C (so Phase 5 parity holds) and use the same for CCC; the datasets expose `already_224` so switching to normalize-only is one flag. Say if you want the no-resize protocol instead.
8. **Source slice caveat (new).** The temperatures and ATC thresholds are fitted on ImageNet *train* as the paper says, but the pretrained members were trained on those images, so they are a bit more confident there and the fitted temperatures may come out low. Default: implement as written and have `fit_temperatures.py` also report the val-fitted values (sensitivity only, never used).
9. **Reset-threshold tuning stream (new).** Default: cycles of the 4 tuning corruptions (speckle, gaussian blur, spatter, saturate) in random order, about 1M images, seeds disjoint from the test streams.
10. **Logits seen by the proxy, gate and temperatures on ImageNet-A/R (new).** Your note covers the TTA objective only. Default: everything downstream of the members' logits (proxy score, gate, pooling, fixed temperatures) uses the masked 200-way logits, because that is what predictions are scored on and it is what the reference benchmark does. The temperatures T_L, T_S are fitted once on 1000-way clean data and applied unchanged to the masked logits. Say if you want the gate to see all 1000 logits instead.
