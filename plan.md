# Plan: one script per paper result

Goal: every result in the paper's Experiments section has its own short script in `scripts/` that outputs its table or figure. All shared logic lives in `src/`. Past-experiment scripts are removed. Shared information (dataset paths etc.) lives in one global config, and each dataset has its own class in `src/utils/`.

Status key: `[x]` done, `[ ]` open, `[?]` needs the author's decision.

## Status (2026-10-07)

| Phase | State |
|---|---|
| 0 Safety net and inventory | done |
| 1 Global config, dataset classes, staging | done (CCC streams still to be generated, Phase 8) |
| 2 Engine (`src/experiments/`) | done, parity with the old engine verified on real models |
| 3 New components (proxies, oracle gate, SAR/EATA/CoTTA/ROID/RDumb, ViT-L/ViT-S loaders, single-model runner) | **on hold at the author's request** |
| Preliminary experiments (inserted before Phase 3) | in progress: `tab_temperatures.py`, `tab_oracles.py` (see below) |
| 4-9 | not started |

Everything above is committed and pushed on `filtered-proxy-soft-weighting` up to the end of Phase 2.

### Preliminary experiments (inserted before Phase 3)

Two analysis scripts that decide which temperatures "Fixed TS" uses and bound what any temperature-based duo method can reach. Both read the same cached member logits (one adaptation run per seed, 20 streams: clean validation, the 4 held-out corruptions, the 15 test corruptions) and write LaTeX tables.

1. `scripts/tab_temperatures.py`: optimal (T_L, T_S) fitted on each stream, next to the single-model accuracies and the accuracy of clean-fitted TS, held-out-corruption-fitted TS and per-stream optimal TS.
2. `scripts/tab_oracles.py`: Fixed TS against four oracles of increasing granularity (corruption-wise TS, batch-wise TS, batch-wise oracle weight, per-sample oracle), all of which use labels.

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

## Phase 2: Engine (`src/experiments/`, no paper experiments yet)  (status: done)

The old loop (`run_duo`, `evaluate_dynamic_duo`, `plot_run_diagnostics`) stays untouched until Phase 9; the new engine sits beside it and is checked against it.

- [x] `specs.py`: `RunSpec` (duo, TTA method + kwargs, mode, protocol, seed, sample cap) with a fingerprint over everything that changes the members' logits (model blocks incl. learning rates, TTA, protocol, seed, batch size, objective-logits mode), and `GateSpec` (fixed TS or the proxy-weighted gate: proxy, beta, b_t, pool, filter, T_L/T_S checkpoint).
- [x] `protocols.py`: episodic, tuning, natural, continual (one of CoTTA's 10 orders), ccc, as lists of `Segment(label, dataset, reset_before)`. Continual is the episodic loop minus the resets, with one loader per corruption.
- [x] `members.py`: `MemberRunner`, a lean independent-adaptation loop (no calibrator, diagnostics or wandb) that reuses the existing `TTAMethod` classes. Duo modes are rejected. It plumbs `objective_logits_200class` (masked/full) into the TTA loss. Tested **bit for bit** against the old `forward_and_adapt` on tiny CPU models.
- [x] `cache.py`: per-segment member-logit cache plus a manifest (spec, batch size, timings, git commit), atomic writes, resume per segment (episodic) or restart (continual).
- [x] `gates.py`: `GateSpec` -> calibrator without needing models. Not supported yet, and raising `NotImplementedError`: source-fitted proxies (ATC etc.) and non-identity calibration maps (Phase 3).
- [x] `evaluate.py`: `StreamMetrics` (accuracy, ECE, NLL, entropy from 5 bytes/sample, equal to the old `get_metrics_dict`), `OnlineEval` (members + any number of gates on the same logits; class mask applied first; gates reset only where the members do).
- [x] `runner.py`: `ensure_members` (compute-or-load), `replay` (touches no model and no dataset), `evaluate_live` (no cache, for CCC), `evaluate`. `--report_only` fails with the list of what is missing.
- [x] `results.py` (tidy rows, per-seed average, mean +- std over seeds, fraction of oracle gap), `cli.py` (`--duo --tta --seeds --num_samples --report_only --shard i/N|auto --only --out`, thread pinning), `src/utils/tables.py` (name-keyed corruption table with mean +- std and a generic table; the generated LaTeX compiles with pdflatex).
- [x] Every module has a `__main__` self-test (`python -m src.experiments.<module>`; `runner` is the end-to-end one: parity with the old engine, resets, resume, replay vs live, report-only, masked objective, 200-class gating).
- [x] **Real-model parity** (`scripts/parity_old_vs_new.py`, temporary; jobs 6019846 and 6020069): old engine vs new engine on ViT-B/16 + ResNet-50, brightness / fog / contrast, 2,000 samples each, two seeds, the gate configured as the existing paper tables were produced (EMA 0.9999, beta 0.5, b_t 128, log pool, the existing T_L/T_S). **Accuracy identical to 0.00 points in all 18 (corruption x series) cells**, covering the large member, the small member and the proxy-weighted duo; ECE within 0.05 points (the fp16 cache). Timing for the same 6,000 images: new members pass 39 s (69 s before the preprocessing moved into the workers); the old engine's whole run took 94-101 s, but that also includes loading the models and building the gate, so it is not a like-for-like speed comparison. The same script at `--num_samples 50000` over all 15 corruptions is the Phase 5 gate.
- [ ] `figures.py` is deliberately not written yet: its first consumers are Phase 6's `fig_gate_weights.py` and Phase 8's stream/reset figures, and the plotting code should be shaped by them.
- Resume for CCC (hours per stream) needs model/optimizer/gate checkpoints every K batches and a start offset in the CCC reader (cheap: the reader is index-addressable). Phase 8.

Findings that changed the design:
- **The gate is not invariant to how the stream is cut into batches.** `JointProxyWeighted` combines the slice that completes a proxy batch with the weight computed from that batch (itself included) and earlier slices with the previous weight. With b_t = 128 and adaptation batch 64, every second adaptation batch is gated with a weight that was computed from its own samples. So a replay must use the original adaptation batch size, which the cache manifest records and `replay` uses. This is existing behaviour (it is what the current 51.3% was computed with), preserved on purpose. Worth a sentence in the paper's method section, and it means the headline numbers depend on BS = 64 relative to b_t.
- **Cache format: centered fp16** (logits minus their row maximum). Everything downstream is invariant to a per-row constant, and centering keeps the decisive logits near zero where fp16 is finest. 3 GB per full IN-C run of both members; argmax and softmax survive to < 2e-3 (tested).
- **Preprocessing moved into the DataLoader workers** (`PairCollate`): both models' resize/crop/normalize ran in the main process and cost as much as the GPU step. Bitwise-identical logits (tested with 0 and 2 workers), about 1.5-1.8x faster end to end. The GPU step is now the limit (see Compute).\n- **CPU thread oversubscription:** a 128x1000 nuclear norm took 38 s on a busy login node (24 threads) and 6 ms with 1-4 threads. `cli.pin_threads()` sets torch's threads from `SLURM_CPUS_PER_TASK`. The Gram-matrix trick is no faster on CPU, so the proxy is unchanged.

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

**Measured** (job 6020117/6020140/6020141; ViT-B/16 + ResNet-50, batch 64, independent Tent, steady state):

| | L40S | H100 |
|---|---|---|
| fp32 (current default) | 202 img/s | 372 img/s |
| TF32 matmuls | 278 img/s | 673 img/s |
| frozen members (forward only) | 471 img/s | 799 img/s |

Data loading (JPEG decode + both models' preprocessing in 8 workers) delivers 1,500 img/s, so the GPU step is the limit. My earlier "800 img/s" was wrong. Moving the preprocessing into the DataLoader workers (done) took the end-to-end loop from about 130 to about 200 img/s on the L40S. TF32 changes the first-step logits by <= 0.005 with identical argmax, but it breaks bitwise reproducibility of the existing numbers (decision 11).

**Estimates from those rates** (H100, fp32; ViT-L/16 + ResNet-50 costs about 3x per image, SAR about 2x, CoTTA about 3-4x):

| Item | Estimate |
|---|---|
| One episodic IN-C run (15 x 50,000 images), ViT-B + RN-50 | 34 min on H100, 62 min on L40S |
| Generality grid, 3 TTA x 3 pairs x 3 seeds | about 35 H100-hours |
| Natural shifts, same grid (about 98k images per run) | about 2 H100-hours |
| CoTTA-orders protocol, 7 duo policies + 6 single-model rows, 10 orders | about 80 H100-hours |
| **CCC, one stream x one row** (7.5M images) | **5.6 H100-hours** (3.1 with TF32) |
| CCC, reduced grid (3 difficulties x 3 seeds, one speed) x ~14 rows | about 700 H100-hours (390 with TF32) |
| CCC, full grid (27 streams) x ~14 rows | about 2,100 H100-hours (1,150 with TF32) |
| CCC reset ablations (~12 runs on one stream) | about 70 H100-hours |
| IN-C logit caches | about 3 GB per run, about 80 GB total |
| CCC shards on disk (27 streams) | about 3-6 TB (a guess, see Phase 8) |

CCC dominates everything else by more than 10x, so decision 1 (develop and report on a reduced CCC grid) matters. Frozen-member rows (no adaptation) run at the forward-only rate.

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
11. **TF32 and GPU type (new).** Enabling TF32 matmuls gives +38% (L40S) / +81% (H100) throughput with logit changes <= 0.005, but the new runs would no longer reproduce the existing fp32 numbers bit for bit, and the parity gate (Phase 5) is easiest to read in fp32. Default: stay fp32 through the Phase 5 parity gate, then decide; request H100 nodes (`--gres=gpu:h100:1`) for long runs, they queued in under a minute.
