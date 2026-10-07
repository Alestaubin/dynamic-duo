# Plan: one script per paper result

Goal: every result in the paper's Experiments section has its own short script in `scripts/` that outputs its table or figure. All shared logic lives in `src/`. Past-experiment scripts are removed. Shared information (dataset paths etc.) lives in one global config, and each dataset has its own class in `src/utils/`.

Status key: `[x]` done, `[ ]` open, `[?]` needs the author's decision.

## What the repo shows (drives the design)

- **Adaptation is decoupled from the gate.** The paper adapts both members independently (`both_indep`), so the calibrator never feeds back into adaptation. One live run per (TTA method, duo, stream, seed) can cache both members' logits, and every gate variant (beta, b_t, proxy, pooling, filter, oracle) is a cheap replay. `slurm/run_paper_tables.sh` already works this way.
- **Reset policies break the replay for continual runs.** A reset changes the member logits, so each reset policy needs its own live run. The gate is then replayed on top of it.
- **CCC (7.5M-image streams) cannot use the current loop.** `run_duo` concatenates all N x K probabilities in memory. CCC needs streaming metrics and compact per-proxy-batch logs.
- **Loader and reset gaps.**
  - `load_imagenetC` shuffles a `ConcatDataset`, which would mix corruptions in a continual stream. Continual needs per-corruption shuffling in a fixed order.
  - `DynamicDuo.reset()` resets all sides, but resets must be per member.
- **Paper-vs-code mismatches.**
  - Temperatures, ATC thresholds and proxy source statistics are fit on `VAL_DIR` (`scripts/fit_fixed_ts.py`, `src/reliability/setup.py:52`). The paper requires a held-out train slice. Every existing `checkpoints/fixed_ts/*`, `checkpoints/proxy_stats/*` and `data/calibration_maps/*` file is therefore stale.
  - The main calib config uses `ema alpha=0.9999` where the paper says "no filter".
  - `zoom_blur` is now in EVAL, which makes 15 corruptions. The paper text says "fourteen", so the reported averages change on rerun.
  - The existing `eata` method has no Fisher term, which makes it ETA. True EATA is only needed as a single-model baseline.
  - CLAUDE.md says SAR is blocked by the stateful calibrator. That does not apply under `*_indep`, where the calibrator is outside the adaptation loop. SAR only needs a closure-style re-forward hook in `TTAMethod`.

## Target layout

```
cfgs/global.yaml            # shared: dataset roots, output roots, seeds, protocol constants, corruption lists
cfgs/duos/*.yaml            # only LARGE/SMALL (name, norm, optim) for the 3 paper pairs
cfgs/tuned/*.yaml           # written by tuning scripts, read by everything else
src/utils/config.py         # load_global(), load_duo(name)
src/utils/datasets/         # one class per dataset
src/reset/                  # reset policies, detectors, targets, diagnostics
src/experiments/            # run/cache/replay engine shared by all scripts
src/utils/tables.py, figures.py, results.py
scripts/                    # one script per paper result, nothing else
out/paper/                  # tab_*.tex, fig_*.pdf, plus the CSV each was built from (tracked)
out/*                       # everything else under out/ is git-ignored
```

## Phase 0: Safety net and inventory  (status: done except the open items)

- [x] Tag `pre-paper-cleanup` (annotated, at `4847a93`). It restores every tracked script, cfg and src file. Local only, not pushed.
- [x] Back up what the tag cannot restore: `archive/pre-paper-cleanup/ignored_files.tar.gz` (32 MB) holds `slurm/*.sh` (30 files), `CLAUDE.md`, `checkpoints/`, `data/calibration_maps/` and `data/proxy_weight_cfg/`. All of these are git-ignored (`*.sh`, `CLAUDE.md`, `checkpoints`, `*.pt`), so deleting or editing them without this backup would be irreversible. `archive/` is itself git-ignored.
- [x] `.gitignore`:
  - `out/*` plus `!out/paper/`: experiment outputs are regenerable and untracked, but the final tables and figures in `out/paper/` stay tracked.
  - `datasets` anchored to `/datasets`. The unanchored rule would have silently git-ignored the planned `src/utils/datasets/` package. No nested directory was newly un-ignored by this change.
  - `/archive/` added.
- [x] Inventory (below).
- [?] 148 files under `out/` are already tracked (93 csv, 50 png, 1 json, 1 tex, 3 txt), and 8 of them have uncommitted modifications. The new ignore rule does not untrack them. To untrack: `git rm -r --cached out/` (stages 148 deletions, files stay on disk), then `git add out/paper` once it exists.
- [?] `*.sh` is git-ignored, so the new per-script launchers in `slurm/` (Phase 9) would be ignored too. Add `!slurm/*.sh` to `.gitignore` if the launchers should be tracked.
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
| `slurm/*.sh` | Backed up. Replace with one launcher per new script in Phase 9. `slurm/README.md` (tracked) keeps the `salloc` recipes |
| `optimal_w_sanity.png`, `results/*.csv` (tracked) | Delete (past experiments) |
| `report/experiments.tex` | Keep |

Library code that looks dead from the paper's point of view, to be removed only after Phase 8 and only with the author's OK: `joint_coca.py`, `calibrators/temp/`, `joint_optimal_w_oracle.py`, the `oracle_ts` / `coca` / `optimal_w_oracle` modes in `dynamic_duo.py`, proxies `prototype` / `ac_mc` / `cot` / `agreement`, calibration maps `linear` / `platt` / `beta` / `isotonic`, filters `kalman` / `running_mean`, `csv_to_latex.py`.

## Phase 1: Global config and dataset classes

- [ ] `cfgs/global.yaml` holds:
  - dataset roots (train, val, IN-C, R, A, Sketch, V2, CCC), filled in by the author;
  - output, cache and checkpoint roots, plus wandb and worker settings;
  - protocol constants: severity 5, adaptation BS 64, default b_t=128, beta=0.5, log pooling, 15 ECE bins, seeds [0,1,2];
  - the 15 test corruptions grouped by family, the 4 tuning corruptions, the CCC corruption lists, and the path to the 10 CoTTA orders.
  - Env override `DUO_GLOBAL_CONFIG`.
- [ ] Per-duo YAMLs drop `TEST_DIR`/`VAL_DIR`. `src/utils/config.py` provides `load_global()` and `load_duo(name)`.
- [ ] `src/utils/datasets/` with a `ShiftDataset` base class (`check()`, `loader(batch_size, num_samples, seed)`, `class_mask`, `stream()`) and a `get_dataset(name)` registry:
  - `ImageNetTrainSlice`: fixed-seed, held-out clean images for temperatures, ATC thresholds and proxy source stats
  - `ImageNetVal`
  - `ImageNetC`: `stream(order, severity, seed)` shuffles within each corruption and keeps the boundary order
  - `ImageNetR`, `ImageNetA`: 200-class subsets, so logits are masked and K=200 is used wherever K matters (log K, nuclear norm)
  - `ImageNetSketch`
  - `ImageNetV2`: folder-index-to-label mapping
  - `CCC`: generator-backed stream by difficulty, speed and seed
- [ ] Done when `scripts/setup_check.py` passes on the imported data.

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
- [ ] TTA methods: alias `eata` as `eta`, add true `EATA` with the Fisher term, add `SAR` (recovery off) with a re-forward closure hook, optional `RMT`.
- [ ] Single-model runner for the reference rows (Tent, ETA, CoTTA, EATA, ROID, RDumb). RDumb is ETA plus a reset every 1000 steps.
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

## Phase 9: Cleanup and consistency

- [ ] Delete the old scripts, cfgs and slurm launchers per the inventory. Add one launcher per new script. Update CLAUDE.md and README.
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

CCC is about 25 times everything else combined, and CPU corruption generation may be the real bottleneck. Option: run all policies of one stream in one process on a shared batch to amortise data generation.

## Decisions (defaults apply unless changed)

1. CCC grid: scripts support the full grid, but develop and report on a reduced one (one speed, three seeds) until throughput is measured.
2. ViT-S/16 source: timm `vit_small_patch16_224` (the paper says ViT-S/16), not the DeiT-S that `model.py` already loads.
3. CoTTA and ROID: port from the official repos rather than vendoring a benchmark repo.
4. CoTTA 10 orders: not to be recreated from memory. The author supplies the order lists, or confirms they should be pulled from the official repo.
5. Corruption count: report 15 and update the "fourteen" wording and averages in the paper text.
6. Old scripts: tag plus delete (done in Phase 0 for the safety net), not an archive folder.
