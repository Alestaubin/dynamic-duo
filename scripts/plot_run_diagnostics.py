#!/usr/bin/env python3
"""
scripts/plot_run_diagnostics.py
================================
Single-config exploratory run: pick a duo config, a set of corruptions, an
adaptation mode, and a calibration framework (fixed_ts or filtered-proxy
soft weighting), run it once end-to-end, and plot the resulting per-batch /
running diagnostics -- accuracy, NLL, entropy for large/small/duo, plus (for
calibration_mode=proxy_weighted) the raw proxy signal and gate weight over
time. Everything is also logged to Weights & Biases (on by default; --no_wandb
to disable): evaluate_dynamic_duo's own per-batch/per-corruption/summary
logging, PLUS this script's own additions (gate weight and raw proxy signal
merged onto the SAME step as each batch's accuracy/NLL/entropy, so they line
up on one x-axis in the wandb UI; the two PNG figures as Images; the full
proxy log as a Table) into one shared run -- see _make_wandb_run.

The calibrator (calibration_mode + all its knobs -- proxy_kind, calib_method,
filter_kind, proxy_batch_size, beta, ...) is specified via --calib_config, a
JSON file holding ONE run_cfg dict -- the exact same shape as one entry in
cfgs/compare_runs/*.json (compare_calibrators.py's --configs_file), so a
config already written for a comparison run works here unmodified, and vice
versa. See cfgs/calib_configs/ for ready-made ones. This keeps "which
calibrator, with which knobs" as a versioned, diffable file rather than a
long CLI invocation that's easy to fat-finger or forget to record.

This is deliberately a THIN driver: model/calibrator construction reuses
compare_calibrators._build_calibrator (same run_cfg dict shape as
cfgs/compare_runs/*.json) so this script can never drift from how a real
comparison run builds a calibrator, and per-batch accuracy/NLL/entropy are
read straight out of DynamicDuo._diag (the same accumulator run_duo already
maintains for wandb logging) rather than recomputed here.

Proxy diagnostics (r_l, r_s, a_l, a_s, x_l, x_s, w_l, and the per-proxy-batch
ground-truth acc_l/acc_s/duo_acc) come from JointProxyWeighted's own CSV
logging (csv_path=... at construction, see joint_proxy_weighted.py's
_CSV_FIELDS) -- this script just points that at out_dir and reads it back
for plotting/wandb, rather than re-deriving the same numbers a second way.

batch_diagnostics.csv and every plot are (re)written after EVERY corruption,
not just once at the end (see _run's _on_corruption_end / _write_plots) --
a long run killed mid-way by a SLURM walltime limit still leaves usable,
up-to-date plots instead of only the calibrator's own incrementally-written
proxy_log CSV.

--compare_configs (optional) points at a JSON file holding a LIST of run_cfg
dicts -- the SAME shape/file compare_calibrators.py's --configs_file takes
(e.g. cfgs/compare_runs/default.json). Each entry is built into its own
calibrator (fit_beta/register_hooks included, via
compare_calibrators._build_calibrator) against the SAME large/small model
instances as this run's own --calib_config duo, then every batch re-
calibrates that SAME batch's z_large/z_small through it (no extra model
forward passes -- see compare_calibrators.py's fixed_ts_reference pattern,
generalized here from one hardcoded reference to an arbitrary list). Each
one's resulting per-batch accuracy is plotted as a dashed line (plus an avg-
accuracy tag) in every per-corruption plot, alongside this run's own duo
accuracy (also newly shown there -- see plot_per_corruption_proxy_vs_accuracy's
`extra_series`) and the two input models' independent accuracy -- so one
figure per corruption compares the calibrator actually driving this run
against a list of alternatives, all on the exact same stream.

Both this run's own --calib_config AND every --compare_configs entry are
built at --fit_num_samples (small, default 5000), not --num_samples (large,
the eval loop's own sample count) -- see --fit_num_samples' own help and the
comment on `calibrator`'s construction in _run for why: a fit_beta=true or
calib_method != 'identity' run_cfg's dev-corruption fitting pass only needs
enough samples to estimate a handful of scalars/maps, not eval-sized data,
and a --compare_configs file with several such entries multiplies that cost.

--cache_logits (optional, off by default) saves every batch's z_large/
z_small/labels to out_dir/logits_cache/ (one file per completed corruption).
This is the ONLY thing that lets you add MORE calibrators to a run's plots
LATER without rerunning the duo: pass --csv_dir <run's out_dir> together
with --compare_configs (instead of --calib_config) and every listed
calibrator is built fresh and replayed against those cached logits -- see
_replay_compare_configs_from_cache. Zero model forward passes over the eval
set either way; only a --fit_num_samples-sized dev pass if a calibrator
needs fit_beta/a calib_map fit. Without --cache_logits on the original run,
there's nothing to replay against and adding a calibrator later needs a
full rerun with an updated --compare_configs instead.

Usage
-----
Corruptions/severities always come from --config's EVAL.CORRUPTIONS/
EVAL.SEVERITIES (cfgs/dynamic_duo_config.yaml) -- edit that file to change
which ones a run covers, rather than passing them on the command line.

    # fixed_ts baseline, no adaptation
    python scripts/plot_run_diagnostics.py --config cfgs/dynamic_duo_config.yaml \
        --calib_config cfgs/calib_configs/fixed_ts_default.json \
        --mode no_adapt --num_samples 10000

    # filtered-proxy soft weighting, both models adapting jointly
    python scripts/plot_run_diagnostics.py --config cfgs/dynamic_duo_config.yaml \
        --calib_config cfgs/calib_configs/nuclear_norm_identity_pbs128.json \
        --mode both_indep --num_samples 10000

    # same, but skip wandb entirely (quick local iteration)
    python scripts/plot_run_diagnostics.py --config cfgs/dynamic_duo_config.yaml \
        --calib_config cfgs/calib_configs/nuclear_norm_identity_pbs128.json \
        --no_wandb --num_samples 500
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import wandb

from src.tta.dynamic_duo import setup_duo, evaluate_dynamic_duo, _MODES, _CALIB_MODES
from src.utils.data import load_config
from src.utils.model import get_model
from src.reliability.proxies.stats import PROXY_KINDS
from src.calibrators.joint_proxy_weighted import JointProxyWeighted
from src.utils.diagnostics_plots import (
    plot_batch_diagnostics, plot_proxy_diagnostics, plot_per_corruption_proxy_vs_accuracy,
    extra_duo_series_from_batch_records, DEFAULT_EMA_WINDOW, C_GATE, EXTRA_SERIES_PALETTE,
)
from scripts._cli import (
    add_duo_config_arg, add_num_samples_arg, add_seed_arg,
    add_wandb_project_group_args, add_out_dir_run_name_args,
)
from scripts.compare_calibrators import _build_calibrator, _load_run_configs

# Private JointProxyWeighted attributes holding the LATEST cached gate
# internals (refreshed at proxy-batch flushes, reused between them) -- no
# public accessor exists for these beyond last_w_l, so this script reads
# them the same way scripts/calibrate_gate_oracle.py reaches into
# calibrator._forward directly: an accepted pattern in this codebase for a
# driver script that needs diagnostic internals, not a public API surface.
_GATE_INTERNALS = [
    ("_cached_r_l", "gate/r_l"), ("_cached_r_s", "gate/r_s"),
    ("_cached_a_l", "gate/a_l"), ("_cached_a_s", "gate/a_s"),
    ("_cached_x_l", "gate/x_l"), ("_cached_x_s", "gate/x_s"),
]


def _quiet_mode() -> None:
    """Suppress noisy default output for THIS SCRIPT specifically, without
    changing any shared module's own defaults for OTHER callers:

    - dynamic_duo.py's own `logging.basicConfig(level=logging.INFO)` (an
      import-time side effect -- see its module docstring) makes every
      `logger.info(...)` call anywhere in the codebase print with a
      timestamp. Rather than edit that shared file, raise the ROOT logger's
      level here, in this process only, after the import has already
      installed its handler.
    - tqdm progress bars (run_duo's per-batch bar; fit_beta's dev-pass
      collection bar in src/reliability/setup.py). Patched in two places:
      dynamic_duo.tqdm (that module did `from tqdm import tqdm` at ITS OWN
      import time, so the name is already bound there -- patching the tqdm
      PACKAGE afterward wouldn't reach it) and the tqdm package's own
      `tqdm` attribute (fit_beta does a LOCAL `from tqdm import tqdm`
      INSIDE the function body, re-importing fresh on every call, so a
      package-level patch made before it's called does reach it).

    Called once at the top of a live run (see main()) -- --csv_dir-only
    replotting/replay never touches the model/eval-loop code paths this
    guards, so it isn't needed there.
    """
    logging.getLogger().setLevel(logging.WARNING)

    import tqdm as _tqdm_pkg
    _real_tqdm_cls = _tqdm_pkg.tqdm

    def _silent_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        return _real_tqdm_cls(*args, **kwargs)

    _tqdm_pkg.tqdm = _silent_tqdm
    import src.tta.dynamic_duo as _dd_mod
    _dd_mod.tqdm = _silent_tqdm


@contextlib.contextmanager
def _quiet_stdout(label: str = ""):
    """Suppress routine stdout chatter from a block of model/calibrator
    construction code -- get_model's "Loading X..."/"Freezing..." banners,
    JointFixedTS.load's "Loaded ... trained on ..." print, JointProxyWeighted's
    ASCII-art config banner, fit_beta's grid-search announcement. These are
    plain print() calls, not logging, so _quiet_mode's logger-level change
    can't reach them.

    Any captured line starting with "WARNING" (this codebase's own
    consistent convention for a real, actionable fallback -- e.g. a missing
    fixed_ts checkpoint silently defaulting to T=1.0) is still printed
    afterward, optionally prefixed with `label` for context (e.g. which
    --compare_configs entry it came from) -- so going quiet never means
    silently losing something that actually matters. The rescue scan runs
    in a `finally` so a captured WARNING is never lost even if the wrapped
    code goes on to raise -- the original exception still propagates after.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            yield
    finally:
        for line in buf.getvalue().splitlines():
            if line.strip().upper().startswith("WARNING"):
                print(f"[{label}] {line}" if label else line)

_CALIB_METHODS = {"identity", "linear", "platt", "beta", "isotonic"}
_FILTER_KINDS = {"none", "running_mean", "ema", "kalman"}


def _load_calib_config(path: str) -> dict:
    """Load and validate a single run_cfg dict (calibration_mode + its knobs)
    from a JSON file -- the same shape as one entry in cfgs/compare_runs/*.json
    (see compare_calibrators._load_run_configs), just not wrapped in a list
    since this script only ever runs one config at a time."""
    with open(path) as f:
        run_cfg = json.load(f)
    if not isinstance(run_cfg, dict):
        raise ValueError(f"{path} must contain a single JSON object (a run_cfg dict), "
                          f"not a {type(run_cfg).__name__} -- see cfgs/calib_configs/ for examples.")
    if run_cfg.get("calibration_mode") not in _CALIB_MODES:
        raise ValueError(f"{path}: 'calibration_mode' must be one of {sorted(_CALIB_MODES)}, "
                          f"got {run_cfg.get('calibration_mode')!r}")
    if run_cfg["calibration_mode"] == "proxy_weighted":
        if run_cfg.get("proxy_kind") not in PROXY_KINDS:
            raise ValueError(f"{path}: 'proxy_kind' must be one of {sorted(PROXY_KINDS)}, "
                              f"got {run_cfg.get('proxy_kind')!r}")
        calib_method = run_cfg.get("calib_method", "identity")
        if calib_method not in _CALIB_METHODS:
            raise ValueError(f"{path}: 'calib_method' must be one of {sorted(_CALIB_METHODS)}, "
                              f"got {calib_method!r}")
        filter_kind = run_cfg.get("filter_kind", "none")
        if filter_kind not in _FILTER_KINDS:
            raise ValueError(f"{path}: 'filter_kind' must be one of {sorted(_FILTER_KINDS)}, "
                              f"got {filter_kind!r}")
    run_cfg.setdefault("name", Path(path).stem)
    return run_cfg


def _resolve_fixed_ts_config(path: str | None) -> str | None:
    """None if the checkpoint doesn't exist, so _build_calibrator's
    JointFixedTS.load(...) call is never handed a path it will raise on --
    prints a warning and falls back to T_l=T_s=1.0 instead."""
    if path is None:
        return None
    if not (Path(path) / "config.json").exists():
        print(f"WARNING: fixed_ts_config={path!r} not found; combining at T_l=T_s=1.0 "
              f"(or, for proxy_weighted, gating with no base_ts prior).")
        return None
    return path


def _default_calib_map(cfg: dict, proxy_kind: str, calib_method: str) -> str:
    return f"{cfg['LARGE']['NAME']}_{cfg['SMALL']['NAME']}_{proxy_kind}_{calib_method}"


def _load_batch_diagnostics_csv(path: Path) -> list[dict]:
    """Read a previously-written batch_diagnostics.csv back into the same
    shape _run's on_batch hook builds live (global_idx an int, everything
    else -- including "duo_*" columns, still present in the CSV even though
    the plots no longer draw them -- a float), so it can be handed straight
    to plot_batch_diagnostics/plot_per_corruption_proxy_vs_accuracy as if
    this were a fresh run."""
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k != "corruption":
                r[k] = int(v) if k == "global_idx" else float(v)
    return rows


def _load_proxy_log_csv(csv_dir: Path) -> list[dict]:
    """The proxy log's filename carries a timestamp suffix (JointProxyWeighted
    appends one at construction) so it can't be located by a fixed name --
    glob for it instead. Empty (not an error) when the run wasn't
    calibration_mode=proxy_weighted, matching plot_proxy_diagnostics' own
    graceful no-proxy-data handling."""
    matches = sorted(csv_dir.glob("proxy_log_*.csv"))
    if not matches:
        return []
    if len(matches) > 1:
        print(f"WARNING: {len(matches)} proxy_log_*.csv files in {csv_dir}; "
              f"using the most recent: {matches[-1].name}")
    with matches[-1].open() as f:
        return list(csv.DictReader(f))


def _boundaries_from_batch_records(batch_records: list[dict]) -> list[dict]:
    """Recover corruption boundaries (see _run's on_corruption_start hook)
    from batch_records alone -- they aren't a column in batch_diagnostics.csv,
    but every row's own "corruption" field is enough to reconstruct exactly
    where each one started."""
    boundaries = []
    last_corr = None
    for i, r in enumerate(batch_records):
        if r["corruption"] != last_corr:
            boundaries.append({"idx": i, "label": r["corruption"]})
            last_corr = r["corruption"]
    return boundaries


def _write_plots(
    batch_records: list[dict], boundaries: list[dict], proxy_rows: list[dict],
    out_dir: Path, ema_window: int,
    extra_series: list[tuple[str, str, str]] | None = None,
) -> tuple[bool, bool]:
    """Write every diagnostics artifact from whatever has been recorded SO
    FAR, overwriting what's already on disk -- called after every corruption
    (see _run's _on_corruption_end) rather than only once at the very end, so
    a long multi-corruption run's plots are visible while it's still going,
    and survive a SLURM walltime kill instead of leaving nothing but a
    partial proxy_log CSV behind. Mirrors run_tent.py's own _write_plots for
    the exact same reason (see its docstring). Returns (has_batch_plot,
    has_proxy_plot), same as the two plotting calls' own return values.

    extra_series (this run's own duo accuracy, plus one per --compare_configs
    entry -- see _run) is only ever forwarded to the per-corruption plot,
    which is the one place a "calibrated duo" comparison line makes sense --
    see plot_per_corruption_proxy_vs_accuracy's own docstring for why this
    isn't the default everywhere.
    """
    has_batch_plot = False
    if batch_records:
        plot_batch_diagnostics(batch_records, boundaries, out_dir / "batch_diagnostics.png",
                                ema_window=ema_window)
        has_batch_plot = True

    has_proxy_plot = plot_proxy_diagnostics(proxy_rows, out_dir / "proxy_diagnostics.png",
                                             ema_window=ema_window)

    per_corruption_dir = out_dir / "per_corruption"
    per_corruption_dir.mkdir(parents=True, exist_ok=True)
    plot_per_corruption_proxy_vs_accuracy(batch_records, proxy_rows, per_corruption_dir,
                                           ema_window=ema_window, extra_series=extra_series)
    return has_batch_plot, has_proxy_plot


def _replot_from_csv_dir(csv_dir: Path, ema_window: int) -> None:
    """Re-run every plotting function against an existing run's own output
    directory instead of re-running the duo -- e.g. after a plotting-only
    change (line styles, --ema_window) that doesn't need fresh model
    forward passes. Overwrites every PNG (and the per-corruption CSVs,
    which are themselves a plot-data export -- see
    plot_per_corruption_proxy_vs_accuracy's docstring) already there."""
    batch_csv = csv_dir / "batch_diagnostics.csv"
    batch_records = _load_batch_diagnostics_csv(batch_csv) if batch_csv.exists() else []
    if not batch_records:
        print(f"No {batch_csv} -- nothing to plot.")
        return
    boundaries = _boundaries_from_batch_records(batch_records)
    proxy_rows = _load_proxy_log_csv(csv_dir)
    # No run_cfg available here to name the main duo line -- "duo" is a
    # generic stand-in; cmp_<name>_acc columns (if any) keep their own names.
    extra_series = extra_duo_series_from_batch_records(batch_records, main_label="duo")

    _write_plots(batch_records, boundaries, proxy_rows, csv_dir, ema_window, extra_series)

    print(f"\nRe-plotted from {csv_dir} (existing PNGs/per-corruption CSVs overwritten).")


def _replay_compare_configs_from_cache(
    csv_dir: Path, config_path: str, compare_configs_path: str,
    fit_num_samples: int | None, seed: int | None, ema_window: int,
) -> None:
    """Add MORE --compare_configs calibrators to an already-finished run's
    plots WITHOUT rerunning the duo. Valid because --compare_configs
    calibrators (see _run's _on_batch) only ever re-calibrate the SAME
    z_large/z_small a batch already produced -- if those were saved during
    the original run (--cache_logits, one file per COMPLETED corruption
    under csv_dir/logits_cache/), a brand new calibrator can be built fresh
    and replayed against them here with ZERO model forward passes over the
    eval set (only a small --fit_num_samples-sized dev pass if it needs
    fit_beta or a calib_map fit).

    Raises if csv_dir/logits_cache/ doesn't exist -- a run from before
    --cache_logits existed, or one that didn't pass it, has nothing to
    replay against; the only way to add calibrators to THAT run's plots is
    a full rerun with --compare_configs.

    Does not support proxy_kind='prototype' compare-config entries: the
    cache holds logits only, not the penultimate features that proxy needs
    a live forward hook for (same limitation as compare_calibrators.py's
    own --use_cache + prototype).
    """
    batch_csv = csv_dir / "batch_diagnostics.csv"
    batch_records = _load_batch_diagnostics_csv(batch_csv) if batch_csv.exists() else []
    if not batch_records:
        raise FileNotFoundError(f"No {batch_csv} -- nothing to replay new calibrators onto.")

    cache_dir = csv_dir / "logits_cache"
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"{cache_dir} not found -- this run wasn't started with --cache_logits, so there are "
            f"no saved z_large/z_small to replay new calibrators against. For THIS run, add the "
            f"new calibrator(s) to --compare_configs and do a full rerun instead; pass "
            f"--cache_logits on the next run to enable this fast path going forward."
        )

    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with _quiet_stdout("model loading"):
        large_model, large_preprocess = get_model(cfg["LARGE"]["NAME"])
        small_model, small_preprocess = get_model(cfg["SMALL"]["NAME"])
    large_model, small_model = large_model.to(device), small_model.to(device)

    cmp_run_cfgs = _load_run_configs(compare_configs_path)

    new_calibrators: list[tuple[str, object]] = []
    used_cmp_run_cfgs: list[dict] = []
    for cmp_cfg in cmp_run_cfgs:
        if cmp_cfg["calibration_mode"] == "proxy_weighted" and cmp_cfg.get("proxy_kind") == "prototype":
            print(f"Skipping {cmp_cfg['name']!r}: proxy_kind='prototype' needs live features, "
                  f"which the logits-only cache can't supply.")
            continue
        if "fixed_ts_config" in cmp_cfg:
            cmp_cfg["fixed_ts_config"] = _resolve_fixed_ts_config(cmp_cfg["fixed_ts_config"])
        with _quiet_stdout(cmp_cfg["name"]):
            calibrator = _build_calibrator(
                cmp_cfg, cfg, large_model, large_preprocess, small_model, small_preprocess,
                device, fit_num_samples, seed,
                csv_path=str(csv_dir / f"compare_{cmp_cfg['name']}"), verbose=False,
            )
        new_calibrators.append((cmp_cfg["name"], calibrator))
        used_cmp_run_cfgs.append(cmp_cfg)

    if not new_calibrators:
        print("No new calibrators to replay -- nothing to do.")
        return

    # Every row gets every new column, defaulted to NaN first, so the CSV
    # stays rectangular even for a corruption whose cache file is missing
    # (e.g. one from before --cache_logits was added mid-run, or a run
    # killed before that corruption's _on_corruption_end saved it).
    for name, _ in new_calibrators:
        for r in batch_records:
            r[f"cmp_{name}_acc"] = float("nan")

    boundaries = _boundaries_from_batch_records(batch_records)
    for i, b in enumerate(boundaries):
        start = b["idx"]
        end = boundaries[i + 1]["idx"] if i + 1 < len(boundaries) else len(batch_records)
        label = b["label"]  # e.g. "brightness/s5" -- matches _on_corruption_end's own naming
        cache_file = cache_dir / f"{label.replace('/', '_')}.pt"
        if not cache_file.exists():
            print(f"WARNING: no cached logits for {label} ({cache_file} missing) -- leaving new "
                  f"calibrator columns as NaN for its rows.")
            continue

        cached = torch.load(cache_file, map_location=device, weights_only=True)
        z_l_all, z_s_all, labels_all = cached["z_l"], cached["z_s"], cached["labels"]
        rows = batch_records[start:end]
        n_cached = sum(int(r["n"]) for r in rows)
        if n_cached != z_l_all.shape[0]:
            print(f"WARNING: {label} cache has {z_l_all.shape[0]} samples but batch_diagnostics."
                  f"csv rows for it sum to {n_cached} -- skipping as stale/mismatched (re-run "
                  f"with --cache_logits to refresh it).")
            continue

        for name, calibrator in new_calibrators:
            if hasattr(calibrator, "set_corruption"):
                calibrator.set_corruption(label)
            pos = 0
            for r in rows:
                n = int(r["n"])
                z_l, z_s, labels = z_l_all[pos:pos + n], z_s_all[pos:pos + n], labels_all[pos:pos + n]
                if hasattr(calibrator, "set_labels"):
                    calibrator.set_labels(labels)
                with torch.no_grad():
                    z_cmp = calibrator.calibrate(z_l, z_s)
                r[f"cmp_{name}_acc"] = float((z_cmp.argmax(1) == labels.to(z_cmp.device)).float().mean())
                pos += n

    with batch_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(batch_records[0].keys()))
        writer.writeheader()
        writer.writerows(batch_records)
    print(f"Merged {len(new_calibrators)} new calibrator column(s) into {batch_csv}")

    proxy_rows = _load_proxy_log_csv(csv_dir)
    extra_series = extra_duo_series_from_batch_records(batch_records, main_label="duo")
    _write_plots(batch_records, boundaries, proxy_rows, csv_dir, ema_window, extra_series)

    # Appended (not overwritten) to the ORIGINAL run's own manifest -- see
    # _format_run_manifest -- so run_config.txt stays a complete history of
    # every calibrator ever added to this run's plots, not just the latest.
    replay_note = [
        "", "-" * 78,
        f"REPLAY at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -- added via --compare_configs "
        f"{compare_configs_path} (fit_num_samples={fit_num_samples}, seed={seed}):",
        "-" * 78,
    ]
    for cmp_cfg in used_cmp_run_cfgs:
        replay_note.append(f"[{cmp_cfg['name']}]")
        replay_note += [f"  {k}: {v}" for k, v in cmp_cfg.items()]
    with (csv_dir / "run_config.txt").open("a") as f:
        f.write("\n".join(replay_note) + "\n")

    print(f"\nReplayed {[n for n, _ in new_calibrators]} against cached logits and re-plotted "
          f"{csv_dir} -- no model forward passes over the eval set were needed.")


def _format_run_manifest(
    args: argparse.Namespace, run_cfg: dict, cfg: dict, cmp_run_cfgs: list[dict], out_dir: Path,
) -> str:
    """One comprehensive, human-readable block covering every hyperparameter
    this run's plots depend on -- the duo (model names/norm/optimizer per
    side), adaptation_batch_size (cfg['BS']) vs. each calibrator's own
    proxy_batch_size (never the same knob -- see JointProxyWeighted's module
    docstring), mode/steps/num_samples/fit_num_samples/seed, EVAL and
    CALIBRATOR corruptions/severities, the PRIMARY calib_config's dict
    verbatim, and every --compare_configs entry's dict verbatim (fixed_ts_
    config already resolved to its actual path-or-None by the time this is
    called -- see _run/main).

    Printed once at the start of a run (see _run) AND saved to
    out_dir/run_config.txt -- one call builds both, so the two can never
    drift apart the way a separately-maintained log message and file would.
    """
    lines = [
        "=" * 78,
        "RUN CONFIGURATION",
        "=" * 78,
        f"out_dir: {out_dir}",
        f"duo: LARGE={cfg['LARGE']['NAME']} (norm={cfg['LARGE']['NORM']})  "
        f"SMALL={cfg['SMALL']['NAME']} (norm={cfg['SMALL']['NORM']})",
        f"mode={args.mode}  steps={args.steps}",
        f"adaptation_batch_size (cfg.BS)={cfg['BS']}  workers={cfg['WORKERS']}",
        f"num_samples={args.num_samples}  fit_num_samples={args.fit_num_samples}  seed={args.seed}",
        f"eval/corruptions={cfg['EVAL']['CORRUPTIONS']}",
        f"eval/severities={cfg['EVAL']['SEVERITIES']}",
        f"calibrator_dev/corruptions={cfg.get('CALIBRATOR', {}).get('CORRUPTIONS')}",
        f"calibrator_dev/severities={cfg.get('CALIBRATOR', {}).get('SEVERITIES')}",
        f"large/optim={cfg['LARGE']['OPTIM']}",
        f"small/optim={cfg['SMALL']['OPTIM']}",
        "",
        "-" * 78,
        f"PRIMARY calib_config: {run_cfg['name']!r}",
        "-" * 78,
    ]
    lines += [f"  {k}: {v}" for k, v in run_cfg.items()]
    if cmp_run_cfgs:
        lines += ["", "-" * 78, f"COMPARE configs ({len(cmp_run_cfgs)}):", "-" * 78]
        for cmp_cfg in cmp_run_cfgs:
            lines.append(f"[{cmp_cfg['name']}]")
            lines += [f"  {k}: {v}" for k, v in cmp_cfg.items()]
            lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


def _wandb_config(args: argparse.Namespace, run_cfg: dict, cfg: dict) -> dict:
    config = {
        "mode": args.mode,
        "steps": args.steps,
        "num_samples": args.num_samples,
        "fit_num_samples": args.fit_num_samples,
        "seed": args.seed,
        "batch_size": cfg["BS"],
        "eval/corruptions": cfg["EVAL"]["CORRUPTIONS"],
        "eval/severities": cfg["EVAL"]["SEVERITIES"],
        "large/name": cfg["LARGE"]["NAME"],
        "small/name": cfg["SMALL"]["NAME"],
    }
    # Whatever the calib_config file actually declared, verbatim -- avoids
    # hand-maintaining a per-calibration_mode field list here that would
    # silently drift out of sync with cfgs/calib_configs/*.json.
    config.update({f"calib/{k}": v for k, v in run_cfg.items() if k != "name"})
    return config


def _make_wandb_run(args: argparse.Namespace, run_cfg: dict, cfg: dict, run_name: str):
    if not args.use_wandb:
        return None
    duo_tag = f"{cfg['LARGE']['NAME']}+{cfg['SMALL']['NAME']}"
    group = f"{duo_tag}__{args.wandb_group}" if args.wandb_group else None
    return wandb.init(
        project=args.wandb_project, name=run_name, group=group,
        tags=[cfg["LARGE"]["NAME"], cfg["SMALL"]["NAME"], run_cfg["calibration_mode"], args.mode],
        config=_wandb_config(args, run_cfg, cfg),
        settings=wandb.Settings(silent=True),  # suppress wandb's own console chatter
    )


def _run(
    args: argparse.Namespace, run_cfg: dict, cfg: dict, out_dir: Path, run_name: str, wandb_run,
) -> tuple[list[dict], list[dict], list[dict], list, bool, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with _quiet_stdout("model loading"):
        large_model, large_preprocess = get_model(cfg["LARGE"]["NAME"])
        small_model, small_preprocess = get_model(cfg["SMALL"]["NAME"])
    large_model, small_model = large_model.to(device), small_model.to(device)

    # --cache_logits: one file per COMPLETED corruption under out_dir/
    # logits_cache/, written at _on_corruption_end below -- lets a LATER
    # invocation (--csv_dir + --compare_configs, see
    # _replay_compare_configs_from_cache) add brand new calibrators to this
    # run's plots with zero model forward passes, by re-calibrating these
    # exact saved z_large/z_small instead of re-running the duo. Valid
    # regardless of --mode: every --compare_configs entry already only ever
    # re-combines the SAME z_large/z_small a batch produced (see below), so
    # caching those same tensors changes nothing about what gets computed,
    # only WHEN. Off by default -- a full multi-corruption run's logits can
    # be several GB.
    logits_cache_dir = out_dir / "logits_cache" if args.cache_logits else None
    if logits_cache_dir is not None:
        logits_cache_dir.mkdir(parents=True, exist_ok=True)

    # args.fit_num_samples (NOT args.num_samples) here: any calibrator-
    # construction-time dev fitting a run_cfg triggers -- fit_beta's beta
    # grid search, or _fit_and_save_calibration_maps for a calib_method !=
    # identity -- only needs enough held-out CALIBRATOR.CORRUPTIONS samples
    # to estimate a handful of scalars/maps, not the full eval sample count.
    # args.num_samples is reserved for the actual EVAL loop below (see
    # evaluate_dynamic_duo), which needs many samples for stable
    # per-corruption accuracy curves -- conflating the two meant a
    # fit_beta=true run_cfg re-ran its ENTIRE dev pass at eval-sized
    # num_samples for no benefit (e.g. 50000 samples x 4 dev corruptions to
    # grid-search 7 beta scalars).
    with _quiet_stdout(run_cfg["name"]):
        calibrator = _build_calibrator(
            run_cfg, cfg, large_model, large_preprocess, small_model, small_preprocess,
            device, args.fit_num_samples, args.seed, csv_path=str(out_dir / "proxy_log"), verbose=False,
        )
    # JointProxyWeighted appends its own timestamp suffix to the csv_path
    # given above and writes to it incrementally as the run progresses (see
    # _on_corruption_end below) -- so this path is fixed for the whole run
    # once the calibrator is constructed, unlike _load_proxy_log_csv's glob
    # (used by --csv_dir re-plotting, where no live calibrator instance is
    # around to ask directly).
    proxy_csv_path = getattr(calibrator, "_csv_path", None)

    # --compare_configs: extra calibrators built against the SAME large/small
    # model instances as `calibrator` above (before setup_duo's
    # configure_model/configure_model_frozen calls, matching how `calibrator`
    # itself is built), so each one's accuracy line reflects whatever
    # adaptation state those models are ACTUALLY in at that point in the
    # stream -- compare_calibrators.py's fixed_ts_reference pattern (see its
    # _on_batch), generalized from one hardcoded reference to an arbitrary
    # list. Populated below, after setup_duo, once large/small are finalized
    # and register_hooks (prototype proxies) can target the exact objects
    # the primary calibrator's own hooks were registered on. Also built at
    # args.fit_num_samples, not args.num_samples -- see the comment on
    # `calibrator`'s own construction above; a compare-configs file with
    # several fit_beta=true entries makes this decoupling matter even more,
    # since each one otherwise re-ran its own full-sized dev pass.
    compare_calibrators: list[tuple[str, object]] = []
    cmp_run_cfgs: list[dict] = []
    if args.compare_configs:
        cmp_run_cfgs = _load_run_configs(args.compare_configs)
        for cmp_cfg in cmp_run_cfgs:
            if "fixed_ts_config" in cmp_cfg:
                cmp_cfg["fixed_ts_config"] = _resolve_fixed_ts_config(cmp_cfg["fixed_ts_config"])
            with _quiet_stdout(cmp_cfg["name"]):
                cmp_calibrator = _build_calibrator(
                    cmp_cfg, cfg, large_model, large_preprocess, small_model, small_preprocess,
                    device, args.fit_num_samples, args.seed,
                    csv_path=str(out_dir / f"compare_{cmp_cfg['name']}"), verbose=False,
                )
            if (cmp_cfg["calibration_mode"] == "proxy_weighted"
                    and getattr(cmp_calibrator, "proxy_kind", None) == "prototype"):
                cmp_calibrator.register_hooks(large_model, small_model)
            compare_calibrators.append((cmp_cfg["name"], cmp_calibrator))

    manifest = _format_run_manifest(args, run_cfg, cfg, cmp_run_cfgs, out_dir)
    print(manifest)
    (out_dir / "run_config.txt").write_text(manifest + "\n")

    duo = setup_duo(
        large=large_model, large_preprocess=large_preprocess,
        small=small_model, small_preprocess=small_preprocess,
        mode=args.mode, joint_calibrator=calibrator, calibration_mode=run_cfg["calibration_mode"],
        cfg=cfg, steps=args.steps,
    )

    # This run's own duo (always shown -- see the module docstring) plus one
    # dashed line per --compare_configs entry, in the file's own order, from
    # EXTRA_SERIES_PALETTE (cycled past 5 entries) -- fed to every
    # per-corruption plot via _write_plots below.
    extra_series: list[tuple[str, str, str]] = [("duo_acc", C_GATE, run_cfg["name"])]
    for i, (name, _) in enumerate(compare_calibrators):
        extra_series.append((f"cmp_{name}_acc", EXTRA_SERIES_PALETTE[i % len(EXTRA_SERIES_PALETTE)], name))

    batch_records: list[dict] = []
    corruption_boundaries: list[dict] = []
    proxy_rows: list[dict] = []
    has_batch_plot, has_proxy_plot = False, False
    # --cache_logits buffer for the CURRENT corruption -- reset at
    # _on_corruption_start, appended to in _on_batch, saved+cleared at
    # _on_corruption_end (once the whole stream has actually completed, so a
    # SLURM walltime kill mid-corruption leaves no truncated/misleading cache
    # file for it -- same convention as batch_diagnostics.csv itself).
    cache_buf = {"z_l": [], "z_s": [], "labels": []}

    def _on_corruption_start(corruption, severity):
        corruption_boundaries.append({"idx": len(batch_records), "label": f"{corruption}/s{severity}"})
        for _, cmp_calibrator in compare_calibrators:
            if hasattr(cmp_calibrator, "set_corruption"):
                cmp_calibrator.set_corruption(f"{corruption}/s{severity}")
        cache_buf["z_l"].clear(); cache_buf["z_s"].clear(); cache_buf["labels"].clear()

    def _on_batch(batch_idx, prefix, duo, outputs, z_large, z_small, labels):
        row = {"global_idx": len(batch_records), "corruption": prefix.rstrip("/"), "n": labels.shape[0]}
        for name in ("large", "small", "duo"):
            d = duo._diag[name]
            row[f"{name}_acc"] = d["acc_last"]
            row[f"{name}_nll"] = d["nll_last"]
            row[f"{name}_ent"] = d["ent_last"]
            row[f"{name}_acc_run"] = d["acc_sum"] / d["n"] if d["n"] > 0 else float("nan")
            row[f"{name}_nll_run"] = d["nll_sum"] / d["n"] if d["n"] > 0 else float("nan")
            row[f"{name}_ent_run"] = d["ent_sum"] / d["n"] if d["n"] > 0 else float("nan")
        w_l = getattr(duo.joint_calibrator, "last_w_l", None)
        if w_l is not None:
            row["w_l"] = w_l

        # --compare_configs: re-calibrate this SAME batch's z_large/z_small
        # through every extra calibrator (no extra model forward pass) and
        # record its accuracy -- see plot_per_corruption_proxy_vs_accuracy's
        # extra_series for how these get plotted.
        for name, cmp_calibrator in compare_calibrators:
            if hasattr(cmp_calibrator, "set_labels"):
                cmp_calibrator.set_labels(labels)
            with torch.no_grad():
                z_cmp = cmp_calibrator.calibrate(z_large, z_small)
            labels_dev = labels.to(z_cmp.device)
            row[f"cmp_{name}_acc"] = float((z_cmp.argmax(1) == labels_dev).float().mean())

        if logits_cache_dir is not None:
            cache_buf["z_l"].append(z_large.detach().cpu())
            cache_buf["z_s"].append(z_small.detach().cpu())
            cache_buf["labels"].append(labels.detach().cpu())

        batch_records.append(row)

        if wandb_run is not None:
            # commit=False: buffered into the SAME step evaluate_dynamic_duo's
            # own (unconditional, every batch) wandb_run.log() call flushes
            # right after this returns -- so gate/proxy internals land on
            # exactly the same x-axis step as that batch's accuracy/NLL/
            # entropy, instead of splitting into two adjacent steps.
            log_dict = {}
            if w_l is not None:
                log_dict["gate/w_l"] = w_l
            if run_cfg["calibration_mode"] == "proxy_weighted":
                for attr, key in _GATE_INTERNALS:
                    val = getattr(duo.joint_calibrator, attr, None)
                    if val is not None:
                        log_dict[key] = val
            if log_dict:
                wandb_run.log(log_dict, commit=False)

    def _on_corruption_end(corruption, severity, metrics_by_model):
        # Overwrite batch_diagnostics.csv and every plot from whatever's been
        # recorded so far, after EVERY corruption rather than only once at
        # the very end -- see _write_plots' docstring for why (a SLURM
        # walltime kill mid-run, like logs/5394206_plot_diagnostics.err,
        # otherwise leaves nothing behind but the calibrator's own
        # incrementally-written proxy_log CSV).
        nonlocal proxy_rows, has_batch_plot, has_proxy_plot
        if batch_records:
            with (out_dir / "batch_diagnostics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(batch_records[0].keys()))
                writer.writeheader()
                writer.writerows(batch_records)

        if logits_cache_dir is not None and cache_buf["z_l"]:
            cache_file = logits_cache_dir / f"{corruption}_s{severity}.pt"
            torch.save({
                "z_l": torch.cat(cache_buf["z_l"]),
                "z_s": torch.cat(cache_buf["z_s"]),
                "labels": torch.cat(cache_buf["labels"]),
            }, cache_file)

        if proxy_csv_path is not None and Path(proxy_csv_path).exists():
            with Path(proxy_csv_path).open() as f:
                proxy_rows = list(csv.DictReader(f))

        # Quiet: _write_plots (re)writes EVERY plot/CSV seen so far, called
        # after EVERY corruption -- its own per-file "wrote ..." lines would
        # otherwise dominate the log with a growing, mostly-redundant flood.
        # evaluate_dynamic_duo's own "{corruption}/s{severity}: duo=... "
        # line (unaffected by this) is this run's per-corruption heartbeat.
        with _quiet_stdout("plots"):
            has_batch_plot, has_proxy_plot = _write_plots(
                batch_records, corruption_boundaries, proxy_rows, out_dir, args.ema_window,
                extra_series,
            )

    results_rows = evaluate_dynamic_duo(
        duo, cfg, num_samples=args.num_samples, seed=args.seed,
        wandb_run=wandb_run, run_name=run_name,
        on_corruption_start=_on_corruption_start, on_batch=_on_batch,
        on_corruption_end=_on_corruption_end,
    )

    if batch_records:
        print(f"Wrote {len(batch_records)} rows to {out_dir / 'batch_diagnostics.csv'}")
    else:
        print("No batches were recorded -- nothing was plotted.")

    # Plots/CSVs were already written after each corruption (see
    # _on_corruption_end above) -- batch_records/proxy_rows haven't changed
    # since the last one ran, so has_batch_plot/has_proxy_plot from that
    # final call are already the complete, final state.
    return batch_records, corruption_boundaries, proxy_rows, results_rows, has_batch_plot, has_proxy_plot


def _log_wandb_artifacts(
    wandb_run, out_dir: Path, has_batch_plot: bool, has_proxy_plot: bool,
    proxy_rows: list[dict], results_rows: list[dict],
) -> None:
    """Everything that can only be logged AFTER the run finishes and the PNGs
    exist: the figures as Images, the full proxy log as a Table, and the
    final per-corruption average as run.summary (for quick scanning in the
    wandb Runs table without opening the run)."""
    media = {}
    if has_batch_plot:
        media["plots/batch_diagnostics"] = wandb.Image(str(out_dir / "batch_diagnostics.png"))
    if has_proxy_plot:
        media["plots/proxy_diagnostics"] = wandb.Image(str(out_dir / "proxy_diagnostics.png"))
    if media:
        wandb_run.log(media)

    if proxy_rows:
        # proxy_rows came back from csv.DictReader -- every value is still a
        # str. "corruption" is the only genuinely textual column; cast
        # everything else back to int/float so the wandb Table's columns are
        # numeric (sortable, plottable) rather than opaque text.
        _INT_FIELDS = {"n_refreshes", "n"}
        table = wandb.Table(columns=JointProxyWeighted._CSV_FIELDS)
        for row in proxy_rows:
            values = []
            for c in JointProxyWeighted._CSV_FIELDS:
                if c == "corruption":
                    values.append(row[c])
                elif c in _INT_FIELDS:
                    values.append(int(row[c]))
                else:
                    values.append(float(row[c]))
            table.add_data(*values)
        wandb_run.log({"proxy_diagnostics_table": table})

    avg_row = next((r for r in results_rows if r.get("corruption") == "average"), None)
    if avg_row is not None:
        for k, v in avg_row.items():
            if isinstance(v, (int, float)):
                wandb_run.summary[k] = v


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv_dir", type=str, default=None,
                    help="Skip running the duo entirely and instead re-plot from an existing "
                         "run's own output directory (reads batch_diagnostics.csv and "
                         "proxy_log_*.csv back in) -- e.g. after a plotting-only change like "
                         "--ema_window or a line-style tweak that doesn't need fresh model "
                         "forward passes. OVERWRITES every PNG (and the per-corruption CSVs) "
                         "already in that directory. Normally every other argument below except "
                         "--ema_window is ignored (no duo config/calibrator is built) -- UNLESS "
                         "--compare_configs is ALSO given, which instead replays those new "
                         "calibrators against that run's cached logits (requires the original "
                         "run to have used --cache_logits) and merges the new columns in before "
                         "plotting -- see --compare_configs and _replay_compare_configs_from_cache.")
    add_duo_config_arg(p)
    p.add_argument("--calib_config", type=str, default=None,
                    help="Path to a JSON file holding ONE run_cfg dict -- calibration_mode plus "
                         "all its knobs (proxy_kind, calib_method, filter_kind, proxy_batch_size, "
                         "beta, fixed_ts_config, ...). Same shape as one entry in "
                         "cfgs/compare_runs/*.json -- see cfgs/calib_configs/ for ready-made ones. "
                         "Required unless --csv_dir is given.")
    p.add_argument("--compare_configs", type=str, default=None,
                    help="Path to a JSON file holding a LIST of run_cfg dicts (compare_calibrators."
                         "py's --configs_file shape, e.g. cfgs/compare_runs/default.json). Each one "
                         "is re-evaluated on this run's own z_large/z_small every batch (no extra "
                         "model forward passes) and its resulting duo accuracy is plotted, alongside "
                         "this run's own duo accuracy and the two input models, as one dashed line "
                         "(plus an avg-accuracy tag) per calibrator in every per-corruption plot -- "
                         "see plot_per_corruption_proxy_vs_accuracy's extra_series. Optional. Also "
                         "usable WITH --csv_dir (instead of --calib_config) to add these calibrators "
                         "to an already-finished run's plots without rerunning the duo -- requires "
                         "that run to have used --cache_logits; --config must point at the SAME "
                         "duo config that run used (for model names + CALIBRATOR corruptions).")
    p.add_argument("--cache_logits", action="store_true",
                    help="Save z_large/z_small/labels for every batch to out_dir/logits_cache/ "
                         "(one file per COMPLETED corruption). Lets a LATER invocation "
                         "(--csv_dir + --compare_configs) add brand new calibrators to this run's "
                         "plots afterward with NO model forward passes over the eval set -- see "
                         "--compare_configs. Off by default: a full multi-corruption run's logits "
                         "can be several GB. Not needed for --compare_configs entries known at "
                         "THIS run's launch time -- those are already computed live regardless.")
    p.add_argument("--mode", type=str, default="no_adapt", choices=sorted(_MODES))
    p.add_argument("--steps", type=int, default=1)
    add_num_samples_arg(p)
    p.add_argument("--fit_num_samples", type=int, default=5000,
                    help="Samples per CALIBRATOR.CORRUPTIONS dev stream used ONLY for "
                         "calibrator-construction-time fitting -- fit_beta's beta grid search, "
                         "and _fit_and_save_calibration_maps for a calib_method != 'identity' -- "
                         "for BOTH --calib_config and every --compare_configs entry. Decoupled "
                         "from --num_samples (the eval loop's own sample count, which needs to "
                         "be large for stable per-corruption accuracy curves): grid-searching a "
                         "handful of beta scalars or fitting a calibration map doesn't need "
                         "nearly that many dev samples, and reusing --num_samples for both meant "
                         "a fit_beta=true run_cfg re-ran its entire dev pass at eval-sized "
                         "num_samples for no benefit. Pass the same value as --num_samples to "
                         "restore the old behavior.")
    add_seed_arg(p)
    p.add_argument("--batch_size", type=int, default=None, help="Overrides cfg['BS'].")

    add_out_dir_run_name_args(
        p, out_dir_default="out/run_diagnostics",
        run_name_help="Subdirectory name under --out_dir, and the wandb run name. Default: "
                       "auto-generated from the duo's model names/calib_config name/mode/timestamp.",
    )
    p.add_argument("--ema_window", type=int, default=DEFAULT_EMA_WINDOW,
                    help="Span (in points) of the EMA smoothing applied to every plotted line "
                         "(accuracy, NLL, entropy, proxy scores, gate weight) across all three "
                         "diagnostics plots -- alpha = 2/(window+1). Purely a plotting knob; "
                         "unrelated to a proxy_weighted calib_config's own 'filter_kind': 'ema' "
                         "gate-smoothing knob (Section 4), which affects the logged values "
                         "themselves, not just how they're plotted.")

    wandb_group_args = p.add_argument_group("wandb options")
    wandb_group_args.add_argument("--use_wandb", dest="use_wandb", action="store_true", default=True,
                                   help="Log everything to Weights & Biases (default: on).")
    wandb_group_args.add_argument("--no_wandb", dest="use_wandb", action="store_false",
                                   help="Disable wandb logging entirely (quick local iteration).")
    add_wandb_project_group_args(
        wandb_group_args,
        group_help="Optional shared group tag (e.g. to cluster several manual invocations in "
                    "the W&B UI). Always prefixed with the duo's model names. Default: "
                    "ungrouped (a standalone run).",
    )
    args = p.parse_args()
    _quiet_mode()

    if args.csv_dir is not None:
        csv_dir = Path(args.csv_dir)
        if not csv_dir.is_dir():
            p.error(f"--csv_dir {csv_dir} is not a directory.")
        if args.compare_configs:
            _replay_compare_configs_from_cache(
                csv_dir, args.config, args.compare_configs,
                args.fit_num_samples, args.seed, args.ema_window,
            )
        else:
            _replot_from_csv_dir(csv_dir, args.ema_window)
        return

    if args.calib_config is None:
        p.error("--calib_config is required unless --csv_dir is given.")

    cfg = load_config(args.config)
    if args.batch_size:
        cfg["BS"] = args.batch_size

    run_cfg = _load_calib_config(args.calib_config)
    if "fixed_ts_config" in run_cfg:
        run_cfg["fixed_ts_config"] = _resolve_fixed_ts_config(run_cfg["fixed_ts_config"])
    if (run_cfg["calibration_mode"] == "proxy_weighted"
            and run_cfg.get("calib_map") is None
            and run_cfg.get("calib_method", "identity") != "identity"):
        run_cfg["calib_map"] = _default_calib_map(cfg, run_cfg["proxy_kind"], run_cfg["calib_method"])
        print(f"No 'calib_map' in {args.calib_config} with calib_method={run_cfg['calib_method']!r}; "
              f"auto-naming it {run_cfg['calib_map']!r} (fit fresh if not already cached).")

    duo_tag = f"{cfg['LARGE']['NAME']}+{cfg['SMALL']['NAME']}"
    run_name = args.run_name or (
        f"{duo_tag}__{run_cfg['name']}__{args.mode}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = _make_wandb_run(args, run_cfg, cfg, run_name)
    if wandb_run is not None:
        print(f"wandb run: {wandb_run.url}")

    # Plots/CSVs are written incrementally, after every corruption (see
    # _run's _on_corruption_end) -- has_batch_plot/has_proxy_plot below
    # already reflect the final, complete state; nothing left to (re)write
    # here.
    _, _, proxy_rows, results_rows, has_batch_plot, has_proxy_plot = _run(
        args, run_cfg, cfg, out_dir, run_name, wandb_run,
    )

    if wandb_run is not None:
        _log_wandb_artifacts(wandb_run, out_dir, has_batch_plot, has_proxy_plot, proxy_rows, results_rows)
        wandb_run.finish()

    avg_row = next((r for r in results_rows if r.get("corruption") == "average"), None)
    if avg_row is not None:
        print(f"\nFinal average: duo={avg_row['duo/accuracy']:.4f}  "
              f"large={avg_row['large/accuracy']:.4f}  small={avg_row['small/accuracy']:.4f}")
    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
