"""
scripts/calibrate_gate_oracle.py
=================================
Isolates the SECOND half of filtered-proxy soft weighting -- the gate

    w_l = sigmoid(beta * (x_l - x_s))

plus the combination -- from the FIRST half (proxy
score -> calibration -> temporal filter), by driving the gate with the
ORACLE proxy_kind (src/reliability/proxies/oracle.py): the model's literal
per-batch accuracy, read straight from labels instead of estimated from a
label-free score.

Why
---
sweep_proxies.py answers "how good is a *label-free* proxy at tracking who's
actually more accurate" (Stage 1: proxy + calibration). This script answers
the complementary question: "given a PERFECT reliability signal, what is the
best this gate + combination mechanism can possibly do" (Stage 2's ceiling).
Any gap between this oracle ceiling and the real proxy_weighted numbers from
compare_calibrators.py / sweep_proxies.py is attributable to Stage 1 (the
proxy itself is imperfect), not Stage 2 (the gate math) -- those two
failure modes need completely different fixes, so it matters which one
you're actually looking at.

Even under oracle, a batch's true accuracy is a noisy estimate of the
corruption's long-run accuracy whenever the batch is small (a 128-image
batch's accuracy jitters batch to batch even for a fixed, non-adapting
model). So sweeping filter_kind here isolates how much of the temporal
filter's benefit is genuine batch-noise smoothing versus (in the real,
non-oracle setting) compensating for a bad proxy: if "none" wins under
oracle but "ema"/"kalman" win with real proxies (see sweep_proxies.py /
compare_calibrators.py results), the filter's value is entirely about proxy
noise, not batch noise -- and vice versa.

Design
------
Every (beta, filter_kind, proxy_batch_size) combination in the sweep
sees IDENTICAL model logits: mode=no_adapt means frozen models with pure
batch-stat BatchNorm (see CLAUDE.md's TENT gotcha -- with track_running_stats
disabled, a batch's output depends only on that batch's own composition,
never on the order or history of prior calls). So a corruption's logits only
depend on (corruption, severity, num_samples, seed, batch_size), never on
the calibrator being swept. Logits are therefore collected ONCE per
(corruption, severity) via a single forward pass and cached in memory; every
sweep point just replays the cheap gate+combine math against those cached
tensors, re-chunked into adaptation batches in their original order so the
stateful temporal filters see the exact same batch sequence a real run
would (this matters: EMA/Kalman are order-dependent).

Reference numbers printed alongside the sweep table: single-model baselines
(large/small accuracy alone, the floor) and the fixed_ts/coca_ts baseline
rows folded into the same table (see _baseline_accuracy) -- these fuse both
models' logits continuously rather than switching between them per chunk, so
they're the right bar to compare the oracle gate against. (A prior version
of this script also printed a "hard-selection ceiling" -- beta -> infinity,
pick whichever model is more accurate per chunk with certainty -- as an
upper bound on the sweep. It isn't one: fixed_ts/coca_ts routinely beat it,
since continuous soft pooling exploits per-sample complementary evidence
from both models that a hard per-chunk switch structurally cannot, even
with perfect oracle labels. Removed to avoid the false impression of a
ceiling.)

--optimal_w: an even tighter ceiling than the beta sweep
--------------------------------------------------------
The beta sweep above still constrains w_l to the PARAMETERIZED FORM
sigmoid(beta * (x_l - x_s)) -- it asks "what's the best beta", not "what's
the best possible w_l". --optimal_w (see src.calibrators.
joint_optimal_w_oracle.optimal_w_nll) removes that constraint: for every
adaptation batch it directly solves for the scalar
w_l in [0, 1] that minimizes THAT batch's own NLL, via a bounded 1-D
optimizer. This is exact, not a heuristic: for fixed T_l/T_s,
z_duo(w_l) = w_l*(z_l/T_l)+(1-w_l)*(z_s/T_s) is AFFINE in w_l, and cross-
entropy is convex in its logit argument, so NLL(w_l) is provably convex on
[0, 1] -- a bounded scalar optimizer is guaranteed to find the GLOBAL
optimum, no local-minima risk. Any gap between --optimal_w's ceiling and
the beta sweep's best row is attributable to the sigmoid FORM itself being
a worse fit than the true per-batch optimum -- a THIRD failure mode
alongside Stage 1 (proxy imperfection) and Stage 2's beta/filter choices.
Like proxy_kind='oracle', this cheats by construction (solves using the
batch's own test labels) -- an upper-bound diagnostic only, never a
deployable method. Keeps T_l/T_s fixed from the same base_ts as everything
else (never re-optimized here), so it isolates the gating ceiling
specifically, not a re-calibrated one.

Usage
-----
python scripts/calibrate_gate_oracle.py --config cfgs/dynamic_duo_config.yaml \
    --num_samples 5000 --seed 0 --fixed_ts_config checkpoints/fixed_ts/default \
    --csv_path out/gate_calibration_oracle.csv

# Also sweep how much aggregating more true-accuracy samples per gate update
# helps, beyond the default (one adaptation batch per gate refresh):
python scripts/calibrate_gate_oracle.py --config cfgs/dynamic_duo_config.yaml \
    --proxy_batch_sizes 128 512 1024 --csv_path out/gate_calibration_oracle.csv

# Add the optimal-w-per-batch ceiling row alongside the beta sweep and
# fixed_ts/coca_ts baselines:
python scripts/calibrate_gate_oracle.py --config cfgs/dynamic_duo_config.yaml \
    --fixed_ts_config checkpoints/fixed_ts/default --run_baselines --optimal_w \
    --csv_path out/gate_calibration_oracle.csv
"""

from __future__ import annotations

import argparse
import csv as csv_module
import itertools
from pathlib import Path

import calibration as cal
import torch
import torch.nn.functional as F

from src.utils.data import load_config, load_imagenetC
from src.utils.model import get_model
from src.tta.tent import configure_model_frozen, softmax_entropy
from src.utils.stream_cache import duo_cache_dir, stream_key, collect_stream, load_or_collect_stream
from src.calibrators.joint_fixed_TS import JointFixedTS
from src.calibrators.joint_coca import JointCoca
from src.calibrators.joint_proxy_weighted import JointProxyWeighted
from src.calibrators.joint_optimal_w_oracle import combine as _combine, optimal_w_nll as _optimal_w_nll
from src.reliability.proxies.stats import ProxyStats, FeatureExtractor
from src.reliability.calibration.logit import to_logit

_TABLE_COLUMNS = [
    "beta", "filter_kind", "proxy_batch_size",
    "duo_acc", "duo_nll", "duo_ece", "duo_entropy", "mean_w_l", "std_w_l", "large_acc", "small_acc",
]


def _build_models(cfg: dict, device: torch.device):
    """Frozen, batch-stat-BN models -- mirrors src.utils.logits.get_model_logits'
    tent_mode setup exactly, so cached logits match what a real no_adapt run
    (evaluate_dynamic_duo / compare_calibrators.py) would produce."""
    large, large_preprocess = get_model(cfg["LARGE"]["NAME"], freeze=True)
    small, small_preprocess = get_model(cfg["SMALL"]["NAME"], freeze=True)
    large = large.to(device).eval()
    small = small.to(device).eval()
    large = configure_model_frozen(large, norm_type=cfg["LARGE"]["NORM"])
    small = configure_model_frozen(small, norm_type=cfg["SMALL"]["NORM"])
    return large, large_preprocess, small, small_preprocess


def _collect_eval_streams(
    cfg, device, num_samples, seed, large, large_preprocess, small, small_preprocess,
    use_cache: bool = False, overwrite_cache: bool = False,
):
    """One forward pass per (corruption, severity) in cfg['EVAL'] -- cached in
    CPU memory, reused by every sweep config below. Goes through
    src.utils.stream_cache (the same disk cache sweep_proxies.py and
    compare_calibrators.py use) so a repeat run against the same duo/data
    skips the model forward pass entirely; this script only needs (z_l, z_s,
    labels), so the cached penultimate features are simply discarded."""
    cache_dir = duo_cache_dir(cfg["LARGE"]["NAME"], cfg["SMALL"]["NAME"]) if use_cache else None
    ext_l = FeatureExtractor(large, cfg["LARGE"]["NAME"])
    ext_s = FeatureExtractor(small, cfg["SMALL"]["NAME"])
    streams = {}
    try:
        for severity in cfg["EVAL"]["SEVERITIES"]:
            for corruption in cfg["EVAL"]["CORRUPTIONS"]:
                key = stream_key(f"{corruption}_s{severity}", num_samples, seed)

                def _collect(corruption=corruption, severity=severity):
                    loader = load_imagenetC(
                        cfg["TEST_DIR"], severities=severity, corruption_types=[corruption],
                        device=device, batch_size=cfg["BS"], num_workers=cfg["WORKERS"],
                        num_samples=num_samples, seed=seed,
                    )
                    return collect_stream(loader, large_preprocess, small_preprocess, ext_l, ext_s, device)

                print(f"Collecting eval stream {corruption}/s{severity}...")
                z_l, z_s, _f_l, _f_s, labels = load_or_collect_stream(
                    cache_dir, key, _collect, use_cache=use_cache, overwrite_cache=overwrite_cache,
                )
                streams[(corruption, severity)] = (z_l, z_s, labels)
    finally:
        ext_l.remove()
        ext_s.remove()
    return streams


def _diagnose_signal(
    streams: dict, proxy_batch_sizes: list[int],
    fixed_ts_baseline: dict | None = None, coca_baseline: dict | None = None,
) -> None:
    """Print the RAW oracle accuracy gap (acc_l - acc_s) per (corruption,
    severity, proxy_batch_size) chunk, computed directly from the cached
    streams -- bypassing JointProxyWeighted/calibration/filter/beta/sigmoid
    entirely. This is the signal the gate sees BEFORE any of that, so it
    isolates whether a flat mean_w_l~=0.5 sweep result is:
      (a) a real, consistently-signed per-corruption gap that just cancels
          out when macro-averaged across corruptions with opposite favorites
          (mean gap column would vary in SIGN across corruptions), or
      (b) chunk-to-chunk noise already swamping a small true gap even within
          a single corruption (std gap >> |mean gap| -> low signal/noise),
          in which case no beta can help -- half the chunks have the "wrong"
          sign no matter how sharply beta reacts to it, or
      (c) neither of the above (mean gap sizeable and consistent, signal/
          noise not small) -- which would point back at a real bug in the
          gate/beta plumbing rather than a statistical noise-floor issue.

    fixed_ts_baseline/coca_baseline (see _baseline_accuracy) are printed
    alongside purely for reference -- they cost nothing extra to show here
    since main() already needs them for the full sweep's baseline rows, and
    having them next to the raw gap stats means --diagnose_only alone (which
    returns before the expensive beta/filter sweep) still tells you
    where the existing calibrators land, not just the raw signal quality.
    """
    print(f"\n{'=' * 100}\nDIAGNOSTIC: raw oracle accuracy gap (acc_l - acc_s), before any "
          f"calibration/filter/gate\n{'=' * 100}")
    if fixed_ts_baseline is not None:
        print(f"Reference: fixed_ts duo_acc={fixed_ts_baseline['duo_acc']:.4f}  "
              f"duo_nll={fixed_ts_baseline['duo_nll']:.4f}  duo_ece={fixed_ts_baseline['duo_ece']:.4f}")
    else:
        print("Reference: fixed_ts skipped (no --fixed_ts_config given)")
    if coca_baseline is not None:
        print(f"Reference: coca_ts  duo_acc={coca_baseline['duo_acc']:.4f}  "
              f"duo_nll={coca_baseline['duo_nll']:.4f}  duo_ece={coca_baseline['duo_ece']:.4f}")
    header = (f"{'corruption':<22} {'sev':>4} {'pbs':>5} {'chunks':>7}  "
              f"{'mean acc_l':>11} {'mean acc_s':>11} {'mean gap':>9} {'std gap':>8}  "
              f"{'%L wins':>8} {'%S wins':>8} {'%tie':>6}")
    print(header)
    print("-" * len(header))
    for pbs in proxy_batch_sizes:
        pooled_gaps: list[float] = []
        for (corruption, severity), (z_l, z_s, labels) in streams.items():
            n = z_l.shape[0]
            gaps, accs_l, accs_s = [], [], []
            for start in range(0, n, pbs):
                sl = slice(start, min(start + pbs, n))
                acc_l = float((z_l[sl].argmax(1) == labels[sl]).float().mean())
                acc_s = float((z_s[sl].argmax(1) == labels[sl]).float().mean())
                accs_l.append(acc_l); accs_s.append(acc_s); gaps.append(acc_l - acc_s)
            gaps_t = torch.tensor(gaps)
            n_l = sum(1 for g in gaps if g > 1e-9)
            n_s = sum(1 for g in gaps if g < -1e-9)
            n_tie = len(gaps) - n_l - n_s
            print(f"{corruption:<22} {severity:>4} {pbs:>5} {len(gaps):>7}  "
                  f"{sum(accs_l) / len(accs_l):>11.4f} {sum(accs_s) / len(accs_s):>11.4f} "
                  f"{gaps_t.mean().item():>9.4f} {gaps_t.std().item():>8.4f}  "
                  f"{100 * n_l / len(gaps):>7.1f}% {100 * n_s / len(gaps):>7.1f}% {100 * n_tie / len(gaps):>5.1f}%")
            pooled_gaps.extend(gaps)

        pooled = torch.tensor(pooled_gaps)
        std = pooled.std().item()
        snr = abs(pooled.mean().item()) / std if std > 1e-9 else float("inf")
        print(f"  -> pooled @ pbs={pbs}: mean gap={pooled.mean().item():.4f}  "
              f"std gap={std:.4f}  |mean|/std={snr:.3f} "
              f"(<<1 means chunk noise likely swamps the true gap at this pbs)\n")


def _run_config(streams, base_ts, beta, filter_kind, filter_kwargs, proxy_batch_size, batch_size):
    """Replay one (beta, filter_kind, proxy_batch_size) config across all
    cached eval streams, adaptation-batch by adaptation-batch, through the
    REAL JointProxyWeighted gate/combine logic (proxy_kind='oracle'). Uses
    the calibrator's private _forward() directly (bypassing calibrate() /
    calibrate_with_grad()'s per-batch console logging, which would otherwise
    flood stdout across a large sweep) -- the same pattern src.reliability.
    setup.fit_beta already uses for its own beta grid search.

    Returns macro-averaged (over streams) duo accuracy/NLL/ECE/entropy/mean
    gate weight. ECE is computed once per stream over that stream's full
    concatenated predictions (it bins over a distribution, so it can't be
    accumulated batch-by-batch like a sum), then macro-averaged across
    streams -- same convention as accuracy/NLL/mean_w_l here.
    """
    cfg_l = ProxyStats(name="large", num_classes=1000)
    cfg_s = ProxyStats(name="small", num_classes=1000)
    calibrator = JointProxyWeighted(
        proxy_kind="oracle", cfg_l=cfg_l, cfg_s=cfg_s,
        beta=beta, filter_kind=filter_kind, filter_kwargs=filter_kwargs,
        prior_l=to_logit(0.5), prior_s=to_logit(0.5),
        base_ts=base_ts, proxy_batch_size=proxy_batch_size, log_every=0,
    )

    per_stream_rows = []
    for (corruption, severity), (z_l, z_s, labels) in streams.items():
        n = z_l.shape[0]
        # total_samples lets the calibrator flush a trailing proxy-batch
        # remainder instead of leaving it stale (see
        # JointProxyWeighted._maybe_update_gate).
        calibrator.set_corruption(f"{corruption}/s{severity}", total_samples=n)
        correct, nll_sum, entropy_sum, w_l_sum, total = 0.0, 0.0, 0.0, 0.0, 0
        z_duo_chunks = []
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            zl_b, zs_b, y_b = z_l[sl], z_s[sl], labels[sl]
            calibrator.set_labels(y_b)
            with torch.no_grad():
                z_duo, r_l, r_s, a_l, a_s, x_l, x_s, w_l = calibrator._forward(zl_b, zs_b)
            bs = sl.stop - sl.start
            correct += float((z_duo.argmax(1) == y_b).float().sum())
            nll_sum += float(F.cross_entropy(z_duo, y_b, reduction="sum"))
            entropy_sum += float(softmax_entropy(z_duo).sum())
            w_l_sum += w_l * bs
            total += bs
            z_duo_chunks.append(z_duo)
        probs = F.softmax(torch.cat(z_duo_chunks, dim=0), dim=1).numpy()
        ece = cal.get_ece(probs, labels.numpy(), num_bins=15)
        per_stream_rows.append({
            "acc": correct / total, "nll": nll_sum / total, "mean_w_l": w_l_sum / total,
            "entropy": entropy_sum / total, "ece": ece,
        })

    n_streams = len(per_stream_rows)
    return {
        "duo_acc": sum(r["acc"] for r in per_stream_rows) / n_streams,
        "duo_nll": sum(r["nll"] for r in per_stream_rows) / n_streams,
        "mean_w_l": sum(r["mean_w_l"] for r in per_stream_rows) / n_streams,
        "duo_entropy": sum(r["entropy"] for r in per_stream_rows) / n_streams,
        "duo_ece": sum(r["ece"] for r in per_stream_rows) / n_streams,
    }


@torch.no_grad()
def _optimal_weight_accuracy(streams: dict, base_ts, batch_size: int) -> dict:
    """Per-batch ceiling: chunk each stream by the adaptation batch_size (no
    proxy-batch aggregation -- this tests the pure per-batch mixing ceiling,
    not the aggregation/filter stage) and solve _optimal_w_nll (see
    src.calibrators.joint_optimal_w_oracle) fresh for every batch, instead
    of routing through JointProxyWeighted's gate at all. Same
    per-stream-macro-average convention as _run_config/_baseline_accuracy,
    so directly comparable in the same table. Also
    reports std_w_l (across every batch, pooled over all streams): a
    near-zero std here would say the optimum barely moves batch to batch
    (a near-constant w_l might already be close to optimal, hard-selection-
    ceiling style); a wide std says the ceiling genuinely needs to react
    per batch, which the sigmoid gate is at least structurally able to do.
    """
    T_l = float(base_ts.Tl.item()) if base_ts is not None else 1.0
    T_s = float(base_ts.Ts.item()) if base_ts is not None else 1.0

    per_stream_rows = []
    all_w_l: list[float] = []
    for (z_l, z_s, labels) in streams.values():
        n = z_l.shape[0]
        correct, nll_sum, entropy_sum, w_l_sum, total = 0.0, 0.0, 0.0, 0.0, 0
        z_duo_chunks = []
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            zl_b, zs_b, y_b = z_l[sl], z_s[sl], labels[sl]
            w_l, _ = _optimal_w_nll(zl_b, zs_b, y_b, T_l, T_s)
            z_duo = _combine(zl_b, zs_b, w_l, T_l, T_s)
            bs = sl.stop - sl.start
            correct += float((z_duo.argmax(1) == y_b).float().sum())
            nll_sum += float(F.cross_entropy(z_duo, y_b, reduction="sum"))
            entropy_sum += float(softmax_entropy(z_duo).sum())
            w_l_sum += w_l * bs
            total += bs
            all_w_l.append(w_l)
            z_duo_chunks.append(z_duo)
        probs = F.softmax(torch.cat(z_duo_chunks, dim=0), dim=1).numpy()
        ece = cal.get_ece(probs, labels.numpy(), num_bins=15)
        per_stream_rows.append({
            "acc": correct / total, "nll": nll_sum / total, "mean_w_l": w_l_sum / total,
            "entropy": entropy_sum / total, "ece": ece,
        })

    n_streams = len(per_stream_rows)
    return {
        "duo_acc": sum(r["acc"] for r in per_stream_rows) / n_streams,
        "duo_nll": sum(r["nll"] for r in per_stream_rows) / n_streams,
        "mean_w_l": sum(r["mean_w_l"] for r in per_stream_rows) / n_streams,
        "duo_entropy": sum(r["entropy"] for r in per_stream_rows) / n_streams,
        "duo_ece": sum(r["ece"] for r in per_stream_rows) / n_streams,
        "std_w_l": float(torch.tensor(all_w_l).std()) if len(all_w_l) > 1 else float("nan"),
    }


def _baseline_accuracy(streams: dict, calibrator) -> dict:
    """Macro-averaged duo accuracy/NLL (same streams, same per-stream-macro-
    average convention as _run_config, so directly comparable to the
    oracle-gate sweep rows) for a calibrator that needs no per-batch gate
    state -- JointFixedTS (a fixed per-call formula) or JointCoca (re-
    optimizes its temperature fresh from scratch on whatever's passed in,
    internally re-chunked by its own chunk_size) -- so the whole cached
    stream can be passed in one call rather than looping adaptation batches."""
    per_stream = []
    for (z_l, z_s, labels) in streams.values():
        with torch.no_grad():
            z_duo = calibrator.calibrate(z_l, z_s)
        # JointFixedTS.calibrate() always moves its inputs to cuda internally
        # and returns the result there (see JointFixedTS.calibrate), regardless
        # of what device z_l/z_s came in on -- unlike JointProxyWeighted, which
        # only pulls T_l/T_s out as floats. Align labels to z_duo's device
        # rather than assuming streams and calibrator output share one.
        labels = labels.to(z_duo.device)
        acc = float((z_duo.argmax(1) == labels).float().mean())
        nll = float(F.cross_entropy(z_duo, labels, reduction="mean"))
        entropy = float(softmax_entropy(z_duo).mean())
        probs = F.softmax(z_duo, dim=1).cpu().numpy()
        ece = cal.get_ece(probs, labels.cpu().numpy(), num_bins=15)
        per_stream.append({"acc": acc, "nll": nll, "entropy": entropy, "ece": ece})
    n = len(per_stream)
    return {
        "duo_acc": sum(r["acc"] for r in per_stream) / n,
        "duo_nll": sum(r["nll"] for r in per_stream) / n,
        "duo_entropy": sum(r["entropy"] for r in per_stream) / n,
        "duo_ece": sum(r["ece"] for r in per_stream) / n,
    }


def _print_table(rows: list[dict]) -> None:
    header = (f"{'beta':>6} {'filter':<13} {'pbs':>5}  "
              f"{'duo_acc':>8} {'duo_nll':>8} {'duo_ece':>8} {'duo_ent':>8} {'mean_w_l':>9} {'std_w_l':>8}   "
              f"{'large_acc':>9} {'small_acc':>9}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['beta']:>6.2f} {r['filter_kind']:<13} {r['proxy_batch_size']:>5}  "
              f"{r['duo_acc']:>8.4f} {r['duo_nll']:>8.4f} {r['duo_ece']:>8.4f} {r['duo_entropy']:>8.4f} "
              f"{r['mean_w_l']:>9.4f} {r['std_w_l']:>8.4f}   {r['large_acc']:>9.4f} {r['small_acc']:>9.4f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="cfgs/dynamic_duo_config.yaml")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed_ts_config", type=str, default=None,
                         help="JointFixedTS checkpoint folder supplying (T_l, T_s) for the "
                              "Section-5 combination. Omit for T_l=T_s=1.0 (not recommended -- "
                              "must be a checkpoint fit for THIS duo.")
    parser.add_argument("--betas", type=float, nargs="+",
                         default=[0.0, 0.001, 0.01, 0.1, 0.5, 0.75, 1.0])
    parser.add_argument("--filter_kinds", type=str, nargs="+",
                         default=["none", "running_mean", "ema", "kalman"],
                         choices=["none", "running_mean", "ema", "kalman"])
    parser.add_argument("--proxy_batch_sizes", type=int, nargs="+", default=None,
                         help="Defaults to [BS] from the config (gate refreshes every "
                              "adaptation batch). Pass larger values to also see how much "
                              "aggregating more true-accuracy samples per gate update helps.")
    parser.add_argument("--ema_alpha", type=float, default=0.5)
    parser.add_argument("--kalman_q", type=float, default=1e-3)
    parser.add_argument("--kalman_r", type=float, default=1e-1)
    parser.add_argument("--sort_by", type=str, default="duo_acc", choices=_TABLE_COLUMNS)
    parser.add_argument("--csv_path", type=str, default="out/gate_calibration_oracle.csv")
    parser.add_argument("--use_cache", action="store_true",
                         help="Cache each eval (corruption, severity) stream's logits (see "
                              "src.utils.stream_cache — automatic, duo-specific directory, no "
                              "path to pick) so a repeat run against the same duo/--num_samples/"
                              "--seed skips the model forward pass entirely.")
    parser.add_argument("--overwrite_cache", action="store_true",
                         help="With --use_cache, always recompute and overwrite any existing "
                              "cache entries instead of reusing them.")
    parser.add_argument("--diagnose", action="store_true",
                         help="Print the raw oracle accuracy gap (acc_l - acc_s) per corruption/"
                              "severity/proxy_batch_size, bypassing calibration/filter/gate "
                              "entirely (see _diagnose_signal) -- for investigating a flat/"
                              "insensitive-to-beta sweep result.")
    parser.add_argument("--diagnose_only", action="store_true",
                         help="Print the diagnostic and exit before running the (expensive) "
                              "full sweep. Implies --diagnose.")
    parser.add_argument("--use_wandb", action="store_true",
                         help="Log the sweep as one wandb.Table (all rows, every "
                              "_TABLE_COLUMNS field) to the proxy-weighted-duo-calibration "
                              "project, built and pushed incrementally (one new table version "
                              "per config, including the fixed_ts/coca_ts baseline rows) so you "
                              "can watch it fill in live rather than only seeing it once the "
                              "whole sweep finishes — click any column header in the table UI "
                              "to sort by it, rather than being limited to --sort_by's one ranking.")
    parser.add_argument("--wandb_project", type=str, default="proxy-weighted-duo-calibration")
    parser.add_argument("--wandb_group", type=str, default=None,
                         help="Defaults to a timestamp. Always prefixed with the duo's model "
                              "names so two duos' sweeps can never mix in the same wandb group.")
    parser.add_argument("--run_baselines", action="store_true",
                         help="Run the fixed_ts/coca_ts baseline rows (see _baseline_accuracy) "
                              "even if --diagnose_only is set, so you can see where the "
                              "existing calibrators land relative to the raw gap signal.")
    parser.add_argument("--optimal_w", action="store_true",
                         help="Add the optimal-w-per-batch oracle ceiling row (see "
                              "_optimal_weight_accuracy and the module docstring's "
                              "'--optimal_w' section) -- solves for the exact NLL-minimizing "
                              "w_l per adaptation batch directly, unconstrained by the "
                              "sigmoid(beta*(x_l-x_s)) form the rest of this script sweeps, "
                              "isolating whether that form itself (not just its beta) is "
                              "leaving accuracy/NLL on the table.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    proxy_batch_sizes = args.proxy_batch_sizes or [cfg["BS"]]

    base_ts = JointFixedTS.load(args.fixed_ts_config) if args.fixed_ts_config else None
    if base_ts is None:
        print("WARNING: no --fixed_ts_config given; combining at T_l=T_s=1.0. "
              "Pass a checkpoint fit for THIS duo for realistic numbers.")

    print(f"Using device: {device}  |  duo: {cfg['LARGE']['NAME']}+{cfg['SMALL']['NAME']}")
    large, large_preprocess, small, small_preprocess = _build_models(cfg, device)
    streams = _collect_eval_streams(
        cfg, device, args.num_samples, args.seed, large, large_preprocess, small, small_preprocess,
        use_cache=args.use_cache, overwrite_cache=args.overwrite_cache,
    )

    large_acc = sum(
        float((z_l.argmax(1) == labels).float().mean()) for z_l, z_s, labels in streams.values()
    ) / len(streams)
    small_acc = sum(
        float((z_s.argmax(1) == labels).float().mean()) for z_l, z_s, labels in streams.values()
    ) / len(streams)
    print(f"\nSingle-model baselines (macro-avg over {len(streams)} eval streams): "
          f"large={large_acc:.4f}  small={small_acc:.4f}")
    if args.run_baselines:
        print("Running baselines...")
        fixed_ts_baseline = _baseline_accuracy(streams, base_ts) if base_ts is not None else None
        coca = JointCoca(num_steps=10, lr=5e-2, chunk_size=cfg["BS"])
        coca_baseline = _baseline_accuracy(streams, coca)
    else:
        print("Skipping baselines (--run_baselines to enable).")
        fixed_ts_baseline = None
        coca_baseline = None

    if args.diagnose or args.diagnose_only:
        _diagnose_signal(streams, proxy_batch_sizes, fixed_ts_baseline, coca_baseline)
        if args.diagnose_only:
            return

    rows: list[dict] = []
    run = None
    table = None
    group = None
    if args.use_wandb:
        import wandb
        from datetime import datetime
        duo_tag = f"{cfg['LARGE']['NAME']}+{cfg['SMALL']['NAME']}"
        group = f"{duo_tag}__{args.wandb_group or datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run = wandb.init(
            project=args.wandb_project, group=group, name=f"gate_calibration_oracle_{duo_tag}",
            job_type="gate_calibration_oracle", tags=[cfg["LARGE"]["NAME"], cfg["SMALL"]["NAME"]],
        )
        run.summary["large_acc"] = large_acc
        run.summary["small_acc"] = small_acc
        table = wandb.Table(columns=_TABLE_COLUMNS, log_mode="MUTABLE")

    def _add_row(row: dict) -> None:
        """Append to rows and, if --use_wandb, push an updated table version
        right away -- so the wandb UI fills in row-by-row over the course of
        the sweep instead of only appearing once everything finishes."""
        rows.append(row)
        if table is not None:
            table.add_data(*[row[c] for c in _TABLE_COLUMNS])
            run.log({"gate_calibration_oracle": table})

    filter_kwargs = {"alpha": args.ema_alpha, "q": args.kalman_q, "r": args.kalman_r}
    configs = list(itertools.product(proxy_batch_sizes, args.filter_kinds, args.betas))
    total = len(configs)
    for i, (pbs, filter_kind, beta) in enumerate(configs, start=1):
        print(f"[{i}/{total}] pbs={pbs} filter={filter_kind} beta={beta}", end="  ", flush=True)
        result = _run_config(
            streams, base_ts, beta, filter_kind, filter_kwargs, pbs, cfg["BS"],
        )
        print(f"-> duo_acc={result['duo_acc']:.4f} duo_nll={result['duo_nll']:.4f} "
              f"duo_ece={result['duo_ece']:.4f} duo_entropy={result['duo_entropy']:.4f} "
              f"mean_w_l={result['mean_w_l']:.4f}", flush=True)
        _add_row({
            "beta": beta, "filter_kind": filter_kind, "proxy_batch_size": pbs,
            "duo_acc": result["duo_acc"], "duo_nll": result["duo_nll"],
            "duo_ece": result["duo_ece"], "duo_entropy": result["duo_entropy"],
            "mean_w_l": result["mean_w_l"], "std_w_l": float("nan"),
            "large_acc": large_acc, "small_acc": small_acc,
        })

    # Baseline rows (no gate at all) folded into the SAME table so they sort
    # alongside the oracle-gate sweep — beta/proxy_batch_size don't apply
    # to these, filled with sentinels (nan / -1) rather than a separate
    # "config" column, to keep the schema/CSV/wandb Table unchanged.
    if fixed_ts_baseline is not None:
        _add_row({
            "beta": float("nan"), "filter_kind": "fixed_ts", "proxy_batch_size": -1,
            "duo_acc": fixed_ts_baseline["duo_acc"], "duo_nll": fixed_ts_baseline["duo_nll"],
            "duo_ece": fixed_ts_baseline["duo_ece"], "duo_entropy": fixed_ts_baseline["duo_entropy"],
            "mean_w_l": float("nan"), "std_w_l": float("nan"), "large_acc": large_acc, "small_acc": small_acc,
        })
    else:
        print("Skipping fixed_ts baseline row: no --fixed_ts_config given.")

    if coca_baseline is not None:
        _add_row({
            "beta": float("nan"), "filter_kind": "coca_ts", "proxy_batch_size": -1,
            "duo_acc": coca_baseline["duo_acc"], "duo_nll": coca_baseline["duo_nll"],
            "duo_ece": coca_baseline["duo_ece"], "duo_entropy": coca_baseline["duo_entropy"],
            "mean_w_l": float("nan"), "std_w_l": float("nan"), "large_acc": large_acc, "small_acc": small_acc,
        })
    else:
        print("Skipping coca_ts baseline row: --run_baselines not set.")

    if args.optimal_w:
        print("Running optimal-w-per-batch oracle ceiling...")
        opt = _optimal_weight_accuracy(streams, base_ts, cfg["BS"])
        print(f"[optimal_w] duo_acc={opt['duo_acc']:.4f} duo_nll={opt['duo_nll']:.4f} "
              f"duo_ece={opt['duo_ece']:.4f} duo_entropy={opt['duo_entropy']:.4f} "
              f"mean_w_l={opt['mean_w_l']:.4f} std_w_l={opt['std_w_l']:.4f}")
        _add_row({
            "beta": float("nan"), "filter_kind": "optimal_w", "proxy_batch_size": -1,
            "duo_acc": opt["duo_acc"], "duo_nll": opt["duo_nll"],
            "duo_ece": opt["duo_ece"], "duo_entropy": opt["duo_entropy"],
            "mean_w_l": opt["mean_w_l"], "std_w_l": opt["std_w_l"],
            "large_acc": large_acc, "small_acc": small_acc,
        })
    else:
        print("Skipping optimal-w-per-batch oracle ceiling (--optimal_w to enable).")

    rows.sort(key=lambda r: r[args.sort_by], reverse=True)
    _print_table(rows)

    best = rows[0]
    print(f"\nBest config: beta={best['beta']} filter={best['filter_kind']} "
          f"proxy_batch_size={best['proxy_batch_size']} -> duo_acc={best['duo_acc']:.4f} "
          f"(vs large={large_acc:.4f} small={small_acc:.4f})")

    if args.csv_path:
        path = Path(args.csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=_TABLE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {path}")

    if run is not None:
        run.finish()
        print(f"\nLogged {len(rows)} rows to wandb project '{args.wandb_project}' "
              f"(group='{group}') — click any column header in the table UI to sort by it "
              f"(table was built and pushed incrementally, one version per config).")


if __name__ == "__main__":
    main()
