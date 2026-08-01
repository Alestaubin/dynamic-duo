"""
scripts/calibrate_gate_oracle.py
=================================
Isolates the SECOND half of filtered-proxy soft weighting -- the gate

    w_l = sigmoid(beta * (x_l - x_s))

plus the combination (log/linear pooling) -- from the FIRST half (proxy
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
proxy itself is imperfect), not Stage 2 (the gate/pool math) -- those two
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
Every (beta, pool, filter_kind, proxy_batch_size) combination in the sweep
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

Two reference numbers are printed alongside the sweep table:
  - single-model baselines (large/small accuracy alone), the floor.
  - a hard-selection ceiling per proxy_batch_size: for each proxy-batch-sized
    chunk, pretend the gate picks (with certainty) whichever model is
    actually more accurate on that exact chunk. This is the beta -> infinity
    limit -- no real sigmoid gate at that batching granularity can beat it,
    so it upper-bounds the entire sweep for a given proxy_batch_size.

Usage
-----
python scripts/calibrate_gate_oracle.py --config cfgs/dynamic_duo_config.yaml \
    --num_samples 5000 --seed 0 --fixed_ts_config checkpoints/fixed_ts/default \
    --csv_path out/gate_calibration_oracle.csv

# Also sweep how much aggregating more true-accuracy samples per gate update
# helps, beyond the default (one adaptation batch per gate refresh):
python scripts/calibrate_gate_oracle.py --config cfgs/dynamic_duo_config.yaml \
    --proxy_batch_sizes 128 512 1024 --csv_path out/gate_calibration_oracle.csv
"""

from __future__ import annotations

import argparse
import csv as csv_module
from pathlib import Path

import torch
import torch.nn.functional as F

from src.utils.data import load_config, load_imagenetC
from src.utils.model import get_model
from src.tta.tent import configure_model_frozen
from src.tta.dynamic_duo import collect_logits
from src.calibrators.joint_fixed_TS import JointFixedTS
from src.calibrators.joint_proxy_weighted import JointProxyWeighted
from src.reliability.proxies.stats import ProxyStats
from src.reliability.calibration.logit import to_logit

_TABLE_COLUMNS = [
    "beta", "pool", "filter_kind", "proxy_batch_size",
    "duo_acc", "duo_nll", "mean_w_l", "large_acc", "small_acc",
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


def _collect_eval_streams(cfg, device, num_samples, seed, large, large_preprocess, small, small_preprocess):
    """One forward pass per (corruption, severity) in cfg['EVAL'] -- cached in
    CPU memory (see collect_logits), reused by every sweep config below."""
    streams = {}
    for severity in cfg["EVAL"]["SEVERITIES"]:
        for corruption in cfg["EVAL"]["CORRUPTIONS"]:
            print(f"Collecting eval stream {corruption}/s{severity}...")
            loader = load_imagenetC(
                cfg["TEST_DIR"], severities=severity, corruption_types=[corruption],
                device=device, batch_size=cfg["BS"], num_workers=cfg["WORKERS"],
                num_samples=num_samples, seed=seed,
            )
            z_l, z_s, labels = collect_logits(large, large_preprocess, small, small_preprocess, loader)
            streams[(corruption, severity)] = (z_l, z_s, labels)
    return streams


def _hard_selection_ceiling(streams: dict, chunk_size: int) -> float:
    """beta -> infinity reference: for each chunk_size-sample chunk, pick
    (with certainty) whichever model is actually more accurate on that exact
    chunk. Macro-averaged across streams, matching the sweep's own averaging."""
    per_stream_acc = []
    for (z_l, z_s, labels) in streams.values():
        n = z_l.shape[0]
        correct, total = 0.0, 0
        for start in range(0, n, chunk_size):
            sl = slice(start, min(start + chunk_size, n))
            acc_l = float((z_l[sl].argmax(1) == labels[sl]).float().mean())
            acc_s = float((z_s[sl].argmax(1) == labels[sl]).float().mean())
            picked = max(acc_l, acc_s)
            correct += picked * (sl.stop - sl.start)
            total += (sl.stop - sl.start)
        per_stream_acc.append(correct / total)
    return sum(per_stream_acc) / len(per_stream_acc)


def _run_config(streams, base_ts, beta, pool, filter_kind, filter_kwargs, proxy_batch_size, batch_size):
    """Replay one (beta, pool, filter_kind, proxy_batch_size) config across all
    cached eval streams, adaptation-batch by adaptation-batch, through the
    REAL JointProxyWeighted gate/combine logic (proxy_kind='oracle'). Uses
    the calibrator's private _forward() directly (bypassing calibrate() /
    calibrate_with_grad()'s per-batch console logging, which would otherwise
    flood stdout across a large sweep) -- the same pattern src.reliability.
    setup.fit_beta already uses for its own beta grid search.

    Returns macro-averaged (over streams) duo accuracy/NLL/mean gate weight.
    """
    cfg_l = ProxyStats(name="large", num_classes=1000)
    cfg_s = ProxyStats(name="small", num_classes=1000)
    calibrator = JointProxyWeighted(
        proxy_kind="oracle", cfg_l=cfg_l, cfg_s=cfg_s,
        beta=beta, pool=pool, filter_kind=filter_kind, filter_kwargs=filter_kwargs,
        prior_l=to_logit(0.5), prior_s=to_logit(0.5),
        base_ts=base_ts, proxy_batch_size=proxy_batch_size, log_every=0,
    )

    per_stream_rows = []
    for (corruption, severity), (z_l, z_s, labels) in streams.items():
        calibrator.set_corruption(f"{corruption}/s{severity}")
        n = z_l.shape[0]
        correct, nll_sum, w_l_sum, total = 0.0, 0.0, 0.0, 0
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            zl_b, zs_b, y_b = z_l[sl], z_s[sl], labels[sl]
            calibrator.set_labels(y_b)
            with torch.no_grad():
                z_duo, r_l, r_s, a_l, a_s, x_l, x_s, w_l = calibrator._forward(zl_b, zs_b)
            bs = sl.stop - sl.start
            correct += float((z_duo.argmax(1) == y_b).float().sum())
            nll_sum += float(F.cross_entropy(z_duo, y_b, reduction="sum"))
            w_l_sum += w_l * bs
            total += bs
        per_stream_rows.append({
            "acc": correct / total, "nll": nll_sum / total, "mean_w_l": w_l_sum / total,
        })

    n_streams = len(per_stream_rows)
    return {
        "duo_acc": sum(r["acc"] for r in per_stream_rows) / n_streams,
        "duo_nll": sum(r["nll"] for r in per_stream_rows) / n_streams,
        "mean_w_l": sum(r["mean_w_l"] for r in per_stream_rows) / n_streams,
    }


def _print_table(rows: list[dict]) -> None:
    header = (f"{'beta':>6} {'pool':<7} {'filter':<13} {'pbs':>5}  "
              f"{'duo_acc':>8} {'duo_nll':>8} {'mean_w_l':>9}   "
              f"{'large_acc':>9} {'small_acc':>9}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['beta']:>6.2f} {r['pool']:<7} {r['filter_kind']:<13} {r['proxy_batch_size']:>5}  "
              f"{r['duo_acc']:>8.4f} {r['duo_nll']:>8.4f} {r['mean_w_l']:>9.4f}   "
              f"{r['large_acc']:>9.4f} {r['small_acc']:>9.4f}")


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
                              "must be a checkpoint fit for THIS duo, see CLAUDE.md's "
                              "checkpoint/duo mismatch gotcha).")
    parser.add_argument("--betas", type=float, nargs="+",
                         default=[0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0])
    parser.add_argument("--pools", type=str, nargs="+", default=["log", "linear"],
                         choices=["log", "linear"])
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
    )

    large_acc = sum(
        float((z_l.argmax(1) == labels).float().mean()) for z_l, z_s, labels in streams.values()
    ) / len(streams)
    small_acc = sum(
        float((z_s.argmax(1) == labels).float().mean()) for z_l, z_s, labels in streams.values()
    ) / len(streams)
    print(f"\nSingle-model baselines (macro-avg over {len(streams)} eval streams): "
          f"large={large_acc:.4f}  small={small_acc:.4f}")
    for pbs in proxy_batch_sizes:
        ceiling = _hard_selection_ceiling(streams, pbs)
        print(f"Hard-selection ceiling @ proxy_batch_size={pbs}: {ceiling:.4f}  "
              f"(picks whichever model is actually more accurate on each {pbs}-sample chunk; "
              f"no gate at that granularity can beat this)")

    rows = []
    filter_kwargs = {"alpha": args.ema_alpha, "q": args.kalman_q, "r": args.kalman_r}
    for pbs in proxy_batch_sizes:
        for filter_kind in args.filter_kinds:
            for pool in args.pools:
                for beta in args.betas:
                    result = _run_config(
                        streams, base_ts, beta, pool, filter_kind, filter_kwargs, pbs, cfg["BS"],
                    )
                    rows.append({
                        "beta": beta, "pool": pool, "filter_kind": filter_kind, "proxy_batch_size": pbs,
                        "duo_acc": result["duo_acc"], "duo_nll": result["duo_nll"],
                        "mean_w_l": result["mean_w_l"],
                        "large_acc": large_acc, "small_acc": small_acc,
                    })

    rows.sort(key=lambda r: r[args.sort_by], reverse=True)
    _print_table(rows)

    best = rows[0]
    print(f"\nBest config: beta={best['beta']} pool={best['pool']} filter={best['filter_kind']} "
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


if __name__ == "__main__":
    main()
