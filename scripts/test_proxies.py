#!/usr/bin/env python3
"""
test_proxies.py
================
Validates a reliability proxy (paper Sections 2-3): does it actually
indicate which of the two models (large/small) is more accurate on a given
proxy batch — both from the raw score alone, and after Section-3
calibration onto a common predicted-accuracy scale?

For each (corruption, severity) stream, groups samples into
--proxy_batch_size chunks (Section 1's b_t, independent of any adaptation
batch size) and computes, per proxy batch:
  - raw proxy scores r_l, r_s
  - true accuracy acc_l, acc_s (from labels)
  - calibrated predicted accuracy \\hat a_l, \\hat a_s, if --calib_map is
    given (falls back to the raw score otherwise — Section 3's identity
    baseline)

then reports, per corruption and overall, for BOTH the raw score and the
calibrated score:
  - selection accuracy: fraction of non-tie proxy batches where the score
    correctly identifies the more-accurate model (sign(r_l - r_s) or
    sign(a_l - a_s) matches sign(acc_l - acc_s))
  - R^2, Pearson r, Spearman rho against the true per-batch accuracy, for
    each model

A proxy that's doing its job should show selection accuracy well above 0.5
and a strong positive correlation, and calibration should not make either
number worse (if it does, the calibration map is overfit or mis-specified).

Usage
-----
    python scripts/test_proxies.py --config cfgs/dynamic_duo_config.yaml \\
        --proxy_kind nuclear_norm --proxy_batch_size 128

    # with a calibration map (loaded if it exists, else fit fresh and saved):
    python scripts/test_proxies.py --config cfgs/dynamic_duo_config.yaml \\
        --proxy_kind atc --calib_map resnet50_vitb16_dev --calib_method isotonic

    # restrict to specific corruptions/severities, save per-batch rows:
    python scripts/test_proxies.py --config cfgs/dynamic_duo_config.yaml \\
        --proxy_kind prototype --corruptions fog snow --severities 3 5 \\
        --csv_path out/prototype_check
"""

from __future__ import annotations

import argparse
import csv as csv_module
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from src.utils.data import load_config, load_imagenetC
from src.utils.model import get_model
from src.reliability.proxies.stats import FeatureExtractor
from src.reliability.setup import _build_proxy_stats, _fit_and_save_calibration_maps
from src.reliability.calibration.maps import load_calibration_maps
from scripts._cli import add_duo_config_arg, add_num_samples_arg, add_seed_arg, add_proto_metric_arg

_VALIDATABLE_PROXY_KINDS = ["nuclear_norm", "atc", "prototype", "ac_mc", "cot"]


def _corr_stats(xs: list[float], ys: list[float]) -> dict:
    """R^2, Pearson r, Spearman rho between xs and ys. nan when n < 3."""
    n = len(xs)
    nan_result = {"r2": float("nan"), "pearson_r": float("nan"), "spearman_rho": float("nan"), "n": n}
    if n < 3:
        return nan_result
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.std() < 1e-8 or y.std() < 1e-8:
        return nan_result
    pr, _ = pearsonr(x, y)
    sr, _ = spearmanr(x, y)
    return {"r2": float(pr ** 2), "pearson_r": float(pr), "spearman_rho": float(sr), "n": n}


def _selection_accuracy(score_l: list[float], score_s: list[float], acc_l: list[float], acc_s: list[float]) -> tuple[float, int, int]:
    """Fraction of non-tie batches where sign(score_l - score_s) matches
    sign(acc_l - acc_s). Returns (accuracy, n_correct, n_total_non_tie)."""
    correct = total = 0
    for sl, ss, al, as_ in zip(score_l, score_s, acc_l, acc_s):
        if abs(al - as_) < 1e-9:
            continue  # tie: no "better" model to select
        total += 1
        true_better_large = al > as_
        pred_better_large = sl > ss
        if true_better_large == pred_better_large:
            correct += 1
    acc = correct / total if total > 0 else float("nan")
    return acc, correct, total


@torch.no_grad()
def _iter_proxy_batches(loader, preprocess_l, preprocess_s, ext_l, ext_s, device, proxy_batch_size, need_features):
    """Yield (z_l, z_s, f_l, f_s, labels) aggregated to ~proxy_batch_size
    samples each (the final chunk of a stream may be smaller)."""
    buf_z_l, buf_z_s, buf_f_l, buf_f_s, buf_labels, n = [], [], [], [], [], 0
    for imgs, labels in loader:
        xl = torch.stack([preprocess_l(img) for img in imgs]).to(device)
        xs = torch.stack([preprocess_s(img) for img in imgs]).to(device)
        zl, fl = ext_l(xl)
        zs, fs = ext_s(xs)
        buf_z_l.append(zl.cpu()); buf_z_s.append(zs.cpu())
        if need_features:
            buf_f_l.append(fl.cpu()); buf_f_s.append(fs.cpu())
        buf_labels.append(labels)
        n += zl.shape[0]

        if n >= proxy_batch_size:
            yield (
                torch.cat(buf_z_l), torch.cat(buf_z_s),
                torch.cat(buf_f_l) if need_features else None,
                torch.cat(buf_f_s) if need_features else None,
                torch.cat(buf_labels),
            )
            buf_z_l, buf_z_s, buf_f_l, buf_f_s, buf_labels, n = [], [], [], [], [], 0

    if n > 0:
        yield (
            torch.cat(buf_z_l), torch.cat(buf_z_s),
            torch.cat(buf_f_l) if need_features else None,
            torch.cat(buf_f_s) if need_features else None,
            torch.cat(buf_labels),
        )


def _fmt(v: float) -> str:
    return f"{v:6.3f}" if v == v else "   nan"


def main():
    parser = argparse.ArgumentParser(
        description="Check whether a reliability proxy indicates the better model, before/after calibration."
    )
    add_duo_config_arg(parser, required=True)
    parser.add_argument("--proxy_kind", type=str, required=True, choices=_VALIDATABLE_PROXY_KINDS)
    add_proto_metric_arg(parser)
    parser.add_argument("--proxy_cache", type=str, default=None)
    parser.add_argument("--calib_map", type=str, default=None,
                        help="Name of a calibration-map file to load, or fit fresh "
                             "(on the config's CALIBRATOR corruptions) and save under "
                             "this name if it doesn't exist yet. If omitted, the raw "
                             "score is used directly (Section 3's identity baseline).")
    parser.add_argument("--calib_method", type=str, default="isotonic",
                        choices=["identity", "linear", "platt", "beta", "isotonic"])
    parser.add_argument("--proxy_batch_size", type=int, default=128,
                        help="Section-1 proxy batch size b_t.")
    parser.add_argument("--corruptions", type=str, nargs="+", default=None,
                        help="Defaults to the config's EVAL.CORRUPTIONS.")
    parser.add_argument("--severities", type=int, nargs="+", default=None,
                        help="Defaults to the config's EVAL.SEVERITIES.")
    add_num_samples_arg(parser, default=None)
    add_seed_arg(parser, default=None)
    parser.add_argument("--csv_path", type=str, default=None,
                        help="If given, write one row per proxy batch to this CSV path.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(args.config)
    large_model, large_preprocess = get_model(config["LARGE"]["NAME"])
    small_model, small_preprocess = get_model(config["SMALL"]["NAME"])
    large_model = large_model.to(device).eval()
    small_model = small_model.to(device).eval()

    cfg_l, cfg_s = _build_proxy_stats(
        args.proxy_kind, config,
        large_model, large_preprocess, small_model, small_preprocess,
        device, cache_path=args.proxy_cache, proto_metric=args.proto_metric,
    )

    if args.calib_map is not None:
        try:
            maps = load_calibration_maps(args.calib_map)
        except FileNotFoundError:
            maps = _fit_and_save_calibration_maps(
                args.calib_map, cfg_l, cfg_s,
                large_model, large_preprocess, small_model, small_preprocess,
                config, device,
                proxy_name=args.proxy_kind,
                num_samples=args.num_samples, seed=args.seed,
                calib_method=args.calib_method,
            )
        maps.attach(cfg_l, cfg_s)
        print(f"Attached calibration map '{args.calib_map}' (method={maps.method})")
    else:
        print(f"No --calib_map given; calibrated == raw score (identity baseline).")

    corruptions = args.corruptions or config["EVAL"]["CORRUPTIONS"]
    severities = args.severities or config["EVAL"]["SEVERITIES"]
    need_features = args.proxy_kind == "prototype"

    csv_path = None
    if args.csv_path:
        csv_path = Path(args.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    ext_l = FeatureExtractor(large_model, cfg_l.name)
    ext_s = FeatureExtractor(small_model, cfg_s.name)

    overall = {k: [] for k in ("raw_l", "raw_s", "cal_l", "cal_s", "acc_l", "acc_s")}

    try:
        for severity in severities:
            for corruption in corruptions:
                loader = load_imagenetC(
                    config["TEST_DIR"], severities=severity, corruption_types=[corruption],
                    device=device, batch_size=config["BS"],
                    num_samples=args.num_samples, seed=args.seed,
                )
                per_corr = {k: [] for k in overall}
                for z_l, z_s, f_l, f_s, labels in tqdm(
                    _iter_proxy_batches(loader, large_preprocess, small_preprocess, ext_l, ext_s, device, args.proxy_batch_size, need_features),
                    desc=f"{corruption}/s{severity}",
                ):
                    r_l = cfg_l.score(args.proxy_kind, z_l, f_l)
                    r_s = cfg_s.score(args.proxy_kind, z_s, f_s)
                    a_l = cfg_l.predicted_acc(args.proxy_kind, r_l)
                    a_s = cfg_s.predicted_acc(args.proxy_kind, r_s)
                    acc_l = float((z_l.argmax(1) == labels).float().mean())
                    acc_s = float((z_s.argmax(1) == labels).float().mean())

                    for k, v in (("raw_l", r_l), ("raw_s", r_s), ("cal_l", a_l), ("cal_s", a_s),
                                 ("acc_l", acc_l), ("acc_s", acc_s)):
                        per_corr[k].append(v)
                        overall[k].append(v)

                    if csv_path is not None:
                        need_header = not csv_path.exists()
                        with csv_path.open("a", newline="") as f:
                            writer = csv_module.DictWriter(
                                f, fieldnames=["corruption", "severity", "r_l", "r_s", "a_l", "a_s", "acc_l", "acc_s"]
                            )
                            if need_header:
                                writer.writeheader()
                            writer.writerow({
                                "corruption": corruption, "severity": severity,
                                "r_l": r_l, "r_s": r_s, "a_l": a_l, "a_s": a_s,
                                "acc_l": acc_l, "acc_s": acc_s,
                            })

                _report(f"{corruption}/s{severity}", per_corr)
    finally:
        ext_l.remove()
        ext_s.remove()

    print("\n" + "=" * 70)
    _report("OVERALL", overall)


def _report(label: str, d: dict) -> None:
    n = len(d["acc_l"])
    if n == 0:
        print(f"[{label}] no batches")
        return

    raw_sel_acc, raw_correct, raw_total = _selection_accuracy(d["raw_l"], d["raw_s"], d["acc_l"], d["acc_s"])
    cal_sel_acc, cal_correct, cal_total = _selection_accuracy(d["cal_l"], d["cal_s"], d["acc_l"], d["acc_s"])
    raw_l_stats = _corr_stats(d["raw_l"], d["acc_l"])
    raw_s_stats = _corr_stats(d["raw_s"], d["acc_s"])
    cal_l_stats = _corr_stats(d["cal_l"], d["acc_l"])
    cal_s_stats = _corr_stats(d["cal_s"], d["acc_s"])

    print(
        f"[{label}] n={n} proxy batches\n"
        f"  selection accuracy   raw={_fmt(raw_sel_acc)} ({raw_correct}/{raw_total})   "
        f"calibrated={_fmt(cal_sel_acc)} ({cal_correct}/{cal_total})\n"
        f"  large  raw:  R2={_fmt(raw_l_stats['r2'])}  r={_fmt(raw_l_stats['pearson_r'])}  rho={_fmt(raw_l_stats['spearman_rho'])}\n"
        f"  large  cal:  R2={_fmt(cal_l_stats['r2'])}  r={_fmt(cal_l_stats['pearson_r'])}  rho={_fmt(cal_l_stats['spearman_rho'])}\n"
        f"  small  raw:  R2={_fmt(raw_s_stats['r2'])}  r={_fmt(raw_s_stats['pearson_r'])}  rho={_fmt(raw_s_stats['spearman_rho'])}\n"
        f"  small  cal:  R2={_fmt(cal_s_stats['r2'])}  r={_fmt(cal_s_stats['pearson_r'])}  rho={_fmt(cal_s_stats['spearman_rho'])}"
    )


if __name__ == "__main__":
    main()
