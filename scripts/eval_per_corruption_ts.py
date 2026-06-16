#!/usr/bin/env python3
"""
For each EVAL corruption, fit a JointFixedTS calibrator on that corruption's
logits and immediately stream one LaTeX table row with Tl, Ts, acc_large,
acc_small, acc_duo.

Usage:
    python scripts/eval_per_corruption_ts.py \
        --config cfgs/dynamic_duo_config.yaml
"""

import argparse
import sys
import torch

from src.utils.data import load_config
from src.utils.logits import get_model_logits
from src.calibrators.joint_fixed_TS import JointFixedTS


def _acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    device = logits.device
    return (logits.argmax(1) == labels.to(device)).float().mean().item()


def _fit_and_eval(zl, zs, y, device):
    cal = JointFixedTS(verbose=False)
    cal.tune(logits_l=zl, logits_s=zs, labels=y)
    zl_d, zs_d, y_d = zl.to(device), zs.to(device), y.to(device)
    acc_l = _acc(zl_d, y_d)
    acc_s = _acc(zs_d, y_d)
    acc_duo = _acc(cal.calibrate(zl, zs), y_d)
    return cal.Tl.item(), cal.Ts.item(), acc_l, acc_s, acc_duo


def _latex_row(label, tl, ts, acc_l, acc_s, acc_duo):
    return (
        f"{label:<22} & {tl:.4f} & {ts:.4f}"
        f" & {acc_l * 100:.2f} & {acc_s * 100:.2f} & {acc_duo * 100:.2f} \\\\"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="cfgs/dynamic_duo_config.yaml")
    parser.add_argument("--cache_dir", default="cache/logits")
    args = parser.parse_args()

    cfg = load_config(args.config)
    large_name  = cfg["LARGE"]["NAME"]
    small_name  = cfg["SMALL"]["NAME"]
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corruptions = cfg["EVAL"]["CORRUPTIONS"]
    severities  = cfg["EVAL"]["SEVERITIES"]

    def get_logits(model_name, corruption, severity):
        return get_model_logits(
            model_name=model_name,
            val_dir=cfg["VAL_DIR"],
            test_dir=cfg["TEST_DIR"],
            cache_dir=args.cache_dir,
            batch_size=cfg["BS"],
            num_workers=cfg["WORKERS"],
            corruption=corruption,
            severity=severity,
            device=device,
            verbose = False,
            tent_mode=True,
            norm_type=cfg["LARGE"]["NORM"] if model_name == large_name else cfg["SMALL"]["NORM"],
        )

    multi_sev = len(severities) > 1

    print(r"\begin{tabular}{l" + ("r" * (6 if multi_sev else 5)) + "}")
    print(r"\toprule")
    sev_col = r"Sev & " if multi_sev else ""
    print(
        r"Corruption & " + sev_col +
        r"$T_l$ & $T_s$ & "
        r"Acc$_{\text{large}}$ & Acc$_{\text{small}}$ & Acc$_{\text{duo}}$ \\"
    )
    print(r"\midrule")
    sys.stdout.flush()

    pool_l, pool_s, pool_y = [], [], []

    for severity in severities:
        for corruption in corruptions:
            zl, y  = get_logits(large_name, corruption, severity)
            zs, _  = get_logits(small_name, corruption, severity)

            tl, ts, acc_l, acc_s, acc_duo = _fit_and_eval(zl, zs, y, device)

            label = corruption.replace("_", " ") + (f" s{severity}" if multi_sev else "")
            row = _latex_row(label, tl, ts, acc_l, acc_s, acc_duo)
            print(row)
            sys.stdout.flush()

            pool_l.append(zl); pool_s.append(zs); pool_y.append(y)

    if len(pool_y) > 1:
        zl_all = torch.cat(pool_l)
        zs_all = torch.cat(pool_s)
        y_all  = torch.cat(pool_y)

        tl, ts, acc_l, acc_s, acc_duo = _fit_and_eval(zl_all, zs_all, y_all, device)

        print(r"\midrule")
        print(_latex_row("Average", tl, ts, acc_l, acc_s, acc_duo))
        sys.stdout.flush()

    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
