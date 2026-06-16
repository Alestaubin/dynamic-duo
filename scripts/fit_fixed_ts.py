"""
Fit a JointFixedTS calibrator on clean val and/or calibrator corruptions, then save.

Usage
-----
# Val + calibrator corruptions (all severities from config)
python scripts/fit_fixed_ts.py --config cfgs/dynamic_duo_config.yaml \
    --out checkpoints/fixed_ts/default

# Val only
python scripts/fit_fixed_ts.py --config cfgs/dynamic_duo_config.yaml --out checkpoints/fixed_ts/test
"""

import argparse
import os
import torch

from src.utils.data import load_config
from src.utils.logits import get_model_logits
from src.calibrators.joint_fixed_TS import JointFixedTS
from src.utils.logit_transforms import logit_pnorm
from src.utils.metrics import get_metrics_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cfgs/dynamic_duo_config.yaml")
    parser.add_argument("--out", type=str, required=True,
                        help="Folder to save the calibrator (config.json)")
    parser.add_argument("--cache_dir", type=str, default="cache/logits",
                        help="Directory to store/read cached model logits")
    parser.add_argument("--clean_only", action="store_true",
                        help="Train on clean val only; ignore calibrator corruptions")
    parser.add_argument("--corruptions_only", action="store_true",
                        help="Train on calibrator corruptions only; skip clean val")
    parser.add_argument("--no_save_if_exists", action="store_true",
                        help="Skip if output already exists")
    parser.add_argument("--norm_logits", action="store_true", help="Whether to apply logit normalization (p-norm) before fitting the calibrator.")
    parser.add_argument("--test", action="store_true",
                        help="After fitting, evaluate the temperatures on the EVAL "
                             "corruptions from the config and report accuracy.")
    args = parser.parse_args()

    if args.clean_only and args.corruptions_only:
        parser.error("--clean_only and --corruptions_only are mutually exclusive")

    if args.no_save_if_exists and os.path.exists(os.path.join(args.out, "config.json")):
        print(f"Already exists: {args.out} — skipping")
        return

    cfg         = load_config(args.config)  
    large_name  = cfg["LARGE"]["NAME"]
    small_name  = cfg["SMALL"]["NAME"]
    val_dir     = cfg["VAL_DIR"]
    test_dir    = cfg["TEST_DIR"]
    batch_size  = cfg["BS"]
    num_workers = cfg["WORKERS"]
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_logits(model_name, corruption=None, severity=None):
        return get_model_logits(
            model_name=model_name, val_dir=val_dir, test_dir=test_dir,
            cache_dir=args.cache_dir, batch_size=batch_size,
            num_workers=num_workers, corruption=corruption,
            severity=severity, device=device,
        )

    large_l, small_l, labels_l = [], [], []

    # Val
    use_val = not args.corruptions_only
    if use_val:
        print("Loading val logits...")
        zl, y = get_logits(large_name)
        zs, _ = get_logits(small_name)
        if args.norm_logits:
            print("Normalizing val logits")
            zl = logit_pnorm(zl, p=2.0, tau=1.0)
            zs = logit_pnorm(zs, p=2.0, tau=1.0)
        large_l.append(zl); small_l.append(zs); labels_l.append(y)

    # Calibrator corruptions
    cal_corruptions = []
    if not args.clean_only:
        cal_corruptions = cfg["CALIBRATOR"].get("CORRUPTIONS", [])
        severities = cfg["CALIBRATOR"].get("SEVERITIES", [])
        for corruption in cal_corruptions:
            for sev in severities:
                print(f"  + calibrator: {corruption} sev={sev}")
                zl, y = get_logits(large_name, corruption, sev)
                zs, _ = get_logits(small_name, corruption, sev)
                if args.norm_logits:
                    print(f"Normalizing {corruption} logits")
                    zl = logit_pnorm(zl, p=2.0, tau=1.0)
                    zs = logit_pnorm(zs, p=2.0, tau=1.0)
                large_l.append(zl); small_l.append(zs); labels_l.append(y)


    # Fit
    if not labels_l:
        parser.error(
            "No data to fit on: --corruptions_only was set but CALIBRATOR.CORRUPTIONS "
            "is empty in the config."
        )
    logits_l = torch.cat(large_l)
    logits_s = torch.cat(small_l)
    labels   = torch.cat(labels_l)
    print(f"\nFitting JointFixedTS on {len(labels):,} samples...")

    cal = JointFixedTS()
    cal.tune(logits_l=logits_l, logits_s=logits_s, labels=labels)

    trained_on = {
        "large_model": large_name,
        "small_model": small_name,
        "clean_val": use_val,
        "cal_corruptions": cal_corruptions,
        "severities": cfg["CALIBRATOR"].get("SEVERITIES", []),
    }
    cal.save(args.out, trained_on=trained_on)

    # Evaluate the fitted temperatures on the EVAL corruptions.
    if args.test:
        eval_corruptions = cfg.get("EVAL", {}).get("CORRUPTIONS", [])
        eval_severities  = cfg.get("EVAL", {}).get("SEVERITIES", [])
        if not eval_corruptions or not eval_severities:
            print("\n--test: EVAL.CORRUPTIONS / EVAL.SEVERITIES is empty — nothing to test.")
            return

        def duo_acc(zl, zs, y):
            logits = cal.calibrate(logits_l=zl, logits_s=zs)
            m = get_metrics_dict(logits.softmax(dim=1), y)
            y_dev = y.to(zl.device)
            la = (zl.argmax(1) == y_dev).float().mean().item()
            sa = (zs.argmax(1) == y_dev).float().mean().item()
            return m, la, sa

        print(f"\nTesting Tl={cal.Tl.item():.4f}, Ts={cal.Ts.item():.4f} on EVAL corruptions...")
        header = (f"{'corruption':<18}{'sev':>4}{'duo_acc':>10}{'large_acc':>11}"
                  f"{'small_acc':>11}{'duo_ece':>10}{'duo_nll':>10}")
        print(header)
        print("-" * len(header))

        all_l, all_s, all_y = [], [], []
        for corruption in eval_corruptions:
            for sev in eval_severities:
                zl, y = get_logits(large_name, corruption, sev)
                zs, _ = get_logits(small_name, corruption, sev)
                if args.norm_logits:
                    zl = logit_pnorm(zl, p=2.0, tau=1.0)
                    zs = logit_pnorm(zs, p=2.0, tau=1.0)
                m, la, sa = duo_acc(zl, zs, y)
                print(f"{corruption:<18}{sev:>4}{m['accuracy']:>10.4f}{la:>11.4f}"
                      f"{sa:>11.4f}{m['ece']:>10.4f}{m['nll']:>10.4f}")
                all_l.append(zl); all_s.append(zs); all_y.append(y)

        if len(all_y) > 1:
            m, la, sa = duo_acc(torch.cat(all_l), torch.cat(all_s), torch.cat(all_y))
            print("-" * len(header))
            print(f"{'OVERALL':<18}{'':>4}{m['accuracy']:>10.4f}{la:>11.4f}"
                  f"{sa:>11.4f}{m['ece']:>10.4f}{m['nll']:>10.4f}")


if __name__ == "__main__":
    main()
