"""
Fit a JointFixedTS calibrator on clean val and/or calibrator corruptions, then save.

Usage
-----
# Val + calibrator corruptions (all severities from config)
python scripts/fit_fixed_ts.py --config cfgs/dynamic_duo_config.yaml \
    --out checkpoints/fixed_ts/default

# Val only
python scripts/fit_fixed_ts.py --config cfgs/dynamic_duo_config.yaml \
    --out checkpoints/fixed_ts/clean --clean_only
"""

import argparse
import os
import torch

from src.utils.data import load_config
from src.utils.logits import get_model_logits
from src.calibrators.fixed_TS import JointFixedTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cfgs/dynamic_duo_config.yaml")
    parser.add_argument("--out", type=str, required=True,
                        help="Folder to save the calibrator (config.json)")
    parser.add_argument("--cache_dir", type=str, default="cache/logits",
                        help="Directory to store/read cached model logits")
    parser.add_argument("--clean_only", action="store_true",
                        help="Train on clean val only; ignore calibrator corruptions")
    parser.add_argument("--no_save_if_exists", action="store_true",
                        help="Skip if output already exists")
    args = parser.parse_args()

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
    print("Loading val logits...")
    zl, y = get_logits(large_name)
    zs, _ = get_logits(small_name)
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
                large_l.append(zl); small_l.append(zs); labels_l.append(y)


    # Fit
    logits_l = torch.cat(large_l)
    logits_s = torch.cat(small_l)
    labels   = torch.cat(labels_l)
    print(f"\nFitting JointFixedTS on {len(labels):,} samples...")

    cal = JointFixedTS()
    cal.tune(logits_l=logits_l, logits_s=logits_s, labels=labels)

    trained_on = {
        "large_model": large_name,
        "small_model": small_name,
        "clean_val": True,
        "cal_corruptions": cal_corruptions,
        "severities": cfg["CALIBRATOR"].get("SEVERITIES", []),
    }
    cal.save(args.out, trained_on=trained_on)


if __name__ == "__main__":
    main()
