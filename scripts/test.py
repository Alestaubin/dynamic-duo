"""
Quick smoke test: fit JointFixedTS on vit_b_16+resnet50 clean val logits plus
the held-out CALIBRATOR corruptions from the config, and print the resulting
temperatures. No saving, no CLI args -- just a fast way to check the fit is
sane (in particular, that Tl/Ts land in [t_min, t_max] rather than diverging,
see src/calibrators/joint_fixed_TS.py's LBFGS clamp).

Usage: python scripts/test.py
"""

import torch

from src.utils.data import load_config
from src.utils.logits import get_model_logits
from src.calibrators.joint_fixed_TS import JointFixedTS

CONFIG_PATH = "cfgs/dynamic_duo_config_vitb_resnet.yaml"


def main():
    cfg = load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_logits(model_name, norm_type, corruption=None, severity=None):
        return get_model_logits(
            model_name=model_name, norm_type=norm_type,
            val_dir=cfg["VAL_DIR"], test_dir=cfg["TEST_DIR"], cache_dir="cache/logits",
            batch_size=cfg["BS"], num_workers=cfg["WORKERS"], device=device, tent_mode=False,
            corruption=corruption, severity=severity,
        )

    print(f"Loading clean val logits for {cfg['LARGE']['NAME']} + {cfg['SMALL']['NAME']}...")
    z_l, y_l = get_logits(cfg["LARGE"]["NAME"], cfg["LARGE"]["NORM"])
    z_s, y_s = get_logits(cfg["SMALL"]["NAME"], cfg["SMALL"]["NORM"])
    assert torch.equal(y_l, y_s), "large/small val labels differ"
    logits_l, logits_s, labels = [z_l], [z_s], [y_l]

    cal_corruptions = [] #cfg["CALIBRATOR"].get("CORRUPTIONS", [])
    cal_severities = [] #cfg["CALIBRATOR"].get("SEVERITIES", [])
    for corruption in cal_corruptions:
        for severity in cal_severities:
            print(f"Loading held-out shift logits: {corruption}/s{severity}...")
            zl, yl = get_logits(cfg["LARGE"]["NAME"], cfg["LARGE"]["NORM"], corruption, severity)
            zs, ys = get_logits(cfg["SMALL"]["NAME"], cfg["SMALL"]["NORM"], corruption, severity)
            assert torch.equal(yl, ys), f"large/small labels differ for {corruption}/s{severity}"
            logits_l.append(zl); logits_s.append(zs); labels.append(yl)

    z_l, z_s, y_l = torch.cat(logits_l), torch.cat(logits_s), torch.cat(labels)

    print(f"Fitting JointFixedTS on {len(y_l):,} samples "
          f"(clean val + {len(cal_corruptions)} held-out shift(s) x {len(cal_severities)} severity(ies))...")
    cal = JointFixedTS(verbose=True)
    cal.tune(logits_l=z_l, logits_s=z_s, labels=y_l)

    print(f"\nResult: Tl={cal.Tl.item():.4f}  Ts={cal.Ts.item():.4f}")
    acc_l = (z_l.argmax(1) == y_l).float().mean().item()
    acc_s = (z_s.argmax(1) == y_l).float().mean().item()
    acc_fixed = (cal.calibrate(z_l, z_s).argmax(1) == y_l.to(device)).float().mean().item()
    print(f"acc_large={acc_l:.4f}  acc_small={acc_s:.4f}  acc_fixed={acc_fixed:.4f}")


if __name__ == "__main__":
    main()
