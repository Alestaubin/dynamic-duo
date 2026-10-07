"""
scripts/test_tta_parity.py
==========================
Regression check for the TTA-method refactor (src/tta/methods/): drive a real
duo over a tiny ImageNet-C stream in every adaptation mode, record every
batch's logits (large/small/duo) plus a checksum of the models' parameters
after each stream, and compare two such recordings.

The point is to run the SAME script against two source trees -- the pre-refactor
commit and the working tree -- and diff the recordings (see
slurm/test_tta_refactor.sh, which does exactly that on a GPU node):

    # record (run from inside a source tree, PYTHONPATH=.)
    python scripts/test_tta_parity.py --config cfgs/dynamic_duo_config_vitb_resnet.yaml --out old.pt
    python scripts/test_tta_parity.py --config cfgs/dynamic_duo_config_vitb_resnet.yaml --out new.pt
    # compare
    python scripts/test_tta_parity.py --compare old.pt new.pt

Two streams per case, with duo.reset() in between, so a reset that fails to
restore the pre-adaptation state shows up as the SECOND stream's logits
diverging. Deterministic cuDNN/cuBLAS settings are forced so identical code
gives bitwise-identical output; --tol only exists to report (not hide) the
residual if the GPU still isn't deterministic.

Deliberately avoids anything the pre-refactor tree doesn't have: the TTA flags
are only passed when explicitly given (--tta_method), never by default.
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede CUDA init

import argparse
import sys

import torch

# (mode, steps) cases -- every _MODES entry once, plus a multi-step duo case
# (DynamicDuo.forward loops forward_and_adapt `steps` times).
CASES = [
    ("no_adapt", 1), ("both_duo", 1), ("large_duo", 1), ("small_duo", 1),
    ("both_indep", 1), ("large_indep", 1), ("small_indep", 1), ("both_duo", 2),
]


def _param_checksums(model) -> dict:
    """float64 sum and abs-sum over ALL parameters (frozen ones included: they
    must not have moved either)."""
    ps = [p.detach().double() for p in model.parameters()]
    return {"sum": sum(p.sum().item() for p in ps), "abs_sum": sum(p.abs().sum().item() for p in ps)}


def record(args) -> None:
    from src.calibrators.joint_fixed_TS import JointFixedTS
    from src.tta.dynamic_duo import setup_duo
    from src.utils.data import load_config, load_imagenetC
    from src.utils.model import get_model

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    # ViT's memory-efficient SDPA backward is non-deterministic even under the
    # flag above (warn_only just warns); the math kernel is deterministic.
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  config={args.config}  corruptions={args.corruptions}")
    cfg = load_config(args.config)

    tta_kwargs = {}
    if args.tta_method is not None:
        tta_kwargs["tta_method"] = args.tta_method

    recording = {}
    for mode, steps in CASES:
        case = f"{mode}_steps{steps}"
        torch.manual_seed(0)
        large, large_pre = get_model(cfg["LARGE"]["NAME"], verbose=False)
        small, small_pre = get_model(cfg["SMALL"]["NAME"], verbose=False)
        large, small = large.to(device), small.to(device)
        # Non-identity temperatures so the calibrated-output path is exercised
        # (the duo loss backpropagates through the combine step).
        calibrator = JointFixedTS(Tl=1.5, Ts=0.8, verbose=False)
        duo = setup_duo(
            large=large, large_preprocess=large_pre, small=small, small_preprocess=small_pre,
            mode=mode, joint_calibrator=calibrator, calibration_mode="fixed_ts",
            cfg=cfg, steps=steps, **tta_kwargs,
        )

        rec = {"batches": [], "streams": {}}
        for corruption in args.corruptions:
            duo.reset()
            loader = load_imagenetC(
                cfg["TEST_DIR"], severities=5, corruption_types=[corruption], device=device,
                batch_size=cfg["BS"], num_workers=cfg["WORKERS"],
                num_samples=args.num_samples, seed=args.seed,
            )
            for imgs, labels in loader:
                out, z_l, z_s = duo(imgs, labels=labels)
                rec["batches"].append({
                    "corruption": corruption, "out": out.detach().cpu(),
                    "z_l": z_l.detach().cpu(), "z_s": z_s.detach().cpu(), "labels": labels.cpu(),
                })
            rec["streams"][corruption] = {
                "large": _param_checksums(large), "small": _param_checksums(small),
            }
        recording[case] = rec
        print(f"recorded {case}: {len(rec['batches'])} batches")

        del duo, large, small, calibrator
        if device.type == "cuda":
            torch.cuda.empty_cache()

    torch.save(recording, args.out)
    print(f"saved {args.out}")


def compare(path_a: str, path_b: str, tol: float) -> int:
    a = torch.load(path_a, map_location="cpu", weights_only=False)
    b = torch.load(path_b, map_location="cpu", weights_only=False)
    if set(a) != set(b):
        print(f"FAIL: case sets differ: {sorted(set(a) ^ set(b))}")
        return 1

    n_bad = 0
    print(f"{'case':<20} {'batches':>7} {'max|d out|':>12} {'max|d z_l|':>12} {'max|d z_s|':>12} "
          f"{'max|d params|':>14}  status")
    for case in sorted(a):
        ra, rb = a[case], b[case]
        if len(ra["batches"]) != len(rb["batches"]):
            print(f"{case:<20} FAIL: batch counts differ ({len(ra['batches'])} vs {len(rb['batches'])})")
            n_bad += 1
            continue
        d = {"out": 0.0, "z_l": 0.0, "z_s": 0.0}
        labels_match = True
        for ba, bb in zip(ra["batches"], rb["batches"]):
            labels_match &= torch.equal(ba["labels"], bb["labels"])
            for k in d:
                d[k] = max(d[k], (ba[k] - bb[k]).abs().max().item())
        dp = 0.0
        for corruption, sa in ra["streams"].items():
            for side in ("large", "small"):
                for k in ("sum", "abs_sum"):
                    dp = max(dp, abs(sa[side][k] - rb["streams"][corruption][side][k]))
        ok = labels_match and max(*d.values(), dp) <= tol
        n_bad += not ok
        status = "ok" if ok else ("LABELS DIFFER" if not labels_match else "MISMATCH")
        print(f"{case:<20} {len(ra['batches']):>7} {d['out']:>12.3e} {d['z_l']:>12.3e} {d['z_s']:>12.3e} "
              f"{dp:>14.3e}  {status}")

    # Sanity: adapting modes must actually differ from no_adapt, otherwise this
    # test could pass vacuously with adaptation silently disabled in BOTH trees.
    base = a["no_adapt_steps1"]["batches"]
    for case in sorted(a):
        if case.startswith("no_adapt"):
            continue
        moved = max(
            (x["out"] - y["out"]).abs().max().item() for x, y in zip(a[case]["batches"], base)
        )
        if moved == 0.0:
            print(f"FAIL: {case} is bitwise identical to no_adapt -- adaptation did nothing")
            n_bad += 1

    print("PASS" if n_bad == 0 else f"FAIL ({n_bad} problem case(s))")
    return 1 if n_bad else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="cfgs/dynamic_duo_config_vitb_resnet.yaml")
    p.add_argument("--out", help="Where to save the recording (record mode).")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Compare two recordings instead of recording.")
    p.add_argument("--tol", type=float, default=0.0,
                   help="Max allowed abs difference when comparing (default 0 = bitwise identical).")
    p.add_argument("--num_samples", type=int, default=192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corruptions", nargs="+", default=["gaussian_noise", "fog"])
    p.add_argument("--tta_method", default=None,
                   help="Only passed to setup_duo when given (the pre-refactor tree has no such argument).")
    args = p.parse_args()

    if args.compare:
        return compare(*args.compare, tol=args.tol)
    if not args.out:
        p.error("--out is required unless --compare is given.")
    record(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
