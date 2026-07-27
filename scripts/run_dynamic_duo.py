from src.tta.dynamic_duo import setup_duo, evaluate_dynamic_duo
from src.utils.model import get_model
from src.utils.data import load_config
from src.calibrators.joint_fixed_TS import JointFixedTS, PreScaledCalibrator
from src.reliability.setup import build_proxy_weighted_calibrator, fit_beta

import argparse
import torch

"""
source /scratch0/alxstaub/ddenv/bin/activate
export PYTHONPATH=$PYTHONPATH:~/dynamic-duo

CUDA_VISIBLE_DEVICES=0

# fixed temperature-scaling baseline:
python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode no_adapt \
    --seed 0 \
    --calibration_mode fixed_ts \
    --fixed_ts_config checkpoints/naive_ts/clean

# filtered-proxy soft weighting (Sections 2-5):
python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode no_adapt \
    --calibration_mode proxy_weighted \
    --proxy_kind nuclear_norm \
    --calib_map resnet50_vitb16_dev \
    --calib_method isotonic \
    --filter kalman --kalman_q 1e-3 --kalman_r 1e-1 \
    --gate_beta 4.0 --pool linear \
    --seed 0
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dynamic Duo TTA on ImageNet-C")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="both_duo")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--calibration_mode", type=str, default="fixed_ts",
                        choices=["fixed_ts", "oracle_ts", "proxy_weighted"])
    parser.add_argument("--norm_logits", action="store_true",
                        help="Whether to L2-normalize the logits before calibration. ")
    parser.add_argument("--proxy_kind", type=str, default="prototype",
                        choices=["nuclear_norm", "atc", "prototype", "ac_mc", "cot"])
    parser.add_argument("--proto_metric", type=str, default="cosine",
                        choices=["cosine", "mahalanobis"],
                        help="Distance for the prototype proxy: cosine similarity to "
                             "L2-normalised class means (default), or tied-covariance "
                             "Mahalanobis to raw class means. Only used when "
                             "--proxy_kind prototype.")
    parser.add_argument("--proxy_cache", type=str, default=None)
    parser.add_argument("--calib_map", type=str, default=None, help="Path to a CSV file. ")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--precalibrate", action="store_true",
                        help="Whether to precalibrate the duo with a fixed temperature. ")
    parser.add_argument("--fixed_ts_config", type=str, default=None,
                        help="Path to a file containing the fixed temperature configuration.")

    # --- proxy_weighted (filtered-proxy soft weighting, paper Sections 2-5) ---
    parser.add_argument("--calib_method", type=str, default="isotonic",
                        choices=["identity", "linear", "platt", "beta", "isotonic"],
                        help="Section-3 calibration ladder: raw proxy -> predicted accuracy. "
                             "Only used when --calibration_mode proxy_weighted.")
    parser.add_argument("--filter", type=str, default="none",
                        choices=["none", "running_mean", "ema", "kalman"],
                        help="Section-4 temporal filter denoising the calibrated score "
                             "across batches. Only used when --calibration_mode proxy_weighted.")
    parser.add_argument("--ema_alpha", type=float, default=0.1,
                        help="EMA smoothing factor (--filter ema).")
    parser.add_argument("--kalman_q", type=float, default=1e-3,
                        help="Kalman process noise (--filter kalman).")
    parser.add_argument("--kalman_r", type=float, default=1e-1,
                        help="Kalman observation noise (--filter kalman).")
    parser.add_argument("--gate_beta", type=float, default=4.0,
                        help="Section-5 gate sharpness: w_l = sigmoid(beta * gap).")
    parser.add_argument("--pool", type=str, default="linear", choices=["log", "linear"],
                        help="Section-5 combination: 'log' (product-of-experts) or "
                             "'linear' (collapse-robust mixture).")
    parser.add_argument("--prior_l", type=float, default=0.5,
                        help="Large model's clean-source accuracy, used as the "
                             "ema/kalman filter's reset prior. Defaults to a neutral 0.5.")
    parser.add_argument("--prior_s", type=float, default=0.5,
                        help="Small model's clean-source accuracy, used as the "
                             "ema/kalman filter's reset prior. Defaults to a neutral 0.5.")
    parser.add_argument("--fit_beta", action="store_true",
                        help="Grid-search --gate_beta against held-out dev-shift NLL "
                             "(config's CALIBRATOR corruptions/severities) before eval.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(args.config)
    large_model, large_preprocess = get_model(config["LARGE"]["NAME"])
    small_model, small_preprocess = get_model(config["SMALL"]["NAME"])
    large_model = large_model.to(device)
    small_model = small_model.to(device)

    if args.calibration_mode == "proxy_weighted":
        try:
            base_ts = JointFixedTS.load(args.fixed_ts_config) if args.fixed_ts_config else None
            filter_kwargs = {
                "alpha": args.ema_alpha,
                "q": args.kalman_q,
                "r": args.kalman_r,
            }
            calibrator = build_proxy_weighted_calibrator(
                proxy_kind=args.proxy_kind,
                proxy_cache=args.proxy_cache,
                calib_map=args.calib_map,
                calib_method=args.calib_method,
                filter_kind=args.filter,
                filter_kwargs=filter_kwargs,
                beta=args.gate_beta,
                pool=args.pool,
                prior_l=args.prior_l,
                prior_s=args.prior_s,
                base_ts=base_ts,
                csv_path=args.csv_path,
                config=config,
                large_model=large_model, large_preprocess=large_preprocess,
                small_model=small_model, small_preprocess=small_preprocess,
                device=device,
                num_samples=args.num_samples,
                seed=args.seed,
                proto_metric=args.proto_metric,
            )
        except ValueError as e:
            parser.error(str(e))
        if args.fit_beta:
            fit_beta(
                calibrator,
                large_model, large_preprocess, small_model, small_preprocess,
                config, device, num_samples=args.num_samples, seed=args.seed,
            )
    elif args.calibration_mode == "fixed_ts":
        if args.fixed_ts_config is None:
            parser.error("--fixed_ts_config is required when --calibration_mode is fixed_ts")
        calibrator = JointFixedTS.load(args.fixed_ts_config)
    elif args.calibration_mode == "oracle_ts":
        calibrator = JointFixedTS()
    else:
        raise ValueError(f"Invalid calibration mode: {args.calibration_mode}")

    if args.precalibrate:
        assert args.fixed_ts_config is not None, "--fixed_ts_config is required when --precalibrate is set"
        fixed_ts = JointFixedTS.load(args.fixed_ts_config)
        fixed_ts.requires_grad_(False)
        calibrator = PreScaledCalibrator(fixed_ts, calibrator)
        print(f"Pre-scaling with JointFixedTS: Tl={fixed_ts.Tl.item():.4f}  Ts={fixed_ts.Ts.item():.4f}")

    duo = setup_duo(
        large=large_model,
        large_preprocess=large_preprocess,
        small=small_model,
        small_preprocess=small_preprocess,
        mode=args.mode,
        joint_calibrator=calibrator,
        calibration_mode=args.calibration_mode,
        cfg=config,
        steps=args.steps,
        norm_logits=args.norm_logits,
    )
    evaluate_dynamic_duo(duo, config, num_samples=args.num_samples, seed=args.seed, use_wandb=args.wandb)
