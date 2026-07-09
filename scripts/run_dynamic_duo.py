from src.tta.dynamic_duo import setup_duo, evaluate_dynamic_duo
from src.utils.model import get_model
from src.utils.data import load_config
from src.calibrators.joint_fixed_TS import JointFixedTS, PreScaledCalibrator
from src.calibrators.joint_coca import JointCoca
from src.calibrators.joint_sample_nll_oracle import JointSampleNLLOracle
from src.calibrators.joint_relative_entropy import JointRelativeEntropy
from src.calibrators.joint_lambda_entropy import JointLambdaEntropy
from src.proxies.calibrator_setup import build_proxy_calibrator, build_proxy_weighted_calibrator

import argparse
import torch

"""
source /scratch0/alxstaub/ddenv/bin/activate
export PYTHONPATH=$PYTHONPATH:~/dynamic-duo

CUDA_VISIBLE_DEVICES=0
python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode no_adapt \
    --seed 0 \
    --calibration_mode coca

# proxy-anchor COCA with calibrated anchor selection:
python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode no_adapt \
    --calibration_mode proxy_anchor_coca \
    --proxy_kind prototype \
    --proxy_cache resnet50_vitb16 \
    --calib_map resnet50_vitb16_dev \
    --calibrated_selection

# proxy-anchor COCA with cumulative nuclear norm proxy:
python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode no_adapt \
    --calibration_mode proxy_anchor_coca \
    --proxy_kind nuclear_norm_cum \
    --seed 0 \
    --wandb

# proxy-weighted (soft, continuous reliability weighting) with the suggested
# defaults (filter=kalman denoises the per-batch proxy, pool=linear is
# collapse-robust, calib=zscore is the lightweight val-based calibration).
# beta is fit automatically on --calib_map's dev-shift corruptions and cached
# under --weight_cfg if given; pass --proxy_beta to skip fitting/override it.
python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode both_duo \
    --calibration_mode proxy_weighted \
    --proxy_kind nuclear_norm_cum \
    --fixed_ts_config checkpoints/naive_ts/clean \
    --calib_map resnet50_vitb16_dev \
    --weight_cfg resnet50_vitb16_nuclear_norm_cum \
    --seed 0 \
    --wandb

# proxy-weighted with EMA-Gram denoising instead of Kalman, and the log pool:
python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode both_duo \
    --calibration_mode proxy_weighted \
    --proxy_kind nuclear_norm \
    --proxy_filter ema_gram --ema_decay 0.1 \
    --proxy_weight_pool log \
    --fixed_ts_config checkpoints/naive_ts/clean \
    --proxy_beta 4.0
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dynamic Duo TTA on ImageNet-C")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="both_duo")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--calibration_mode", type=str, default="fixed")
    parser.add_argument("--norm_logits", action="store_true", 
                        help="Whether to L2-normalize the logits before calibration. ")
    parser.add_argument("--coca_bs", type=int, default=None, 
                        help="Batch size for COCA calibration, independent of the TENT batch size.")
    parser.add_argument("--proxy_kind", type=str, default="prototype",
                        choices=["nuclear_norm", "nuclear_norm_cum", "atc", "prototype"])
    parser.add_argument("--proto_metric", type=str, default="cosine",
                        choices=["cosine", "mahalanobis"],
                        help="Distance for the prototype proxy: cosine similarity to "
                             "L2-normalised class means (default), or tied-covariance "
                             "Mahalanobis to raw class means. Only used when "
                             "--proxy_kind prototype.")
    parser.add_argument("--proxy_cache", type=str, default=None)
    parser.add_argument("--calib_map", type=str, default=None, help="Path to a CSV file. ")
    parser.add_argument("--calibrated_selection", action="store_true",
                        help="Whether to use calibrated selection."
                        "If set, ")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--precalibrate", action="store_true", 
                        help="Whether to precalibrate the duo with a fixed temperature. ")
    parser.add_argument("--fixed_ts_config", type=str, default=None,
                        help="Path to a file containing the fixed temperature configuration.")
    parser.add_argument("--proxy_weight_pool", type=str, default="linear",
                        choices=["log", "linear"],
                        help="proxy_weighted pooling: 'log' (sharpening, matches "
                             "today's product-of-experts aggregation) or 'linear' "
                             "(mixture, collapse-robust — default).")
    parser.add_argument("--proxy_calib", type=str, default="zscore",
                        choices=["zscore", "isotonic"],
                        help="proxy_weighted per-model calibration onto a common "
                             "reliability scale: 'zscore' (default, val mean/std) "
                             "or 'isotonic' (needs --calib_map).")
    parser.add_argument("--proxy_filter", type=str, default="kalman",
                        choices=["none", "ema_gram", "kalman"],
                        help="proxy_weighted per-batch proxy denoising: 'none', "
                             "'ema_gram' (recency-weighted nuclear norm, needs "
                             "--proxy_kind nuclear_norm[_cum]), or 'kalman' "
                             "(default; needs val stats from the source pass).")
    parser.add_argument("--proxy_beta", type=float, default=None,
                        help="proxy_weighted gate sharpness. If unset, fit on "
                             "--calib_map's dev-shift corruptions (and cache "
                             "under --weight_cfg if given).")
    parser.add_argument("--ema_decay", type=float, default=0.1,
                        help="Decay lambda for --proxy_filter ema_gram.")
    parser.add_argument("--kalman_q", type=float, default=1e-3,
                        help="Process noise for --proxy_filter kalman.")
    parser.add_argument("--kalman_r", type=float, default=1e-1,
                        help="Observation noise for --proxy_filter kalman.")
    parser.add_argument("--proxy_precision_weight", action="store_true",
                        help="Temper --proxy_beta by the kalman filters' "
                             "posterior variance. Only with --proxy_filter kalman.")
    parser.add_argument("--weight_cfg", type=str, default=None,
                        help="Name for the fitted beta sidecar "
                             "(data/proxy_weight_cfg/<name>.weightcfg.json). "
                             "Loaded if it exists, else fit and saved under this name.")

    args = parser.parse_args()

    if args.calibrated_selection and args.calibration_mode != "proxy_anchor_coca":
        parser.error("--calibrated_selection only applies to --calibration_mode proxy_anchor_coca")
    if args.precalibrate and args.calibration_mode == "proxy_weighted":
        parser.error("--precalibrate is redundant with --calibration_mode proxy_weighted: "
                      "it already applies --fixed_ts_config internally as its base scale.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(args.config)
    large_model, large_preprocess = get_model(config["LARGE"]["NAME"])
    small_model, small_preprocess = get_model(config["SMALL"]["NAME"])
    large_model = large_model.to(device)
    small_model = small_model.to(device)

    if args.calibration_mode in {"soft_anchor", "proxy_anchor_coca"}:
        try:
            calibrator = build_proxy_calibrator(
                calibration_mode=args.calibration_mode,
                proxy_kind=args.proxy_kind,
                proxy_cache=args.proxy_cache,
                calib_map=args.calib_map,
                calibrated_selection=args.calibrated_selection,
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
    elif args.calibration_mode == "proxy_weighted":
        filter_kwargs = {}
        if args.proxy_filter == "ema_gram":
            filter_kwargs["decay"] = args.ema_decay
        elif args.proxy_filter == "kalman":
            filter_kwargs["q"] = args.kalman_q
            filter_kwargs["r"] = args.kalman_r
        try:
            calibrator = build_proxy_weighted_calibrator(
                proxy_kind=args.proxy_kind,
                config=config,
                large_model=large_model, large_preprocess=large_preprocess,
                small_model=small_model, small_preprocess=small_preprocess,
                device=device,
                pool=args.proxy_weight_pool,
                calib_mode=args.proxy_calib,
                filter=args.proxy_filter,
                filter_kwargs=filter_kwargs,
                beta=args.proxy_beta,
                precision_weight=args.proxy_precision_weight,
                proxy_cache=args.proxy_cache,
                calib_map=args.calib_map,
                weight_cfg_name=args.weight_cfg,
                fixed_ts_config=args.fixed_ts_config,
                csv_path=args.csv_path,
                num_samples=args.num_samples,
                seed=args.seed,
                proto_metric=args.proto_metric,
            )
        except ValueError as e:
            parser.error(str(e))
    elif args.calibration_mode == "fixed_ts":
        if args.fixed_ts_config is None:
            parser.error("--fixed_ts_config is required when --calibration_mode is fixed_ts")
        calibrator = JointFixedTS.load(args.fixed_ts_config)
    elif args.calibration_mode == "coca_entropy":
        calibrator = JointCoca(num_steps=5, lr=5e-2, loss="entropy", chunk_size=args.coca_bs)
    elif args.calibration_mode == "coca":
        calibrator = JointCoca(num_steps=5, lr=5e-2, chunk_size=args.coca_bs)
    elif args.calibration_mode == "oracle_ts":
        calibrator = JointFixedTS()
    elif args.calibration_mode == "sample_oracle_ts":
        calibrator = JointSampleNLLOracle(num_steps=20, lr=5e-2)
    elif args.calibration_mode == "relative_entropy":
        calibrator = JointRelativeEntropy(init_w=0.0, t_max=10.0)
    elif args.calibration_mode == "lambda_entropy":
        calibrator = JointLambdaEntropy(init_lambda=0.5)
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