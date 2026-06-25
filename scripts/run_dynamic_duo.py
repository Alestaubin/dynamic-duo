from src.tta.dynamic_duo import setup_duo, evaluate_dynamic_duo
from src.utils.model import get_model
from src.utils.data import load_config
from src.calibrators.joint_fixed_TS import JointFixedTS
from src.calibrators.joint_coca import JointCoca
from src.calibrators.joint_sample_nll_oracle import JointSampleNLLOracle
from src.calibrators.joint_relative_entropy import JointRelativeEntropy
from src.calibrators.joint_lambda_entropy import JointLambdaEntropy
from src.proxies.calibrator_setup import build_proxy_calibrator

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
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dynamic Duo TTA on ImageNet-C")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="both_duo")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--calibration_mode", type=str, default="fixed")
    parser.add_argument("--norm_logits", action="store_true")
    parser.add_argument("--coca_bs", type=int, default=None)
    parser.add_argument("--fixed_ts_config", type=str, default=None)
    parser.add_argument("--proxy_kind", type=str, default="prototype",
                        choices=["nuclear_norm", "atc", "prototype"])
    parser.add_argument("--proxy_cache", type=str, default=None)
    parser.add_argument("--calib_map", type=str, default=None)
    parser.add_argument("--calibrated_selection", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--csv_path", type=str, default=None)

    args = parser.parse_args()

    if args.calibrated_selection and args.calibration_mode != "proxy_anchor_coca":
        parser.error("--calibrated_selection only applies to --calibration_mode proxy_anchor_coca")

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