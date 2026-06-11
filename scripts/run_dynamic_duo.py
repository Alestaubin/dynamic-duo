from src.tta.dynamic_duo import setup_duo, evaluate_dynamic_duo
from src.utils.model import get_model
from src.utils.data import load_config
from src.calibrators.joint_fixed_TS import JointFixedTS
from src.calibrators.joint_coca import JointCoca
from src.calibrators.joint_duo_entropy import JointDuoEntropy
from src.calibrators.joint_batch_nll_oracle import JointBatchNLLOracle
from src.calibrators.joint_sample_nll_oracle import JointSampleNLLOracle
import argparse
import torch

"""
python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode no_adapt \
    --steps 1 \
    --num_samples 1000 \
    --seed 0 \
    --calibration_mode duo_entropy

source /scratch0/alxstaub/ddenv/bin/activate
export PYTHONPATH=$PYTHONPATH:~/dynamic-duo

CUDA_VISIBLE_DEVICES=0 
python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode no_adapt \
    --steps 1 \
    --seed 0 \
    --calibration_mode fixed_ts

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dynamic Duo TTA on ImageNet-C")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--mode", type=str, default="both_duo", help="Dynamic Duo mode to run")
    parser.add_argument("--steps", type=int, default=1, help="Number of adaptation steps per batch")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use from each corruption/severity subset (default: all)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (when using fraction < 1.0)")  
    parser.add_argument("--calibration_mode", type=str, default="fixed", help="Calibrator mode for combining the duo logits.")
    parser.add_argument("--norm_logits", action="store_true", help="Whether to apply logit normalization (p-norm) before feeding into the calibrator.")
    parser.add_argument("--fixed_ts_config", type=str, default=None, help="Path to a directory containing config.json with pre-tuned temperatures (required when --calibration_mode is fixed_ts).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(args.config)
    # Load models and preprocessors
    large_model, large_preprocess = get_model(config["LARGE"]["NAME"])
    small_model, small_preprocess = get_model(config["SMALL"]["NAME"])
    large_model = large_model.to(device)
    small_model = small_model.to(device)

    if args.calibration_mode == "fixed_ts":
        if args.fixed_ts_config is None:
            parser.error("--fixed_ts_config is required when --calibration_mode is fixed_ts")
        calibrator = JointFixedTS.load(args.fixed_ts_config)
    elif args.calibration_mode == "coca":
        calibrator = JointCoca(num_steps=5, lr=5e-2)
    elif args.calibration_mode == "duo_entropy":
        calibrator = JointDuoEntropy(num_steps=10, lr=5e-2)
    elif args.calibration_mode == "oracle_ts":
        calibrator = JointFixedTS()  # temperatures fitted per-corruption by evaluate_dynamic_duo
    elif args.calibration_mode == "batch_oracle_ts":
        calibrator = JointBatchNLLOracle(num_steps=20, lr=5e-2)
    elif args.calibration_mode == "sample_oracle_ts":
        calibrator = JointSampleNLLOracle(num_steps=20, lr=5e-2)
    else:
        raise ValueError(f"Invalid calibration mode: {args.calibration_mode}")

    # Set up Dynamic Duo

    duo = setup_duo(
        large=large_model, 
        large_preprocess=large_preprocess,
        small=small_model, 
        small_preprocess=small_preprocess,
        mode=args.mode,
        joint_calibrator = calibrator,
        calibration_mode=args.calibration_mode,
        cfg = config,
        steps=args.steps,
        norm_logits=args.norm_logits
    )
    # Evaluate Dynamic Duo on ImageNet-C
    evaluate_dynamic_duo(duo, config, num_samples=args.num_samples, seed=args.seed)
