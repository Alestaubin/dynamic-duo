from src.tta.dynamic_duo import setup_duo, evaluate_dynamic_duo
from src.utils.model import get_model
from src.utils.data import load_config
from src.calibrators.fixed_TS import JointFixedTS
from src.calibrators.joint_coca import JointCoca
import argparse
import torch

"""
python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode both_duo \
    --steps 1 \
    --num_samples 1000 \
    --seed 0 \
    --calibration_mode coca
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dynamic Duo TTA on ImageNet-C")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--mode", type=str, default="both_duo", help="Dynamic Duo mode to run")
    parser.add_argument("--steps", type=int, default=1, help="Number of adaptation steps per batch")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use from each corruption/severity subset (default: all)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (when using fraction < 1.0)")  
    parser.add_argument("--calibration_mode", type=str, default="fixed", help="Calibrator mode for combining the duo logits.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    # Load models and preprocessors
    large_model, large_preprocess = get_model(config["LARGE"]["NAME"])
    small_model, small_preprocess = get_model(config["SMALL"]["NAME"])
    large_model = large_model.to(device)
    small_model = small_model.to(device)
    
    if args.calibration_mode == "fixed": 
        calibrator = JointFixedTS(Tl=config["CALIBRATOR"]["TL"], Ts=config["CALIBRATOR"]["TS"])
    elif args.calibration_mode == "coca": 
        calibrator = JointCoca(num_steps=5, lr=5e-2)

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
    )
    # Evaluate Dynamic Duo on ImageNet-C
    evaluate_dynamic_duo(duo, config, num_samples=args.num_samples, seed=args.seed)
