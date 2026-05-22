from src.tta.dynamic_duo import setup_duo, evaluate_dynamic_duo
from src.utils.model import get_model
from src.utils.data import load_config
from src.calibrators.fixed_TS import JointFixedTS
import argparse
import torch

"""
python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode both_duo \
    --steps 1
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dynamic Duo TTA on ImageNet-C")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--mode", type=str, default="both_duo", help="Dynamic Duo mode to run")
    parser.add_argument("--steps", type=int, default=1, help="Number of adaptation steps per batch")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    # Load models and preprocessors
    large_model, large_preprocess = get_model(config["LARGE"]["NAME"])
    small_model, small_preprocess = get_model(config["SMALL"]["NAME"])
    large_model = large_model.to(device)
    small_model = small_model.to(device)

    calibrator = JointFixedTS(Tl=config["CALIBRATOR"]["TL"], Ts=config["CALIBRATOR"]["TS"])
    
    # Set up Dynamic Duo
    duo = setup_duo(
        large=large_model, 
        large_preprocess=large_preprocess,
        small=small_model, 
        small_preprocess=small_preprocess,
        mode=args.mode,
        joint_calibrator = calibrator,
        cfg = config,
        steps=args.steps,
    )

    # Evaluate Dynamic Duo on ImageNet-C
    evaluate_dynamic_duo(duo, config)
