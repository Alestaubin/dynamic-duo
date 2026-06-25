
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import datasets
from src.utils.data import _pil_collate_fn
import argparse
from src.tta.dynamic_duo import setup_duo, evaluate_dynamic_duo
from src.utils.model import get_model
from src.utils.data import load_config
from src.calibrators.joint_fixed_TS import JointFixedTS
from src.calibrators.joint_coca import JointCoca
from src.calibrators.joint_sample_nll_oracle import JointSampleNLLOracle
from src.calibrators.joint_relative_entropy import JointRelativeEntropy
from src.calibrators.joint_lambda_entropy import JointLambdaEntropy
from src.calibrators.joint_soft_anchor import JointSoftAnchor
from src.calibrators.joint_proxy_anchor_coca import JointProxyAnchorCoca
from src.proxies.proxies import build_proxy_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--mode", type=str, default="both_duo", help="Dynamic Duo mode to run")
    parser.add_argument("--steps", type=int, default=1, help="Number of adaptation steps per batch")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use from each corruption/severity subset (default: all)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (when using fraction < 1.0)")  
    parser.add_argument("--norm_logits", action="store_true", help="Whether to apply logit normalization (p-norm) before feeding into the calibrator.")
    parser.add_argument("--proxy_kind", type=str, default="prototype",
                        choices=["nuclear_norm", "atc", "prototype"],
                        help="Proxy for JointSoftAnchor (default: prototype).")
    parser.add_argument("--proxy_cache", type=str, default=None,
                        help="Path to a .pt file for caching source-pass proxy configs "
                             "(ATC thresholds + prototypes). Created on first run, "
                             "loaded on subsequent runs. Ignored for nuclear_norm.")
    parser.add_argument("--wandb", action="store_true",
                        help="Log results to Weights & Biases (default: off).")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Path to a CSV file for per-batch diagnostics "
                             "(proxy_anchor_coca only). Created/appended on each run.")

    args = parser.parse_args()

    config = load_config(args.config)
    # Load models and preprocessors
    large_model, large_preprocess = get_model(config["LARGE"]["NAME"])
    small_model, small_preprocess = get_model(config["SMALL"]["NAME"])
    large_model = large_model.to(device)
    small_model = small_model.to(device)


    print(f"Building proxy configs from source data (In-Distribution)...")
    src_ds = datasets.ImageFolder(config["VAL_DIR"])
    src_loader = DataLoader(
        src_ds, batch_size=config["BS"], shuffle=False,
        num_workers=config["WORKERS"], pin_memory=(device.type == "cuda"),
        collate_fn=_pil_collate_fn,
    )
    
    cfg_l, cfg_s = build_proxy_configs(
        large_model, large_preprocess, config["LARGE"]["NAME"],
        small_model, small_preprocess, config["SMALL"]["NAME"],
        src_loader, device,
        cache_path=args.proxy_cache,
    )

    calibrator = JointProxyAnchorCoca(
        proxy_kind=args.proxy_kind,
        cfg_l=cfg_l,
        cfg_s=cfg_s,
        csv_path=args.csv_path,
    )
