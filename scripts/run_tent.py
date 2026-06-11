"""
python scripts/run_tent.py --config cfgs/dynamic_duo_config.yaml --mode large --steps 1
python scripts/run_tent.py --config cfgs/dynamic_duo_config.yaml --mode small --steps 1 --num_samples 5000 --seed 0

python scripts/run_tent.py --config cfgs/dynamic_duo_config.yaml --mode small --steps 1 --num_samples 5000 --seed 0
"""
from src.utils.model import get_model, _preprocess_batch
from src.utils.data import load_config, load_imagenetC
from src.utils.metrics import get_metrics_dict
from src.calibrators.joint_fixed_TS import JointFixedTS
from src.tta.tent import (setup_tent, copy_model_and_optimizer,
                          load_model_and_optimizer, forward_and_adapt, softmax_entropy)
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

_MODES = {"large", "small"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-model TENT TTA on ImageNet-C")
    parser.add_argument("--config",   type=str,   required=True)
    parser.add_argument("--mode",     type=str,   default="large", choices=list(_MODES))
    parser.add_argument("--steps",    type=int,   default=1,  help="Adaptation steps per batch")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use from each corruption/severity subset (default: all)")
    parser.add_argument("--seed",     type=int,   default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)

    calibrator = JointFixedTS(config["CALIBRATOR"]["TL"], config["CALIBRATOR"]["TS"])

    # Configure TENT on the adapted model; keep the other frozen in eval.
    if args.mode == "large":
        model_cfg = config["LARGE"]
        model, preprocess = get_model(config["LARGE"]["NAME"])
        model = model.to(device)
        model = setup_tent(model, norm_type="BN", cfg=model_cfg["OPTIM"])
    else:
        model_cfg = config["SMALL"]
        model, preprocess = get_model(config["SMALL"]["NAME"])
        model = model.to(device)
        model = setup_tent(model, norm_type="BN", cfg=model_cfg["OPTIM"])
    
    model.eval()

    run_name = (
        f"tent-{args.mode} | "
        f"{config['LARGE']['NAME']}+{config['SMALL']['NAME']} | "
        f"lr={model_cfg['OPTIM']['LR']} {model_cfg['OPTIM']['METHOD']} | steps={args.steps}"
    )
    wandb_run = wandb.init(
        project="dynamic-duos",
        name=run_name,
        config={
            "mode": args.mode,
            "steps": args.steps,
            "large/name": config["LARGE"]["NAME"],
            "large/norm": config["LARGE"]["NORM"],
            "small/name": config["SMALL"]["NAME"],
            "small/norm": config["SMALL"]["NORM"],
            f"{args.mode}/lr":    model_cfg["OPTIM"]["LR"],
            f"{args.mode}/optim": model_cfg["OPTIM"]["METHOD"],
            "calibrator/Tl": config["CALIBRATOR"]["TL"],
            "calibrator/Ts": config["CALIBRATOR"]["TS"],
            "eval/corruptions": config["EVAL"]["CORRUPTIONS"],
            "eval/severities":  config["EVAL"]["SEVERITIES"],
        },
    )
    results_rows = []
    for severity in config["EVAL"]["SEVERITIES"]:
        for corruption in config["EVAL"]["CORRUPTIONS"]:
            logger.info(f"TENT ({args.mode}) | {corruption} severity {severity}")

            model.reset()  # reset TENT

            loader = load_imagenetC(
                config["TEST_DIR"], severity, [corruption],
                device=device,
                batch_size=config["BS"],
                num_samples=args.num_samples,
                seed=args.seed,
            )
            prefix = f"{corruption}/s{severity}/"
            all_probs, all_labels = [], []
            diag = {"large": {"n": 0, "acc_sum": 0.0, "ent_sum": 0.0},
                    "small": {"n": 0, "acc_sum": 0.0, "ent_sum": 0.0},
                    "duo":   {"n": 0, "acc_sum": 0.0, "ent_sum": 0.0}}

            for imgs, labels in tqdm(loader, desc=f"{corruption} s{severity}"):
                x = _preprocess_batch(imgs, preprocess, device)

                z_out = model.forward(x)

                batch_labels_cpu = labels.cpu()
                log_dict = {}
                name = args.mode
                z = z_out
                probs = F.softmax(z.cpu(), dim=1)
                acc = (probs.argmax(1) == batch_labels_cpu).float().mean().item()
                nll = F.nll_loss(torch.log(probs.clamp(min=1e-8)), batch_labels_cpu).item()
                ent = softmax_entropy(z).mean().item()
                d = diag[name]
                d["n"] += 1
                d["acc_sum"] += acc
                d["ent_sum"] += ent
                log_dict[f"{prefix}{name}/batch_acc"] = acc
                log_dict[f"{prefix}{name}/avg_acc"]   = d["acc_sum"] / d["n"]
                log_dict[f"{prefix}{name}/batch_nll"] = nll
                log_dict[f"{prefix}{name}/batch_ent"] = ent
                log_dict[f"{prefix}{name}/avg_ent"]   = d["ent_sum"] / d["n"]
                wandb_run.log(log_dict)

                all_probs.append(F.softmax(z_out.cpu(), dim=1))
                all_labels.append(batch_labels_cpu)

            metrics = get_metrics_dict(torch.cat(all_probs), torch.cat(all_labels))
            logger.info(f"Results for {corruption} severity {severity}: {metrics}")
            wandb_run.log({f"{prefix}{k}": v for k, v in metrics.items()})
            results_rows.append({"corruption": corruption, "severity": severity, **metrics})

    cols = list(results_rows[0].keys())
    table = wandb.Table(columns=cols)
    for row in results_rows:
        table.add_data(*[row[c] for c in cols])
    wandb_run.log({"summary/results": table})
    wandb_run.finish()
