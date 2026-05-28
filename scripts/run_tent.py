"""
python scripts/run_tent.py --config cfgs/dynamic_duo_config.yaml --mode large --steps 1
python scripts/run_tent.py --config cfgs/dynamic_duo_config.yaml --mode small --steps 1 --fraction 0.1 --seed 0
"""
from src.utils.model import get_model, _preprocess_batch
from src.utils.data import load_config, load_imagenetC, _norm_logits
from src.utils.metrics import get_metrics_dict
from src.calibrators.fixed_TS import JointFixedTS
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
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed",     type=int,   default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)

    large_model, large_preprocess = get_model(config["LARGE"]["NAME"])
    small_model, small_preprocess = get_model(config["SMALL"]["NAME"])
    large_model = large_model.to(device)
    small_model = small_model.to(device)

    calibrator = JointFixedTS(config["CALIBRATOR"]["TL"], config["CALIBRATOR"]["TS"])

    # Configure TENT on the adapted model; keep the other frozen in eval.
    if args.mode == "large":
        model_cfg = config["LARGE"]
        large_model, optimizer = setup_tent(large_model, model_cfg["NORM"], config, opt_cfg=model_cfg["OPTIM"])
        model_state, opt_state = copy_model_and_optimizer(large_model, optimizer)
        small_model.eval()
    else:
        model_cfg = config["SMALL"]
        small_model, optimizer = setup_tent(small_model, model_cfg["NORM"], config, opt_cfg=model_cfg["OPTIM"])
        model_state, opt_state = copy_model_and_optimizer(small_model, optimizer)
        large_model.eval()

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

            if args.mode == "large":
                load_model_and_optimizer(large_model, optimizer, model_state, opt_state)
            else:
                load_model_and_optimizer(small_model, optimizer, model_state, opt_state)

            loader = load_imagenetC(
                config["TEST_DIR"], severity, [corruption],
                device=device,
                batch_size=config["LARGE"]["BS"],
                fraction=args.fraction,
                seed=args.seed,
            )
            prefix = f"{corruption}/s{severity}/"
            all_probs, all_labels = [], []
            diag = {"large": {"n": 0, "acc_sum": 0.0, "ent_sum": 0.0},
                    "small": {"n": 0, "acc_sum": 0.0, "ent_sum": 0.0},
                    "duo":   {"n": 0, "acc_sum": 0.0, "ent_sum": 0.0}}

            for imgs, labels in tqdm(loader, desc=f"{corruption} s{severity}"):
                x_large = _preprocess_batch(imgs, large_preprocess, device)
                x_small = _preprocess_batch(imgs, small_preprocess, device)

                # Adapt for `steps` iterations, then do a clean final inference.
                for _ in range(args.steps):
                    if args.mode == "large":
                        forward_and_adapt(x_large, large_model, optimizer)
                    else:
                        forward_and_adapt(x_small, small_model, optimizer)

                with torch.no_grad():
                    z_large = _norm_logits(large_model(x_large))
                    z_small = _norm_logits(small_model(x_small))
                    z_duo   = calibrator(z_large, z_small)

                batch_labels_cpu = labels.cpu()
                log_dict = {}
                for name, z in (("large", z_large), ("small", z_small), ("duo", z_duo)):
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

                z_out = z_large if args.mode == "large" else z_small
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
