"""
python scripts/run_baselines.py --config cfgs/dynamic_duo_config.yaml --mode joint
python scripts/run_baselines.py --config cfgs/dynamic_duo_config.yaml --mode large --fraction 0.1 --seed 0
"""
from src.utils.model import get_model, _preprocess_batch
from src.utils.data import load_config, load_imagenetC, _norm_logits
from src.utils.metrics import get_metrics_dict
from src.calibrators.fixed_TS import JointFixedTS
from src.tta.tent import softmax_entropy
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

"""
python scripts/run_baselines.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode small
"""

_MODES = {"large", "small", "joint"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baselines on ImageNet-C")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--mode", type=str, default="joint", choices=list(_MODES),
                        help="Which model(s) to evaluate: large, small, or joint (calibrated ensemble)")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to evaluate on")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for fraction sampling")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)

    large_model, large_preprocess = get_model(config["LARGE"]["NAME"])
    small_model, small_preprocess = get_model(config["SMALL"]["NAME"])
    large_model = large_model.to(device).eval()
    small_model = small_model.to(device).eval()

    calibrator = JointFixedTS(config["CALIBRATOR"]["TL"], config["CALIBRATOR"]["TS"])

    run_name = (
        f"baseline-{args.mode} | "
        f"{config['LARGE']['NAME']}+{config['SMALL']['NAME']} | "
        f"Tl={config['CALIBRATOR']['TL']} Ts={config['CALIBRATOR']['TS']}"
    )
    wandb_run = wandb.init(
        project="dynamic-duos",
        name=run_name,
        config={
            "mode": args.mode,
            "large/name": config["LARGE"]["NAME"],
            "small/name": config["SMALL"]["NAME"],
            "calibrator/Tl": config["CALIBRATOR"]["TL"],
            "calibrator/Ts": config["CALIBRATOR"]["TS"],
            "fraction": args.fraction,
            "seed": args.seed,
            "eval/corruptions": config["EVAL"]["CORRUPTIONS"],
            "eval/severities": config["EVAL"]["SEVERITIES"],
        },
    )

    results_rows = []
    for severity in config["EVAL"]["SEVERITIES"]:
        for corruption in config["EVAL"]["CORRUPTIONS"]:
            logger.info(f"Evaluating baseline ({args.mode}) on {corruption} severity {severity}")
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

            with torch.no_grad():
                for imgs, labels in tqdm(loader, desc=f"{corruption} s{severity}"):
                    x_large = _preprocess_batch(imgs, large_preprocess, device)
                    x_small = _preprocess_batch(imgs, small_preprocess, device)
                    z_large = _norm_logits(large_model(x_large))
                    z_small = _norm_logits(small_model(x_small))
                    z_duo   = calibrator(z_large, z_small)

                    if args.mode == "large":
                        logits = z_large
                    elif args.mode == "small":
                        logits = z_small
                    else:  # joint
                        logits = z_duo
                    components = [("large", z_large), ("small", z_small), ("duo", z_duo)]

                    batch_labels_cpu = labels.cpu()
                    log_dict = {}
                    for name, z in components:
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

                    all_probs.append(F.softmax(logits.cpu(), dim=1))
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
