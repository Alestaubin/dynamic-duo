import argparse
import wandb

from src.utils.data import load_config, load_imagenetC
from src.utils.metrics import get_metrics_dict
from src.utils.model import get_model
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from copy import deepcopy
from torch.nn import functional as F
from src.utils.model import _preprocess_batch
from tqdm import tqdm

"""
python src/tta/tent.py  --config cfgs/dynamic_duo_config.yaml --model large --steps 1 --fraction 0.1 --seed 0
"""

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    loss = softmax_entropy(outputs).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs

def configure_model(model, norm_type):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    logger.info(f"Configuring TENT with norm type: {norm_type}")

    assert norm_type in {"BN", "LN", "GN"}, f"Unsupported norm type {norm_type}"

    # 2. Enable grads for normalization layers
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and norm_type == "BN" or \
            isinstance(m, nn.LayerNorm) and norm_type == "LN" or \
            isinstance(m, nn.GroupNorm) and norm_type == "GN":
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def collect_params(model, norm_type):
    """Collect the affine scale + shift parameters from norms.

    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and norm_type == "BN" or \
            isinstance(m, nn.LayerNorm) and norm_type == "LN" or \
            isinstance(m, nn.GroupNorm) and norm_type == "GN":
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def setup_optimizer(params, cfg):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg["METHOD"] == 'Adam':
        return optim.Adam(params,
                    lr=float(cfg["LR"]),
                    betas=(float(cfg["BETA"]), 0.999),
                    weight_decay=float(cfg["WD"]))
    elif cfg["METHOD"] == 'SGD':
        return optim.SGD(params,
                   lr=cfg["LR"],
                   momentum=cfg["MOMENTUM"],
                   dampening=cfg["DAMPENING"],
                   weight_decay=cfg["WD"],
                   nesterov=cfg["NESTEROV"])
    else:
        raise NotImplementedError

def setup_tent(model, norm_type, cfg, opt_cfg=None):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    tented_model = configure_model(model, norm_type)
    params, param_names = collect_params(tented_model, norm_type)
    
    if not params:
        raise ValueError("No parameters found for adaptation. Check if model has Norm layers.")
    
    optimizer = setup_optimizer(params, opt_cfg)

    logger.info(f"model for adaptation: %s", tented_model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)

    return tented_model, optimizer

     
def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TENT on ImageNet-C")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--model", type=str, default="large", help="small or large model to adapt")
    parser.add_argument("--steps", type=int, default=1, help="Number of adaptation steps per batch")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of ImageNet-C to evaluate on (0 < fraction <= 1.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (when using fraction < 1.0)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    # Load models and preprocessors
    if args.model == "large":
        model_name = config["LARGE"]["NAME"]
        norm_type = config["LARGE"]["NORM"]
        opt_cfg = config["LARGE"]["OPTIM"]
        bs = config["LARGE"]["BS"]
    elif args.model == "small":
        model_name = config["SMALL"]["NAME"]
        norm_type = config["SMALL"]["NORM"]
        opt_cfg = config["SMALL"]["OPTIM"]
        bs = config["SMALL"]["BS"]
    else:
        raise ValueError("Invalid model choice. Use 'small' or 'large'.")
    model, preprocess = get_model(model_name)
    
    # Set up TENT
    model, optimizer = setup_tent(model.to(device), norm_type, config, opt_cfg=opt_cfg)

    tented_model = Tent(model, optimizer, steps=args.steps, episodic=False)

    device = next(tented_model.parameters()).device
    logger.info(f"Device for TENT: {device}")

    run_name = (
        f"tent | {model_name} | {norm_type} | "
        f"lr={opt_cfg['LR']} {opt_cfg['METHOD']} | steps={args.steps}"
    )
    wandb_run = wandb.init(
        project="dynamic-duos",
        name=run_name,
        config={
            "model": args.model,
            "model_name": model_name,
            "norm_type": norm_type,
            "steps": args.steps,
            "optim/method": opt_cfg["METHOD"],
            "optim/lr": opt_cfg["LR"],
            "calibrator/Tl": config["CALIBRATOR"]["TL"],
            "calibrator/Ts": config["CALIBRATOR"]["TS"],
            "eval/corruptions": config["EVAL"]["CORRUPTIONS"],
            "eval/severities": config["EVAL"]["SEVERITIES"],
        },
    )

    results_rows = []
    for corruption in config["EVAL"]["CORRUPTIONS"]:
        for severity in config["EVAL"]["SEVERITIES"]:
            logger.info(f"Evaluating TENT on {corruption}, severity {severity}...")
            test_loader = load_imagenetC(config["TEST_DIR"], severity, [corruption],
                                    device=device,
                                    batch_size=bs, 
                                    fraction=args.fraction, seed=args.seed)
            prefix = f"{corruption}/s{severity}/"
            all_probs = []
            all_labels = []
            with torch.no_grad():
                for images, labels in tqdm(test_loader, desc="Evaluating"):
                    images = _preprocess_batch(images, preprocess, device)
                    outputs = tented_model(images)
                    batch_probs = F.softmax(outputs.detach().cpu(), dim=1)
                    batch_labels_cpu = labels.cpu()
                    batch_acc = (batch_probs.argmax(1) == batch_labels_cpu).float().mean().item()
                    batch_nll = F.nll_loss(torch.log(batch_probs.clamp(min=1e-8)), batch_labels_cpu).item()
                    logger.info(f"Batch Metrics - Accuracy: {batch_acc:.4f}, NLL: {batch_nll:.4f}")
                    wandb_run.log({
                        f"{prefix}batch_acc": batch_acc,
                        f"{prefix}batch_nll": batch_nll,
                    })
                    all_probs.append(batch_probs)
                    all_labels.append(batch_labels_cpu)
            all_probs = torch.cat(all_probs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            metrics = get_metrics_dict(all_probs, all_labels)
            logger.info(f"Metrics for {corruption} severity {severity}: {metrics}")
            wandb_run.log({f"{prefix}{k}": v for k, v in metrics.items()})
            results_rows.append({"corruption": corruption, "severity": severity, **metrics})

    cols = list(results_rows[0].keys())
    table = wandb.Table(columns=cols)
    for row in results_rows:
        table.add_data(*[row[c] for c in cols])
    wandb_run.log({"summary/results": table})
    wandb_run.finish()