import torch
import torch.jit
import torch.nn as nn
import torch.optim as optim
from src.tta.tent import copy_model_and_optimizer, load_model_and_optimizer, setup_tent, softmax_entropy
from src.utils.data import load_imagenetC
from src.utils.metrics import get_metrics_dict
from src.utils.model import _preprocess_batch
import logging
import torch.nn.functional as F
from tqdm import tqdm
import wandb


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("src.tta.dynamic_duo").setLevel(logging.DEBUG)


_MODES = {"both_duo", "large_duo", "small_duo",
          "large_indep", "small_indep", "both_indep"}

# (adapt_large, adapt_small, signal) for each mode.
_MODE_SPEC = {
    "both_duo":     (True,  True,  "duo"),
    "large_duo":    (True,  False, "duo"),
    "small_duo":    (False, True,  "duo"),
    "large_indep":  (True,  False, "indep"),
    "small_indep":  (False, True,  "indep"),
    "both_indep":   (True,  True,  "indep"),
}

class DynamicDuo(nn.Module):
    """Asymmetric Duo Test-Time Adaptation.

    Wraps a large model and a small model (each already configured for TENT:
    BN affine params require grad, batch stats forced) and drives adaptation
    according to the selected mode. This module is the sole owner of the
    backward/step calls — the inner models should NOT be Tent-wrapped.
    """
    def __init__(
        self,
        large: nn.Module,
        large_preprocess,
        large_optimizer: optim.Optimizer | None,
        small: nn.Module,
        small_preprocess,
        small_optimizer: optim.Optimizer | None,
        joint_calibrator: nn.Module,
        mode: str = "both_duo",
        steps: int = 1,
    ):
        super().__init__()
        assert mode in _MODES, f"Invalid mode {mode}. Must be one of {_MODES}."
        self.large = large
        self.large_preprocess = large_preprocess
        self.small = small
        self.small_preprocess = small_preprocess
        self.joint_calibrator = joint_calibrator
        self.large_optimizer = large_optimizer
        self.small_optimizer = small_optimizer
        self.mode = mode
        self.steps = steps

        self.adapt_large, self.adapt_small, _ = _MODE_SPEC[mode]
        logger.info(f"Initialized DynamicDuo | mode={mode} steps={steps} adapt_large={self.adapt_large} adapt_small={self.adapt_small}")
        self._reset_diagnostics()
        if self.adapt_large:
            assert large_optimizer is not None, f"{mode} needs a large optimizer"
            self.large_model_state, self.large_optimizer_state = \
                copy_model_and_optimizer(self.large, self.large_optimizer)

        if self.adapt_small:
            assert small_optimizer is not None, f"{mode} needs a small optimizer"
            self.small_model_state, self.small_optimizer_state = \
                copy_model_and_optimizer(self.small, self.small_optimizer)

    def forward(self, x, labels=None):
        for _ in range(self.steps):
            outputs, z_large, z_small = forward_and_adapt(
                x,
                self.large, self.large_preprocess, self.large_optimizer,
                self.small, self.small_preprocess, self.small_optimizer,
                self.joint_calibrator,
                self.mode,
            )
        if logger.isEnabledFor(logging.DEBUG) and labels is not None:
            self._log_batch_diagnostics(z_large, z_small, outputs, labels)
        return outputs

    def _reset_diagnostics(self):
        self._diag = {k: {"n": 0, "acc_sum": 0.0, "nll_sum": 0.0}
                      for k in ("large", "small", "duo")}

    def _log_batch_diagnostics(self, z_large, z_small, z_duo, labels):
        labels_cpu = labels.cpu()
        for name, logits in (("large", z_large), ("small", z_small), ("duo", z_duo)):
            probs = F.softmax(logits.detach().cpu(), dim=1)
            acc = (probs.argmax(1) == labels_cpu).float().mean().item()
            nll = F.nll_loss(torch.log(probs.clamp(min=1e-8)), labels_cpu).item()
            d = self._diag[name]
            d["n"] += 1
            d["acc_sum"] += acc
            d["nll_sum"] += nll
            avg_acc = d["acc_sum"] / d["n"]
            avg_nll = d["nll_sum"] / d["n"]
            logger.debug(f"  {name:5s}: acc={acc:.4f} (avg={avg_acc:.4f})  nll={nll:.4f} (avg={avg_nll:.4f})")

    def reset(self):
        """Reset the model and optimizer states to before adaptation."""
        self._reset_diagnostics()
        if self.adapt_large:
            logger.info("Resetting large model to pre-adaptation state")
            load_model_and_optimizer(
                self.large, self.large_optimizer,
                self.large_model_state, self.large_optimizer_state
            )

        if self.adapt_small:
            logger.info("Resetting small model to pre-adaptation state")
            load_model_and_optimizer(
                self.small, self.small_optimizer,
                self.small_model_state, self.small_optimizer_state
            )

@torch.enable_grad()
def forward_and_adapt(x, large, large_preprocess, large_optimizer,
                      small, small_preprocess, small_optimizer,
                      joint_calibrator, mode):
    adapt_large, adapt_small, signal = _MODE_SPEC[mode]
    device = next(large.parameters()).device
    x_large = _preprocess_batch(x, large_preprocess, device)
    x_small = _preprocess_batch(x, small_preprocess, device)

    z_large = large(x_large)
    z_small = small(x_small)

    if signal == "duo":
        zl = z_large if adapt_large else z_large.detach()
        zs = z_small if adapt_small else z_small.detach()
        z_bar = joint_calibrator(zl, zs)
        loss = softmax_entropy(z_bar).mean(0)
        logger.debug(f"duo loss={loss.item():.4f}")

        loss.backward()
        for opt, do in ((large_optimizer, adapt_large),
                        (small_optimizer, adapt_small)):
            if do and opt is not None:
                opt.step()
        for opt, do in ((large_optimizer, adapt_large),
                        (small_optimizer, adapt_small)):
            if do and opt is not None:
                opt.zero_grad()

    else:  # indep
        if adapt_large:
            large_loss = softmax_entropy(z_large).mean(0)
            logger.debug(f"large indep loss={large_loss.item():.4f}")
            large_loss.backward()
            large_optimizer.step(); large_optimizer.zero_grad()
        if adapt_small:
            small_loss = softmax_entropy(z_small).mean(0)
            logger.debug(f"small indep loss={small_loss.item():.4f}")
            small_loss.backward()
            small_optimizer.step(); small_optimizer.zero_grad()

    with torch.no_grad():
        z_large_final = large(x_large)
        z_small_final = small(x_small)
        return joint_calibrator(z_large_final, z_small_final), z_large_final, z_small_final
    
def setup_duo(large, large_preprocess, small, small_preprocess, joint_calibrator, mode, cfg, steps):
    """Configure a DynamicDuo for TENT adaptation.

    This is a helper function to set up the DynamicDuo with the appropriate
    optimizers and calibrator for the selected mode. It will tent the inner
    models, set up the optimizers, and then wrap them in a DynamicDuo.
    """
    assert mode in _MODES, f"Invalid mode {mode}. Must be one of {_MODES}."
    adapt_large, adapt_small, signal = _MODE_SPEC[mode]
    logger.info(f"Setting up DynamicDuo | mode={mode} steps={steps} adapt_large={adapt_large} adapt_small={adapt_small}")

    large_optimizer, small_optimizer = None, None

    if adapt_large:
        logger.info(f"Configuring large model with norm={cfg['LARGE']['NORM']}")
        large, large_optimizer = setup_tent(large, cfg["LARGE"]["NORM"], cfg, opt_cfg=cfg["LARGE"]["OPTIM"])
    if adapt_small:
        logger.info(f"Configuring small model with norm={cfg['SMALL']['NORM']}")
        small, small_optimizer = setup_tent(small, cfg["SMALL"]["NORM"], cfg, opt_cfg=cfg["SMALL"]["OPTIM"])

    dynamic_duo = DynamicDuo(
        large=large,
        large_preprocess=large_preprocess,
        large_optimizer=large_optimizer,
        small=small,
        small_preprocess=small_preprocess,
        small_optimizer=small_optimizer,
        joint_calibrator=joint_calibrator,
        mode=mode,
        steps=steps
    )
    return dynamic_duo

def run_duo(duo, data_loader, wandb_run=None, wandb_prefix=""):
    """
    Run a forward pass through the DynamicDuo on the given data loader.

    """
    all_outputs = []
    all_labels = []
    for imgs, labels in tqdm(data_loader, desc="run"):
        outputs = duo(imgs, labels=labels)
        all_outputs.append(outputs.cpu())
        all_labels.append(labels.cpu())

        if wandb_run is not None:
            with torch.no_grad():
                batch_probs = F.softmax(outputs.detach().cpu(), dim=1)
                batch_labels_cpu = labels.cpu()
                batch_acc = (batch_probs.argmax(1) == batch_labels_cpu).float().mean().item()
                batch_nll = F.nll_loss(torch.log(batch_probs.clamp(min=1e-8)), batch_labels_cpu).item()
                logger.info(f"{wandb_prefix} batch_acc={batch_acc:.4f} batch_nll={batch_nll:.4f}")
            wandb_run.log({
                f"{wandb_prefix}batch_acc": batch_acc,
                f"{wandb_prefix}batch_nll": batch_nll,
            })

    logits = torch.cat(all_outputs)
    probs = F.softmax(logits, dim=1)

    return probs, torch.cat(all_labels)

def evaluate_dynamic_duo(duo, cfg, wandb_project="dynamic-duos", fraction=1.0, seed=None):
    adapt_large, adapt_small, signal = _MODE_SPEC[duo.mode]
    run_name = (
        f"{duo.mode} | {cfg['LARGE']['NAME']}+{cfg['SMALL']['NAME']} | "
        f"Tl={cfg['CALIBRATOR']['TL']} Ts={cfg['CALIBRATOR']['TS']} | steps={duo.steps}"
    )
    wandb_run = wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "mode": duo.mode,
            "adapt_large": adapt_large,
            "adapt_small": adapt_small,
            "signal": signal,
            "steps": duo.steps,
            "large/name": cfg["LARGE"]["NAME"],
            "large/norm": cfg["LARGE"]["NORM"],
            "large/lr": cfg["LARGE"]["OPTIM"]["LR"],
            "large/optim": cfg["LARGE"]["OPTIM"]["METHOD"],
            "small/name": cfg["SMALL"]["NAME"],
            "small/norm": cfg["SMALL"]["NORM"],
            "small/lr": cfg["SMALL"]["OPTIM"]["LR"],
            "small/optim": cfg["SMALL"]["OPTIM"]["METHOD"],
            "calibrator/Tl": cfg["CALIBRATOR"]["TL"],
            "calibrator/Ts": cfg["CALIBRATOR"]["TS"],
            "eval/corruptions": cfg["EVAL"]["CORRUPTIONS"],
            "eval/severities": cfg["EVAL"]["SEVERITIES"],
        },
    )

    results_rows = []
    for severity in cfg["EVAL"]["SEVERITIES"]:
        for corruption_type in cfg["EVAL"]["CORRUPTIONS"]:
            logger.info(f"Evaluating corruption {corruption_type} severity {severity}")
            try:
                duo.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
            loader = load_imagenetC(cfg["TEST_DIR"], severity, [corruption_type],
                                    device=next(duo.parameters()).device,
                                    batch_size=cfg["LARGE"]["BS"], fraction=fraction, seed=seed)
            prefix = f"{corruption_type}/s{severity}/"
            probs, labels = run_duo(duo, loader, wandb_run=wandb_run, wandb_prefix=prefix)
            metrics = get_metrics_dict(probs, labels)
            logger.info(f"Results for {corruption_type} severity {severity}: {metrics}")

            wandb_run.log({f"{prefix}{k}": v for k, v in metrics.items()})
            results_rows.append({"corruption": corruption_type, "severity": severity, **metrics})

    if results_rows:
        cols = list(results_rows[0].keys())
        table = wandb.Table(columns=cols)
        for row in results_rows:
            table.add_data(*[row[c] for c in cols])
        wandb_run.log({"summary/results": table})

    wandb_run.finish()


def tune_duo(duo, data_loader):
    pass

