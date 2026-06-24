import torch
import torch.nn as nn
import torch.optim as optim
from src.tta.tent import configure_model, copy_model_and_optimizer, load_model_and_optimizer, setup_optimizer, softmax_entropy, collect_params, configure_model_frozen
from src.utils.data import load_imagenetC
from src.utils.metrics import get_metrics_dict, get_intersection_metrics
from src.utils.model import _preprocess_batch
from src.utils.logit_transforms import logit_pnorm, normalize
import logging
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from src.utils.logits import get_model_logits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
# logging.getLogger("src.tta.dynamic_duo").setLevel(logging.DEBUG)


_MODES = {"both_duo", "large_duo", "small_duo",
          "large_indep", "small_indep", "both_indep", "no_adapt"}

# (adapt_large, adapt_small, signal) for each mode.
_MODE_SPEC = {
    "both_duo":     (True,  True,  "duo"),
    "large_duo":    (True,  False, "duo"),
    "small_duo":    (False, True,  "duo"),
    "large_indep":  (True,  False, "indep"),
    "small_indep":  (False, True,  "indep"),
    "both_indep":   (True,  True,  "indep"),
    "no_adapt":     (False, False, None),
}

_CALIB_MODES = {"fixed_ts", "coca", "duo_entropy", "oracle_ts", "batch_oracle_ts", "sample_oracle_ts", "relative_entropy", "coca_entropy", "lambda_entropy", "soft_anchor"}

class DynamicDuo(nn.Module):
    """Asymmetric Duo Test-Time Adaptation.

    Wraps a large model and a small model (each already configured for TENT:
    BN affine params require grad, batch stats forced) and drives adaptation
    according to the selected mode.

    Calibrators
    -----------
    `joint_calibrator` is a `BaseJointCalibrator`, used through its interface:
      * `calibrate_with_grad(z_l, z_s)` on the adaptation (loss) path,
      * `calibrate(z_l, z_s)` on the no-grad inference path.
    The first logits argument is the large (anchor) model.

    Calibration modes (`calibration_mode`):
      * "fixed_ts"   -> JointFixedTS pre-tuned on held-out data. Frozen here;
                        only BN params adapt. Never reset.
      * "coca"       -> JointCoca: self-adapting per batch (owns its optimizer).
      * "duo_entropy"-> JointDuoEntropy: like coca but minimises ensemble
                        entropy with two temperatures.
      * "oracle_ts"      -> JointFixedTS fitted per-corruption on test data
                           (uses test labels — cheating). evaluate_dynamic_duo
                           handles the tuning loop.
      * "batch_oracle_ts" -> JointBatchNLLOracle: two shared scalars per batch,
                            minimises NLL against batch labels (cheating).
      * "sample_oracle_ts"-> JointSampleNLLOracle: one (T_l, T_s) pair per
                            sample per batch — the tightest per-instance oracle.
                            Both oracle variants use set_labels() injected by
                            DynamicDuo.forward().
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
        calibration_mode: str = "fixed_ts",
        norm_logits: bool = False,
    ):
        super().__init__()
        assert mode in _MODES, f"Invalid mode {mode}. Must be one of {_MODES}."
        assert calibration_mode in _CALIB_MODES, \
            f"Invalid calibration mode {calibration_mode}. Must be one of {_CALIB_MODES}"
        self.large = large
        self.large_preprocess = large_preprocess
        self.small = small
        self.small_preprocess = small_preprocess
        self.joint_calibrator = joint_calibrator
        self.large_optimizer = large_optimizer
        self.small_optimizer = small_optimizer
        self.mode = mode
        self.steps = steps
        self.calibration_mode = calibration_mode
        self.norm_logits = norm_logits

        self.adapt_large, self.adapt_small, _ = _MODE_SPEC[mode]
        logger.info(
            f"Initialized DynamicDuo | mode={mode} steps={steps} "
            f"adapt_large={self.adapt_large} adapt_small={self.adapt_small} "
            f"calibration_mode={calibration_mode}"
        )
        self._reset_diagnostics()

        device = next(self.large.parameters()).device
        self.joint_calibrator.to(device)

        if self.adapt_large:
            assert large_optimizer is not None, f"{mode} needs a large optimizer"
            self.large_model_state, self.large_optimizer_state = \
                copy_model_and_optimizer(self.large, self.large_optimizer)

        if self.adapt_small:
            assert small_optimizer is not None, f"{mode} needs a small optimizer"
            self.small_model_state, self.small_optimizer_state = \
                copy_model_and_optimizer(self.small, self.small_optimizer)

        if calibration_mode == "fixed_ts":
            n_frozen = 0
            for p in self.joint_calibrator.parameters():
                p.requires_grad_(False)
                n_frozen += 1
            logger.info(
                f"Calibrator FIXED (fixed_ts) | froze {n_frozen} param tensors "
                f"(assumed pre-tuned on held-out data)"
            )
        elif calibration_mode in {"coca", "duo_entropy"}:
            # Self-adapting: owns its optimization internally, fits per batch.
            logger.info(f"Calibrator SELF-ADAPTING ({calibration_mode}) | fits its temperature(s) per batch")
        elif calibration_mode == "oracle_ts":
            # Oracle: temperatures fitted per-corruption by evaluate_dynamic_duo.
            # Left unfrozen here; evaluate_dynamic_duo freezes after each fit.
            logger.info("Calibrator ORACLE (oracle_ts) | will be fitted per-corruption on test data")
        elif calibration_mode == "batch_oracle_ts":
            # Per-batch oracle: uses test labels injected via set_labels() each forward.
            logger.info("Calibrator ORACLE (batch_oracle_ts) | fits T_l, T_s per batch using test labels")
        elif calibration_mode == "sample_oracle_ts":
            # Per-sample oracle: one (T_l, T_s) per sample, injected via set_labels().
            logger.info("Calibrator ORACLE (sample_oracle_ts) | fits per-sample T_l, T_s using test labels")
        elif calibration_mode == "soft_anchor":
            proxy_kind = getattr(joint_calibrator, "proxy_kind", "unknown")
            logger.info(
                "Calibrator SOFT-ANCHOR (soft_anchor) | proxy=%s | "
                "self-adapting T_l, T_s per batch via KL to proxy-weighted anchor",
                proxy_kind,
            )

    def forward(self, x, labels=None):
        if self.calibration_mode in {""
        "", "sample_oracle_ts"} and labels is not None:
            self.joint_calibrator.set_labels(labels)
        for _ in range(self.steps):
            outputs, z_large, z_small = forward_and_adapt(
                x,
                self.large, self.large_preprocess, self.large_optimizer,
                self.small, self.small_preprocess, self.small_optimizer,
                self.joint_calibrator,
                self.mode,
                norm_logits=self.norm_logits
            )
        if labels is not None:
            self._log_batch_diagnostics(z_large, z_small, outputs, labels)
        return outputs, z_large, z_small

    def _reset_diagnostics(self):
        self._diag = {k: {"n": 0, "acc_sum": 0.0, "nll_sum": 0.0, "ent_sum": 0.0,
                          "acc_last": 0.0, "nll_last": 0.0, "ent_last": 0.0}
                      for k in ("large", "small", "duo")}

    def _log_batch_diagnostics(self, z_large, z_small, z_duo, labels):
        labels_cpu = labels.cpu()
        for name, logits in (("large", z_large), ("small", z_small), ("duo", z_duo)):
            probs = F.softmax(logits.detach().cpu(), dim=1)
            acc = (probs.argmax(1) == labels_cpu).float().mean().item()
            nll = F.nll_loss(torch.log(probs.clamp(min=1e-8)), labels_cpu).item()
            ent = softmax_entropy(logits.detach()).mean().item()
            d = self._diag[name]
            d["n"] += 1
            d["acc_sum"] += acc
            d["nll_sum"] += nll
            d["ent_sum"] += ent
            d["acc_last"] = acc
            d["nll_last"] = nll
            d["ent_last"] = ent
            if logger.isEnabledFor(logging.DEBUG):
                avg_acc = d["acc_sum"] / d["n"]
                avg_nll = d["nll_sum"] / d["n"]
                avg_ent = d["ent_sum"] / d["n"]
                logger.debug(f"  {name:5s}: acc={acc:.4f} (avg={avg_acc:.4f})  nll={nll:.4f} (avg={avg_nll:.4f})  ent={ent:.4f} (avg={avg_ent:.4f})")

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
        # Calibrator: fixed_ts is frozen (nothing to reset); coca self-adapts
        # fresh each batch (reset_each_batch), so there is nothing to restore.


@torch.enable_grad()
def forward_and_adapt(x, large, large_preprocess, large_optimizer,
                      small, small_preprocess, small_optimizer,
                      joint_calibrator, mode, norm_logits=False):
    adapt_large, adapt_small, signal = _MODE_SPEC[mode]
    device = next(large.parameters()).device
    x_large = _preprocess_batch(x, large_preprocess, device)
    x_small = _preprocess_batch(x, small_preprocess, device)

    z_large = large(x_large)
    z_small = small(x_small)

    if norm_logits:
        z_large = normalize(logits=z_large, p=2.0, centralize_logits=True)
        z_small = normalize(logits=z_small, p=2.0, centralize_logits=True)

    if signal == "duo":
        zl = z_large if adapt_large else z_large.detach()
        zs = z_small if adapt_small else z_small.detach()
        # Differentiable calibration path. fixed_ts: temps frozen, grad flows
        # only to the un-detached model logits. coca: fits its tau internally on
        # detached logits (separate optimizer/loss), returns aggregated logits
        # that are differentiable w.r.t. the model logits.
        z_bar = joint_calibrator.calibrate_with_grad(logits_l=zl, logits_s=zs)
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
                opt.zero_grad(set_to_none=True)
        # No calibrator grad to clear: fixed_ts is frozen, coca manages its own.

    elif signal == "indep":  # indep
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
    elif signal is None:
        pass  # no adaptation, just forward
    else:
        raise ValueError(f"Invalid signal {signal} in mode {mode}")

    with torch.no_grad():
        # Inference path: calibrate() (no-grad combination). For coca this reuses
        # the tau already fit on this batch by calibrate_with_grad (same logits).
        return joint_calibrator.calibrate(logits_l=z_large, logits_s=z_small), z_large, z_small

def setup_duo(large, large_preprocess, small, small_preprocess, joint_calibrator, calibration_mode, mode, cfg, steps, norm_logits=False):
    """
    Configure a DynamicDuo for TENT adaptation.
    """
    assert mode in _MODES, f"Invalid mode {mode}. Must be one of {_MODES}."
    adapt_large, adapt_small, signal = _MODE_SPEC[mode]
    logger.info(f"Setting up DynamicDuo | mode={mode} steps={steps} adapt_large={adapt_large} adapt_small={adapt_small}")

    large_optimizer, small_optimizer = None, None

    if adapt_large:
        logger.info(f"Configuring large model with norm={cfg['LARGE']['NORM']}")
        large_model = configure_model(large, cfg["LARGE"]["NORM"])
        params, param_names = collect_params(large_model, cfg["LARGE"]["NORM"])

        if not params:
            raise ValueError("No parameters found for adaptation. Check if model has Norm layers.")

        large_optimizer = setup_optimizer(params, cfg["LARGE"]["OPTIM"])

        logger.info(f"model for adaptation: %s", large_model)
        logger.info(f"params for adaptation: %s", param_names)
        logger.info(f"optimizer for adaptation: %s", large_optimizer)
    else: 
        logger.info(f"Configuring large model (frozen, batch stats) with norm={cfg['LARGE']['NORM']}")
        configure_model_frozen(large, cfg["LARGE"]["NORM"])

    if adapt_small:
        logger.info(f"Configuring small model with norm={cfg['SMALL']['NORM']}")
        small_model = configure_model(small, cfg["SMALL"]["NORM"])
        params, param_names = collect_params(small_model, cfg["SMALL"]["NORM"])

        if not params:
            raise ValueError("No parameters found for adaptation. Check if model has Norm layers.")

        small_optimizer = setup_optimizer(params, cfg["SMALL"]["OPTIM"])
        logger.info(f"model for adaptation: %s", small_model)
        logger.info(f"params for adaptation: %s", param_names)
        logger.info(f"optimizer for adaptation: %s", small_optimizer)
    else: 
        logger.info(f"Configuring small model (frozen, batch stats) with norm={cfg['SMALL']['NORM']}")
        configure_model_frozen(small, cfg["SMALL"]["NORM"])


    # For soft_anchor with prototype proxy: register feature hooks on the
    # (now TENT-configured) models so the calibrator can read features each forward.
    if calibration_mode == "soft_anchor" and getattr(joint_calibrator, "proxy_kind", None) == "prototype":
        joint_calibrator.register_hooks(large, small)
        logger.info("setup_duo: registered prototype feature hooks on large and small models")

    dynamic_duo = DynamicDuo(
        large=large,
        large_preprocess=large_preprocess,
        large_optimizer=large_optimizer,
        small=small,
        small_preprocess=small_preprocess,
        small_optimizer=small_optimizer,
        joint_calibrator=joint_calibrator,
        calibration_mode=calibration_mode,
        mode=mode,
        steps=steps,
        norm_logits=norm_logits,
    )
    return dynamic_duo

def run_duo(duo, data_loader, wandb_run=None, wandb_prefix=""):
    """
    Run a forward pass through the DynamicDuo on the given data loader.
    Returns a dict of {model_name: probs} and labels.
    """
    all_duo, all_large, all_small = [], [], []
    all_labels = []
    for imgs, labels in tqdm(data_loader, desc="run"):
        outputs, z_large, z_small = duo(imgs, labels=labels)
        all_duo.append(outputs.detach().cpu())
        all_large.append(z_large.detach().cpu())
        all_small.append(z_small.detach().cpu())
        all_labels.append(labels.cpu())

        if wandb_run is not None:
            log_dict = {}
            for name in ("large", "small", "duo"):
                d = duo._diag[name]
                if d["n"] > 0:
                    avg_acc = d["acc_sum"] / d["n"]
                    avg_ent = d["ent_sum"] / d["n"]
                    avg_nll = d["nll_sum"] / d["n"]
                    log_dict[f"{wandb_prefix}{name}/batch_acc"] = d["acc_last"]
                    log_dict[f"{wandb_prefix}{name}/avg_acc"] = avg_acc
                    log_dict[f"{wandb_prefix}{name}/batch_nll"] = d["nll_last"]
                    log_dict[f"{wandb_prefix}{name}/avg_nll"] = avg_nll
                    log_dict[f"{wandb_prefix}{name}/batch_ent"] = d["ent_last"]
                    log_dict[f"{wandb_prefix}{name}/avg_ent"] = avg_ent
            wandb_run.log(log_dict)

    labels = torch.cat(all_labels)
    probs = {
        "duo":   F.softmax(torch.cat(all_duo),   dim=1),
        "large": F.softmax(torch.cat(all_large), dim=1),
        "small": F.softmax(torch.cat(all_small), dim=1),
    }
    return probs, labels

@torch.no_grad()
def collect_logits(large, large_preprocess, small, small_preprocess, data_loader):
    """Forward pass over data_loader without any adaptation; returns stacked logits and labels."""
    device = next(large.parameters()).device
    all_z_l, all_z_s, all_labels = [], [], []
    for imgs, labels in tqdm(data_loader, desc="collect logits"):
        x_l = _preprocess_batch(imgs, large_preprocess, device)
        x_s = _preprocess_batch(imgs, small_preprocess, device)
        all_z_l.append(large(x_l).cpu())
        all_z_s.append(small(x_s).cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_z_l), torch.cat(all_z_s), torch.cat(all_labels)


def evaluate_dynamic_duo(duo, cfg, wandb_project="dynamic-duos", num_samples=None, seed=None):
    adapt_large, adapt_small, signal = _MODE_SPEC[duo.mode]
    calibration_name = duo.calibration_mode if duo.calibration_mode != "fixed_ts" else "fixed_ts Tl=" + str(duo.joint_calibrator.Tl.item()) + ", Ts=" + str(duo.joint_calibrator.Ts.item())
    run_name = (
        f"{duo.mode} | {calibration_name}{' normalized' if duo.norm_logits else ' '}| {cfg['LARGE']['NAME']}+{cfg['SMALL']['NAME']} | steps={duo.steps}"
    )
    wandb_run = wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "mode": duo.mode,
            "calibration_mode": duo.calibration_mode,
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
            **({
                "calibrator/Tl": duo.joint_calibrator.Tl.item(),
                "calibrator/Ts": duo.joint_calibrator.Ts.item(),
            } if duo.calibration_mode == "fixed_ts" else {}),
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

            prefix = f"{corruption_type}/s{severity}/"
            device = next(duo.parameters()).device
            loader_kwargs = dict(
                severities=severity, corruption_types=[corruption_type],
                device=device, batch_size=cfg["BS"],
                num_samples=num_samples, seed=seed,
            )

            if duo.calibration_mode == "oracle_ts":
                z_l, labels_l = get_model_logits(
                    model_name=cfg["LARGE"]["NAME"], val_dir=cfg["VAL_DIR"],
                    test_dir=cfg["TEST_DIR"], cache_dir="cache/logits",
                    batch_size=cfg["BS"], num_workers=cfg["WORKERS"],
                    corruption=corruption_type, severity=severity, device=device,
                    tent_mode=True, norm_type=cfg["LARGE"]["NORM"],
                )
                z_s, labels_s = get_model_logits(
                    model_name=cfg["SMALL"]["NAME"], val_dir=cfg["VAL_DIR"],
                    test_dir=cfg["TEST_DIR"], cache_dir="cache/logits",
                    batch_size=cfg["BS"], num_workers=cfg["WORKERS"],
                    corruption=corruption_type, severity=severity, device=device,
                    tent_mode=True, norm_type=cfg["SMALL"]["NORM"],
                )
                assert torch.equal(labels_l, labels_s), "Logit collection mismatch: large and small labels differ"
                cal = duo.joint_calibrator
                for p in cal.parameters():
                    p.requires_grad_(True)
                cal.tune(z_l, z_s, labels_l)
                for p in cal.parameters():
                    p.requires_grad_(False)
                logger.info(
                    f"Oracle TS fitted | Tl={cal.Tl.item():.4f}  Ts={cal.Ts.item():.4f}"
                )

            loader = load_imagenetC(cfg["TEST_DIR"], **loader_kwargs)
            probs_dict, labels = run_duo(duo, loader, wandb_run=wandb_run, wandb_prefix=prefix)
            metrics_by_model = {name: get_metrics_dict(p, labels) for name, p in probs_dict.items()}
            intersection_metrics = get_intersection_metrics(probs_dict, labels)

            wandb_log = {}
            for model_name, metrics in metrics_by_model.items():
                wandb_log.update({f"{prefix}{model_name}/{k}": v for k, v in metrics.items()})
            wandb_log.update({f"{prefix}intersection/{k}": v for k, v in intersection_metrics.items()})
            if duo.calibration_mode == "oracle_ts":
                wandb_log[f"{prefix}oracle/Tl"] = cal.Tl.item()
                wandb_log[f"{prefix}oracle/Ts"] = cal.Ts.item()
            wandb_run.log(wandb_log)

            logger.info(f"Results for {corruption_type} severity {severity}: {metrics_by_model['duo']}")
            row = {"mode": duo.mode, "corruption": corruption_type, "severity": severity}
            for model_name, metrics in metrics_by_model.items():
                row.update({f"{model_name}/{k}": v for k, v in metrics.items()})
            row.update({f"intersection/{k}": v for k, v in intersection_metrics.items()})
            results_rows.append(row)

    if results_rows:
        cols = list(results_rows[0].keys())
        numeric_cols = [c for c in cols if isinstance(results_rows[0][c], (int, float)) and c != "severity"]
        avg_row = {c: results_rows[0][c] for c in cols}
        avg_row["corruption"] = "average"
        avg_row["severity"] = 0
        for c in numeric_cols:
            avg_row[c] = sum(r[c] for r in results_rows) / len(results_rows)
        table = wandb.Table(columns=cols)
        for row in results_rows:
            table.add_data(*[row[c] for c in cols])
        table.add_data(*[avg_row[c] for c in cols])
        wandb_run.log({"summary/results": table})

    wandb_run.finish()

