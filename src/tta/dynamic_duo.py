import torch
import torch.jit
import torch.nn as nn
import torch.optim as optim
from src.tta.tent_utils import configure_model, collect_params, setup_optimizer, copy_model_and_optimizer, load_model_and_optimizer
import logging


logger = logging.getLogger(__name__)

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

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


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
        large_optimizer: optim.Optimizer | None,
        small: nn.Module,
        small_optimizer: optim.Optimizer | None,
        joint_calibrator: nn.Module,
        mode: str = "both_duo",
        steps: int = 1,
    ):
        super().__init__()
        assert mode in _MODES, f"Invalid mode {mode}. Must be one of {_MODES}."
        self.large = large
        self.small = small
        self.joint_calibrator = joint_calibrator
        self.large_optimizer = large_optimizer
        self.small_optimizer = small_optimizer
        self.mode = mode
        self.steps = steps

        self.adapt_large, self.adapt_small, _ = _MODE_SPEC[mode]

        if self.adapt_large:
            assert large_optimizer is not None, f"{mode} needs a large optimizer"
            self.large_model_state, self.large_optimizer_state = \
                copy_model_and_optimizer(self.large, self.large_optimizer)

        if self.adapt_small:
            assert small_optimizer is not None, f"{mode} needs a small optimizer"
            self.small_model_state, self.small_optimizer_state = \
                copy_model_and_optimizer(self.small, self.small_optimizer)


    def forward(self, x):
        for _ in range(self.steps):
            outputs = forward_and_adapt(
                x,
                self.large, self.large_optimizer,
                self.small, self.small_optimizer,
                self.joint_calibrator,
                self.mode,
            )
        return outputs
    
    def reset(self):
        """Reset the model and optimizer states to before adaptation."""

        if self.adapt_large:
            load_model_and_optimizer(
                self.large, self.large_optimizer,
                self.large_model_state, self.large_optimizer_state
            )

        if self.adapt_small:
            load_model_and_optimizer(
                self.small, self.small_optimizer,
                self.small_model_state, self.small_optimizer_state
            )

@torch.enable_grad()
def forward_and_adapt(x, large, large_optimizer, small, small_optimizer,
                      joint_calibrator, mode):
    adapt_large, adapt_small, signal = _MODE_SPEC[mode]

    z_large = large(x)
    z_small = small(x)

    if signal == "duo":
        zl = z_large if adapt_large else z_large.detach()
        zs = z_small if adapt_small else z_small.detach()
        z_bar = joint_calibrator(zl, zs)
        loss = softmax_entropy(z_bar).mean(0)

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
            softmax_entropy(z_large).mean(0).backward()
            large_optimizer.step(); large_optimizer.zero_grad()
        if adapt_small:
            softmax_entropy(z_small).mean(0).backward()
            small_optimizer.step(); small_optimizer.zero_grad()

    with torch.no_grad():
        return joint_calibrator(large(x), small(x))
    
def setup_duo(large, small, joint_calibrator, mode, cfg, steps):
    """Configure a DynamicDuo for TENT adaptation.

    This is a helper function to set up the DynamicDuo with the appropriate
    optimizers and calibrator for the selected mode. It will tent the inner
    models, set up the optimizers, and then wrap them in a DynamicDuo.
    """
    assert mode in _MODES, f"Invalid mode {mode}. Must be one of {_MODES}."
    adapt_large, adapt_small, signal = _MODE_SPEC[mode]

    large_optimizer, small_optimizer = None, None

    if adapt_large:
        large = configure_model(large, cfg["LARGE"]["NORM"])
        large_optimizer = setup_optimizer(
            collect_params(large, cfg["LARGE"]["NORM"])[0], cfg)
    if adapt_small:
        logger.info(f"Configuring small model for TENT adaptation with norm type: {cfg['SMALL']['NORM']}")
        small = configure_model(small, cfg["SMALL"]["NORM"])
        small_optimizer = setup_optimizer(
            collect_params(small, cfg["SMALL"]["NORM"])[0], cfg)
    
    dynamic_duo = DynamicDuo(
        large=large,
        large_optimizer=large_optimizer,
        small=small,
        small_optimizer=small_optimizer,
        joint_calibrator=joint_calibrator,
        mode=mode,
        steps=steps
    )
    return dynamic_duo

def evaluate_dynamic_duo(cfg, duo):
    for severity in cfg["CORRUPTION"]["SEVERITY"]:
        for corruption_type in cfg["CORRUPTION"]["TYPE"]:
            # reset adaptation for each combination of corruption x severity
            # note: for evaluation protocol, but not necessarily needed
            try:
                duo.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
            x_test, y_test = load_imagenetC(cfg["CORRUPTION"]["NUM_EX"],
                                           severity, cfg["DATA_DIR"], False,
                                           [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            acc = accuracy(duo, x_test, y_test, cfg["TEST"]["BATCH_SIZE"])
            err = 1. - acc
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")


def tune_duo(duo, data_loader):
    pass

