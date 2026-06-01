import torch
import torch.nn as nn
import torch.optim as optim
import logging
from copy import deepcopy
from torch.nn import functional as F
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
    print(cfg)
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

def setup_tent(model, norm_type, cfg):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = configure_model(model, norm_type)
    params, param_names = collect_params(model, norm_type)
    
    if not params:
        raise ValueError("No parameters found for adaptation. Check if model has Norm layers.")
    
    optimizer = setup_optimizer(params, cfg)

    # logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)

    tented_model = Tent(model, optimizer,
                        steps=cfg["STEPS"],
                        episodic=False)
    
    return tented_model

     
def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)