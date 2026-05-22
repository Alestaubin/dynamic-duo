import torch
import torch.nn as nn
import torch.optim as optim
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

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
    if cfg["OPTIM"]["METHOD"] == 'Adam':
        return optim.Adam(params,
                    lr=float(cfg["OPTIM"]["LR"]),
                    betas=(float(cfg["OPTIM"]["BETA"]), 0.999),
                    weight_decay=float(cfg["OPTIM"]["WD"]))
    elif cfg["OPTIM"]["METHOD"] == 'SGD':
        return optim.SGD(params,
                   lr=cfg["OPTIM"]["LR"],
                   momentum=cfg["OPTIM"]["MOMENTUM"],
                   dampening=cfg["OPTIM"]["DAMPENING"],
                   weight_decay=cfg["OPTIM"]["WD"],
                   nesterov=cfg["OPTIM"]["NESTEROV"])
    else:
        raise NotImplementedError

def setup_tent(model, norm_type, cfg):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    tented_model = configure_model(model, norm_type)
    params, param_names = collect_params(tented_model, norm_type)
    
    if not params:
        raise ValueError("No parameters found for adaptation. Check if model has Norm layers.")
    
    optimizer = setup_optimizer(params, cfg)

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
