from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import torch.optim as optim
import logging

from tqdm import tqdm
from src.utils.model import get_model, _preprocess_batch
from src.utils.data import load_imagenetC, load_config
from src.tta.tent import setup_tent
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


OPTIM = "Adam"
LR = 0.0001
BETA = 0.9
WD = 0.0
MOMENTUM = 0.9
DAMPENING = 0.0
NESTEROV = False

def evaluate():
    # configure model
    model_name = "resnet50"
    base_model, preprocess = get_model(model_name)
    logger.info("test-time adaptation: TENT")
    model = setup_tent(base_model).cuda()
    severities = [5]
    corruptions = ['contrast']
    test_dir = "/home/alxstaub/projects/aip-evanesce/alxstaub/dynamic-duos/data"
    bs = 128
    from robustbench.data import load_imagenetc
    from robustbench.utils import clean_accuracy as accuracy

    # evaluate on each severity and type of corruption in turn
    for severity in severities:
        for corruption_type in corruptions:
            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
            x_test, y_test = load_imagenetc(n_examples = 5000,
                                           severity=severity, data_dir=test_dir, shuffle=False, corruptions=[corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            acc = accuracy(model, x_test, y_test, bs)
            err = 1. - acc
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
            
def evaluate_2():
    cfg = load_config("./cfgs/dynamic_duo_config.yaml")["SMALL"]["OPTIM"]
    print(cfg)
    model_name = "resnet50"
    base_model, preprocess = get_model(model_name)
    logger.info("test-time adaptation: TENT")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_tent(base_model, norm_type="BN", cfg=cfg).to(device)
    severities = [5]
    corruptions = ['contrast']
    test_dir = "./data/ImageNet-C"
    bs = 128

    for severity in severities:
        for corruption_type in corruptions:
            model.reset()
            logger.info("resetting model")
            loader = load_imagenetC(
                test_dir, severity, [corruption_type],
                device=device, batch_size=bs, num_samples=5000, shuffle=False, seed=None
            )
            all_probs, all_labels = [], []
            for imgs, labels in tqdm(loader, desc=f"{corruption_type} s{severity}"):
                x = _preprocess_batch(imgs, preprocess, device)
                logits = model(x)
                all_probs.append(F.softmax(logits.detach().cpu(), dim=1))
                all_labels.append(labels.cpu())
            acc = (torch.cat(all_probs).argmax(1) == torch.cat(all_labels)).float().mean().item()
            err = 1. - acc
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")


def evaluate_3():
    from src.tta.tent import configure_model, collect_params, setup_optimizer
    from src.tta.dynamic_duo import DynamicDuo
    from src.calibrators.fixed_TS import JointFixedTS

    cfg = load_config("./cfgs/dynamic_duo_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    large_model, large_preprocess = get_model(cfg["LARGE"]["NAME"])
    small_model, small_preprocess = get_model(cfg["SMALL"]["NAME"])
    large_model, small_model = large_model.to(device), small_model.to(device)

    large_model = configure_model(large_model, cfg["LARGE"]["NORM"])
    large_params, _ = collect_params(large_model, cfg["LARGE"]["NORM"])
    large_optimizer = setup_optimizer(large_params, cfg["LARGE"]["OPTIM"])

    small_model = configure_model(small_model, cfg["SMALL"]["NORM"])
    small_params, _ = collect_params(small_model, cfg["SMALL"]["NORM"])
    small_optimizer = setup_optimizer(small_params, cfg["SMALL"]["OPTIM"])

    calibrator = JointFixedTS(cfg["CALIBRATOR"]["TL"], cfg["CALIBRATOR"]["TS"])

    duo = DynamicDuo(
        large=large_model, large_preprocess=large_preprocess, large_optimizer=large_optimizer,
        small=small_model, small_preprocess=small_preprocess, small_optimizer=small_optimizer,
        joint_calibrator=calibrator,
        mode="both_duo",
        steps=1,
    )

    severities = [5]
    corruptions = ['contrast']
    test_dir = "./data/ImageNet-C"
    bs = 128

    for severity in severities:
        for corruption_type in corruptions:
            duo.reset()
            logger.info("resetting model")
            loader = load_imagenetC(
                test_dir, severity, [corruption_type],
                device=device, batch_size=bs, num_samples=5000, shuffle=False, seed=None
            )
            all_probs, all_labels = [], []
            for imgs, labels in tqdm(loader, desc=f"{corruption_type} s{severity}"):
                logits = duo(imgs, labels=labels)
                all_probs.append(F.softmax(logits.detach().cpu(), dim=1))
                all_labels.append(labels.cpu())
            acc = (torch.cat(all_probs).argmax(1) == torch.cat(all_labels)).float().mean().item()
            err = 1. - acc
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")


if __name__ == '__main__':
    evaluate_3()