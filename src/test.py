
import math

from src.tta.dynamic_duo import setup_duo
import torch
import torch.nn.functional as F
import logging
import numpy as np
from sklearn.metrics import accuracy_score

from tqdm import tqdm
from src.utils.model import get_model, _preprocess_batch
from src.utils.data import load_imagenetC, load_config
from src.tta.tent import setup_tent
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
from robustbench.data import load_imagenetc


OPTIM = "Adam"
LR = 0.0001
BETA = 0.9
WD = 0.0
MOMENTUM = 0.9
DAMPENING = 0.0
NESTEROV = False

def evaluate_1(corruptions, severities, bs, test_dir):
    cfg = load_config("./cfgs/dynamic_duo_config.yaml")["SMALL"]["OPTIM"]
    print(cfg)
    model_name = "resnet50"
    model, preprocess = get_model(model_name)
    print(preprocess)
    logger.info("test-time adaptation: TENT")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_tent(model, norm_type="BN", cfg=cfg)
    model.to(device).eval()
    test_dir = "./data/ImageNet-C"
    bs = 128
    acc = 0.
    for severity in severities:
        for corruption_type in corruptions:
            try:
                model.reset()
            except:
                logger.warning("not resetting model")
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
                acc += (logits.cpu().max(1)[1] == labels.cpu()).float().sum()
            acc_1 = acc.item() / len(loader.dataset)
            print("acc_1: %.2f%%" % (acc_1 * 100))
            all_probs = torch.cat(all_probs)
            all_labels = torch.cat(all_labels)
            logger.info(f"acc_2: {accuracy_score(all_labels, np.argmax(all_probs, axis=1))}")
            acc = (all_probs.argmax(1) == all_labels).float().mean().item()
            err = 1. - acc
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")


def evaluate_2(corruptions, severities, bs, test_dir):
    cfg = load_config("./cfgs/dynamic_duo_config.yaml")["SMALL"]["OPTIM"]
    # configure model
    model_name = "resnet50"
    base_model, _ = get_model(model_name)
    logger.info("test-time adaptation: TENT")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_tent(base_model, norm_type="BN", cfg=cfg).cuda()
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
            acc = 0.
            n_batches = math.ceil(x_test.shape[0] / bs)
            with torch.no_grad():
                for counter in range(n_batches):
                    x_curr = x_test[counter * bs:(counter + 1) * bs].to(device)
                    y_curr = y_test[counter * bs:(counter + 1) * bs].to(device)

                    output = model(x_curr)
                    acc += (output.max(1)[1] == y_curr).float().sum()
            acc = acc.item() / x_test.shape[0]
            err = 1. - acc
            logger.info(f"acc: {acc:.2%}")
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")


if __name__ == '__main__':
    
    corruptions = ['snow']
    severities = [5]
    test_dir = "./data"#/ImageNet-C"
    bs = 128
    model_name = "resnet50"
    model, preprocess = get_model(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xb, _ = load_imagenetc(n_examples=128, severity=3,
                       data_dir=test_dir, corruptions=["gaussian_noise"])
    print("rb   ", xb.shape, xb.min().item(), xb.max().item(),
        xb.mean().item(), xb.std().item())

    # your side
    loader = load_imagenetC("./data/ImageNet-C", 3, ["gaussian_noise"],
                            device=device, batch_size=128,
                            num_samples=128, shuffle=False, seed=None)
    imgs, _ = next(iter(loader))
    xs = _preprocess_batch(imgs, preprocess, device)
    print("ours ", xs.shape, xs.min().item(), xs.max().item(),
        xs.mean().item(), xs.std().item())

    evaluate_2(corruptions, severities, bs, test_dir)
    evaluate_1(corruptions, severities, bs, test_dir)

# def evaluate_3(corruptions, severities, bs, test_dir):
#     from src.calibrators.fixed_TS import JointFixedTS

#     cfg = load_config("./cfgs/dynamic_duo_config.yaml")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     large_model, large_preprocess = get_model(cfg["LARGE"]["NAME"])
#     small_model, small_preprocess = get_model(cfg["SMALL"]["NAME"])
#     large_model, small_model = large_model.to(device), small_model.to(device)

#     calibrator = JointFixedTS(cfg["CALIBRATOR"]["TL"], cfg["CALIBRATOR"]["TS"])

#     duo = setup_duo(large_model, large_preprocess, small_model, small_preprocess, calibrator, "both_indep", cfg=cfg, steps=1)

#     for severity in severities:
#         for corruption_type in corruptions:
#             duo.reset()
#             logger.info("resetting model")
#             loader = load_imagenetC(
#                 test_dir, severity, [corruption_type],
#                 device=device, batch_size=bs, num_samples=5000, shuffle=False, seed=None
#             )
#             all_probs, all_labels = [], []
#             for imgs, labels in tqdm(loader, desc=f"{corruption_type} s{severity}"):
#                 logits = duo(imgs, labels=labels)
#                 all_probs.append(F.softmax(logits.detach().cpu(), dim=1))
#                 all_labels.append(labels.cpu())
#             acc = (torch.cat(all_probs).argmax(1) == torch.cat(all_labels)).float().mean().item()
#             err = 1. - acc
#             logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
