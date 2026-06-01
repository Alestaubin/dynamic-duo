import math
import logging

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# from robustbench.data import load_imagenetc


from src.tta.dynamic_duo import setup_duo
from src.tta.tent import setup_tent
from src.utils.model import get_model, _preprocess_batch
from src.utils.data import load_imagenetC, load_config, load_imagenetc
from src.calibrators.fixed_TS import JointFixedTS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


# ----------------------------------------------------------------------------
# Core eval loop, shared by both data sources so the ONLY variable is the loader
# ----------------------------------------------------------------------------
def _run_eval(model, batches, device, desc=""):
    """Run TENT adaptation + scoring over an iterable of (x, y) batches.

    x is expected to already be a normalized/scaled tensor in the model's
    input space (i.e. [0,1] if the model normalizes internally). Adaptation
    happens inside model(x) because forward_and_adapt is @torch.enable_grad().
    """
    all_probs, all_labels = [], []
    correct, total = 0, 0
    for x, y in tqdm(batches, desc=desc):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)                       # <-- adapts here (grad forced on internally)
        preds = logits.argmax(1)
        correct += (preds == y).float().sum().item()
        total += y.numel()
        all_probs.append(F.softmax(logits.detach().cpu(), dim=1))
        all_labels.append(y.cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    acc = correct / total
    return acc, all_probs, all_labels


def _custom_batches(test_dir, severity, corruption_type, preprocess, device, bs, num_samples):
    """Yield (preprocessed_x, y) batches from your custom loader."""
    loader = load_imagenetC(
        test_dir, severity, [corruption_type],
        device=device, batch_size=bs, num_samples=num_samples, seed=0,
    )
    for imgs, labels in loader:
        x = _preprocess_batch(imgs, preprocess, device)
        yield x, labels


def _robustbench_batches(test_dir, severity, corruption_type, bs, num_samples, preprocess):
    """Yield (x, y) batches from robustbench's pre-loaded tensors."""
    x_test, y_test = load_imagenetc(
        n_examples=num_samples, severity=severity, data_dir=test_dir,
        shuffle=False, corruptions=[corruption_type], prepr=preprocess,
    )
    n_batches = math.ceil(x_test.shape[0] / bs)
    for i in range(n_batches):
        yield x_test[i * bs:(i + 1) * bs], y_test[i * bs:(i + 1) * bs]


# ----------------------------------------------------------------------------
# Unified TENT eval. source="custom" reproduces evaluate_2, source="robustbench"
# reproduces evaluate_4 -- but through the SAME model and SAME loop, so any
# remaining accuracy gap is attributable purely to the data source.
# ----------------------------------------------------------------------------
def evaluate_tent(corruptions, severities, bs, test_dir, source="robustbench", num_samples=5000):
    cfg = load_config("./cfgs/dynamic_duo_config.yaml")["SMALL"]["OPTIM"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, preprocess = get_model("resnet50")
    model = setup_tent(model, norm_type="BN", cfg=cfg).to(device).eval()

    logger.info(f"test-time adaptation: TENT | data source: {source}")

    for severity in severities:
        for corruption_type in corruptions:
            try:
                model.reset()
            except Exception:
                logger.warning("not resetting model")

            if source == "custom":
                batches = _custom_batches(test_dir, severity, corruption_type,
                                          preprocess, device, bs, num_samples)
            elif source == "robustbench":
                batches = _robustbench_batches(test_dir, severity, corruption_type,
                                               bs, num_samples, preprocess)
            else:
                raise ValueError(f"unknown source: {source}")

            acc, probs, labels = _run_eval(
                model, batches, device, desc=f"{corruption_type} s{severity}"
            )
            # accuracy_score on the collected probs is a cross-check of acc
            acc_check = accuracy_score(labels.numpy(), np.argmax(probs.numpy(), axis=1))
            err = 1. - acc
            logger.info(f"acc [{corruption_type}{severity}]: {acc:.2%} (check {acc_check:.2%})")
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")


# ----------------------------------------------------------------------------
# Diagnostic: feed the SAME model+TENT path robustbench tensors vs your loader,
# and directly compare labels for the first matched image. This isolates whether
# the gap is the data pipeline (images) or the labels.
# ----------------------------------------------------------------------------
def diagnose(corruption_type, severity, test_dir_custom, test_dir_rb, bs=128, num_samples=5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, preprocess = get_model("resnet50")

    # --- input range / stats comparison (already confirmed [0,1] for both) ---
    xb, yb_rb = load_imagenetc(n_examples=bs, severity=severity,
                               data_dir=test_dir_rb, shuffle=False,
                               corruptions=[corruption_type], prepr=preprocess)
    print("rb   ", xb.shape, xb.min().item(), xb.max().item(),
          xb.mean().item(), xb.std().item())

    loader = load_imagenetC(test_dir_custom, severity, [corruption_type],
                            device=device, batch_size=bs, seed=0,
                            num_samples=bs)
    imgs, yb_custom = next(iter(loader))
    xs = _preprocess_batch(imgs, preprocess, device)
    print("ours ", xs.shape, xs.min().item(), xs.max().item(),
          xs.mean().item(), xs.std().item())

    # --- LABEL comparison: the actual suspect now -------------------------
    yb_rb = yb_rb[:bs].cpu()
    yb_custom = yb_custom[:bs].cpu()
    n_match = (yb_rb == yb_custom).sum().item()
    print(f"label agreement on first {bs}: {n_match}/{bs}")
    print("rb labels   [:10]:", yb_rb[:128].tolist())
    print("ours labels [:10]:", yb_custom[:128].tolist())
    if n_match < bs:
        print(">>> LABELS DISAGREE. Your custom loader's class indices do not "
              "match torchvision's output space (sorted-WNID order). This is "
              "the accuracy gap, not the model or adaptation.")

    # --- same-model sanity: run TENT on robustbench tensors only ----------
    cfg = load_config("./cfgs/dynamic_duo_config.yaml")["SMALL"]["OPTIM"]
    model, _ = get_model("resnet50")
    model = setup_tent(model, norm_type="BN", cfg=cfg).to(device).eval()
    try:
        model.reset()
    except Exception:
        pass
    x_test, y_test = load_imagenetc(n_examples=num_samples, severity=severity,
                                    data_dir=test_dir_rb, shuffle=False,
                                    corruptions=[corruption_type], prepr=preprocess)
    acc, _, _ = _run_eval(
        model,
        ((x_test[i*bs:(i+1)*bs], y_test[i*bs:(i+1)*bs])
         for i in range(math.ceil(x_test.shape[0] / bs))),
        device, desc=f"rb-only {corruption_type} s{severity}",
    )
    print(f"robustbench-tensor TENT accuracy: {acc:.2%}")


# ----------------------------------------------------------------------------
# Duo eval (formerly evaluate_3), unchanged in behaviour, tidied.
# ----------------------------------------------------------------------------
def evaluate_duo(corruptions, severities, bs, test_dir, num_samples=5000):
    cfg = load_config("./cfgs/dynamic_duo_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    large_model, large_preprocess = get_model(cfg["LARGE"]["NAME"])
    small_model, small_preprocess = get_model(cfg["SMALL"]["NAME"])
    large_model, small_model = large_model.to(device), small_model.to(device)

    calibrator = JointFixedTS(cfg["CALIBRATOR"]["TL"], cfg["CALIBRATOR"]["TS"])
    duo = setup_duo(large_model, large_preprocess, small_model, small_preprocess,
                    calibrator, "both_indep", cfg=cfg, steps=1)

    for severity in severities:
        for corruption_type in corruptions:
            duo.reset()
            logger.info("resetting model")
            loader = load_imagenetC(
                test_dir, severity, [corruption_type],
                device=device, batch_size=bs, num_samples=num_samples,
                seed=0,
            )
            all_probs, all_labels = [], []
            for imgs, labels in tqdm(loader, desc=f"{corruption_type} s{severity}"):
                logits = duo(imgs, labels=labels)
                all_probs.append(F.softmax(logits.detach().cpu(), dim=1))
                all_labels.append(labels.cpu())
            acc = (torch.cat(all_probs).argmax(1) == torch.cat(all_labels)).float().mean().item()
            logger.info(f"error % [{corruption_type}{severity}]: {1. - acc:.2%}")


if __name__ == "__main__":
    corruptions = ["snow"]
    severities = [5]
    bs = 128

    # Run the diagnostic FIRST -- it tells you whether the gap is labels.
    diagnose("snow", 5, test_dir_custom="./data/ImageNet-C", test_dir_rb="./data/ImageNet-C", bs=bs, num_samples=10000)

    # Then the two matched runs (same model + loop, only the loader differs):
    print("\n=== robustbench source ===")
    evaluate_tent(corruptions, severities, bs, test_dir="./data/ImageNet-C", source="robustbench", num_samples=10000)

    print("\n=== custom source ===")
    evaluate_tent(corruptions, severities, bs, test_dir="./data/ImageNet-C", source="custom", num_samples=10000)