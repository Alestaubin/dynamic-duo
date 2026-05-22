# from https://github.com/tochris/pts-uncertainty/blob/main/metrics.py

import numpy as np
import calibration as cal
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss, log_loss, roc_auc_score
import torch

def get_metrics_dict(probs, labels) -> dict:

    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    labels = np.array(labels).astype(int)
    
    probs = probs / probs.sum(axis=1, keepdims=True)

    num_classes = probs.shape[1]
    preds = np.argmax(probs, axis=1)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    top_k_idx = np.argsort(probs, axis=1)[:, ::-1]          # (N, C) descending
    acc3 = np.mean((top_k_idx[:, :min(3, num_classes)] == labels[:, None]).any(axis=1))
    acc5 = np.mean((top_k_idx[:, :min(5, num_classes)] == labels[:, None]).any(axis=1))
    
    y_true_one_hot = np.zeros_like(probs)
    y_true_one_hot[np.arange(len(labels)), labels] = 1
    brier = np.mean(np.sum((probs - y_true_one_hot)**2, axis=1))

    nll = log_loss(labels, probs, labels=list(range(num_classes)))
    
    ece = cal.get_ece(probs, labels, num_bins=15)

    return {
        'ece': ece,
        'nll': nll,
        'accuracy': acc,
        'acc3': acc3,
        'acc5': acc5,
        'f1': f1,
        'brier': brier
    }
