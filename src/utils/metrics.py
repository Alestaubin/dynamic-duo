# from https://github.com/tochris/pts-uncertainty/blob/main/metrics.py

import numpy as np
import calibration as cal #pip install uncertainty-calibration
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss, log_loss, roc_auc_score
import torch


def _compute_rc_metrics(probs: np.ndarray, labels: np.ndarray):
    """AURC, E-AURC, AUGRC for selective prediction via max-confidence abstention.

    All three are lower-is-better and lie in [0, 1].

    AURC  — area under the risk-coverage curve (risk = error rate of selected).
    E-AURC — AURC minus the oracle lower bound (excess above optimal).
    AUGRC — area under the *generalized* risk-coverage curve (risk normalized by
             total n, not by selected k), equivalent to a weighted AUC where
             high-confidence wrong predictions are penalized more.
    """
    n = len(labels)
    preds = np.argmax(probs, axis=1)
    mistakes = (preds != labels).astype(float)
    m = int(mistakes.sum())

    # sort by max-prob confidence descending
    confidence = probs.max(axis=1)
    order = np.argsort(confidence)[::-1]
    sm = mistakes[order]  # sorted mistakes

    # AURC: (1/n) * sum_k  [cumulative_errors_k / k]
    risks = np.cumsum(sm) / np.arange(1, n + 1)
    aurc = float(risks.mean())

    # ideal AURC: oracle places all m wrong samples last
    ideal_risks = np.concatenate([
        np.zeros(n - m),
        np.arange(1, m + 1) / np.arange(n - m + 1, n + 1),
    ]) if m > 0 else np.zeros(n)
    e_aurc = aurc - float(ideal_risks.mean())

    # AUGRC: (1/n^2) * sum_i  sorted_mistakes[i] * (n - i)
    weights = np.arange(n, 0, -1)
    augrc = float(np.dot(sm, weights) / (n ** 2))

    return aurc, e_aurc, augrc


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

    aurc, e_aurc, augrc = _compute_rc_metrics(probs, labels)
    auroc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')

    return {
        'ece': ece,
        'nll': nll,
        'accuracy': acc,
        'acc3': acc3,
        'acc5': acc5,
        'f1': f1,
        'brier': brier,
        'aurc': aurc,
        'e_aurc': e_aurc,
        'augrc': augrc,
        'auroc': auroc,
    }


def get_intersection_metrics(probs_dict: dict, labels) -> dict:
    """Compute intersection metrics between large, small, and duo predictions.

    Answers:
      - How often do large and small agree, and how accurate are they when they do?
      - When they disagree, who does the duo follow? How accurate is each choice?
      - When only one model is correct on a disagreement, does the duo pick the right one
        (rescue rate)? This is the key signal for calibration quality.
      - When both models agree and are wrong, can the duo still override them?

    Args:
        probs_dict: {"duo": (N,C) tensor, "large": (N,C) tensor, "small": (N,C) tensor}
        labels: (N,) ground-truth class indices

    Returns:
        Flat dict of scalar metrics.
    """
    preds = {}
    for k, p in probs_dict.items():
        if torch.is_tensor(p):
            p = p.detach().cpu().numpy()
        preds[k] = np.argmax(p, axis=1)

    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    labels = np.array(labels).astype(int)

    pred_l, pred_s, pred_d = preds["large"], preds["small"], preds["duo"]

    agree    = pred_l == pred_s   # (N,) bool — large and small top-1 match
    disagree = ~agree

    correct_l = pred_l == labels
    correct_s = pred_s == labels
    correct_d = pred_d == labels

    def _safe_mean(mask):
        return float(mask.mean()) if mask.any() else float("nan")

    # ── Agreement / disagreement accuracy ────────────────────────────────────
    agree_acc_large = _safe_mean(correct_l[agree])   # == agree_acc_small (same pred)
    agree_acc_duo   = _safe_mean(correct_d[agree])
    dis_acc_large   = _safe_mean(correct_l[disagree])
    dis_acc_small   = _safe_mean(correct_s[disagree])
    dis_acc_duo     = _safe_mean(correct_d[disagree])
    # Oracle: on disagreement pick whichever model is right (upper bound)
    dis_oracle_acc  = _safe_mean((correct_l | correct_s)[disagree])

    # ── Duo alignment ─────────────────────────────────────────────────────────
    follows_l = pred_d == pred_l
    follows_s = pred_d == pred_s
    follows_neither = ~follows_l & ~follows_s

    # Given agreement, does duo match the consensus?
    consensus_follow_rate = _safe_mean(follows_l[agree])
    # Overall rates (unconditional)
    follows_l_rate       = float(follows_l.mean())
    follows_s_rate       = float(follows_s.mean())
    follows_neither_rate = float(follows_neither.mean())

    # On disagreement: who does duo follow?
    dis_follows_l_rate       = _safe_mean(follows_l[disagree])
    dis_follows_s_rate       = _safe_mean(follows_s[disagree])
    dis_follows_neither_rate = _safe_mean(follows_neither[disagree])

    # ── Rescue rates (key calibration quality signal) ─────────────────────────
    # When large is the ONLY correct model on a disagreement, does duo follow large?
    only_l_correct = disagree & correct_l & ~correct_s
    only_s_correct = disagree & correct_s & ~correct_l
    rescue_by_large = _safe_mean(follows_l[only_l_correct])
    rescue_by_small = _safe_mean(follows_s[only_s_correct])

    # ── Consensus override ────────────────────────────────────────────────────
    # When large==small and they're WRONG, does duo still get it right?
    agree_wrong = agree & ~correct_l
    consensus_override_rate = _safe_mean(correct_d[agree_wrong])

    return {
        # ── Base rates ──────────────────────────────────────────
        "agree_rate":    float(agree.mean()),
        "disagree_rate": float(disagree.mean()),

        # ── Accuracy by slice ───────────────────────────────────
        "agree_acc_large":   agree_acc_large,   # == agree_acc_small (same prediction)
        "agree_acc_duo":     agree_acc_duo,
        "dis_acc_large":     dis_acc_large,
        "dis_acc_small":     dis_acc_small,
        "dis_acc_duo":       dis_acc_duo,
        "dis_oracle_acc":    dis_oracle_acc,    # upper bound on disagreement

        # ── Duo alignment (unconditional) ───────────────────────
        "follows_large_rate":   follows_l_rate,
        "follows_small_rate":   follows_s_rate,
        "follows_neither_rate": follows_neither_rate,

        # ── Given agreement, does duo agree with consensus? ─────
        "consensus_follow_rate": consensus_follow_rate,

        # ── On disagreement: who does duo follow? ───────────────
        "dis_follows_large_rate":   dis_follows_l_rate,
        "dis_follows_small_rate":   dis_follows_s_rate,
        "dis_follows_neither_rate": dis_follows_neither_rate,

        # ── Rescue rates: calibration quality signal ────────────
        # P(duo=large | disagree, only large correct)
        "rescue_by_large_rate": rescue_by_large,
        # P(duo=small | disagree, only small correct)
        "rescue_by_small_rate": rescue_by_small,

        # ── Consensus override ──────────────────────────────────
        # P(duo correct | large==small, both wrong)
        "consensus_override_rate": consensus_override_rate,
    }

