"""
logit.py
========
From accuracy to log-odds (eq. 8): places a predicted batch accuracy on the
additive log-odds scale, where equal steps correspond to equal multiplicative
changes in the odds of a correct prediction — the scale on which Section 4's
temporal filters and Section 5's gate operate (comparing two probabilities by
a difference of logits is the log odds ratio, the standard effect measure for
binary outcomes).
"""

from __future__ import annotations

import math


def to_logit(p: float, eps: float = 1e-3) -> float:
    """logit(clip(p, eps, 1 - eps)) = log(p / (1 - p))."""
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


if __name__ == "__main__":
    assert abs(to_logit(0.5)) < 1e-9
    assert to_logit(0.9) > 0.0
    assert to_logit(0.1) < 0.0
    assert abs(to_logit(0.9) - (-to_logit(0.1))) < 1e-9  # symmetry around 0.5
    # Clipping keeps the output finite at the extremes.
    assert math.isfinite(to_logit(0.0))
    assert math.isfinite(to_logit(1.0))
    # Monotone increasing.
    xs = [0.01, 0.2, 0.5, 0.8, 0.99]
    ys = [to_logit(x) for x in xs]
    assert all(ys[i] < ys[i + 1] for i in range(len(ys) - 1))
    print("to_logit self-test passed")
