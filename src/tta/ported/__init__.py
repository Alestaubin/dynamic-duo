"""
Test-time adaptation algorithms ported from their official code, one MEMBER (one model) at a time (Phase 3, Stage D).

Unlike src/tta/methods/ (a loss and an update step driven by DynamicDuo or MemberRunner), these own the whole step, which
SAR needs (two forward and backward passes) and the later algorithms (CoTTA, ROID, DeYO) will too. Each member offers

    reset()                      the source weights back, and every piece of algorithm state cleared (episodic)
    step(x, class_mask=None)     one batch of [0, 1] pixels -> the SCORED logits (the first forward pass, before the update),
                                 detached; `class_mask` restricts the loss to the dataset's classes (ImageNet-A/R)

The model is expected to normalize inside itself (a `Normalize` layer in front, src/experiments/algorithms.py builds that),
which is the convention of the upstream benchmark. Every file names its upstream source and commit and keeps its license.
"""
