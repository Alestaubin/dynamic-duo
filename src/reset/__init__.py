"""Reset rules for Dynamic Duos (plan.md, Phase 4; docs/design_resets.md).

  detectors   streaming one-sided drop detectors (level, imm_vs_ema, fast_vs_slow, slope, cusum): update(x) -> bool, restart(), state
  controller  one rule per member on the proxy stream: own / gap quantities, cooldown, the partner-gap restart, the reset log
  actions     reset ONE member to its source state (parameters and optimizer) without touching the other
  live        the stream loop that runs Vanilla DD or Joint DD without episodic resets and lets a controller reset members

Each module has a `__main__` self-test: `python -m src.reset.detectors`, `... controller`, `... actions`, `... live`.
"""
