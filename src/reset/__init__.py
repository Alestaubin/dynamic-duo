"""Reset rules for Dynamic Duos (plan.md, Phase 4; docs/design_resets.md; the tuning protocol is docs/reset_tuning.md).

  detectors     the fast-vs-slow drop detector of `own_fast_vs_slow`: update(x) -> bool, restart(), state
  controller    one rule per member on its own proxy stream: warm-up, cooldown, the reset log
  member_rules  the per-member settings (averaging speeds, warm-up, threshold) of that rule and the condition names that carry them
  actions       reset ONE member to its source state (parameters and optimizer) without touching the other
  live          the stream loop that runs Vanilla DD or Joint DD without episodic resets and lets a controller reset members
  live_runs     the triggers of a live run (rule, fixed schedule, random control, boundary), their names, and a loop that stores per-sample correctness
  live_report   file names, tables and figures of the stored live runs
  screening     the offline screen: a detector setting replayed over stored proxy sequences (delay, misses, false alarms)
  tuning        the selection rules of the held-out tuning, as code (rates and detector settings)
  gate_variants other gates scored on the same members' logits (beta, routing after a reset)
  layout        where every file of the tuning lives, for any duo (no torch)
  jobs          the units of work (a job = one run, a JSON spec) and their idempotent submission to Slurm (no torch)
  workers       what a Slurm job runs: the temperatures, a continual-rate rung, a calibration stream, a live reset run
  screens       the two offline screens of the detector on a duo's calibration streams (the shared setting; each member's candidates)
  pipeline      the whole tuning of a duo as one resumable procedure: `scripts/tune_reset.py --duo <duo>`

Each module has a `__main__` self-test: `python -m src.reset.detectors`, `... controller`, `... actions`, `... live`, ... Rules other than
`own_fast_vs_slow` were screened and dropped; their code is in git at the tag `pre-reset-cleanup`.
"""
