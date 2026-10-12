"""The experiment engine shared by every script in scripts/.

  specs      what a run is (RunSpec) and how the gate is configured (GateSpec)
  protocols  the streams a run visits: episodic, natural, continual, ccc (Segment lists)
  members    MemberRunner: the two adapting models, one batch at a time: each on its own entropy (*_indep modes) or
             on the entropy of a static duo (*_duo modes: joint, anchors)
  cache      per-segment member-logit cache (centered fp16), the thing gate variants are replayed on
  gates      GateSpec -> calibrator (fixed TS, proxy-weighted gate, a single member)
  evaluate   streaming metrics + OnlineEval (many calibrators on the same member logits)
  runner     ensure_members / evaluate: compute-or-load, then replay
  results    tidy CSV rows and mean +- std over seeds
  cli        the flags every script takes
  adaptation_modes  scripts/03: the six ways of adapting a duo, the shared temperatures, the two tables
  tuning     scripts/02: one-model Tent tuning on the held-out corruptions, written to cfgs/tuned/

Why this shape: the paper adapts the two members independently, so the gate never feeds back into
adaptation. One live run per (TTA method, duo, stream, seed) therefore yields both members' logits, and
every gate variant (beta, b_t, proxy, pooling, filter, oracle) is a cheap replay of them. The duo-entropy
modes put a STATIC calibrator (fixed temperatures) in the loop; its temperatures are then part of the run
(RunSpec.loop_ts), and replaying the same static calibrator still reproduces the run's predictions.
"""
