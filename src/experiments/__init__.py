"""The experiment engine shared by every script in scripts/.

  specs      what a run is (RunSpec) and how the gate is configured (GateSpec)
  protocols  the streams a run visits: episodic, natural, continual, ccc (Segment lists)
  members    MemberRunner: the two adapting models, independent adaptation, one batch at a time
  cache      per-segment member-logit cache (centered fp16), the thing gate variants are replayed on
  gates      GateSpec -> calibrator (fixed TS, proxy-weighted gate, a single member)
  evaluate   streaming metrics + OnlineEval (many calibrators on the same member logits)
  runner     ensure_members / evaluate: compute-or-load, then replay
  results    tidy CSV rows and mean +- std over seeds
  cli        the flags every script takes

Why this shape: the paper adapts the two members independently, so the gate never feeds back into
adaptation. One live run per (TTA method, duo, stream, seed) therefore yields both members' logits, and
every gate variant (beta, b_t, proxy, pooling, filter, oracle) is a cheap replay of them.
"""
