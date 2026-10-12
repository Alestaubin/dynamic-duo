"""
stream_cache.py
================
Shared disk cache for one data stream's (z_l, z_s, f_l, f_s, labels) — the
large/small models' logits AND penultimate features, plus labels — used by
both scripts/compare_calibrators.py and scripts/sweep_proxies.py so a repeat
run against the same duo/data skips the model forward pass entirely.

Before this module existed the two scripts had separate, inconsistent
caching:
  - compare_calibrators.py's _CachedModel cached logits ONLY, keyed by a
    user-supplied --logits_cache_dir, and explicitly skipped caching
    whenever proxy_kind="prototype" (a logits-only cache can't supply the
    penultimate features that proxy's live forward hook needs).
  - sweep_proxies.py had no caching at all.

This module caches logits AND features together (so no caller needs a
prototype-shaped special case) and derives the cache directory automatically
from the duo's model names — callers never pick a directory name, just
whether caching is on (see duo_cache_dir).

Two ways to consume the cache:
  - load_or_collect_stream(...): returns the whole-stream tensors directly.
    What sweep_proxies.py wants — it works entirely in memory over full
    (corruption, severity) streams anyway.
  - CachedModel + DuoStreamCache: an nn.Module replay wrapper pair for
    compare_calibrators.py, which drives batches through a live DynamicDuo/
    calibrator one batch at a time. DuoStreamCache.set_stream() eagerly
    collects (or loads) the WHOLE stream once per (corruption, severity) —
    same as load_or_collect_stream — then each CachedModel.forward() call
    just replays the next slice; no per-batch cache bookkeeping is needed
    because nothing is missing by the time batches start arriving.

Safety notes
------------
- Only valid for frozen models (mode=no_adapt): configure_model_frozen puts
  the model in train() with batch statistics (see src/tta/tent.py) so a
  batch's output depends only on that batch's own composition, never on call
  history — this is what makes whole-stream replay safe regardless of
  whether a batch was originally computed live or replayed from cache.
- load_imagenetC's DataLoader shuffle is seeded deterministically, so the
  SAME (corruption, severity, num_samples, seed) reproduces the exact same
  sample order every time — required for replaying a cached stream back
  against a separately-constructed live loader for the same parameters.
- The cache key bundles num_samples and seed. Omitting either was the exact
  bug that once desynced src.utils.logits.get_model_logits's large/small
  labels across differently-sampled runs (see CLAUDE.md) — this cache must
  not repeat it.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.reliability.proxies.stats import FeatureExtractor

CACHE_ROOT = Path("cache/stream_cache")


def duo_cache_dir(large_name: str, small_name: str, root: Path | str = CACHE_ROOT) -> Path:
    """Automatic, duo-specific cache directory — callers never choose a name."""
    return Path(root) / f"{large_name}+{small_name}"


def stream_key(tag: str, num_samples: int | None, seed: int | None) -> str:
    """Deterministic filename stem for one stream. tag identifies the stream
    itself (e.g. "fog_s5", or a combined-corruption calibration tag) —
    num_samples/seed are appended since they change WHICH samples are
    selected/ordered (see module docstring)."""
    parts = [tag]
    if num_samples is not None:
        parts.append(f"n{num_samples}")
    if seed is not None:
        parts.append(f"seed{seed}")
    return "_".join(parts)


@torch.no_grad()
def collect_stream(loader, preprocess_l, preprocess_s, ext_l: FeatureExtractor, ext_s: FeatureExtractor, device):
    """Run both models over one loader; return (z_l, z_s, f_l, f_s, labels),
    each concatenated across every batch. ext_l/ext_s must be FeatureExtractor
    instances wrapping the RAW (unwrapped) models, since this is exactly what
    populates the cache on a miss."""
    zl_all, zs_all, fl_all, fs_all, labels_all = [], [], [], [], []
    for imgs, labels in tqdm(loader, desc="collecting", leave=False):
        xl = torch.stack([preprocess_l(img) for img in imgs]).to(device)
        xs = torch.stack([preprocess_s(img) for img in imgs]).to(device)
        zl, fl = ext_l(xl)
        zs, fs = ext_s(xs)
        zl_all.append(zl.cpu()); zs_all.append(zs.cpu())
        fl_all.append(fl.cpu()); fs_all.append(fs.cpu())
        labels_all.append(labels)
    return (torch.cat(zl_all), torch.cat(zs_all), torch.cat(fl_all), torch.cat(fs_all), torch.cat(labels_all))


def load_or_collect_stream(
    cache_dir: Path | None,
    key: str,
    collect_fn,
    use_cache: bool = True,
    overwrite_cache: bool = False,
):
    """Return (z_l, z_s, f_l, f_s, labels) for one stream.

    collect_fn is a zero-arg thunk that actually runs the model forward pass
    (e.g. via collect_stream) — only called on a miss, or when caching is
    disabled/forced, so a hit never constructs/iterates a data loader at all.

    cache_dir=None or use_cache=False disables caching entirely (every call
    runs collect_fn). overwrite_cache=True always (re)computes and re-saves,
    even over an existing cache file — for deliberately refreshing a stale
    entry without deleting it by hand.
    """
    cache_file = (Path(cache_dir) / f"{key}.pt") if (cache_dir is not None and use_cache) else None

    if cache_file is not None and cache_file.exists() and not overwrite_cache:
        print(f"[stream cache] HIT  {cache_file}")
        data = torch.load(cache_file, map_location="cpu", weights_only=True)
        return data["z_l"], data["z_s"], data["f_l"], data["f_s"], data["labels"]

    if cache_file is not None:
        reason = "overwrite requested" if cache_file.exists() else "miss"
        print(f"[stream cache] {reason.upper()} {cache_file} — collecting...")
    z_l, z_s, f_l, f_s, labels = collect_fn()

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"z_l": z_l, "z_s": z_s, "f_l": f_l, "f_s": f_s, "labels": labels}, cache_file)
        print(f"[stream cache] saved {cache_file}")

    return z_l, z_s, f_l, f_s, labels


class _StreamState:
    """One stream's (z_l, z_s, f_l, f_s, labels) plus independent read
    cursors for the large/small sides (they're always advanced by the same
    amount per batch in lockstep, since both CachedModels are driven by the
    same live batch, but tracking them separately avoids any implicit
    call-order assumption between the two)."""

    __slots__ = ("z_l", "z_s", "f_l", "f_s", "labels", "pos_l", "pos_s")

    def __init__(self, z_l, z_s, f_l, f_s, labels):
        self.z_l, self.z_s, self.f_l, self.f_s, self.labels = z_l, z_s, f_l, f_s, labels
        self.pos_l = 0
        self.pos_s = 0


class CachedModel(nn.Module):
    """Wraps a FROZEN model (mode=no_adapt only) so forward() replays a
    pre-fetched whole-stream cache instead of recomputing. Always built in
    matched pairs by DuoStreamCache, which is what actually populates the
    shared _StreamState — do not construct directly.

    Registers its OWN FeatureExtractor on the raw model (independent of any
    proxy-specific hook a calibrator might register), so logits+features are
    always captured on a miss regardless of which proxy_kind (if any) this
    run actually uses — a cache built once is reusable by any later run.

    On a replay hit, the cached feature slice is pushed directly into this
    extractor AND into any "extra" extractors attached via
    attach_extra_feature_extractor (e.g. a JointProxyWeighted calibrator's
    own _ext_l/_ext_s, built by register_hooks for proxy_kind="prototype")
    — faking what their live forward hook would have captured, since no real
    forward happens on a hit for them to hook into.
    """

    def __init__(self, model: nn.Module, model_name: str, which: str):
        super().__init__()
        assert which in ("large", "small"), which
        self.model = model
        self.which = which
        self._own_ext = FeatureExtractor(model, model_name)
        self._extra_exts: list[FeatureExtractor] = []
        self._state: _StreamState | None = None

    def attach_extra_feature_extractor(self, ext) -> None:
        if ext is not None:
            self._extra_exts.append(ext)

    def set_state(self, state: _StreamState | None) -> None:
        self._state = state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._state is None:
            # No stream set (or caching not in play for this call) — just run
            # the real model; its own hook (and any extras') fires normally.
            return self.model(x)

        n = x.shape[0]
        pos = self._state.pos_l if self.which == "large" else self._state.pos_s
        z_all = self._state.z_l if self.which == "large" else self._state.z_s
        f_all = self._state.f_l if self.which == "large" else self._state.f_s

        z = z_all[pos:pos + n].to(x.device)
        f = f_all[pos:pos + n].to(x.device)
        for ext in (self._own_ext, *self._extra_exts):
            ext._feats = f

        if self.which == "large":
            self._state.pos_l += n
        else:
            self._state.pos_s += n
        return z

    def remove_hooks(self) -> None:
        self._own_ext.remove()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class DuoStreamCache:
    """Owns a matched CachedModel pair (.large, .small) that replay/record a
    SHARED per-stream cache together, so both models' logits and features
    stay batch-aligned. One DuoStreamCache per run: call set_stream() once
    per (corruption, severity) BEFORE that stream's batches start flowing
    (e.g. from an evaluate_dynamic_duo on_corruption_start callback), and
    finish() once at the end to remove the feature hooks.
    """

    def __init__(
        self,
        large_model: nn.Module, large_name: str, large_preprocess,
        small_model: nn.Module, small_name: str, small_preprocess,
        device: torch.device,
        cache_dir: Path | None,
        num_samples: int | None,
        seed: int | None,
        use_cache: bool = True,
        overwrite_cache: bool = False,
    ):
        self.large = CachedModel(large_model, large_name, "large")
        self.small = CachedModel(small_model, small_name, "small")
        self.large_preprocess = large_preprocess
        self.small_preprocess = small_preprocess
        self.device = device
        self.cache_dir = cache_dir
        self.num_samples = num_samples
        self.seed = seed
        self.use_cache = use_cache
        self.overwrite_cache = overwrite_cache

    def attach_extra_feature_extractors(self, ext_l, ext_s) -> None:
        self.large.attach_extra_feature_extractor(ext_l)
        self.small.attach_extra_feature_extractor(ext_s)

    def set_stream(self, tag: str, loader_factory) -> None:
        """tag identifies the stream (e.g. "fog_s5"). loader_factory() builds
        the DataLoader fresh — only called on a cache miss."""
        key = stream_key(tag, self.num_samples, self.seed)

        def _collect():
            loader = loader_factory()
            return collect_stream(
                loader, self.large_preprocess, self.small_preprocess,
                self.large._own_ext, self.small._own_ext, self.device,
            )

        z_l, z_s, f_l, f_s, labels = load_or_collect_stream(
            self.cache_dir, key, _collect,
            use_cache=self.use_cache, overwrite_cache=self.overwrite_cache,
        )
        state = _StreamState(z_l, z_s, f_l, f_s, labels)
        self.large.set_state(state)
        self.small.set_state(state)

    def finish(self) -> None:
        self.large.remove_hooks()
        self.small.remove_hooks()


if __name__ == "__main__":
    import tempfile

    torch.manual_seed(0)

    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)

        def forward(self, x):
            return self.fc(x)

    # --- load_or_collect_stream: hit/miss/overwrite/disabled semantics ---
    calls = {"n": 0}

    def _fake_collect():
        calls["n"] += 1
        return (torch.randn(4, 3), torch.randn(4, 3),
                torch.randn(4, 8), torch.randn(4, 8), torch.randint(0, 3, (4,)))

    with tempfile.TemporaryDirectory() as tmp:
        d = duo_cache_dir("large", "small", root=tmp)
        key = stream_key("fog_s5", num_samples=100, seed=0)

        z_l1, *_ = load_or_collect_stream(d, key, _fake_collect)
        assert calls["n"] == 1
        z_l2, *_ = load_or_collect_stream(d, key, _fake_collect)
        assert calls["n"] == 1, "cache hit must not call collect_fn again"
        assert torch.equal(z_l1, z_l2)

        load_or_collect_stream(d, key, _fake_collect, overwrite_cache=True)
        assert calls["n"] == 2, "overwrite_cache must always recompute"

        load_or_collect_stream(d, key, _fake_collect, use_cache=False)
        assert calls["n"] == 3, "use_cache=False must always recompute"

        load_or_collect_stream(None, key, _fake_collect)
        assert calls["n"] == 4, "cache_dir=None must always recompute"

    # --- CachedModel/DuoStreamCache: replay matches a real forward pass,
    # and features are correctly pushed into an "extra" extractor on a hit
    # (simulating a calibrator's own prototype hook). ---
    large = _TinyModel()
    small = _TinyModel()

    def _preprocess(img):
        return img

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = duo_cache_dir("large", "small", root=tmp)
        dsc = DuoStreamCache(
            large, "large", _preprocess, small, "small", _preprocess,
            device=torch.device("cpu"), cache_dir=cache_dir,
            num_samples=8, seed=0,
        )

        extra_ext_l = FeatureExtractor(large, "large")
        dsc.attach_extra_feature_extractors(extra_ext_l, None)

        imgs = [torch.randn(4) for _ in range(8)]
        labels = torch.randint(0, 3, (8,))

        def _loader_factory():
            xb = torch.stack(imgs)
            return [(list(xb.split(4)[0]), labels[:4]), (list(xb.split(4)[1]), labels[4:])]

        with torch.no_grad():
            expected_z = large(torch.stack(imgs))

        dsc.set_stream("fog_s5", _loader_factory)  # miss -> real forward, caches to disk
        replay_out = []
        for xb, _ in _loader_factory():
            x = torch.stack(xb)
            replay_out.append(dsc.large(x))
        replay_out = torch.cat(replay_out)
        assert torch.allclose(replay_out, expected_z, atol=1e-5), \
            "replay after a cache miss's own eager collection must match a real forward pass"
        assert extra_ext_l._feats is not None, \
            "an attached extra extractor must receive features on replay, not just the owned one"

        # Second DuoStreamCache instance, same key -> hit, no model call needed.
        large2 = _TinyModel()
        large2.load_state_dict(large.state_dict())
        small2 = _TinyModel()
        small2.load_state_dict(small.state_dict())
        dsc2 = DuoStreamCache(
            large2, "large", _preprocess, small2, "small", _preprocess,
            device=torch.device("cpu"), cache_dir=cache_dir,
            num_samples=8, seed=0,
        )
        dsc2.set_stream("fog_s5", _loader_factory)
        replay_out2 = []
        for xb, _ in _loader_factory():
            x = torch.stack(xb)
            replay_out2.append(dsc2.large(x))
        replay_out2 = torch.cat(replay_out2)
        assert torch.allclose(replay_out2, expected_z, atol=1e-5), \
            "a fresh DuoStreamCache hitting the same cache file must replay identical logits"

        dsc.finish()
        dsc2.finish()

    print("stream_cache self-test passed")
