"""
calibrator_setup.py
===================
Factory helpers for proxy-based calibrators (proxy_anchor_coca, soft_anchor,
proxy_weighted).

Keeps all proxy-stats building, calibration-map fitting/loading, and calibrator
construction out of the top-level run script.

Public API
----------
build_proxy_calibrator(calibration_mode, proxy_kind, ...) -> BaseJointCalibrator
build_proxy_weighted_calibrator(...) -> JointProxyWeighted
fit_beta(...) -> float
save_weight_cfg / load_weight_cfg
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.proxies.proxies import build_proxy_stats, ProxyStats
from src.proxies.calibration import load_calibration_maps

# Dedicated, sidecar-only directory + distinctive suffix, mirroring the
# proxy-stats / calib-map folders: every file here is a fitted weight config
# (beta + pooling/calibration/filter choices) for one (proxy_kind, models) pair.
DEFAULT_WEIGHT_CFG_DIR = Path("data/proxy_weight_cfg")
WEIGHT_CFG_SUFFIX = ".weightcfg.json"

_BETA_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def _build_proxy_stats(
    proxy_kind: str,
    config: dict,
    large_model, large_preprocess,
    small_model, small_preprocess,
    device: torch.device,
    cache_path: str | None,
    proto_metric: str = "cosine",
) -> tuple[ProxyStats, ProxyStats]:
    """Return (cfg_l, cfg_s). atc/prototype run a source pass; nuclear_norm is free.

    proto_metric ("cosine" | "mahalanobis") selects the prototype-proxy distance.
    """
    if proxy_kind in {"atc", "prototype"}:
        from torch.utils.data import DataLoader
        from torchvision import datasets
        from src.utils.data import _pil_collate_fn
        print(f"Building proxy stats (proxy={proxy_kind}, metric={proto_metric}) "
              f"from source data...")
        src_ds = datasets.ImageFolder(config["VAL_DIR"])
        src_loader = DataLoader(
            src_ds, batch_size=config["BS"], shuffle=False,
            num_workers=config["WORKERS"], pin_memory=(device.type == "cuda"),
            collate_fn=_pil_collate_fn,
        )
        return build_proxy_stats(
            large_model, large_preprocess, config["LARGE"]["NAME"],
            small_model, small_preprocess, config["SMALL"]["NAME"],
            src_loader, device,
            proto_metric=proto_metric,
            cache_path=cache_path,
        )
    return (
        ProxyStats(name=config["LARGE"]["NAME"], num_classes=1000),
        ProxyStats(name=config["SMALL"]["NAME"], num_classes=1000),
    )


def _fit_and_save_calibration_maps(
    name: str,
    cfg_l: ProxyStats,
    cfg_s: ProxyStats,
    large_model, large_preprocess,
    small_model, small_preprocess,
    config: dict,
    device: torch.device,
    proxy_name: str,
    num_samples: int | None = None,
    seed: int | None = None,
):
    """Collect records over CALIBRATOR corruptions in one combined loader, fit, save."""
    from tqdm import tqdm
    from src.utils.data import load_imagenetC
    from src.proxies.proxies import FeatureExtractor
    from src.proxies.calibration import make_record, fit_calibration_maps, save_calibration_maps

    corruptions = config["CALIBRATOR"]["CORRUPTIONS"]
    severities  = config["CALIBRATOR"]["SEVERITIES"]
    print(f"[calib map] '{name}' not found — fitting now over "
          f"{len(corruptions)} corruptions × {len(severities)} severities ...")

    loader = load_imagenetC(
        config["TEST_DIR"],
        severities=severities,
        corruption_types=corruptions,
        device=device,
        batch_size=config["BS"],
        num_workers=config["WORKERS"],
        num_samples=num_samples,
        seed=seed,
    )

    ext_l = FeatureExtractor(large_model, cfg_l.name)
    ext_s = FeatureExtractor(small_model, cfg_s.name)
    records = []
    try:
        for imgs, labels in tqdm(loader, desc="collecting calibration records"):
            xl = torch.stack([large_preprocess(img) for img in imgs]).to(device)
            xs = torch.stack([small_preprocess(img) for img in imgs]).to(device)
            zl, fl = ext_l(xl)
            zs, fs = ext_s(xs)
            records.append(make_record(
                cfg_l, cfg_s,
                zl.cpu(), zs.cpu(), fl.cpu(), fs.cpu(),
                labels,
                corruption="mixed", severity=0,
            ))
    finally:
        ext_l.remove()
        ext_s.remove()

    maps = fit_calibration_maps(records, proxy_name, cfg_l.name, cfg_s.name)
    save_calibration_maps(maps, name)
    print(f"[calib map] Fitted on {len(records)} batches, proxy={proxy_name}")
    return maps


def build_proxy_calibrator(
    calibration_mode: str,
    proxy_kind: str,
    proxy_cache: str | None,
    calib_map: str | None,
    calibrated_selection: bool,
    csv_path: str | None,
    config: dict,
    large_model, large_preprocess,
    small_model, small_preprocess,
    device: torch.device,
    num_samples: int | None = None,
    seed: int | None = None,
    proto_metric: str = "cosine",
):
    """Build a proxy-based calibrator (proxy_anchor_coca or soft_anchor).

    Handles proxy-stats building, calibration-map loading/fitting, and
    calibrated-selection validation before constructing the calibrator.

    Raises ValueError for invalid argument combinations.
    """
    cfg_l, cfg_s = _build_proxy_stats(
        proxy_kind, config,
        large_model, large_preprocess, small_model, small_preprocess,
        device, cache_path=proxy_cache, proto_metric=proto_metric,
    )

    if calib_map is not None:
        try:
            maps = load_calibration_maps(calib_map)
        except FileNotFoundError:
            maps = _fit_and_save_calibration_maps(
                calib_map, cfg_l, cfg_s,
                large_model, large_preprocess,
                small_model, small_preprocess,
                config, device,
                proxy_name=proxy_kind,
                num_samples=num_samples,
                seed=seed,
            )
        maps.attach(cfg_l, cfg_s)
        print(f"Attached calibration map '{calib_map}' (proxy={maps.proxy_name})")

    if calibrated_selection:
        if calib_map is None:
            raise ValueError("--calibrated_selection requires --calib_map")
        if proxy_kind not in cfg_l.calib or proxy_kind not in cfg_s.calib:
            raise ValueError(
                f"--calibrated_selection set but calib map has no '{proxy_kind}' "
                f"entry for both models "
                f"(map proxies: {sorted(set(cfg_l.calib) | set(cfg_s.calib))})"
            )

    if calibration_mode == "proxy_anchor_coca":
        from src.calibrators.joint_proxy_anchor_coca import JointProxyAnchorCoca
        return JointProxyAnchorCoca(
            proxy_kind=proxy_kind,
            cfg_l=cfg_l,
            cfg_s=cfg_s,
            csv_path=csv_path,
            calibrated_selection=calibrated_selection,
        )
    else:  # soft_anchor
        raise NotImplementedError(
            "JointSoftAnchor is not currently supported. Use proxy_anchor_coca."
        )


# ─── Weight-config sidecar persistence (beta + pool/calib/filter choices) ────

def _resolve_weight_cfg_path(name_or_path: str | Path, directory: str | Path) -> Path:
    p = Path(name_or_path)
    if p.name.endswith(WEIGHT_CFG_SUFFIX):
        return p
    return Path(directory) / f"{p.name}{WEIGHT_CFG_SUFFIX}"


def save_weight_cfg(
    cfg: dict,
    name: str | Path,
    directory: str | Path = DEFAULT_WEIGHT_CFG_DIR,
) -> Path:
    """Save a fitted weight config (beta, pool, calib_mode, filter_kwargs, ...)
    as a small JSON sidecar, mirroring the calib-map persistence pattern."""
    path = _resolve_weight_cfg_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[weight cfg] saved → {path}  (beta={cfg.get('beta')})")
    return path


def load_weight_cfg(
    name: str | Path,
    directory: str | Path = DEFAULT_WEIGHT_CFG_DIR,
) -> dict:
    path = _resolve_weight_cfg_path(name, directory)
    with path.open() as f:
        cfg = json.load(f)
    print(f"[weight cfg] loaded ← {path}  (beta={cfg.get('beta')})")
    return cfg


def list_weight_cfgs(directory: str | Path = DEFAULT_WEIGHT_CFG_DIR) -> list[str]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        p.name[: -len(WEIGHT_CFG_SUFFIX)]
        for p in directory.glob(f"*{WEIGHT_CFG_SUFFIX}")
    )


# ─── Beta fitting on dev-shift (CALIBRATOR) corruptions ──────────────────────

@torch.no_grad()
def _collect_beta_records(probe, large_model, large_preprocess,
                           small_model, small_preprocess, loader, device):
    """One (z_l, z_s, labels, s_l, s_s) record per dev batch. Runs `probe`'s
    raw-score + calibrate/filter pipeline (stateful, in stream order) so the
    filter's denoising sees the same batch sequence a real run would."""
    from src.utils.model import _preprocess_batch

    records = []
    for imgs, labels in loader:
        x_l = _preprocess_batch(imgs, large_preprocess, device)
        x_s = _preprocess_batch(imgs, small_preprocess, device)
        z_l = large_model(x_l)
        z_s = small_model(x_s)
        r_l, r_s = probe._raw_scores(z_l, z_s)
        s_l, s_s = probe._calibrated_scores(r_l, r_s)
        records.append((z_l.cpu(), z_s.cpu(), labels.cpu(), s_l, s_s))
    return records


def _grid_search_beta(probe, records, beta_grid=_BETA_GRID) -> tuple[float, float, dict]:
    """Pick the beta in `beta_grid` minimizing mean duo NLL over `records`.

    Cheap: beta only affects _weights (the gate), not the raw scores or the
    filter state, so the stateful pipeline runs once (_collect_beta_records)
    and each candidate beta only re-pools already-computed logits.
    """
    nll_by_beta = {}
    best_beta, best_nll = beta_grid[0], float("inf")
    for beta in beta_grid:
        total_nll, total_n = 0.0, 0
        for z_l, z_s, labels, s_l, s_s in records:
            w_l, w_s = probe._weights(s_l, s_s, beta=beta)
            z_duo = probe._combine(z_l, z_s, w_l, w_s)
            total_nll += F.cross_entropy(z_duo, labels, reduction="sum").item()
            total_n += labels.shape[0]
        mean_nll = total_nll / max(total_n, 1)
        nll_by_beta[beta] = mean_nll
        print(f"[fit_beta] beta={beta:<5} dev NLL={mean_nll:.4f}")
        if mean_nll < best_nll:
            best_nll, best_beta = mean_nll, beta
    return best_beta, best_nll, nll_by_beta


def fit_beta(
    proxy_kind: str,
    cfg_l: ProxyStats,
    cfg_s: ProxyStats,
    large_model, large_preprocess,
    small_model, small_preprocess,
    config: dict,
    device: torch.device,
    pool: str = "linear",
    calib_mode: str = "zscore",
    filter: str = "kalman",
    filter_kwargs: dict | None = None,
    base_ts=None,
    num_samples: int | None = None,
    seed: int | None = None,
    beta_grid=_BETA_GRID,
    loader=None,
):
    """Grid-search beta minimizing mean duo NLL over CALIBRATOR (dev-shift)
    corruptions/severities — the same principle used to fit the base
    temperatures, applied to beta instead, and on dev-shift data (not clean
    val) because both models look reliable on clean data, making beta
    unidentifiable there.

    `loader` lets callers (tests) inject a synthetic dev stream; by default
    one is built from config["TEST_DIR"] / CALIBRATOR corruptions+severities.
    """
    from src.calibrators.joint_proxy_weighted import JointProxyWeighted

    probe = JointProxyWeighted(
        proxy_kind, cfg_l, cfg_s, beta=1.0, pool=pool, calib_mode=calib_mode,
        filter=filter, filter_kwargs=filter_kwargs, base_ts=base_ts, log_every=0,
    )
    if proxy_kind == "prototype":
        probe.register_hooks(large_model, small_model)

    if loader is None:
        from src.utils.data import load_imagenetC
        loader = load_imagenetC(
            config["TEST_DIR"],
            severities=config["CALIBRATOR"]["SEVERITIES"],
            corruption_types=config["CALIBRATOR"]["CORRUPTIONS"],
            device=device, batch_size=config["BS"], num_workers=config["WORKERS"],
            num_samples=num_samples, seed=seed,
        )

    try:
        records = _collect_beta_records(
            probe, large_model, large_preprocess, small_model, small_preprocess,
            loader, device,
        )
    finally:
        if proxy_kind == "prototype":
            probe.remove_hooks()

    best_beta, best_nll, _ = _grid_search_beta(probe, records, beta_grid)
    print(f"[fit_beta] selected beta={best_beta} (dev NLL={best_nll:.4f}, "
          f"n_records={len(records)})")
    return best_beta


# ─── build_proxy_weighted_calibrator (Task 3.1) ───────────────────────────────

@torch.no_grad()
def _val_pass_for_reliability(
    cfg_l: ProxyStats, cfg_s: ProxyStats,
    large_model, large_preprocess, small_model, small_preprocess,
    config: dict, device: torch.device, proxy_kind: str,
) -> None:
    """One pass over VAL_DIR filling cfg_l/cfg_s .val_acc/.proxy_mean/.proxy_std
    for `proxy_kind` (proxies.fit_val_reliability) — independent of whatever
    source pass `_build_proxy_stats` may already have run for atc/prototype
    threshold/prototype fitting, since those don't retain the raw val
    logits/features needed here."""
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from src.proxies.proxies import FeatureExtractor, fit_val_reliability
    from src.utils.data import _pil_collate_fn
    from src.utils.model import _preprocess_batch

    src_ds = datasets.ImageFolder(config["VAL_DIR"])
    src_loader = DataLoader(
        src_ds, batch_size=config["BS"], shuffle=False,
        num_workers=config["WORKERS"], pin_memory=(device.type == "cuda"),
        collate_fn=_pil_collate_fn,
    )

    need_feats = proxy_kind == "prototype"
    ext_l = FeatureExtractor(large_model, cfg_l.name) if need_feats else None
    ext_s = FeatureExtractor(small_model, cfg_s.name) if need_feats else None

    z_l_all, f_l_all, z_s_all, f_s_all, labs_all = [], [], [], [], []
    try:
        for imgs, labels in src_loader:
            x_l = _preprocess_batch(imgs, large_preprocess, device)
            x_s = _preprocess_batch(imgs, small_preprocess, device)
            if need_feats:
                zl, fl = ext_l(x_l)
                zs, fs = ext_s(x_s)
                f_l_all.append(fl.cpu()); f_s_all.append(fs.cpu())
            else:
                zl, zs = large_model(x_l).detach(), small_model(x_s).detach()
            z_l_all.append(zl.cpu()); z_s_all.append(zs.cpu())
            labs_all.append(labels.cpu())
    finally:
        if need_feats:
            ext_l.remove(); ext_s.remove()

    z_l, z_s, labels = torch.cat(z_l_all), torch.cat(z_s_all), torch.cat(labs_all)
    f_l = torch.cat(f_l_all) if need_feats else None
    f_s = torch.cat(f_s_all) if need_feats else None

    fit_val_reliability(cfg_l, z_l, f_l, labels, proxy_kind, batch_size=config["BS"])
    fit_val_reliability(cfg_s, z_s, f_s, labels, proxy_kind, batch_size=config["BS"])
    print(f"[proxy weighted] val reliability fit: "
          f"large val_acc={cfg_l.val_acc:.4f}  small val_acc={cfg_s.val_acc:.4f}")


def build_proxy_weighted_calibrator(
    proxy_kind: str,
    config: dict,
    large_model, large_preprocess,
    small_model, small_preprocess,
    device: torch.device,
    *,
    pool: str = "linear",
    calib_mode: str = "zscore",
    filter: str = "kalman",
    filter_kwargs: dict | None = None,
    beta: float | None = None,
    precision_weight: bool = False,
    proxy_cache: str | None = None,
    calib_map: str | None = None,
    weight_cfg_name: str | None = None,
    fixed_ts_config: str | None = None,
    csv_path: str | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
    proto_metric: str = "cosine",
):
    """Build a JointProxyWeighted calibrator end to end:
      1. build/load proxy stats (atc threshold / prototypes, if needed),
      2. fill the val reliability prior (Task 2.1),
      3. attach an isotonic calib map if calib_mode="isotonic",
      4. load the base temperatures from --fixed_ts_config (the val-tuned
         prior scale; JointProxyWeighted holds these fixed),
      5. load the beta sidecar, or fit it on dev-shift corruptions and save it
         (Task 2.5) if `weight_cfg_name` is given but no sidecar exists yet.
    """
    from src.calibrators.joint_fixed_TS import JointFixedTS
    from src.calibrators.joint_proxy_weighted import JointProxyWeighted

    cfg_l, cfg_s = _build_proxy_stats(
        proxy_kind, config,
        large_model, large_preprocess, small_model, small_preprocess,
        device, cache_path=proxy_cache, proto_metric=proto_metric,
    )
    _val_pass_for_reliability(
        cfg_l, cfg_s, large_model, large_preprocess, small_model, small_preprocess,
        config, device, proxy_kind,
    )

    if calib_mode == "isotonic":
        if calib_map is None:
            raise ValueError("--proxy_calib isotonic requires --calib_map")
        try:
            maps = load_calibration_maps(calib_map)
        except FileNotFoundError:
            maps = _fit_and_save_calibration_maps(
                calib_map, cfg_l, cfg_s,
                large_model, large_preprocess, small_model, small_preprocess,
                config, device, proxy_name=proxy_kind,
                num_samples=num_samples, seed=seed,
            )
        maps.attach(cfg_l, cfg_s)
        print(f"Attached calibration map '{calib_map}' (proxy={maps.proxy_name})")

    base_ts = None
    if fixed_ts_config is not None:
        base_ts = JointFixedTS.load(fixed_ts_config)
        base_ts.requires_grad_(False)

    if beta is None:
        weight_cfg = None
        if weight_cfg_name is not None:
            try:
                weight_cfg = load_weight_cfg(weight_cfg_name)
            except FileNotFoundError:
                weight_cfg = None
        if weight_cfg is not None:
            beta = weight_cfg["beta"]
            # A saved sidecar's pool/calib_mode/filter win, so a fitted beta
            # stays reproducible even if the CLI flags used to fit it drift.
            pool = weight_cfg.get("pool", pool)
            calib_mode = weight_cfg.get("calib_mode", calib_mode)
            filter = weight_cfg.get("filter", filter)
            filter_kwargs = weight_cfg.get("filter_kwargs", filter_kwargs)
        else:
            beta = fit_beta(
                proxy_kind, cfg_l, cfg_s,
                large_model, large_preprocess, small_model, small_preprocess,
                config, device,
                pool=pool, calib_mode=calib_mode, filter=filter,
                filter_kwargs=filter_kwargs, base_ts=base_ts,
                num_samples=num_samples, seed=seed,
            )
            if weight_cfg_name is not None:
                save_weight_cfg(
                    {"beta": beta, "pool": pool, "calib_mode": calib_mode,
                     "filter": filter, "filter_kwargs": filter_kwargs or {}},
                    weight_cfg_name,
                )

    return JointProxyWeighted(
        proxy_kind, cfg_l, cfg_s,
        beta=beta, pool=pool, calib_mode=calib_mode, filter=filter,
        filter_kwargs=filter_kwargs, base_ts=base_ts,
        precision_weight=precision_weight, csv_path=csv_path,
    )


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.proxies.proxies import ProxyStats

    torch.manual_seed(0)
    C = 10

    def _mk_cfg(name, val_acc):
        cfg = ProxyStats(name=name, num_classes=C)
        cfg.val_acc = val_acc
        cfg.proxy_mean["nuclear_norm"] = 0.5
        cfg.proxy_std["nuclear_norm"] = 0.1
        return cfg

    cfg_l = _mk_cfg("large", val_acc=0.8)
    cfg_s = _mk_cfg("small", val_acc=0.6)

    from src.calibrators.joint_proxy_weighted import JointProxyWeighted
    probe = JointProxyWeighted(
        "nuclear_norm", cfg_l, cfg_s, beta=1.0, pool="linear",
        calib_mode="zscore", filter="none", log_every=0,
    )

    # Monotonic case: large model is always right (huge margin), small always
    # wrong (huge margin), and the score gap always favours large. Larger beta
    # should strictly help -> grid search must select the grid's max beta.
    records = []
    for _ in range(20):
        label = torch.randint(0, C, (4,))
        z_l = F.one_hot(label, C).float() * 20.0
        wrong = (label + 1) % C
        z_s = F.one_hot(wrong, C).float() * 20.0
        records.append((z_l, z_s, label, 5.0, -5.0))  # s_l >> s_s

    best_beta, best_nll, nll_by_beta = _grid_search_beta(probe, records, _BETA_GRID)
    # NLL saturates near 0 for beta large enough to fully trust the (always
    # correct) large model, so ties go to the smallest such beta — just check
    # the grid search picked a large-beta regime and strictly beat beta=0.
    assert best_beta >= 1.0, (best_beta, nll_by_beta)
    assert nll_by_beta[best_beta] < nll_by_beta[0.0], nll_by_beta
    assert all(nll_by_beta[b] >= nll_by_beta[best_beta] for b in _BETA_GRID), nll_by_beta
    print("fit_beta monotonic grid-search self-test passed "
          f"(best_beta={best_beta}, nll={best_nll:.4f})")

    # Reproducibility: same records -> same beta every time.
    best_beta2, *_ = _grid_search_beta(probe, records, _BETA_GRID)
    assert best_beta2 == best_beta
    print("fit_beta reproducibility self-test passed")

    # Weight-config sidecar round-trip.
    import tempfile
    cfg_out = {"beta": best_beta, "pool": "linear", "calib_mode": "zscore",
               "filter": "kalman", "filter_kwargs": {"q": 1e-3, "r": 1e-1}}
    with tempfile.TemporaryDirectory() as d:
        save_weight_cfg(cfg_out, "selftest", directory=d)
        assert list_weight_cfgs(d) == ["selftest"]
        loaded = load_weight_cfg("selftest", directory=d)
    assert loaded == cfg_out
    print("weight-config sidecar round-trip self-test passed")

    print("calibrator_setup self-test passed")