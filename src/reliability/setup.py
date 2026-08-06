"""
setup.py
========
Factory helpers for the filtered-proxy soft-weighting calibrator
(JointProxyWeighted, paper Sections 2-5).

Keeps all proxy-stats building, calibration-map fitting/loading, and calibrator
construction out of the top-level run script.

Public API
----------
build_proxy_weighted_calibrator(proxy_kind, ...) -> JointProxyWeighted
fit_beta(...) -> float
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.reliability.proxies.stats import build_proxy_stats, ProxyStats
from src.reliability.calibration.maps import load_calibration_maps

# Proxy kinds that need a source-data pass to fit state before scoring:
# atc (threshold), prototype (class means), cot (source label marginal).
# nuclear_norm and ac_mc are stateless.
_SOURCE_FIT_KINDS = {"atc", "prototype", "cot"}


def _build_proxy_stats(
    proxy_kind: str,
    config: dict,
    large_model, large_preprocess,
    small_model, small_preprocess,
    device: torch.device,
    cache_path: str | None,
    proto_metric: str = "cosine",
) -> tuple[ProxyStats, ProxyStats]:
    """Return (cfg_l, cfg_s). atc/prototype/cot run a source pass; nuclear_norm
    and ac_mc are stateless and free.

    proto_metric ("cosine" | "mahalanobis") selects the prototype-proxy distance.
    """
    if proxy_kind in _SOURCE_FIT_KINDS:
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
    calib_method: str = "isotonic",
):
    """Collect records over CALIBRATOR corruptions in one combined loader, fit, save."""
    from tqdm import tqdm
    from src.utils.data import load_imagenetC
    from src.reliability.proxies.stats import FeatureExtractor
    from src.reliability.calibration.maps import make_record, fit_calibration_maps, save_calibration_maps

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

    maps = fit_calibration_maps(records, proxy_name, cfg_l.name, cfg_s.name, method=calib_method)
    save_calibration_maps(maps, name)
    print(f"[calib map] Fitted on {len(records)} batches, proxy={proxy_name}")
    return maps


def build_proxy_weighted_calibrator(
    proxy_kind: str,
    proxy_cache: str | None,
    calib_map: str | None,
    calib_method: str,
    filter_kind: str,
    filter_kwargs: dict | None,
    beta: float,
    pool: str,
    prior_l: float,
    prior_s: float,
    base_ts,
    csv_path: str | None,
    config: dict,
    large_model, large_preprocess,
    small_model, small_preprocess,
    device: torch.device,
    num_samples: int | None = None,
    seed: int | None = None,
    proto_metric: str = "cosine",
    proxy_batch_size: int = 1,
):
    """Build a JointProxyWeighted calibrator (paper Sections 2-5).

    prior_l, prior_s are each model's clean-source accuracy in [0, 1]
    (converted to the ema/kalman filters' logit-space reset prior — see
    src.reliability.calibration.logit.to_logit); default 0.5 (neutral) if
    unknown. base_ts is a frozen JointFixedTS supplying the (T_l, T_s) prior
    for the Section-5 combination — load one the same way the fixed_ts
    calibration mode does, or pass None for T_l = T_s = 1.0.
    proxy_batch_size is the proxy batch size b_t (Section 1), independent of
    the adaptation batch size — see JointProxyWeighted's docstring.
    """
    from src.calibrators.joint_proxy_weighted import JointProxyWeighted
    from src.reliability.calibration.logit import to_logit

    assert proxy_kind != "agreement", (
        "proxy_kind='agreement' scores the model pair, not a single model, "
        "and cannot drive JointProxyWeighted's gate; see "
        "src/reliability/proxies/agreement.py."
    )

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
                calib_method=calib_method,
            )
        maps.attach(cfg_l, cfg_s)
        print(f"Attached calibration map '{calib_map}' (proxy={maps.proxy_name}, method={maps.method})")
    else:
        print(f"[proxy_weighted] no --calib_map given; gating on the raw '{proxy_kind}' "
              f"score directly (Section 3's identity baseline).")

    return JointProxyWeighted(
        proxy_kind=proxy_kind,
        cfg_l=cfg_l,
        cfg_s=cfg_s,
        beta=beta,
        pool=pool,
        filter_kind=filter_kind,
        filter_kwargs=filter_kwargs,
        prior_l=to_logit(prior_l),
        prior_s=to_logit(prior_s),
        base_ts=base_ts,
        proxy_batch_size=proxy_batch_size,
        csv_path=csv_path,
    )


def fit_beta(
    calibrator,
    large_model, large_preprocess,
    small_model, small_preprocess,
    config: dict,
    device: torch.device,
    betas: list[float] | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
) -> float:
    """Grid-search calibrator.beta against held-out dev-shift NLL (Section 5).

    Runs the CALIBRATOR corruptions/severities once, caching each batch's
    filtered scores (x_l, x_s) and logits so every candidate beta can be
    re-scored (gate + combine only) without re-running the models or the
    stateful temporal filters more than once per batch. Sets and returns the
    best beta; mutates calibrator.beta in place.
    """
    from src.utils.data import load_imagenetC

    if betas is None:
        betas = [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]

    print(f"[fit_beta] grid-searching beta over {len(betas)} candidates: {betas}")

    corruptions = config["CALIBRATOR"]["CORRUPTIONS"]
    severities = config["CALIBRATOR"]["SEVERITIES"]
    T_l = float(calibrator.base_ts.Tl.item()) if calibrator.base_ts is not None else 1.0
    T_s = float(calibrator.base_ts.Ts.item()) if calibrator.base_ts is not None else 1.0

    cached = []  # (x_l, x_s, z_l, z_s, labels), one entry per dev batch
    for corruption in corruptions:
        loader = load_imagenetC(
            config["TEST_DIR"], severities=severities, corruption_types=[corruption],
            device=device, batch_size=config["BS"], num_workers=config["WORKERS"],
            num_samples=num_samples, seed=seed,
        )
        # total_samples lets the calibrator flush a trailing proxy-batch
        # remainder instead of leaving it stale (see
        # JointProxyWeighted._maybe_update_gate).
        calibrator.set_corruption(corruption, total_samples=len(loader.dataset))
        for imgs, labels in loader:
            xl = torch.stack([large_preprocess(img) for img in imgs]).to(device)
            xs = torch.stack([small_preprocess(img) for img in imgs]).to(device)
            calibrator.set_labels(labels)  # oracle proxy_kind requires labels to score at all
            with torch.no_grad():
                zl = large_model(xl)
                zs = small_model(xs)
                _, _, _, _, _, x_l, x_s, _ = calibrator._forward(zl, zs)
            cached.append((x_l, x_s, zl.cpu(), zs.cpu(), labels))

    best_beta, best_nll = betas[0], float("inf")
    for candidate in betas:
        total_nll, n = 0.0, 0
        for x_l, x_s, zl, zs, labels in cached:
            w_l = 1.0 / (1.0 + math.exp(-candidate * (x_l - x_s)))
            w_s = 1.0 - w_l
            if calibrator.pool == "log":
                z_duo = w_l * (zl / T_l) + w_s * (zs / T_s)
            else:
                p_duo = w_l * F.softmax(zl / T_l, dim=1) + w_s * F.softmax(zs / T_s, dim=1)
                z_duo = torch.log(p_duo.clamp(min=1e-8))
            total_nll += F.cross_entropy(z_duo, labels, reduction="sum").item()
            n += len(labels)
        avg_nll = total_nll / n
        if avg_nll < best_nll:
            best_nll, best_beta = avg_nll, candidate

    calibrator.beta = best_beta
    print(f"[fit_beta] best beta={best_beta} (dev NLL={best_nll:.4f})")
    return best_beta