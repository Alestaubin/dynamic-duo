"""
calibrator_setup.py
===================
Factory helpers for proxy-based calibrators (proxy_anchor_coca, soft_anchor).

Keeps all proxy-stats building, calibration-map fitting/loading, and calibrator
construction out of the top-level run script.

Public API
----------
build_proxy_calibrator(calibration_mode, proxy_kind, ...) -> BaseJointCalibrator
"""

from __future__ import annotations

import torch

from src.proxies.proxies import build_proxy_stats, ProxyStats
from src.proxies.calibration import load_calibration_maps


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