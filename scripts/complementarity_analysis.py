"""
Complementarity / oracle ceiling analysis.

For each eval corruption × severity from the config, bucket every test sample
into the four agreement cells:

    ViT✓R✓  both correct
    ViT✓R✗  only ViT correct
    ViT✗R✓  only ResNet correct  ← ResNet-rescue headroom
    ViT✗R✗  neither correct

Reports:
  * Per-cell counts and fractions.
  * Oracle accuracy (correct if either model is).
  * ResNet rescue headroom (vR cell fraction).
  * If --heavy_tl / --heavy_ts are supplied: samples that the Ts=Tl=1 duo gets
    right but the heavy-ViT duo gets wrong, broken down by agreement cell.
    A high vR fraction there directly attributes the gain to ResNet rescuing ViT.

Usage
-----
# Cell counts + oracle only (no comparison duo):
python scripts/complementarity_analysis.py \
    --config cfgs/dynamic_duo_config.yaml \
    --cache_dir cache/logits

# Full comparison with fitted heavy-ViT temperatures:
python scripts/complementarity_analysis.py \
    --config cfgs/dynamic_duo_config.yaml \
    --cache_dir cache/logits \
    --heavy_tl 0.4637 --heavy_ts 4.8508
"""

import argparse
import torch

from src.utils.data import load_config
from src.utils.logits import get_model_logits


def _duo_preds(z_l: torch.Tensor, z_s: torch.Tensor, Tl: float, Ts: float) -> torch.Tensor:
    return ((z_l / Tl + z_s / Ts) / 2).argmax(dim=1)

def _cells(l_correct: torch.Tensor, s_correct: torch.Tensor):
    return {
        "VR": (l_correct  &  s_correct).sum().item(),
        "Vr": (l_correct  & ~s_correct).sum().item(),
        "vR": (~l_correct &  s_correct).sum().item(),
        "vr": (~l_correct & ~s_correct).sum().item(),
    }


def _rescue_breakdown(rescued: torch.Tensor, l_correct: torch.Tensor, s_correct: torch.Tensor):
    """Cell breakdown for a rescued-sample boolean mask."""
    if not rescued.any():
        return {"VR": 0, "Vr": 0, "vR": 0, "vr": 0}
    return {
        "VR": ( l_correct &  s_correct)[rescued].sum().item(),
        "Vr": ( l_correct & ~s_correct)[rescued].sum().item(),
        "vR": (~l_correct &  s_correct)[rescued].sum().item(),
        "vr": (~l_correct & ~s_correct)[rescued].sum().item(),
    }


def _pct(n, d):
    return n / d if d else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    type=str,   default="cfgs/dynamic_duo_config.yaml")
    parser.add_argument("--cache_dir", type=str,   default="cache/logits")
    parser.add_argument("--heavy_tl",  type=float, default=None,
                        help="Tl for the heavy-ViT comparison duo")
    parser.add_argument("--heavy_ts",  type=float, default=None,
                        help="Ts for the heavy-ViT comparison duo")
    args = parser.parse_args()

    cfg         = load_config(args.config)
    large_name  = cfg["LARGE"]["NAME"]
    small_name  = cfg["SMALL"]["NAME"]
    val_dir     = cfg["VAL_DIR"]
    test_dir    = cfg["TEST_DIR"]
    batch_size  = cfg["BS"]
    num_workers = cfg["WORKERS"]
    device      = torch.device("cpu")  # logits already cached; no GPU needed

    heavy_tl = args.heavy_tl
    heavy_ts = args.heavy_ts
    do_rescue = (heavy_tl is not None) and (heavy_ts is not None)

    def get_logits(model_name, corruption=None, severity=None):
        return get_model_logits(
            model_name=model_name, val_dir=val_dir, test_dir=test_dir,
            cache_dir=args.cache_dir, batch_size=batch_size,
            num_workers=num_workers, corruption=corruption,
            severity=severity, device=device,
        )

    print(f"\n{'='*76}")
    print("Complementarity / Oracle Ceiling Analysis")
    print(f"  Large (ViT): {large_name}")
    print(f"  Small  (R):  {small_name}")
    print(f"  Equal  duo:  Tl=1.0  Ts=1.0")
    if do_rescue:
        print(f"  Heavy-ViT:   Tl={heavy_tl:.4f}  Ts={heavy_ts:.4f}")
    print(f"{'='*76}\n")

    # ------------------------------------------------------------------ header
    col_w = 22
    header = f"{'Corruption':<{col_w}} {'Sv':>2}  {'N':>6}  {'ViT':>6}  {'R':>6}  {'Oracle':>6}  {'Equal':>6}"
    if do_rescue:
        header += f"  {'Heavy':>6}  {'Rescued':>7}  {'%in vR':>7}"
    print(header)
    print("-" * len(header))

    # --------------------------------------------------------- per-corruption
    agg = dict(N=0, VR=0, Vr=0, vR=0, vr=0,
               oracle=0, vit=0, r=0, equal=0,
               heavy=0, rescued=0, res_VR=0, res_Vr=0, res_vR=0, res_vr=0)

    for severity in cfg["EVAL"]["SEVERITIES"]:
        for corruption in cfg["EVAL"]["CORRUPTIONS"]:
            z_l, labels = get_logits(large_name, corruption, severity)
            z_s, _      = get_logits(small_name, corruption, severity)

            l_correct   = z_l.argmax(1) == labels
            s_correct   = z_s.argmax(1) == labels
            equal_preds = _duo_preds(z_l, z_s, 1.0, 1.0)

            cells      = _cells(l_correct, s_correct)
            N          = len(labels)
            oracle_n   = ((l_correct) | (s_correct)).sum().item()
            vit_n      = l_correct.sum().item()
            r_n        = s_correct.sum().item()
            equal_n    = (equal_preds == labels).sum().item()

            row = (f"{corruption:<{col_w}} {severity:>2}  {N:>6}  "
                   f"{vit_n/N:>6.3f}  {r_n/N:>6.3f}  "
                   f"{oracle_n/N:>6.3f}  {equal_n/N:>6.3f}")

            agg["N"]      += N
            agg["oracle"] += oracle_n
            agg["vit"]    += vit_n
            agg["r"]      += r_n
            agg["equal"]  += equal_n
            for k in ("VR", "Vr", "vR", "vr"):
                agg[k] += cells[k]

            if do_rescue:
                heavy_preds = _duo_preds(z_l, z_s, heavy_tl, heavy_ts)
                heavy_n     = (heavy_preds == labels).sum().item()
                rescued     = (equal_preds == labels) & (heavy_preds != labels)
                n_res       = rescued.sum().item()
                rb          = _rescue_breakdown(rescued, l_correct, s_correct)
                pct_vR      = _pct(rb["vR"], n_res)

                row += (f"  {heavy_n/N:>6.3f}  {n_res:>7}  "
                        f"{pct_vR:>7.1%}" if n_res > 0 else
                        f"  {heavy_n/N:>6.3f}  {n_res:>7}  {'n/a':>7}")

                agg["heavy"]   += heavy_n
                agg["rescued"] += n_res
                for k in ("VR", "Vr", "vR", "vr"):
                    agg[f"res_{k}"] += rb[k]

            print(row)

            # Cell breakdown line
            total = sum(cells[k] for k in ("VR", "Vr", "vR", "vr"))
            print(f"  {'':>{col_w+2}}  cells  "
                  f"VR={cells['VR']:>5}({cells['VR']/total:>4.0%})  "
                  f"Vr={cells['Vr']:>5}({cells['Vr']/total:>4.0%})  "
                  f"vR={cells['vR']:>5}({cells['vR']/total:>4.0%})  "
                  f"vr={cells['vr']:>5}({cells['vr']/total:>4.0%})")

    # --------------------------------------------------------- aggregate
    print("-" * len(header))
    N = agg["N"]
    if N == 0:
        print("No data.")
        return

    row_agg = (f"{'AGGREGATE':<{col_w}} {'':>2}  {N:>6}  "
               f"{agg['vit']/N:>6.3f}  {agg['r']/N:>6.3f}  "
               f"{agg['oracle']/N:>6.3f}  {agg['equal']/N:>6.3f}")
    if do_rescue:
        n_res   = agg["rescued"]
        pct_vR  = _pct(agg["res_vR"], n_res)
        row_agg += (f"  {agg['heavy']/N:>6.3f}  {n_res:>7}  "
                    f"{pct_vR:>7.1%}" if n_res > 0 else
                    f"  {agg['heavy']/N:>6.3f}  {n_res:>7}  {'n/a':>7}")
    print(row_agg)

    # --------------------------------------------------------- summary block
    total = sum(agg[k] for k in ("VR", "Vr", "vR", "vr"))
    print(f"\nAgreement cells (aggregate over {total:,} samples):")
    for k, label in [("VR", "ViT✓ R✓  (both correct)       "),
                     ("Vr", "ViT✓ R✗  (only ViT correct)   "),
                     ("vR", "ViT✗ R✓  (only R correct) ←hdroom"),
                     ("vr", "ViT✗ R✗  (neither correct)    ")]:
        print(f"  {label}: {agg[k]:>8,}  ({agg[k]/total:>5.1%})")

    print(f"\n  Oracle accuracy (correct if either):  {agg['oracle']/N:.4f}")
    print(f"  ResNet rescue headroom (vR / total):  {agg['vR']/total:.4f}  "
          f"({agg['vR']:,} samples)")

    if do_rescue:
        n_res = agg["rescued"]
        if n_res > 0:
            print(f"\nRescue analysis — Ts=Tl=1 right, heavy-ViT wrong:")
            print(f"  Total rescued:  {n_res:,}")
            for k, label in [("VR", "ViT✓ R✓"), ("Vr", "ViT✓ R✗"),
                              ("vR", "ViT✗ R✓  ← rescue"), ("vr", "ViT✗ R✗")]:
                n = agg[f"res_{k}"]
                print(f"    {label}: {n:>6,}  ({_pct(n, n_res):>5.1%})")
            vR_frac = _pct(agg["res_vR"], n_res)
            verdict = "YES" if vR_frac > 0.5 else "NO"
            print(f"\n  → {verdict}: rescued samples are "
                  f"{'predominantly' if vR_frac > 0.5 else 'NOT predominantly'} in vR "
                  f"({vR_frac:.1%})")
            print(f"    {'Gain directly attributed to ResNet rescuing ViT errors.' if vR_frac > 0.5 else 'Gain not explained by simple ResNet rescue.'}")
        else:
            print("\n  No rescued samples — Ts=Tl=1 does not outperform heavy-ViT on any sample.")


if __name__ == "__main__":
    main()
