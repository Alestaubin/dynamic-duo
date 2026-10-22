"""
provenance.py
=============
What every table and figure says about where it came from (author's rule, 2026-10-22: all tables and plots are tagged with their
models and the time):

  * the model names of a duo (`models_label`: "ViT-B/16 + ResNet-50 (V1)") and one line naming the run (`run_label`: models, duo name,
    TTA method, `--bn`),
  * the generation time (`tables.timestamp`),
  * LaTeX tables carry them in their first comment line (`tables.table_env(models=...)`), and their captions name the models,
  * figures carry them as a footer line drawn on the figure itself and in the file's metadata (`save_figure`),
  * file names always carry the duo name (`adaptation_modes.output_tag`, `temperature_study.study_tag`), so two duos never share a file.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

from src.utils.tables import timestamp

# display names of the models (captions, figure footers); a name that is not here is shown as it is
MODEL_NAMES = {"vit_b_16": "ViT-B/16", "vit_l_16": "ViT-L/16", "vit_s_16": "ViT-S/16", "resnet18": "ResNet-18", "resnet34": "ResNet-34",
               "resnet50": "ResNet-50", "resnet50_v1": "ResNet-50 (V1)", "vit_b_16_augreg": "ViT-B/16 (AugReg)",
               "vit_b_16_augreg2": "ViT-B/16 (AugReg2)", "vit_b_16_sar": "ViT-B/16 (SAR)", "convnext_base": "ConvNeXt-B",
               "convnext_tiny": "ConvNeXt-T", "deit_small": "DeiT-S", "swin_t": "Swin-T", "efficientnet_b0": "EfficientNet-B0",
               "efficientnet_b4": "EfficientNet-B4"}


def model_name(name: str) -> str:
    return MODEL_NAMES.get(name, name)


def models_label(duo: dict) -> str:
    """'ViT-B/16 + ResNet-50 (V1)' for a duo config (a cfgs/duos dict, or Context.duo)."""
    return f"{model_name(duo['LARGE']['NAME'])} + {model_name(duo['SMALL']['NAME'])}"


def run_label(duo: dict, tta: str | None = None, bn: bool = False) -> str:
    """One line naming a run: 'ViT-B/16 + ResNet-50 (V1) [vitb16_rn50v1], Tent, frozen BN on test-batch statistics'."""
    parts = [f"{models_label(duo)} [{duo['NAME']}]"]
    if tta:
        parts.append("Tent" if tta == "tent" else tta.upper())
    if bn and any(duo[s].get("NORM") == "BN" for s in ("LARGE", "SMALL")):
        parts.append("frozen BN on test-batch statistics")
    return ", ".join(parts)


def footer(duo: dict, tta: str | None = None, bn: bool = False, extra: str = "") -> str:
    """The line a figure carries: the run, an optional `extra` (temperatures, stream) and the generation time."""
    return " | ".join(p for p in (run_label(duo, tta, bn), extra, f"generated {timestamp()}") if p)


def save_figure(fig, path: str | Path, text: str, *, dpi: int = 300, exts: Sequence[str] = ("png", "pdf"), title: str = "") -> list[Path]:
    """Write `fig` as `path` with each of `exts` (the suffix of `path` is replaced), after drawing `text` as a small footer line at the
    bottom left; the same text goes into each file's metadata (PNG text chunk, PDF subject). Returns the paths."""
    import matplotlib.pyplot as plt  # noqa: F401  (imported by the caller's backend choice)

    fig.text(0.004, 0.003, text, fontsize=4.8, color="#6b6a66", ha="left", va="bottom")
    out = []
    for ext in exts:
        p = Path(path).with_suffix("." + ext)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = ({"Title": title or p.stem, "Description": text, "Software": "dynamic-duos"} if ext == "png"
                else {"Title": title or p.stem, "Subject": text, "Creator": "dynamic-duos", "CreationDate": datetime.now().astimezone()}
                if ext == "pdf" else None)
        fig.savefig(p, dpi=dpi, facecolor="white", metadata=meta)
        out.append(p)
    return out


if __name__ == "__main__":
    import re
    import tempfile

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    duo = {"NAME": "vitb16_rn50v1", "LARGE": {"NAME": "vit_b_16", "NORM": "LN"}, "SMALL": {"NAME": "resnet50_v1", "NORM": "BN"}}
    assert models_label(duo) == "ViT-B/16 + ResNet-50 (V1)"
    assert run_label(duo, "tent", True) == "ViT-B/16 + ResNet-50 (V1) [vitb16_rn50v1], Tent, frozen BN on test-batch statistics"
    ln = {"NAME": "convnextb_deits", "LARGE": {"NAME": "convnext_base", "NORM": "LN"}, "SMALL": {"NAME": "deit_small", "NORM": "LN"}}
    assert run_label(ln, "eata", True) == "ConvNeXt-B + DeiT-S [convnextb_deits], EATA", "no BatchNorm: the --bn flag changes nothing and is not named"
    assert models_label({"LARGE": {"NAME": "x_new"}, "SMALL": {"NAME": "resnet34"}}) == "x_new + ResNet-34"
    f = footer(duo, "tent", True, "T_L = 0.651, T_S = 1.0097")
    assert re.search(r"\| T_L = 0.651, T_S = 1.0097 \| generated \d{4}-\d\d-\d\d \d\d:\d\d:\d\d [+-]\d{4}$", f), f
    with tempfile.TemporaryDirectory() as tmp:
        fig = plt.figure(figsize=(3, 2))
        fig.add_subplot().plot([0, 1], [0, 1])
        paths = save_figure(fig, Path(tmp) / "fig.png", f, title="test")
        assert [p.suffix for p in paths] == [".png", ".pdf"] and all(p.stat().st_size > 1000 for p in paths)
        assert b"vitb16_rn50v1" in paths[1].read_bytes() or b"vitb16" in paths[0].read_bytes(), "the footer text is in the files"
        import PIL.Image
        assert PIL.Image.open(paths[0]).text["Description"] == f
        plt.close(fig)
    print("provenance self-test passed")
