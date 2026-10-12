"""
cache.py
========
Per-segment cache of the members' logits: the thing every gate variant is replayed on.

One run (RunSpec) has one directory,
    <cache>/members/<duo>/<tta>/<protocol_tag>_seed<seed>_<fingerprint>/
holding `manifest.json` (the spec, timings, git commit) and one `<segment>.pt` per finished segment.
Files are written to a temp name and renamed, so a killed job never leaves a half-written segment that
looks complete; a re-run recomputes only the missing segments (episodic) or everything (continual/CCC,
where segment k depends on all before it).

Storage format ("centered fp16"): each model's logits minus their per-row maximum, as float16, 2 bytes per
logit (3 GB for a 15-corruption x 50,000-image run of both members). Everything the gate and the metrics
compute from logits is invariant to a per-row constant (softmax, entropy, NLL, ECE, the nuclear norm, a
combination w*z_l/T_l + (1-w)*z_s/T_s -- the shifts add the same constant to every class), and centering
keeps the largest logits, the ones that decide the argmax, at near-zero magnitude where fp16 is finest.
Anything that needs RAW logits (energy scores, logit norms) must not use this cache.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import torch


def center_fp16(z: torch.Tensor) -> torch.Tensor:
    z = z.float()
    out = (z - z.max(dim=1, keepdim=True).values).half()
    if not torch.isfinite(out).all():
        raise ValueError("non-finite logits cannot be cached")
    return out


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


class MemberCache:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def file(self, stem: str) -> Path:
        return self.dir / f"{stem}.pt"

    def has(self, stem: str) -> bool:
        return self.file(stem).is_file()

    def stems(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.pt")) if self.dir.is_dir() else []

    def write(self, stem: str, z_l: torch.Tensor, z_s: torch.Tensor, labels: torch.Tensor) -> Path:
        """z_l, z_s: (N, 1000) raw logits (any dtype, any device); labels: (N,) class indices."""
        self.dir.mkdir(parents=True, exist_ok=True)
        if not (len(z_l) == len(z_s) == len(labels)):
            raise ValueError(f"length mismatch: {len(z_l)}, {len(z_s)}, {len(labels)}")
        payload = {"z_l": center_fp16(z_l.cpu()), "z_s": center_fp16(z_s.cpu()),
                   "labels": labels.cpu().to(torch.int16), "n": len(labels)}
        final = self.file(stem)
        tmp = final.with_suffix(f".tmp{os.getpid()}")
        torch.save(payload, tmp)
        os.replace(tmp, final)
        return final

    def read(self, stem: str) -> dict:
        """{"z_l", "z_s"} float32 (N, 1000) centered logits, {"labels"} int64, {"n"}."""
        d = torch.load(self.file(stem), map_location="cpu", weights_only=True)
        return {"z_l": d["z_l"].float(), "z_s": d["z_s"].float(), "labels": d["labels"].long(), "n": d["n"]}

    def n_samples(self, stem: str) -> int:
        return int(torch.load(self.file(stem), map_location="cpu", weights_only=True)["n"])

    # --- manifest ---------------------------------------------------------------------------------- #
    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def write_manifest(self, info: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        info = {**info, "git_commit": git_commit(), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(info, indent=2, default=str))
        os.replace(tmp, self.manifest_path)

    def read_manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text())


if __name__ == "__main__":
    import tempfile

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        c = MemberCache(Path(tmp) / "run")
        # realistic logit scale: a confident peak (~20) over a noisy floor, a huge-offset row, a flat row
        z_l = torch.randn(512, 1000) * 2.0
        z_l[torch.arange(512), torch.randint(0, 1000, (512,))] += 18.0
        z_l[3] += 300.0
        z_l[4] = 0.0
        z_s = torch.randn(512, 1000) * 1.5 + 7.0
        labels = torch.randint(0, 1000, (512,))
        assert not c.has("fog__s5")
        c.write("fog__s5", z_l, z_s, labels)
        assert c.has("fog__s5") and c.stems() == ["fog__s5"] and c.n_samples("fog__s5") == 512
        assert not list(Path(tmp).rglob("*.tmp*")), "no temp file left behind"
        d = c.read("fog__s5")
        assert d["labels"].equal(labels) and d["z_l"].dtype == torch.float32

        for raw, back in ((z_l, d["z_l"]), (z_s, d["z_s"])):
            # centered: the max of every row is exactly 0
            assert back.max(1).values.eq(0).all()
            # invariants the gate/metrics rely on survive the round trip
            assert back.argmax(1).equal(raw.argmax(1)), "argmax must be unchanged"
            p_raw, p_back = raw.softmax(1), back.softmax(1)
            assert (p_raw - p_back).abs().max() < 2e-3, (p_raw - p_back).abs().max()
            # nuclear norm of the softmax matrix (the paper's proxy)
            nn_raw = torch.linalg.matrix_norm(p_raw[:128], ord="nuc")
            nn_back = torch.linalg.matrix_norm(p_back[:128], ord="nuc")
            assert abs(nn_raw - nn_back) / nn_raw < 2e-3
        # a combination of two models' logits is shift-invariant too: same softmax as from the raw logits
        mix_raw = (0.3 * z_l / 1.7 + 0.7 * z_s / 0.9).softmax(1)
        mix_back = (0.3 * d["z_l"] / 1.7 + 0.7 * d["z_s"] / 0.9).softmax(1)
        assert (mix_raw - mix_back).abs().max() < 2e-3

        try:
            c.write("bad", z_l[:5], z_s, labels)
            raise AssertionError("length mismatch must raise")
        except ValueError:
            pass
        try:
            center_fp16(torch.full((2, 3), float("nan")))
            raise AssertionError("non-finite must raise")
        except ValueError:
            pass
        c.write_manifest({"spec": {"duo": "x"}})
        assert c.read_manifest()["spec"] == {"duo": "x"} and "git_commit" in c.read_manifest()
    print("cache self-test passed")
