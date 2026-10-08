"""
evaluate.py
===========
Scoring a stream without ever holding its N x K probability matrix.

StreamMetrics  one series (a member, or a gate's combined output) on one segment: per-sample confidence and
               correctness (5 bytes/sample, enough for accuracy and top-label ECE with 15 bins), summed NLL and
               entropy, and per-batch correct counts (for accuracy-vs-time curves). A 7.5M-sample CCC stream
               costs ~40 MB, not the 30 GB the old run_duo concatenation would need.
OnlineEval     feeds the SAME member logits, batch after batch, to every gate being compared, so any number of
               gates is evaluated in one pass over the members -- live (CCC) or replayed from the cache.

Logits reaching the gates are the dataset's: `dataset.apply_class_mask` is applied first, so ImageNet-A/R are
scored, gated and temperature-scaled on their 200 classes. Gates are told about segment boundaries only where
the members are reset (Segment.reset_before); in continual and CCC streams they see one unbroken stream.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.datasets.base import ShiftDataset
from src.utils.metrics import top_label_ece


class StreamMetrics:
    def __init__(self, num_bins: int = 15):
        self.num_bins = num_bins
        self.n = 0
        self._conf: list[np.ndarray] = []
        self._correct: list[np.ndarray] = []
        self._nll = 0.0
        self._entropy = 0.0
        self.batch_n: list[int] = []
        self.batch_correct: list[int] = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        logits = logits.float()
        labels = labels.to(logits.device)
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        conf, pred = p.max(dim=1)
        correct = pred == labels
        self._conf.append(conf.cpu().numpy())
        self._correct.append(correct.cpu().numpy())
        self._nll += float(-logp.gather(1, labels[:, None]).sum())
        self._entropy += float(-(p * logp).sum())
        self.batch_n.append(len(labels))
        self.batch_correct.append(int(correct.sum()))
        self.n += len(labels)

    def result(self) -> dict:
        if self.n == 0:
            return {"n": 0, "accuracy": float("nan"), "ece": float("nan"), "nll": float("nan"), "entropy": float("nan")}
        conf, correct = np.concatenate(self._conf), np.concatenate(self._correct)
        return {"n": self.n, "accuracy": float(correct.mean()),
                "ece": float(top_label_ece(conf, correct, self.num_bins)),
                "nll": self._nll / self.n, "entropy": self._entropy / self.n}


class OnlineEval:
    """Evaluate several calibrators (gates) and the two members on one stream, segment by segment.

    gates: {series name: calibrator}. Series "large" and "small" (the members) are always evaluated.
    """

    def __init__(self, gates: dict, num_bins: int = 15, device: torch.device | str | None = None):
        if {"large", "small"} & set(gates):
            raise ValueError("'large' and 'small' are reserved series names")
        self.gates = gates
        self.num_bins = num_bins
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rows: list[dict] = []                         # one per (segment, series)
        self.records: dict[str, list[dict]] = {}           # per-proxy-batch records of gates that keep them
        self.curves: dict[str, dict] = {}                  # series -> {"batch_n": [...], "batch_correct": [...]}
        self.segment_batches: list[tuple[str, int]] = []   # (segment label, first batch index) for the curves
        self._metrics: dict[str, StreamMetrics] = {}
        self._dataset: ShiftDataset | None = None
        self._label = ""
        self._first_segment = True
        for g in gates.values():
            if hasattr(g, "records"):
                g.records = []

    @property
    def series(self) -> list[str]:
        return ["large", "small", *self.gates]

    def start_segment(self, label: str, dataset: ShiftDataset, *, reset: bool, n_samples: int,
                      stream_total: int | None = None) -> None:
        """reset=True: the members were just reset, so gates reset too (episodic). Otherwise the gates keep
        their state and only learn the new label for their records; `stream_total` (the whole stream's
        sample count) is passed at the first segment so the gate can flush its last partial proxy batch."""
        self._dataset, self._label = dataset, label
        self._metrics = {name: StreamMetrics(self.num_bins) for name in self.series}
        n_batches = len(next(iter(self.curves.values()))["batch_n"]) if self.curves else 0
        self.segment_batches.append((label, n_batches))
        for g in self.gates.values():
            if reset or self._first_segment:
                if hasattr(g, "set_corruption"):
                    g.set_corruption(label, total_samples=n_samples if reset else stream_total)
            elif hasattr(g, "mark_segment"):
                g.mark_segment(label)
        self._first_segment = False

    @torch.no_grad()
    def update(self, z_l: torch.Tensor, z_s: torch.Tensor, labels: torch.Tensor) -> None:
        ds = self._dataset
        z_l = ds.apply_class_mask(z_l.to(self.device).float())
        z_s = ds.apply_class_mask(z_s.to(self.device).float())
        labels = labels.to(self.device).long()
        self._metrics["large"].update(z_l, labels)
        self._metrics["small"].update(z_s, labels)
        for name, gate in self.gates.items():
            if hasattr(gate, "set_labels"):
                gate.set_labels(labels)
            self._metrics[name].update(gate.calibrate(z_l, z_s), labels)

    def end_segment(self) -> list[dict]:
        rows = []
        for name, m in self._metrics.items():
            rows.append({"segment": self._label, "series": name, **m.result()})
            curve = self.curves.setdefault(name, {"batch_n": [], "batch_correct": []})
            curve["batch_n"] += m.batch_n
            curve["batch_correct"] += m.batch_correct
        for name, gate in self.gates.items():
            if getattr(gate, "records", None) is not None:
                self.records.setdefault(name, []).extend(gate.records)
                gate.records = []
        self.rows += rows
        return rows

    def window_accuracy(self, series: str, window: int) -> np.ndarray:
        """Accuracy per `window` consecutive samples (rounded to whole batches), for stream plots."""
        c = self.curves[series]
        n, k = np.asarray(c["batch_n"]), np.asarray(c["batch_correct"])
        per = max(1, int(round(window / max(1, int(n.mean())))))
        m = len(n) // per * per
        return k[:m].reshape(-1, per).sum(1) / n[:m].reshape(-1, per).sum(1)


if __name__ == "__main__":
    from src.calibrators.joint_fixed_TS import JointFixedTS
    from src.experiments.gates import MemberOnly
    from src.utils.metrics import get_metrics_dict

    torch.manual_seed(0)

    class Toy(ShiftDataset):          # a dataset whose class mask keeps classes 100..299
        name, config_key, num_classes = "toy", "x", 200
        class_mask = list(range(100, 300))

        def torch_dataset(self):
            raise NotImplementedError

    class Plain(Toy):
        num_classes, class_mask = 1000, None

    def logits(n, k, sharp):
        z = torch.randn(n, k) * 1.5
        z[torch.arange(n), torch.randint(0, k, (n,))] += sharp
        return z

    n = 700
    z_l, z_s = logits(n, 1000, 6.0), logits(n, 1000, 3.0)
    labels = torch.where(torch.rand(n) < 0.6, z_l.argmax(1), torch.randint(0, 1000, (n,)))

    # StreamMetrics == the old get_metrics_dict on the full probability matrix
    sm = StreamMetrics(15)
    for i in range(0, n, 64):
        sm.update(z_l[i:i + 64], labels[i:i + 64])
    old = get_metrics_dict(z_l.softmax(1), labels)
    new = sm.result()
    assert new["n"] == n and abs(new["accuracy"] - old["accuracy"]) < 1e-9
    assert abs(new["ece"] - old["ece"]) < 1e-6 and abs(new["nll"] - old["nll"]) < 1e-3, (new, old)

    # OnlineEval: members + a gate over two segments; the gate state resets only when told to
    ts = JointFixedTS(Tl=1.3, Ts=0.8, verbose=False)
    ev = OnlineEval({"fixed": ts, "large_only": MemberOnly("large")}, device="cpu")
    ds = Plain("x")
    for label in ("a/s5", "b/s5"):
        ev.start_segment(label, ds, reset=True, n_samples=n)
        for i in range(0, n, 64):
            ev.update(z_l[i:i + 64], z_s[i:i + 64], labels[i:i + 64])
        rows = ev.end_segment()
    assert [r["series"] for r in rows] == ["large", "small", "fixed", "large_only"]
    by = {r["series"]: r for r in rows}
    assert by["large"]["accuracy"] == by["large_only"]["accuracy"] and by["large"]["n"] == n
    want = (0.5 * z_l / 1.3 + 0.5 * z_s / 0.8).argmax(1).eq(labels).float().mean().item()
    assert abs(by["fixed"]["accuracy"] - want) < 1e-9
    assert len(ev.rows) == 8 and [b for _, b in ev.segment_batches] == [0, 11]
    w = ev.window_accuracy("large", 128)
    assert w.shape == (2 * 11 // 2,) and 0 <= w.min() and w.max() <= 1

    # the class mask is applied before scoring (ImageNet-A/R): labels live in 0..199
    ev2 = OnlineEval({}, device="cpu")
    toy = Toy("x")
    ev2.start_segment("toy", toy, reset=True, n_samples=64)
    zz = torch.randn(64, 1000)
    ll = torch.randint(0, 200, (64,))
    ev2.update(zz, zz, ll)
    assert abs(ev2.end_segment()[0]["accuracy"] - zz[:, 100:300].argmax(1).eq(ll).float().mean().item()) < 1e-9
    print("evaluate self-test passed")
