import os
import json
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.calibrators.base import BaseJointCalibrator, _NoOpModule
from src.utils.logit_transforms import combine_logits

class JointFixedTS(BaseJointCalibrator):
    """
    Double model naive calibrator as seen in the Asymmetric Duos paper. Finds the two temperatures T_s, T_l that minimize the NLL on the validation set. 
    """
    def __init__(self, Tl=None, Ts=None, verbose=True):
        super().__init__()
        self.Tl = nn.Parameter(torch.ones(1)) if Tl is None else nn.Parameter(torch.tensor([Tl]))
        self.Ts = nn.Parameter(torch.ones(1)) if Ts is None else nn.Parameter(torch.tensor([Ts]))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

    def tune(self, logits_l, logits_s, labels, grid_n: int = 50_000,
             t_min: float = 0.05, t_max: float = 50.0, grid_steps: int = 25,):
        self.to(self.device)
        logits_l, logits_s = logits_l.to(self.device), logits_s.to(self.device)
        labels = labels.to(self.device).long()

        # 1. Grid search on a random subset to initialize L-BFGS.
        N = len(labels)
        if N > grid_n:
            idx = torch.randperm(N, device=self.device)[:grid_n]
            gl_g, gs_g, y_g = logits_l[idx], logits_s[idx], labels[idx]
        else:
            gl_g, gs_g, y_g = logits_l, logits_s, labels

        best_nll = float("inf")
        best_Tl_g, best_Ts_g = 1.0, 1.0
        t_range = torch.logspace(math.log10(t_min), math.log10(t_max),
                                 grid_steps, device=self.device)

        for tl in t_range:
            for ts in t_range:
                nll = F.cross_entropy(combine_logits(z_l=gl_g, z_s=gs_g, tau_l=tl, tau_s=ts), y_g).item()
                if nll < best_nll:
                    best_nll, best_Tl_g, best_Ts_g = nll, tl.item(), ts.item()

        # Warn if the grid optimum is pinned to a boundary
        for name, val in (("Tl", best_Tl_g), ("Ts", best_Ts_g)):
            if val <= t_min * 1.001 or val >= t_max * 0.999:
                warnings.warn(
                    f"{name} grid optimum ({val:.3f}) sits at the search "
                    f"boundary [{t_min}, {t_max}]; consider widening the range "
                    f"— the duo may want to (de)emphasise one model more strongly."
                )
        if self.verbose:
            print(f"Best temperatures found with grid search: Ts={best_Ts_g:.4f}, Tl={best_Tl_g:.4f}")

        # 2. L-BFGS refinement on full data, in log-temperature space
        log_Tl = torch.tensor([math.log(best_Tl_g)], device=self.device, requires_grad=True)
        log_Ts = torch.tensor([math.log(best_Ts_g)], device=self.device, requires_grad=True)
        optimizer = optim.LBFGS([log_Tl, log_Ts], lr=1.0, max_iter=100,
                                line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(
                combine_logits(z_l=logits_l, z_s=logits_s,
                               tau_l=log_Tl.exp(), tau_s=log_Ts.exp()),
                labels,
            )
            loss.backward()
            return loss
        optimizer.step(closure)

        self.Tl.data = log_Tl.detach().exp()
        self.Ts.data = log_Ts.detach().exp()
        if self.verbose:
            print(f"Joint Naive Optimized: Ts={self.Ts.item():.4f}, Tl={self.Tl.item():.4f}")

    def calibrate(self, logits_l, logits_s):
        self.to(self.device)
        logits_l, logits_s = logits_l.to(self.device), logits_s.to(self.device)
        calibrated_logits = combine_logits(z_l=logits_l, z_s=logits_s, tau_l=self.Tl.item(), tau_s=self.Ts.item())
        return calibrated_logits

    def calibrate_with_grad(self, logits_l, logits_s):
        return self.calibrate(logits_l, logits_s)

    def forward(self, logits_l, logits_s):
        return self.calibrate_with_grad(logits_l, logits_s)
        
    @property
    def model(self): return _NoOpModule()
    
    def save(self, folder: str, trained_on: dict | None = None) -> None:
        """
        Save calibrator to a folder containing:
            config.json  — temperatures + metadata about training data
        
        Parameters
        ----------
        folder     : directory to save into (created if needed)
        trained_on : optional dict describing the training data, e.g.
        """
        os.makedirs(folder, exist_ok=True)

        config = {
            "class_name": "JointFixedTS",
            "T_l":        self.Tl.item(),
            "T_s":        self.Ts.item(),
            "trained_on": trained_on or {},
        }
        with open(os.path.join(folder, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        if self.verbose:
            print(f"Saved JointFixedTS to {folder}/  "
                  f"(T_l={self.Tl.item():.4f}, T_s={self.Ts.item():.4f})")

    @classmethod
    def load(cls, folder: str) -> "JointFixedTS":
        """
        Load a calibrator saved with .save().

        Example
        -------
        calibrator = JointFixedTS.load("checkpoints/naive_ts/clean")
        print(calibrator.trained_on)
        """
        config_path = os.path.join(folder, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json found in {folder}")

        with open(config_path) as f:
            config = json.load(f)

        if config.get("class_name") != "JointFixedTS":
            raise ValueError(
                f"Expected JointFixedTS, got {config.get('class_name')}"
            )

        calibrator = cls()
        calibrator.Tl = nn.Parameter(torch.tensor(config["T_l"]))
        calibrator.Ts = nn.Parameter(torch.tensor(config["T_s"]))
        calibrator.trained_on = config.get("trained_on", {})

        if calibrator.verbose:
            print(f"Loaded JointFixedTS from {folder}/  "
                  f"(T_l={calibrator.Tl.item():.4f}, T_s={calibrator.Ts.item():.4f})")
        if calibrator.trained_on:
            if calibrator.verbose:
                print(f"  trained on: {calibrator.trained_on}")

        return calibrator


class PreScaledCalibrator(BaseJointCalibrator):
    """Applies a frozen JointFixedTS pre-scaling before delegating to inner."""

    def __init__(self, fixed_ts: JointFixedTS, inner: BaseJointCalibrator):
        super().__init__()
        self.fixed_ts = fixed_ts
        self.inner = inner

    def _scale(self, logits_l, logits_s):
        return logits_l / self.fixed_ts.Tl, logits_s / self.fixed_ts.Ts

    def calibrate(self, logits_l, logits_s):
        return self.inner.calibrate(*self._scale(logits_l, logits_s))

    def calibrate_with_grad(self, logits_l, logits_s):
        return self.inner.calibrate_with_grad(*self._scale(logits_l, logits_s))

    def forward(self, logits_l, logits_s):
        return self.calibrate_with_grad(logits_l, logits_s)

    def tune(self, *args, **kwargs):
        return self.inner.tune(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules", {})
            if "inner" in modules:
                return getattr(modules["inner"], name)
            raise
