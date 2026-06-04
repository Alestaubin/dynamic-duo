from abc import ABC, abstractmethod
import torch.nn as nn

class BaseCalibrator(nn.Module, ABC):
    """Parent class for all single-model calibrators."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def tune(self, *args, **kwargs):
        pass

    @abstractmethod
    def calibrate(self, logits):
        pass

class BaseJointCalibrator(BaseCalibrator):
    """Parent class for all Duo-based calibrators (Naive or PTS)."""
    @abstractmethod
    def calibrate(self, logits_l, logits_s):
        pass
    @abstractmethod
    def calibrate_with_grad(self, logits_l, logits_s):
        pass


class IdentityCalibrator(BaseCalibrator):
    def calibrate(self, logits):
        return logits
    def tune(self, *args, **kwargs):
        pass

class _NoOpModule(nn.Module):
    def eval(self): return self
