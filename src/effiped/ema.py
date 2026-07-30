"""Exponential moving average support for EffiPed training."""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


class ModelEMA:
    """Maintain a non-trainable moving average of model parameters."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 2000,
        warmup_power: float = 2.0 / 3.0,
        device: Optional[torch.device] = None,
    ):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.warmup_power = warmup_power
        self.updates = 0
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        if device is not None:
            self.ema.to(device)
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def get_decay(self) -> float:
        if self.updates < self.warmup_steps:
            warmup_ratio = self.updates / self.warmup_steps
            return self.decay * (1 - (1 - warmup_ratio) ** self.warmup_power)
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.get_decay()
        model_state = model.state_dict()
        ema_state = self.ema.state_dict()
        for key, value in ema_state.items():
            if key in model_state and value.dtype.is_floating_point:
                value.mul_(decay).add_(model_state[key], alpha=1 - decay)

    def state_dict(self) -> dict:
        return {
            "ema_state_dict": self.ema.state_dict(),
            "updates": self.updates,
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.ema.load_state_dict(state_dict["ema_state_dict"])
        self.updates = state_dict.get("updates", 0)
        self.decay = state_dict.get("decay", self.decay)

    def __call__(self, *args, **kwargs):
        return self.ema(*args, **kwargs)


def copy_params_to_ema(ema_model: nn.Module, model: nn.Module) -> None:
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for key, value in ema_state.items():
        if key in model_state:
            value.copy_(model_state[key])
