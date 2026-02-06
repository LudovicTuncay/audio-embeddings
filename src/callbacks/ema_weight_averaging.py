from typing import Any, Optional, Union

import torch
from torch.optim.swa_utils import get_ema_avg_fn
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from src.callbacks.lightning_weight_averaging import WeightAveraging


class WarmupEMAWeightAveraging(WeightAveraging):
    def __init__(
        self,
        warmup_pct: float,
        enabled: bool = True,
        decay: Optional[float] = None,
        decay_numerator: float = 20.0,
        update_every_n_steps: int = 1,
        update_starting_at_step: Optional[int] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        use_buffers: bool = True,
    ) -> None:
        super().__init__(device=device, use_buffers=use_buffers)
        self.enabled = enabled
        self.warmup_pct = warmup_pct
        self.decay = decay
        self.decay_numerator = decay_numerator
        self.update_every_n_steps = update_every_n_steps
        self.update_starting_at_step = update_starting_at_step

        self.resolved_decay: Optional[float] = None
        self.resolved_start_step: Optional[int] = None

    def setup(self, trainer, pl_module, stage: str) -> None:
        if not self.enabled:
            rank_zero_info("WarmupEMAWeightAveraging is disabled by config.")
            return

        if stage != "fit":
            return super().setup(trainer, pl_module, stage)

        if trainer.max_steps and trainer.max_steps > 0:
            total_steps = trainer.max_steps
        else:
            total_steps = trainer.estimated_stepping_batches

        if total_steps <= 0:
            total_steps = 100000

        warmup_steps = int(total_steps * self.warmup_pct)
        if self.update_starting_at_step is None:
            self.resolved_start_step = warmup_steps
        else:
            self.resolved_start_step = self.update_starting_at_step

        if self.decay is None:
            active_steps = max(1, total_steps - self.resolved_start_step)
            computed_decay = 1.0 - (self.decay_numerator / active_steps)
            self.resolved_decay = min(0.99999, max(0.9, computed_decay))
        else:
            self.resolved_decay = self.decay

        self._kwargs["avg_fn"] = get_ema_avg_fn(decay=self.resolved_decay)

        super().setup(trainer, pl_module, stage)

        rank_zero_info(
            "WarmupEMAWeightAveraging configured: "
            f"total_steps={total_steps}, warmup_steps={warmup_steps}, "
            f"start_step={self.resolved_start_step}, decay={self.resolved_decay:.8f}"
        )

    def should_update(
        self, step_idx: Optional[int] = None, epoch_idx: Optional[int] = None
    ) -> bool:
        if step_idx is None:
            return False

        if not self.enabled:
            return False

        if self.resolved_start_step is None:
            return False

        if step_idx < self.resolved_start_step:
            return False

        if self.update_every_n_steps <= 0:
            return False

        return step_idx % self.update_every_n_steps == 0

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["resolved_decay"] = self.resolved_decay
        state["resolved_start_step"] = self.resolved_start_step
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.resolved_decay = state_dict.get("resolved_decay")
        self.resolved_start_step = state_dict.get("resolved_start_step")
