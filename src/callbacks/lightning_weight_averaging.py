# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Vendored from Lightning-AI/pytorch-lightning commit:
9bcba1c1e82b45e10f948dc28fc12f4cf04ab736

Source:
https://github.com/Lightning-AI/pytorch-lightning/blob/9bcba1c1e82b45e10f948dc28fc12f4cf04ab736/src/lightning/pytorch/callbacks/weight_averaging.py
"""

import itertools
from copy import deepcopy
from typing import Any, Optional, Union

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn
from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
from lightning.pytorch.utilities.types import STEP_OUTPUT


class WeightAveraging(Callback):
    def __init__(
        self,
        device: Optional[Union[torch.device, str, int]] = None,
        use_buffers: bool = True,
        **kwargs: Any,
    ) -> None:
        if isinstance(device, str):
            self._device: Optional[Union[torch.device, int]] = torch.device(device)
        else:
            self._device = device
        self._use_buffers = use_buffers
        self._kwargs = kwargs

        self._average_model: Optional[AveragedModel] = None
        self._latest_update_step = 0
        self._latest_update_epoch = -1

    def should_update(
        self, step_idx: Optional[int] = None, epoch_idx: Optional[int] = None
    ) -> bool:
        return step_idx is not None

    @override
    def setup(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        if stage == "fit":
            device = self._device or pl_module.device

            if is_overridden("configure_model", pl_module):
                rank_zero_warn(
                    "You're using the WeightAveraging callback with a model that overrides the configure_model "
                    "callback. WeightAveraging doesn't support sharding model layers, so you may run out of memory."
                )
                pl_module.configure_model()

            self._average_model = AveragedModel(
                model=pl_module,
                device=device,
                use_buffers=self._use_buffers,
                **self._kwargs,
            )

    @override
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step_idx = trainer.global_step - 1
        if (trainer.global_step > self._latest_update_step) and self.should_update(
            step_idx=step_idx
        ):
            assert self._average_model is not None
            self._average_model.update_parameters(pl_module)
            self._latest_update_step = trainer.global_step

    @override
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if (trainer.current_epoch > self._latest_update_epoch) and self.should_update(
            epoch_idx=trainer.current_epoch
        ):
            assert self._average_model is not None
            self._average_model.update_parameters(pl_module)
            self._latest_update_epoch = trainer.current_epoch

    @override
    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        assert self._average_model is not None
        self._copy_average_to_current(pl_module)

    @override
    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self._average_model is not None:
            self._swap_models(pl_module)

    @override
    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self._average_model is not None:
            self._swap_models(pl_module)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"latest_update_step": self._latest_update_step}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._latest_update_step = state_dict["latest_update_step"]

    @override
    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: dict[str, Any],
    ) -> None:
        if self._average_model is None:
            rank_zero_info(
                "You're using the WeightAveraging callback, but saving a checkpoint outside the 'fit' stage. The state "
                "of the WeightAveraging callback won't be saved in the checkpoint. If training has finished, the "
                "average model parameters will be saved to the state_dict in the checkpoint."
            )
        else:
            average_model_state = self._average_model.state_dict()
            checkpoint["current_model_state"] = checkpoint["state_dict"]
            checkpoint["state_dict"] = {
                name[7:]: value
                for name, value in average_model_state.items()
                if name.startswith("module.")
            }
            checkpoint["averaging_state"] = {
                name: value
                for name, value in average_model_state.items()
                if not name.startswith("module.")
            }

    @override
    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: dict[str, Any],
    ) -> None:
        if self._average_model is None:
            rank_zero_warn(
                "You're using the WeightAveraging callback, but loading a checkpoint outside the 'fit' stage. The "
                "WeightAveraging state cannot be restored. If you're using the checkpoint for prediction or testing, "
                "you can ignore this warning. To disable the warning, remove the WeightAveraging callback."
            )
        elif ("current_model_state" in checkpoint) and (
            "averaging_state" in checkpoint
        ):
            rank_zero_info(
                "Found current_model_state in the checkpoint. This will be used to initialize the model."
            )
            average_model_state = {
                "module." + name: value
                for name, value in checkpoint["state_dict"].items()
            }
            average_model_state |= checkpoint["averaging_state"]
            self._average_model.load_state_dict(average_model_state)
            pl_module.load_state_dict(checkpoint["current_model_state"])
        else:
            rank_zero_warn(
                "The checkpoint was not created with WeightAveraging. Both the current and the average model will be "
                "initialized with state_dict."
            )
            self._average_model.module.load_state_dict(
                deepcopy(checkpoint["state_dict"]), strict=False
            )

    def _swap_models(self, pl_module: "pl.LightningModule") -> None:
        assert self._average_model is not None
        average_params = itertools.chain(
            self._average_model.module.parameters(),
            self._average_model.module.buffers(),
        )
        current_params = itertools.chain(pl_module.parameters(), pl_module.buffers())
        for average_param, current_param in zip(average_params, current_params):
            tmp = average_param.data.clone()
            average_param.data.copy_(current_param.data)
            current_param.data.copy_(tmp)

    def _copy_average_to_current(self, pl_module: "pl.LightningModule") -> None:
        assert self._average_model is not None
        average_params = itertools.chain(
            self._average_model.module.parameters(),
            self._average_model.module.buffers(),
        )
        current_params = itertools.chain(pl_module.parameters(), pl_module.buffers())
        for average_param, current_param in zip(average_params, current_params):
            current_param.data.copy_(average_param.data)


class EMAWeightAveraging(WeightAveraging):
    def __init__(
        self,
        device: Optional[Union[torch.device, str, int]] = None,
        use_buffers: bool = True,
        decay: float = 0.999,
        update_every_n_steps: int = 1,
        update_starting_at_step: Optional[int] = None,
        update_starting_at_epoch: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(
            device=device,
            use_buffers=use_buffers,
            **kwargs,
            avg_fn=get_ema_avg_fn(decay=decay),
        )

        self.update_every_n_steps = update_every_n_steps
        self.update_starting_at_step = update_starting_at_step
        self.update_starting_at_epoch = update_starting_at_epoch

    def should_update(
        self, step_idx: Optional[int] = None, epoch_idx: Optional[int] = None
    ) -> bool:
        if step_idx is not None:
            meets_step_requirement = (
                self.update_starting_at_step is None
                or step_idx >= self.update_starting_at_step
            )
            meets_step_frequency = (
                self.update_every_n_steps > 0
                and step_idx % self.update_every_n_steps == 0
            )
            if meets_step_requirement and meets_step_frequency:
                return True

        if epoch_idx is not None:
            meets_epoch_requirement = (
                self.update_starting_at_epoch is not None
                and epoch_idx >= self.update_starting_at_epoch
            )
            if meets_epoch_requirement:
                return True

        return False
