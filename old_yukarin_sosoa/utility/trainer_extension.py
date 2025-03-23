from pathlib import Path
from typing import Any, Dict

import numpy
import wandb
from pytorch_trainer.training import Extension, Trainer
from tensorboardX import SummaryWriter


def _flatten_dict(dd, separator="/", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in _flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


class TensorboardReport(Extension):
    def __init__(self, writer: SummaryWriter = None):
        self.writer = writer

    def __call__(self, trainer: Trainer):
        if self.writer is None:
            self.writer = SummaryWriter(Path(trainer.out))

        observations = trainer.observation
        n_iter = trainer.updater.iteration
        for n, v in observations.items():
            self.writer.add_scalar(n, v, n_iter)

    def finalize(self):
        super().finalize()
        self.writer.flush()


class WandbReport(Extension):
    def __init__(
        self,
        config_dict: Dict[str, Any],
        project_category: str,
        project_name: str,
        output_dir: Path,
    ):
        self.config_dict = config_dict
        self.project_category = project_category
        self.project_name = project_name
        self.output_dir = output_dir

        self.initialized = False
        self.wandb_id = wandb.util.generate_id()

    def __call__(self, trainer: Trainer):
        if not self.initialized:
            self.initialized = True

            wandb.init(
                id=self.wandb_id,
                project=self.project_category,
                name=self.project_name,
                dir=self.output_dir,
                resume="allow",
            )
            wandb.config.update(_flatten_dict(self.config_dict), allow_val_change=True)

        observations = trainer.observation
        n_iter = trainer.updater.iteration
        wandb.log(observations, step=n_iter)

    def state_dict(self):
        state_dict = {"wandb_id": self.wandb_id}
        return state_dict

    def load_state_dict(self, state_dict):
        self.wandb_id = state_dict["wandb_id"]


class NoamShift(Extension):
    def __init__(self, attr, step, init=None, optimizer=None):
        self._attr = attr
        self._step = step
        self._init = init
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        # ensure that _init is set
        if self._init is None:
            self._init = optimizer.param_groups[0][self._attr]
        if self._last_value is not None:
            value = self._last_value
        else:
            value = self._init
        self._update_value(optimizer, value)

    def __call__(self, trainer):
        self._t += 1
        optimizer = self._get_optimizer(trainer)
        value = self._init ** -0.5 * min(self._t ** -0.5, self._step ** -1.5)
        self._update_value(optimizer, value)

    def state_dict(self):
        return {"t": self._t, "last_value": self._last_value}

    def load_state_dict(self, state_dict):
        self._t = state_dict["t"]
        self._last_value = state_dict["last_value"]
        if isinstance(self._last_value, numpy.ndarray):
            self._last_value = self._last_value.item()

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer("main")

    def _update_value(self, optimizer, value):
        for param_group in optimizer.param_groups:
            param_group[self._attr] = value
        self._last_value = value
