"""Lightning callbacks for Z3 SYNAPSE training.

Provides:
    - EMACallback: Exponential moving average shadow model (mirrors baselines EMAModel)
    - PostTrainEvalCallback: Runs the full evaluation protocol after training completes
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import pytorch_lightning as pl

from synapse.synapse_arch.model import Z3TopologyFirstModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EMA Callback
# ---------------------------------------------------------------------------

class EMACallback(pl.Callback):
    """Exponential moving average of model parameters.

    Maintains a shadow copy of the model weights that is updated as an
    exponential moving average of the live training weights.  At
    validation/test time, the EMA weights are swapped in, then restored
    after evaluation completes.

    Parameters
    ----------
    decay : float
        EMA decay factor (0.999 = slow average, 0.9 = fast).
    every_n_steps : int
        Update EMA every N training steps.
    """

    def __init__(self, decay: float = 0.999, every_n_steps: int = 1) -> None:
        super().__init__()
        self.decay = decay
        self.every_n_steps = every_n_steps
        self._shadow: dict[str, torch.Tensor] | None = None
        self._backup: dict[str, torch.Tensor] | None = None

    def _init_shadow(self, pl_module: pl.LightningModule) -> None:
        """Initialize shadow weights from current model state."""
        model = pl_module.model
        self._shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

    def _update_shadow(self, pl_module: pl.LightningModule) -> None:
        """Update shadow weights with EMA of current weights."""
        if self._shadow is None:
            self._init_shadow(pl_module)
            return

        model = pl_module.model
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self._shadow:
                    self._shadow[name].mul_(self.decay).add_(
                        param.detach(), alpha=1.0 - self.decay
                    )

    def _swap_in_ema(self, pl_module: pl.LightningModule) -> None:
        """Replace model weights with EMA shadow weights."""
        model = pl_module.model
        self._backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        if self._shadow is not None:
            model.load_state_dict(self._shadow, strict=False)

    def _swap_out_ema(self, pl_module: pl.LightningModule) -> None:
        """Restore original model weights."""
        if self._backup is not None:
            pl_module.model.load_state_dict(self._backup, strict=False)
            self._backup = None

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        global_step = trainer.global_step
        if global_step % self.every_n_steps == 0:
            self._update_shadow(pl_module)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._swap_in_ema(pl_module)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._swap_out_ema(pl_module)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._swap_in_ema(pl_module)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._swap_out_ema(pl_module)

    def state_dict(self) -> dict[str, Any]:
        return {"shadow": self._shadow, "decay": self.decay}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._shadow = state_dict["shadow"]


# ---------------------------------------------------------------------------
# Post-Training Evaluation Callback
# ---------------------------------------------------------------------------

class PostTrainEvalCallback(pl.Callback):
    """Run the full evaluation protocol after training completes.

    Mirrors the baselines rollout pattern: after training, automatically
    runs evaluation (metrics, topology alignment, robustness sweeps,
    rollout) and generates reports.

    Parameters
    ----------
    dataset_name : str
        Dataset name for evaluator selection.
    output_root : Path or None
        Root output directory.  If None, uses ``trainer.default_root_dir``.
    """

    def __init__(
        self,
        dataset_name: str = "synthetic",
        output_root: Path | None = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self._output_root = output_root

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        from synapse.synapse.training.evaluation.hooks import run_post_training_evaluation

        output_root = Path(self._output_root or trainer.default_root_dir)
        model = pl_module.model
        config = pl_module.config

        # Get test loader from datamodule
        datamodule = trainer.datamodule
        if datamodule is None or datamodule.test_loader is None:
            log.warning("No test dataloader available — skipping post-training evaluation")
            return

        test_loader = datamodule.test_loader
        bundle = datamodule.bundle

        try:
            eval_results = run_post_training_evaluation(
                model, config, test_loader, bundle, output_root, self.dataset_name,
            )
            log.info("Post-training evaluation complete. Reports at %s", eval_results.get("report_dir"))
        except Exception as e:
            log.warning("Post-training evaluation failed: %s", e)
