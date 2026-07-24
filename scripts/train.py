#!/usr/bin/env python3
"""Pre-training entry point -- ternary language model.

Usage
-----
.. code-block:: bash

    # Single GPU
    python scripts/train.py --config scripts/train_tiny.yaml

    # Multi-GPU via accelerate
    accelerate launch scripts/train.py --config scripts/train_tiny.yaml

    # Torchrun (multi-node)
    torchrun --nproc_per_node=8 scripts/train.py --config scripts/train_one_gb.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the package is importable when running from the repo root
_proj_root = Path(__file__).resolve().parents[1]
_src = _proj_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ternair.training.config import TrainingConfig
from ternair.training.trainer import (
    _train_impl_custom_data,
    build_model,
    freeze_and_export,
)
from ternair.training.optimizer import create_optimizer
from ternair.training.scheduler import WSDScheduler
from ternair.training.utils import setup_logging
from ternair.training.data import build_dataloader

_LOGGER = logging.getLogger("ternair.train")


def _load_config(path: str) -> TrainingConfig:
    """Load YAML config into :class:`TrainingConfig`."""
    try:
        import yaml
    except ImportError:
        raise ImportError("pip install pyyaml to use YAML configs")

    with open(path) as f:
        raw = yaml.safe_load(f)
    return TrainingConfig(**raw)


def _train_impl_custom_data(
    cfg: TrainingConfig,
    model,
    optimizer,
    scheduler,
    dataloader,
    accelerator,
) -> int:
    """Custom training loop that uses ``accelerate.prepare``."""
    from ternair.training.trainer import train_one_epoch

    return train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        accelerator=accelerator,
        global_step=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ternair pre-training")
    parser.add_argument("--config", type=str, default="scripts/train_tiny.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir

    setup_logging("INFO" if "log_level" not in cfg.to_dict() else "INFO")

    _LOGGER.info("Initializing model (profile=%s, storage=%s) …", cfg.model_profile, cfg.model_storage)
    model = build_model(cfg)
    _LOGGER.info("Model built.  Ternary params: %s", model.count_parameters() // 1_000_000)

    _LOGGER.info("Setting up optimizer with decoupled weight decay …")
    optimizer = create_optimizer(
        model,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.epsilon,
    )

    scheduler = WSDScheduler(
        optimizer,
        total_steps=cfg.total_steps,
        warmup_steps=cfg.warmup_steps,
        stable_steps=cfg.stable_steps,
        decay_steps=cfg.decay_steps,
        min_lr=cfg.learning_rate * cfg.min_lr_ratio,
        decay_type=cfg.decay_type,  # type: ignore[arg-type]
    )

    _LOGGER.info("Building dataloader …")
    dataloader = build_dataloader(cfg)

    # ------------------------------------------------------------------
    # Accelerate initialisation
    # ------------------------------------------------------------------
    try:
        from accelerate import Accelerator

        accelerator = Accelerator()
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
    except ImportError:
        accelerator = None
        _LOGGER.warning("accelerate not installed; running on single device")
        model = model.to("cpu")

    _LOGGER.info(
        "Starting training  lr=%.1e  wd=%.1e  max_steps=%d  bs=%d  grad_acc=%d",
        cfg.learning_rate,
        cfg.weight_decay,
        cfg.max_train_steps,
        cfg.batch_size,
        cfg.gradient_accumulation_steps,
    )

    _train_impl_custom_data(cfg, model, optimizer, scheduler, dataloader, accelerator)

    # Export final model with frozen packed storage
    output_model = os.path.join(cfg.output_dir, "final_model.pt")
    freeze_and_export(model, output_model)
    _LOGGER.info("Done!")


if __name__ == "__main__":
    main()
