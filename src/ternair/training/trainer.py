"""Extended trainer — full pre-training loop with accelerate."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ternair.model.modeling import TernairForCausalLM
from ternair.model.size_profiles import tiny_profile, base_profile, one_gb_profile
from ternair.training.config import TrainingConfig
from ternair.training.optimizer import create_optimizer, clip_gradients
from ternair.training.scheduler import WSDScheduler

_LOGGER = logging.getLogger(__name__)

PROFILE_REGISTRY = {
    "tiny": tiny_profile,
    "base": base_profile,
    "one_gb": one_gb_profile,
}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard next-token cross-entropy (shifted inside)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = targets[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
    )


# ---------------------------------------------------------------------------
# Original one-step helper (kept for backward compat / smoke tests)
# ---------------------------------------------------------------------------

def train_one_step(
    model: TernairForCausalLM,
    input_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, torch.Tensor]:
    """One forward + backward + (optional) optimizer step.

    Returns the loss value (float) and the loss tensor.
    """
    model.train()
    logits = model(input_ids)
    loss = cross_entropy(logits, input_ids)
    loss.backward()
    if optimizer is not None:
        optimizer.step()
        optimizer.zero_grad()
    return float(loss.item()), loss.detach()


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(cfg: TrainingConfig) -> TernairForCausalLM:
    profile_fn = PROFILE_REGISTRY.get(cfg.model_profile)
    if profile_fn is None:
        raise ValueError(f"Unknown profile {cfg.model_profile!r}")
    model_cfg = profile_fn(storage=cfg.model_storage)
    return TernairForCausalLM(model_cfg)


# ---------------------------------------------------------------------------
# Full training loop (accelerate-ready)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: TernairForCausalLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WSDScheduler | None,
    cfg: TrainingConfig,
    accelerator,
    global_step: int = 0,
) -> int:
    """Run the training loop over a data loader.

    Uses ``accelerator.backward()`` and gradient clipping.  Returns
    the final ``global_step``.
    """
    model.train()
    log_interval_loss = 0.0
    best_loss = float("inf")

    for step, batch in enumerate(dataloader):
        if global_step >= cfg.max_train_steps:
            break

        input_ids = batch["input_ids"]
        with accelerator.accumulate(model):
            logits = model(input_ids)
            loss = cross_entropy(logits, input_ids)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                grad_norm = clip_gradients(model, max_norm=cfg.max_grad_norm)
                if grad_norm > cfg.max_grad_norm * 5:
                    _LOGGER.warning(
                        "Large grad norm at step %d: %.2f", global_step, grad_norm
                    )

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        log_interval_loss += loss.item()

        if global_step % cfg.log_every == 0 and accelerator.is_main_process:
            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
            avg_loss = log_interval_loss / max(cfg.log_every, 1)
            _LOGGER.info(
                "step=%d  loss=%.4f  lr=%.1e  grad_norm=%.2f",
                global_step, avg_loss, current_lr, grad_norm,
            )
            log_interval_loss = 0.0

        if global_step % cfg.eval_every == 0 and global_step > 0:
            eval_loss = evaluate(model, dataloader, cfg, accelerator)
            model.train()
            if accelerator.is_main_process:
                _LOGGER.info("eval step=%d  eval_loss=%.4f", global_step, eval_loss)
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    _save_checkpoint(model, optimizer, scheduler, global_step, cfg, tag="best")

        if global_step % cfg.save_every == 0 and global_step > 0 and accelerator.is_main_process:
            _save_checkpoint(model, optimizer, scheduler, global_step, cfg, tag=f"step_{global_step}")

        global_step += 1

    return global_step


@torch.no_grad()
def evaluate(
    model: TernairForCausalLM,
    dataloader: DataLoader,
    cfg: TrainingConfig,
    accelerator,
) -> float:
    """Average cross-entropy loss over ``cfg.eval_steps``."""
    model.eval()
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(dataloader):
        if i >= cfg.eval_steps:
            break
        input_ids = batch["input_ids"]
        logits = model(input_ids)
        loss = cross_entropy(logits, input_ids)
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


def _save_checkpoint(
    model: TernairForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    step: int,
    cfg: TrainingConfig,
    tag: str,
) -> None:
    save_dir = os.path.join(cfg.output_dir, tag)
    os.makedirs(save_dir, exist_ok=True)
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": cfg.to_dict(),
    }
    torch.save(state, os.path.join(save_dir, "training_state.pt"))


def freeze_and_export(model: TernairForCausalLM, output_path: str) -> None:
    """Freeze all TernairLinear layers to packed storage and save."""
    model.freeze_storage()
    model.eval()
    torch.save(model.state_dict(), output_path)
    _LOGGER.info("Frozen model saved to %s", output_path)
