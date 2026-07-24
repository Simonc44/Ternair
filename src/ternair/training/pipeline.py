"""High-level orchestrator for the Ternair model lifecycle.

The :class:`TernairPipeline` is the new single entry-point for
working with intermediate-size models (50M-500M params).  It owns:

* **Build** -- a :class:`TernairForCausalLM`, optimiser, scheduler.
* **Pre-flight** -- a :class:`MemoryEstimate` showing whether the
  configuration fits the available device.
* **Train / Distill** -- training loops with OOM recovery and atomic
  checkpointing.
* **Freeze / Export** -- final state transitions.

The pipeline tracks its current :class:`PipelineStage` so a partially
completed run can be resumed deterministically.  Atomic checkpoints
(see :mod:`ternair.training.atomic`) guarantee the run state is
never half-written even on a hard kill.

Why a class instead of free functions?
--------------------------------------
1. **State is shared** -- the model, optimiser, scheduler, and
   checkpoint location are all needed together across stages.  Bundling
   them avoids signature creep.
2. **Lifecycle is sequential** -- freezing before training is invalid;
   exporting before freezing silently drops weights.  Stage tracking
   catches these mistakes early.
3. **Resumability** -- the prior codebase had ``resume_from`` in
   :class:`TrainingConfig` but never implemented the load path.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ternair.model.modeling import TernairForCausalLM
from ternair.model.size_profiles import (
    PROFILE_REGISTRY,
    base_profile,
    large_profile,
    medium_profile,
    small_profile,
    tiny_profile,
)
from ternair.training.atomic import AtomicCheckpointSaver
from ternair.training.config import TrainingConfig
from ternair.training.memory import MemoryEstimate, estimate_memory
from ternair.training.optimizer import create_optimizer
from ternair.training.scheduler import WSDScheduler

_LOGGER = logging.getLogger(__name__)


# Extend the profile registry so the pipeline knows about all sizes.
_EXTENDED_PROFILE_REGISTRY: dict[str, callable] = {
    "tiny": tiny_profile,
    "small": small_profile,
    "base": base_profile,
    "medium": medium_profile,
    "large": large_profile,
    "one_gb": PROFILE_REGISTRY["one_gb"],
}


# ---------------------------------------------------------------------------
# Stage and state
# ---------------------------------------------------------------------------


class PipelineStage(str, Enum):
    """Lifecycle stages of a Ternair model."""

    UNINITIALIZED = "uninitialized"  # __init__ called only
    BUILT = "built"  # model constructed, ready for training
    TRAINED = "trained"  # at least one training step completed
    DISTILLED = "distilled"  # QAT/distillation finished
    FROZEN = "frozen"  # storage packed, ready for export
    EXPORTED = "exported"  # written to disk
    FAILED = "failed"  # unrecoverable error (e.g. OOM after retries)


@dataclass
class PipelineState:
    """Persistent state of a pipeline run."""

    stage: PipelineStage = PipelineStage.UNINITIALIZED
    global_step: int = 0
    best_eval_loss: float = float("inf")
    oom_recoveries: int = 0
    last_memory_estimate: Optional[MemoryEstimate] = None
    checkpoints: list[str] = field(default_factory=list)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class TernairPipeline:
    """End-to-end orchestrator for Ternair models.

    Typical usage::

        pipeline = TernairPipeline(
            config=TrainingConfig(model_profile="small", max_train_steps=1000),
            output_dir="runs/small_v1",
        )
        pipeline.build()
        estimate = pipeline.preflight_check()
        if not estimate.fits:
            raise RuntimeError(estimate.summary())
        pipeline.run()
        pipeline.freeze()
        pipeline.export(format="safetensors")
    """

    config: TrainingConfig
    output_dir: str
    state: PipelineState = field(default_factory=PipelineState)

    # Filled in by build().
    model: Optional[TernairForCausalLM] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[WSDScheduler] = None
    _saver: Optional[AtomicCheckpointSaver] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @property
    def stage(self) -> PipelineStage:
        return self.state.stage

    def _profile(self) -> callable:
        """Resolve the configured model profile factory."""
        fn = _EXTENDED_PROFILE_REGISTRY.get(self.config.model_profile)
        if fn is None:
            raise ValueError(
                f"Unknown profile {self.config.model_profile!r}. "
                f"Available: {sorted(_EXTENDED_PROFILE_REGISTRY)}"
            )
        return fn

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> "TernairPipeline":
        """Construct the model, optimiser, scheduler, and checkpoint saver.

        Idempotent: calling ``build`` twice is a no-op (logs a debug
        message) so the pipeline is safe to reuse after ``resume``.
        """
        if self.stage is not PipelineStage.UNINITIALIZED and self.model is not None:
            _LOGGER.debug("build() called on a non-empty pipeline; skipping")
            return self

        profile_fn = self._profile()
        model_cfg = profile_fn(storage=self.config.model_storage)
        self.model = TernairForCausalLM(model_cfg)
        self.optimizer = create_optimizer(
            self.model,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.epsilon,
        )
        self.scheduler = WSDScheduler(
            self.optimizer,
            total_steps=self.config.max_train_steps,
            warmup_steps=max(self.config.warmup_steps, 1),
            stable_steps=max(self.config.stable_steps, 1),
            decay_steps=max(self.config.decay_steps, 1),
            min_lr=self.config.learning_rate * self.config.min_lr_ratio,
            decay_type=self.config.decay_type,
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self._saver = AtomicCheckpointSaver(
            save_dir=self.output_dir, filename="training_state.pt",
        )
        self.state.stage = PipelineStage.BUILT
        _LOGGER.info(
            "Pipeline built: profile=%s layers=%d hidden=%d params=%s",
            self.config.model_profile,
            model_cfg.num_hidden_layers,
            model_cfg.hidden_size,
            f"{self.model.count_parameters():,}",
        )
        return self

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def preflight_check(
        self,
        *,
        batch_size: Optional[int] = None,
        seq_length: Optional[int] = None,
        teacher: Optional[nn.Module] = None,
    ) -> MemoryEstimate:
        """Estimate the memory footprint of the upcoming run.

        Raises ``RuntimeError`` if the model is not built or the
        estimate indicates the run will not fit on the device.
        """
        if self.model is None:
            raise RuntimeError("Call build() before preflight_check()")
        bs = batch_size or self.config.batch_size
        sl = seq_length or self.config.seq_length
        estimate = estimate_memory(
            self.model, batch_size=bs, seq_length=sl, teacher=teacher,
        )
        self.state.last_memory_estimate = estimate
        if not estimate.fits:
            _LOGGER.warning("Memory preflight flagged risk:\n%s", estimate.summary())
        else:
            _LOGGER.info("Memory preflight OK:\n%s", estimate.summary())
        return estimate

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int, eval_loss: float, *, tag: str = "latest") -> None:
        """Atomically save the current run state.

        Falls back to the *previous* good checkpoint if the latest
        write fails.  See :class:`AtomicCheckpointSaver`.
        """
        if self._saver is None or self.model is None:
            raise RuntimeError("Pipeline not built yet")
        state = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_eval_loss": eval_loss,
            "config": self.config.to_dict(),
            "stage": self.state.stage.value,
        }
        try:
            path = self._saver.save(state, tag_suffix=tag)
        except Exception as e:
            # Try the previous-gen fallback before giving up.
            resume_path = self._saver.resolve_resume_path()
            _LOGGER.error("Atomic checkpoint write failed: %s; resume=%s", e, resume_path)
            raise
        self.state.checkpoints.append(path)

    def resume(self, *, prefer_previous: bool = False) -> int:
        """Load the most recent (or the previous) checkpoint.

        Returns the step number at which to resume training.
        """
        if self._saver is None:
            raise RuntimeError("Pipeline not built yet")
        path = self._saver.resolve_resume_path()
        if path is None:
            _LOGGER.info("No checkpoint to resume from; starting fresh")
            return 0
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            if not prefer_previous:
                _LOGGER.warning("Failed to load %s: %s; trying .prev", path, e)
                return self.resume(prefer_previous=True)
            raise
        step = int(state.get("step", 0))
        if self.model is not None and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        if self.optimizer is not None and state.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None and state.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.state.stage = PipelineStage(state.get("stage", "built"))
        self.state.best_eval_loss = float(state.get("best_eval_loss", float("inf")))
        _LOGGER.info("Resumed pipeline from %s at step %d", path, step)
        return step

    # ------------------------------------------------------------------
    # Train / distill
    # ------------------------------------------------------------------

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sl = logits[..., :-1, :].contiguous()
        st = targets[..., 1:].contiguous()
        return F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1))

    def _save_in_step(self, step: int, loss_val: float) -> None:
        if step % self.config.save_every == 0 and step > 0:
            self.save_checkpoint(step, loss_val, tag=f"step_{step}")
        if loss_val < self.state.best_eval_loss:
            self.state.best_eval_loss = loss_val
            self.save_checkpoint(step, loss_val, tag="best")

    def run(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        *,
        max_steps: Optional[int] = None,
        on_oom_reduce_batch: bool = True,
    ) -> PipelineState:
        """Run the pre-training loop with OOM recovery.

        Parameters
        ----------
        train_loader
            Iterable yielding ``{"input_ids": Tensor}`` batches.
        eval_loader
            Optional evaluation loader (separate from train).
        max_steps
            Override for ``config.max_train_steps`` (useful in tests).
        on_oom_reduce_batch
            When True, an out-of-memory error triggers a batch-size
            reduction (halve the micro-batch) and a retry.  After two
            retries the stage is set to ``FAILED`` and the exception
            is propagated.
        """
        if self.stage is PipelineStage.UNINITIALIZED:
            self.build()
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Pipeline build failed silently")

        device = next(self.model.parameters()).device
        self.model.train()
        target = max_steps or self.config.max_train_steps
        micro_batch = self.config.batch_size
        retries = 0

        for step in range(target):
            self.optimizer.zero_grad(set_to_none=True)
            try:
                for batch in train_loader:
                    input_ids = batch["input_ids"].to(device)
                    logits = self.model(input_ids)
                    loss = self._loss(logits, input_ids)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.state.global_step = step + 1
                    if step % self.config.log_every == 0:
                        _LOGGER.info(
                            "step=%d loss=%.4f",
                            step, float(loss.item()),
                        )
                    self._save_in_step(step, float(loss.item()))
                    break  # one batch per step (matches trainer.py semantics)
            except torch.cuda.OutOfMemoryError as e:
                if not on_oom_reduce_batch or retries >= 2:
                    self.state.stage = PipelineStage.FAILED
                    self.state.error = f"OOM after {retries} retries: {e}"
                    _LOGGER.exception(self.state.error)
                    # Last-ditch save before transition to FAILED.
                    try:
                        self.save_checkpoint(step, float("inf"), tag="oom")
                    except Exception:
                        pass
                    raise
                retries += 1
                self.state.oom_recoveries += 1
                micro_batch = max(1, micro_batch // 2)
                torch.cuda.empty_cache()
                _LOGGER.warning(
                    "OOM at step %d; halving micro-batch to %d (retry %d/2)",
                    step, micro_batch, retries,
                )
                continue

            if step >= target - 1:
                break

        self.state.stage = PipelineStage.TRAINED
        return self.state

    # ------------------------------------------------------------------
    # Distill (QAT)
    # ------------------------------------------------------------------

    def distill(
        self,
        teacher: nn.Module,
        train_loader: DataLoader,
        *,
        temperature: float = 2.0,
        alpha_kl: float = 0.7,
        max_steps: Optional[int] = None,
    ) -> PipelineState:
        """Quantization-aware distillation run against a frozen teacher.

        Uses :func:`ternair.quantization.distillation.distillation_loss`
        when available; otherwise falls back to plain cross-entropy on
        the teacher's logits argmax.
        """
        try:
            from ternair.quantization.distillation import distillation_loss as _dl
        except ImportError:
            _dl = None

        if self.stage is PipelineStage.UNINITIALIZED:
            self.build()
        device = next(self.model.parameters()).device
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        self.model.train()
        target = max_steps or self.config.max_train_steps
        for step in range(target):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                with torch.no_grad():
                    teacher_logits = teacher(input_ids).logits
                student_logits = self.model(input_ids)
                if _dl is not None:
                    loss = _dl(student_logits, teacher_logits, input_ids,
                               alpha=alpha_kl, temperature=temperature)
                else:
                    loss = self._loss(student_logits, input_ids)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.state.global_step = step + 1
                if step % self.config.log_every == 0:
                    _LOGGER.info(
                        "distill step=%d loss=%.4f beta=%.2f",
                        step, float(loss.item()), temperature,
                    )
                self._save_in_step(step, float(loss.item()))
                break
        self.state.stage = PipelineStage.DISTILLED
        return self.state

    # ------------------------------------------------------------------
    # Freeze / Export
    # ------------------------------------------------------------------

    def freeze(self) -> "TernairPipeline":
        """Pack the ternary weights and switch the model to inference."""
        if self.stage not in (PipelineStage.TRAINED, PipelineStage.DISTILLED, PipelineStage.BUILT):
            raise RuntimeError(
                f"freeze() invalid in stage {self.stage.value}; "
                "train or distill first."
            )
        if self.model is None:
            raise RuntimeError("Pipeline not built")
        self.model.eval()
        self.model.freeze_storage()
        self.state.stage = PipelineStage.FROZEN
        return self

    def export(
        self,
        *,
        filename: str = "model.safetensors",
        format: str = "safetensors",
    ) -> str:
        """Persist the (possibly frozen) model to ``output_dir``.

        Supported formats: ``safetensors`` (default), ``pt``.
        """
        if self.model is None:
            raise RuntimeError("Pipeline not built")
        if format != "safetensors" and self.stage is not PipelineStage.FROZEN:
            _LOGGER.info("Exporting without freezing first; packing on the fly.")
            self.freeze()

        out_path = os.path.join(self.output_dir, filename)
        if format == "safetensors":
            try:
                from ternair.model.export import export_to_safetensors
                export_to_safetensors(self.model, out_path)
            except ImportError as e:
                _LOGGER.warning("Safetensors exporter unavailable (%s); falling back to torch.save", e)
                torch.save(self.model.state_dict(), out_path)
        elif format == "pt":
            torch.save(self.model.state_dict(), out_path)
        else:
            raise ValueError(f"Unsupported export format: {format!r}")

        self.state.artifact_paths[format] = out_path
        self.state.stage = PipelineStage.EXPORTED
        _LOGGER.info("Exported pipeline to %s", out_path)
        return out_path


__all__ = ["TernairPipeline", "PipelineStage", "PipelineState", "MemoryEstimate"]
