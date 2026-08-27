"""Verify the multi-step training loop reduces loss (quality path)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_train_loop_reduces_loss() -> None:
    """Run a few real optimizer steps on the toy corpus and assert loss drops."""
    from ternair import TernairForCausalLM, tiny_profile
    from ternair.training.config import TrainingConfig
    from ternair.training.data import build_toy_dataloader, toy_corpus
    from ternair.training.optimizer import create_optimizer
    from ternair.training.scheduler import WSDScheduler
    from ternair.training.trainer import cross_entropy, train_one_epoch

    torch.manual_seed(42)
    cfg = TrainingConfig(
        model_profile="tiny",
        model_storage="packed",
        batch_size=4,
        max_train_steps=10,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_accumulation_steps=1,
        eval_every=0,
        log_every=0,
        save_every=0,
        output_dir="checkpoints",
    )
    model = TernairForCausalLM(tiny_profile(storage="packed"))
    optimizer = create_optimizer(model, lr=cfg.learning_rate, weight_decay=0.0)
    scheduler = WSDScheduler(
        optimizer,
        total_steps=cfg.total_steps,
        warmup_steps=0,
        stable_steps=cfg.stable_steps,
        decay_steps=0,
        min_lr=0.0,
        decay_type="linear",
    )
    dataloader = build_toy_dataloader(
        text=toy_corpus(), n_sequences=32, batch_size=4
    )

    model.eval()
    with torch.no_grad():
        first_batch = next(iter(dataloader))["input_ids"]
        loss0 = float(cross_entropy(model(first_batch), first_batch).item())

    # Real training loop, plain PyTorch (accelerator=None fallback path).
    model.train()
    final_step = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        accelerator=None,
    )

    model.eval()
    with torch.no_grad():
        loss1 = float(cross_entropy(model(first_batch), first_batch).item())

    assert final_step == 10
    assert loss1 < loss0, f"expected loss reduction, got {loss0:.4f} -> {loss1:.4f}"
