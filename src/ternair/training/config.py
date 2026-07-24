"""Training configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Hyper-parameters for pre-training a ternary model.

    All values have sensible defaults targeting the ``tiny`` profile
    for quick smoke tests.
    """

    # Model
    model_profile: str = "tiny"  # tiny / base / one_gb
    model_storage: str = "packed"

    # Optimizer (AdamW)
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # WSD Scheduler
    warmup_ratio: float = 0.05
    stable_ratio: float = 0.80
    decay_ratio: float = 0.15
    min_lr_ratio: float = 0.0  # fraction of peak LR at the end of decay
    decay_type: str = "cosine"  # "cosine" or "linear"

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-100BT"
    dataset_split: str = "train"
    dataset_streaming: bool = True
    dataset_max_samples: int = 10_000  # -1 for all
    dataset_cache_dir: str = "~/.cache/huggingface/datasets"
    tokenizer_name: str = "gpt2"  # HuggingFace tokenizer name

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 100
    max_train_tokens: Optional[int] = None  # alternative to max_train_steps
    seq_length: int = 512
    eval_every: int = 10
    eval_steps: int = 5
    save_every: int = 50
    save_total_limit: int = 3
    output_dir: str = "checkpoints"
    log_every: int = 1
    report_to: str = "none"  # "none", "wandb", "tensorboard"

    # Distributed
    distributed_backend: str = "accelerate"  # "accelerate" or "torchrun"

    # Checkpoint / Resume
    resume_from: Optional[str] = None
    seed: int = 42

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @property
    def total_steps(self) -> int:
        return self.max_train_steps

    @property
    def warmup_steps(self) -> int:
        return int(self.total_steps * self.warmup_ratio)

    @property
    def stable_steps(self) -> int:
        return int(self.total_steps * self.stable_ratio)

    @property
    def decay_steps(self) -> int:
        return max(self.total_steps - self.warmup_steps - self.stable_steps, 0)
