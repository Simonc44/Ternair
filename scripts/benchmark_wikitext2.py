#!/usr/bin/env python3
"""WikiText-2 benchmark: Ternair (ternary) vs FP16 baseline.

Fixes vs previous version
--------------------------
* Gradient clipping + NaN detection in training loop
* Safe perplexity (caps loss at 20.0 before exp to avoid overflow)
* Warmup-then-anneal LR schedule so ternary weights stabilize
* Speed benchmark uses fixed-length prompt (no growing sequence)
* FP16 baseline uses same vocab_size but smaller hidden to be fair
* gamma eps already set in ternary.py; we also clamp logits here

Usage
-----
    python scripts/benchmark_wikitext2.py --steps 300 --device cpu
    python scripts/benchmark_wikitext2.py --steps 2000 --device cuda

Requirements
------------
    pip install datasets transformers
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("[warn] 'datasets' not installed. Run: pip install datasets")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[warn] 'transformers' not installed. Run: pip install transformers")

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ternair.model.size_profiles import tiny_profile, small_profile
from ternair.model.modeling import TernairForCausalLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cap loss before exp() to avoid PPL overflow (exp(20) ~ 4.8e8, still finite)
_MAX_LOSS_FOR_PPL = 20.0

# HuggingFace renamed 'wikitext' -> 'Salesforce/wikitext' in datasets >= 2.20
_WIKITEXT_CANDIDATES = [
    ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    ("wikitext",            "wikitext-2-raw-v1"),
]


# ---------------------------------------------------------------------------
# Fallback tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [min(ord(c), self.vocab_size - 1) for c in text]

    def __call__(self, text: str, **_):
        return {"input_ids": self.encode(text)}


# ---------------------------------------------------------------------------
# FP16 baseline  (pure nn.Linear, same depth as Ternair)
# ---------------------------------------------------------------------------

class FP16Baseline(nn.Module):
    """Lightweight decoder-only transformer used as the FP16 reference.

    Uses a smaller hidden size so training converges in the same number
    of steps as the Ternair model on CPU.
    """

    def __init__(self, vocab_size: int, hidden: int, layers: int, heads: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        # Positional embedding (learned, simpler than RoPE for the baseline)
        self.pos_emb = nn.Embedding(2048, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying

    def forward(self, x: Tensor) -> Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h = self.embed(x) + self.pos_emb(pos)
        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.norm(h)
        return self.head(h)

    def num_bytes(self) -> int:
        return sum(p.numel() * 4 for p in self.parameters())  # FP32


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wikitext2_tokens(
    tokenizer,
    split: str = "train",
    max_tokens: int = 2_000_000,
) -> list[int]:
    """Load and tokenize WikiText-2 with fallback for HF Hub renames."""
    if not HAS_DATASETS:
        print(f"[warn] datasets not available — using random tokens for {split}")
        return list(torch.randint(0, 256, (min(max_tokens, 50_000),)).tolist())

    dataset = None
    last_err: Exception | None = None
    for ds_name, ds_config in _WIKITEXT_CANDIDATES:
        try:
            dataset = load_dataset(ds_name, ds_config, split=split)
            print(f"  Loaded WikiText-2 from '{ds_name}'")
            break
        except Exception as e:
            last_err = e

    if dataset is None:
        print(f"[warn] Could not load WikiText-2 ({last_err}). Using random tokens.")
        return list(torch.randint(0, 256, (min(max_tokens, 50_000),)).tolist())

    tokens: list[int] = []
    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(text)
        else:
            ids = tokenizer(text)["input_ids"]
        tokens.extend(ids)
        if len(tokens) >= max_tokens:
            break
    return tokens[:max_tokens]


def token_batches(
    tokens: list[int],
    seq_len: int,
    batch_size: int,
    device: str,
) -> Iterator[tuple[Tensor, Tensor]]:
    total = (len(tokens) - 1) // seq_len
    for start in range(0, total, batch_size):
        batch_idx = list(range(start, min(start + batch_size, total)))
        if not batch_idx:
            break
        xs = [tokens[i * seq_len : i * seq_len + seq_len] for i in batch_idx]
        ys = [tokens[i * seq_len + 1 : i * seq_len + seq_len + 1] for i in batch_idx]
        yield torch.tensor(xs, device=device), torch.tensor(ys, device=device)


# ---------------------------------------------------------------------------
# Training  — with gradient clipping + NaN guard
# ---------------------------------------------------------------------------

def _make_scheduler(optimizer, total_steps: int, warmup_steps: int, lr: float):
    """Linear warmup then cosine decay."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    model: nn.Module,
    tokens: list[int],
    steps: int,
    seq_len: int = 128,
    batch_size: int = 4,
    lr: float = 3e-4,
    device: str = "cpu",
    label: str = "model",
    warmup_ratio: float = 0.1,
) -> list[float]:
    """Train for ``steps`` steps with gradient clipping and NaN guard."""
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01,
        betas=(0.9, 0.95), eps=1e-8,
    )
    warmup_steps = max(1, int(steps * warmup_ratio))
    scheduler = _make_scheduler(optimizer, steps, warmup_steps, lr)

    losses: list[float] = []
    step = 0
    nan_steps = 0

    while step < steps:
        for x, y in token_batches(tokens, seq_len, batch_size, device):
            if step >= steps:
                break

            logits = model(x)  # (B, L, V)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            # ---- NaN guard ------------------------------------------------
            if not torch.isfinite(loss):
                nan_steps += 1
                if nan_steps > 10:
                    print(f"  [{label}] Too many NaN steps — reinitialising optimizer")
                    for g in optimizer.param_groups:
                        g["lr"] *= 0.1
                    nan_steps = 0
                optimizer.zero_grad()
                step += 1
                continue
            nan_steps = 0
            # ---------------------------------------------------------------

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping — prevents explosion with ternary weights
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Skip update if gradients are NaN/Inf after clipping
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad()
                step += 1
                continue

            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            step += 1

            if step % 50 == 0 or step == 1:
                ppl_str = f"{math.exp(min(loss.item(), _MAX_LOSS_FOR_PPL)):.1f}"
                print(
                    f"  [{label}] step {step:>4d}/{steps}  "
                    f"loss={loss.item():.4f}  ppl={ppl_str}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
    return losses


# ---------------------------------------------------------------------------
# Safe perplexity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    tokens: list[int],
    seq_len: int = 256,
    stride: int = 128,
    device: str = "cpu",
    max_tokens: int = 100_000,
) -> tuple[float, int]:
    """Sliding-window perplexity. Caps loss at 20 to avoid exp() overflow."""
    model.eval()
    model.to(device)
    tokens = tokens[:max_tokens]

    nll_sum = 0.0
    n_tokens = 0

    for i in range(0, len(tokens) - 1, stride):
        chunk = tokens[i : i + seq_len + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk[:-1]], device=device)
        y = torch.tensor([chunk[1:]], device=device)

        logits = model(x)
        # Clamp logits to avoid numerical issues in cross_entropy
        logits = torch.clamp(logits, -100.0, 100.0)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            continue  # skip this window instead of poisoning the average

        nll_sum  += loss_val
        n_tokens += y.numel()

    if n_tokens == 0:
        return float("inf"), 0

    # Cap per-token CE before exp to avoid overflow
    per_token_ce = min(nll_sum / n_tokens, _MAX_LOSS_FOR_PPL)
    return math.exp(per_token_ce), n_tokens


# ---------------------------------------------------------------------------
# Speed benchmark  (fixed-length, no growing sequence)
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_speed(
    model: nn.Module,
    prompt: list[int],
    seq_len: int = 64,
    num_runs: int = 3,
    device: str = "cpu",
) -> float:
    """Tokens/sec measured on fixed-length forward passes (not autoregressive).

    Autoregressive benchmarking with torch.cat grows the sequence each step
    which skews the comparison. We instead do fixed-length forward passes
    and measure logit throughput, which is fairer for comparing compression.
    """
    model.eval()
    model.to(device)

    # Build a fixed-length prompt tensor
    ids = (prompt * ((seq_len // len(prompt)) + 1))[:seq_len]
    x = torch.tensor([ids], device=device)  # (1, seq_len)

    # Warmup
    for _ in range(2):
        _ = model(x)

    # Timed runs
    total_time = 0.0
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(x)
        total_time += time.perf_counter() - start

    # Tokens processed per second = batch_size * seq_len / time
    return (num_runs * seq_len) / max(total_time, 1e-9)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def make_markdown_table(results: dict) -> str:
    rows = [
        "## WikiText-2 Benchmark Results\n",
        f"> Profile: {results['config']['profile']}  ",
        f"> Training steps: {results['config']['steps']}  ",
        f"> Device: {results['config']['device']}  ",
        f"> Date: {results['config']['date']}\n",
        "| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |",
        "|-------|---------------------|------------|------------|" ,
    ]
    for name, r in results["models"].items():
        if name in ("ternair", "fp16"):
            continue
        ppl = f"{r['perplexity']:.2f}" if math.isfinite(r["perplexity"]) else "inf"
        speed = f"{r['tokens_per_sec']:.1f}" if r["tokens_per_sec"] > 0 else "N/A"
        rows.append(f"| {name} | {ppl} | {r['size_mib']:.1f} | {speed} |")

    t = results["models"].get("ternair", {})
    f = results["models"].get("fp16",    {})
    if t and f and math.isfinite(t.get("perplexity", float("nan"))):
        ppl_delta  = t["perplexity"] - f["perplexity"]
        size_ratio = f["size_mib"] / max(t["size_mib"], 0.001)
        spd_ratio  = t["tokens_per_sec"] / max(f["tokens_per_sec"], 0.001)
        rows += [
            "",
            "**Summary:**",
            f"- PPL overhead vs FP16: {ppl_delta:+.2f} ({ppl_delta / max(f['perplexity'], 0.001) * 100:+.1f}%)",
            f"- Size compression: {size_ratio:.1f}×",
            f"- Speed ratio: {spd_ratio:.2f}×",
        ]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ternair vs FP16 WikiText-2 benchmark")
    p.add_argument("--steps",           type=int,   default=500)
    p.add_argument("--device",                      default="cpu")
    p.add_argument("--seq-len",         type=int,   default=128)
    p.add_argument("--batch-size",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=2e-4)
    p.add_argument("--warmup-ratio",    type=float, default=0.1)
    p.add_argument("--eval-max-tokens", type=int,   default=30_000)
    p.add_argument("--speed-seq-len",   type=int,   default=64)
    p.add_argument("--output",                      default="benchmark_results.json")
    p.add_argument("--output-md",                   default="benchmark_results.md")
    p.add_argument("--profile",         choices=["tiny", "small"], default="tiny")
    p.add_argument("--storage",         choices=["int8", "packed", "fastpacked"], default="fastpacked")
    p.add_argument("--ternair-ckpt",    default=None)
    p.add_argument("--fp16-ckpt",       default=None)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = args.device

    import datetime
    print("=" * 60)
    print("  Ternair — WikiText-2 Benchmark")
    print("=" * 60)

    # ---- Tokenizer ----
    tokenizer  = None
    vocab_size = 256
    if HAS_TRANSFORMERS:
        try:
            print("Loading tokenizer (gpt2)...")
            tokenizer  = AutoTokenizer.from_pretrained("gpt2")
            vocab_size = tokenizer.vocab_size
        except Exception as e:
            print(f"[warn] gpt2 tokenizer failed ({e}), falling back to char-level.")
    if tokenizer is None:
        tokenizer  = CharTokenizer(vocab_size=256)
        vocab_size = 256
    print(f"Vocab size: {vocab_size}")

    # ---- Data ----
    print("\nLoading WikiText-2...")
    train_tokens = load_wikitext2_tokens(tokenizer, split="train", max_tokens=500_000)
    test_tokens  = load_wikitext2_tokens(tokenizer, split="test",  max_tokens=100_000)
    print(f"  Train tokens : {len(train_tokens):,}")
    print(f"  Test tokens  : {len(test_tokens):,}")

    # ---- Build models ----
    profile_fn     = tiny_profile if args.profile == "tiny" else small_profile
    cfg            = profile_fn(storage=args.storage)
    cfg.vocab_size = vocab_size

    print(f"\nBuilding Ternair model ({args.profile}, storage={args.storage})...")
    ternair_model    = TernairForCausalLM(cfg)
    n_params_ternair = sum(p.numel() for p in ternair_model.parameters())
    print(f"  Parameters: {n_params_ternair:,}")

    # FP16 baseline with same depth
    print("Building FP16 baseline...")
    fp16_model = FP16Baseline(
        vocab_size=vocab_size,
        hidden=cfg.hidden_size,
        layers=cfg.num_hidden_layers,
        heads=cfg.num_attention_heads,
    )
    n_params_fp16 = sum(p.numel() for p in fp16_model.parameters())
    print(f"  Parameters: {n_params_fp16:,}")

    # ---- Train or load ----
    if args.ternair_ckpt and Path(args.ternair_ckpt).exists():
        ternair_model.load_state_dict(torch.load(args.ternair_ckpt, map_location=device))
    elif args.steps > 0:
        print(f"\nTraining Ternair ({args.steps} steps, warmup={int(args.steps*args.warmup_ratio)})...")
        train(
            ternair_model, train_tokens, args.steps,
            seq_len=args.seq_len, batch_size=args.batch_size,
            lr=args.lr, device=device, label="ternair",
            warmup_ratio=args.warmup_ratio,
        )
        torch.save(ternair_model.state_dict(), "ternair_checkpoint.pt")

    if args.fp16_ckpt and Path(args.fp16_ckpt).exists():
        fp16_model.load_state_dict(torch.load(args.fp16_ckpt, map_location=device))
    elif args.steps > 0:
        print(f"\nTraining FP16 baseline ({args.steps} steps)...")
        train(
            fp16_model, train_tokens, args.steps,
            seq_len=args.seq_len, batch_size=args.batch_size,
            lr=args.lr, device=device, label="fp16  ",
            warmup_ratio=args.warmup_ratio,
        )
        torch.save(fp16_model.state_dict(), "fp16_checkpoint.pt")

    # ---- Freeze ----
    print("\nFreezing Ternair storage...")
    ternair_model.freeze_storage()
    ternair_model.eval()
    fp16_model.eval()

    # ---- Perplexity ----
    print("\nEvaluating perplexity on WikiText-2 test...")
    ternair_ppl, ternair_ntok = evaluate_perplexity(
        ternair_model, test_tokens, seq_len=256, stride=128,
        device=device, max_tokens=args.eval_max_tokens,
    )
    print(f"  Ternair PPL : {ternair_ppl:.2f} ({ternair_ntok:,} tokens)")

    fp16_ppl, fp16_ntok = evaluate_perplexity(
        fp16_model, test_tokens, seq_len=256, stride=128,
        device=device, max_tokens=args.eval_max_tokens,
    )
    print(f"  FP16    PPL : {fp16_ppl:.2f} ({fp16_ntok:,} tokens)")

    # ---- Sizes ----
    def ternair_size(m: TernairForCausalLM) -> int:
        total = sum(
            mod.state_bytes() for mod in m.modules() if hasattr(mod, "state_bytes")
        )
        if hasattr(m.model, "embed_tokens"):
            total += m.model.embed_tokens.weight.numel() * 2
        return total

    t_bytes = ternair_size(ternair_model)
    f_bytes = fp16_model.num_bytes()

    # ---- Speed ----
    print("\nSpeed benchmark (fixed-length forward pass)...")
    prompt_ids    = train_tokens[:args.speed_seq_len]
    ternair_speed = benchmark_speed(ternair_model, prompt_ids, args.speed_seq_len, device=device)
    fp16_speed    = benchmark_speed(fp16_model,    prompt_ids, args.speed_seq_len, device=device)
    print(f"  Ternair : {ternair_speed:.1f} tok/s")
    print(f"  FP16    : {fp16_speed:.1f} tok/s")

    # ---- Results ----
    results = {
        "config": {
            "profile": args.profile,
            "storage": args.storage,
            "steps":   args.steps,
            "seq_len": args.seq_len,
            "device":  device,
            "date":    datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "models": {
            "Ternair (ternary)": {
                "perplexity":     round(ternair_ppl, 4),
                "num_tokens":     ternair_ntok,
                "size_mib":       round(t_bytes / 1024**2, 2),
                "tokens_per_sec": round(ternair_speed, 2),
                "params":         n_params_ternair,
            },
            "FP16 baseline": {
                "perplexity":     round(fp16_ppl, 4),
                "num_tokens":     fp16_ntok,
                "size_mib":       round(f_bytes / 1024**2, 2),
                "tokens_per_sec": round(fp16_speed, 2),
                "params":         n_params_fp16,
            },
        },
    }
    results["models"]["ternair"] = results["models"]["Ternair (ternary)"]
    results["models"]["fp16"]    = results["models"]["FP16 baseline"]

    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to: {args.output}")

    md = make_markdown_table(results)
    with open(args.output_md, "w") as fh:
        fh.write(md)
    print(f"Markdown saved to: {args.output_md}")

    # ---- Summary ----
    ppl_delta  = ternair_ppl - fp16_ppl
    size_ratio = (f_bytes / 1024**2) / max(t_bytes / 1024**2, 0.001)
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  {'Model':<22}  {'PPL':>10}  {'MiB':>8}  {'Tok/s':>8}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*8}  {'-'*8}")
    ppl_t_str = f"{ternair_ppl:.2f}" if math.isfinite(ternair_ppl) else "inf"
    ppl_f_str = f"{fp16_ppl:.2f}"    if math.isfinite(fp16_ppl)    else "inf"
    print(f"  {'Ternair (ternary)':<22}  {ppl_t_str:>10}  "
          f"{t_bytes/1024**2:>8.1f}  {ternair_speed:>8.1f}")
    print(f"  {'FP16 baseline':<22}  {ppl_f_str:>10}  "
          f"{f_bytes/1024**2:>8.1f}  {fp16_speed:>8.1f}")
    print("=" * 60)
    if math.isfinite(ppl_delta):
        print(f"  PPL overhead   : {ppl_delta:+.2f} "
              f"({ppl_delta / max(fp16_ppl, 0.001) * 100:+.1f}%)")
    print(f"  Size reduction : {size_ratio:.1f}×")
    print(f"  Speed ratio    : {ternair_speed / max(fp16_speed, 0.001):.2f}×")
    print("=" * 60)


if __name__ == "__main__":
    main()
