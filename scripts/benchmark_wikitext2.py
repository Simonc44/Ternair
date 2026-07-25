#!/usr/bin/env python3
"""WikiText-2 benchmark: Ternair (ternary) vs FP16 baseline.

What this script does
---------------------
1. Builds a **tiny Ternair model** (8 layers, hidden=256, ~2.6 M params)
   and a matching **FP16 baseline** (same architecture, nn.Linear weights).
2. Trains both models for ``--steps`` steps on a WikiText-2 slice
   (downloaded automatically via HuggingFace datasets).
3. Evaluates perplexity on the WikiText-2 test split.
4. Reports perplexity, model size and tokens/sec.
5. Writes results to ``benchmark_results.json`` and a Markdown table.

Usage
-----
    python scripts/benchmark_wikitext2.py --steps 200 --device cpu
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
from ternair.model.export import print_compression_report


# ---------------------------------------------------------------------------
# Fallback tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [min(ord(c), self.vocab_size - 1) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def __call__(self, text: str, **_):
        return {"input_ids": self.encode(text)}


# ---------------------------------------------------------------------------
# FP16 baseline
# ---------------------------------------------------------------------------

class FP16Baseline(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, layers: int, heads: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=hidden * 2,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        ])
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x: Tensor) -> Tensor:
        B, L = x.shape
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h = self.embed(x)
        mem = torch.zeros(B, 1, h.size(-1), device=x.device, dtype=h.dtype)
        for block in self.blocks:
            h = block(h, mem, tgt_mask=mask, tgt_is_causal=True)
        return self.head(h)

    def num_bytes(self) -> int:
        return sum(p.numel() * 2 for p in self.parameters())


# ---------------------------------------------------------------------------
# Data loading  — robust to HF Hub dataset renames
# ---------------------------------------------------------------------------

# HuggingFace renamed 'wikitext' -> 'Salesforce/wikitext' in datasets >= 2.20.
# We try both names in order so the script works on any version.
_WIKITEXT_CANDIDATES = [
    ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    ("wikitext",            "wikitext-2-raw-v1"),
]


def load_wikitext2_tokens(
    tokenizer,
    split: str = "train",
    max_tokens: int = 2_000_000,
) -> list[int]:
    """Load and tokenize WikiText-2, trying multiple dataset names."""
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
    indices = list(range(total))
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        if not batch_indices:
            break
        xs, ys = [], []
        for i in batch_indices:
            s = i * seq_len
            xs.append(tokens[s : s + seq_len])
            ys.append(tokens[s + 1 : s + seq_len + 1])
        yield torch.tensor(xs, device=device), torch.tensor(ys, device=device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    tokens: list[int],
    steps: int,
    seq_len: int = 128,
    batch_size: int = 4,
    lr: float = 3e-4,
    device: str = "cpu",
    label: str = "model",
) -> list[float]:
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.1)

    losses: list[float] = []
    step = 0
    while step < steps:
        for x, y in token_batches(tokens, seq_len, batch_size, device):
            if step >= steps:
                break
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"  [{label}] step {step:>4d}/{steps}  "
                      f"loss={loss.item():.4f}  ppl={math.exp(loss.item()):.2f}")
    return losses


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    tokens: list[int],
    seq_len: int = 512,
    stride: int = 256,
    device: str = "cpu",
    max_tokens: int = 100_000,
) -> tuple[float, int]:
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
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        )
        nll_sum += loss.item()
        n_tokens += y.numel()
    ce = nll_sum / max(n_tokens, 1)
    return math.exp(ce), n_tokens


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_speed(
    model: nn.Module,
    prompt: list[int],
    max_new_tokens: int = 64,
    num_runs: int = 5,
    device: str = "cpu",
) -> float:
    model.eval()
    model.to(device)
    prompt_t = torch.tensor([prompt], device=device)
    for _ in range(2):
        x = prompt_t.clone()
        for _ in range(8):
            logits = model(x)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_tok], dim=1)
    total_time = 0.0
    total_toks = 0
    for _ in range(num_runs):
        x = prompt_t.clone()
        start = time.perf_counter()
        for _ in range(max_new_tokens):
            logits = model(x)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_tok], dim=1)
        total_time += time.perf_counter() - start
        total_toks += max_new_tokens
    return total_toks / max(total_time, 1e-6)


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
        if name in ("ternair", "fp16"):  # skip alias keys
            continue
        speed = f"{r['tokens_per_sec']:.1f}" if r["tokens_per_sec"] > 0 else "N/A"
        rows.append(f"| {name} | {r['perplexity']:.2f} | {r['size_mib']:.1f} | {speed} |")
    t = results["models"].get("ternair", {})
    f = results["models"].get("fp16",    {})
    if t and f:
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
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--eval-max-tokens", type=int,   default=50_000)
    p.add_argument("--speed-tokens",    type=int,   default=32)
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

    # Tokenizer
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

    # Data
    print("\nLoading WikiText-2...")
    train_tokens = load_wikitext2_tokens(tokenizer, split="train", max_tokens=500_000)
    test_tokens  = load_wikitext2_tokens(tokenizer, split="test",  max_tokens=100_000)
    print(f"  Train tokens : {len(train_tokens):,}")
    print(f"  Test tokens  : {len(test_tokens):,}")

    # Build models
    profile_fn    = tiny_profile if args.profile == "tiny" else small_profile
    cfg           = profile_fn(storage=args.storage)
    cfg.vocab_size = vocab_size

    print(f"\nBuilding Ternair model ({args.profile}, storage={args.storage})...")
    ternair_model   = TernairForCausalLM(cfg)
    n_params_ternair = sum(p.numel() for p in ternair_model.parameters())
    print(f"  Parameters: {n_params_ternair:,}")

    print("Building FP16 baseline...")
    fp16_model  = FP16Baseline(
        vocab_size=vocab_size,
        hidden=cfg.hidden_size,
        layers=cfg.num_hidden_layers,
        heads=cfg.num_attention_heads,
    )
    n_params_fp16 = sum(p.numel() for p in fp16_model.parameters())
    print(f"  Parameters: {n_params_fp16:,}")

    # Train or load
    if args.ternair_ckpt and Path(args.ternair_ckpt).exists():
        ternair_model.load_state_dict(torch.load(args.ternair_ckpt, map_location=device))
    elif args.steps > 0:
        print(f"\nTraining Ternair ({args.steps} steps)...")
        train(ternair_model, train_tokens, args.steps,
              seq_len=args.seq_len, batch_size=args.batch_size,
              lr=args.lr, device=device, label="ternair")
        torch.save(ternair_model.state_dict(), "ternair_checkpoint.pt")

    if args.fp16_ckpt and Path(args.fp16_ckpt).exists():
        fp16_model.load_state_dict(torch.load(args.fp16_ckpt, map_location=device))
    elif args.steps > 0:
        print(f"\nTraining FP16 baseline ({args.steps} steps)...")
        train(fp16_model, train_tokens, args.steps,
              seq_len=args.seq_len, batch_size=args.batch_size,
              lr=args.lr, device=device, label="fp16  ")
        torch.save(fp16_model.state_dict(), "fp16_checkpoint.pt")

    # Freeze
    print("\nFreezing Ternair storage...")
    ternair_model.freeze_storage()
    ternair_model.eval()
    fp16_model.eval()

    # Perplexity
    print("\nEvaluating perplexity on WikiText-2 test...")
    ternair_ppl, ternair_ntok = evaluate_perplexity(
        ternair_model, test_tokens, seq_len=512, stride=256,
        device=device, max_tokens=args.eval_max_tokens)
    print(f"  Ternair PPL : {ternair_ppl:.2f} ({ternair_ntok:,} tokens)")

    fp16_ppl, fp16_ntok = evaluate_perplexity(
        fp16_model, test_tokens, seq_len=512, stride=256,
        device=device, max_tokens=args.eval_max_tokens)
    print(f"  FP16    PPL : {fp16_ppl:.2f} ({fp16_ntok:,} tokens)")

    # Sizes
    def ternair_bytes(m: TernairForCausalLM) -> int:
        total = sum(
            mod.state_bytes() for mod in m.modules() if hasattr(mod, "state_bytes")
        )
        if hasattr(m, "embed"):
            total += m.embed.weight.numel() * 2
        return total

    t_bytes = ternair_bytes(ternair_model)
    f_bytes = fp16_model.num_bytes()

    # Speed
    print("\nSpeed benchmark...")
    prompt_ids    = train_tokens[:16]
    ternair_speed = benchmark_speed(ternair_model, prompt_ids, args.speed_tokens, device=device)
    fp16_speed    = benchmark_speed(fp16_model,    prompt_ids, args.speed_tokens, device=device)
    print(f"  Ternair : {ternair_speed:.1f} tok/s")
    print(f"  FP16    : {fp16_speed:.1f} tok/s")

    # Results
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
                "perplexity":    round(ternair_ppl,   4),
                "num_tokens":    ternair_ntok,
                "size_mib":      round(t_bytes / 1024**2, 2),
                "tokens_per_sec": round(ternair_speed, 2),
                "params":        n_params_ternair,
            },
            "FP16 baseline": {
                "perplexity":    round(fp16_ppl, 4),
                "num_tokens":    fp16_ntok,
                "size_mib":      round(f_bytes / 1024**2, 2),
                "tokens_per_sec": round(fp16_speed, 2),
                "params":        n_params_fp16,
            },
        },
    }
    # Alias keys for inject_benchmark.py
    results["models"]["ternair"] = results["models"]["Ternair (ternary)"]
    results["models"]["fp16"]    = results["models"]["FP16 baseline"]

    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to: {args.output}")

    md = make_markdown_table(results)
    with open(args.output_md, "w") as fh:
        fh.write(md)
    print(f"Markdown saved to: {args.output_md}")

    # Summary
    ppl_delta  = ternair_ppl - fp16_ppl
    size_ratio = (f_bytes / 1024**2) / max(t_bytes / 1024**2, 0.001)
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  {'Model':<22}  {'PPL':>8}  {'Size (MiB)':>10}  {'Tok/s':>8}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*10}  {'-'*8}")
    print(f"  {'Ternair (ternary)':<22}  {ternair_ppl:>8.2f}  "
          f"{t_bytes/1024**2:>10.1f}  {ternair_speed:>8.1f}")
    print(f"  {'FP16 baseline':<22}  {fp16_ppl:>8.2f}  "
          f"{f_bytes/1024**2:>10.1f}  {fp16_speed:>8.1f}")
    print("=" * 60)
    print(f"  PPL overhead   : {ppl_delta:+.2f} ({ppl_delta / max(fp16_ppl, 0.001) * 100:+.1f}%)")
    print(f"  Size reduction : {size_ratio:.1f}×")
    print(f"  Speed ratio    : {ternair_speed / max(fp16_speed, 0.001):.2f}×")
    print("=" * 60)


if __name__ == "__main__":
    main()
