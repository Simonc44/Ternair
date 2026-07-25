#!/usr/bin/env python3
"""WikiText-2 benchmark: Ternair (ternary) vs FP16 baseline.

Key design decisions
---------------------
* Uses a **character-level tokenizer by default** (vocab_size=256).
  This is critical for the tiny profile (hidden=256): with GPT-2's
  vocab (50257 tokens), the embedding alone uses 82% of the model's
  parameters, leaving almost nothing for actual learning.
  With vocab=256, a 300-step run converges to a meaningful PPL.

* Use ``--use-char-tokenizer`` (default in CI) for reproducible results.
  Use ``--no-char-tokenizer`` to benchmark with GPT-2 tokenizer on a
  larger model (small/medium profile).

Usage
-----
    # Fast CI run (char tokenizer, tiny model, ~3 min CPU)
    python scripts/benchmark_wikitext2.py --steps 500 --use-char-tokenizer

    # Full run with GPT-2 tokenizer (needs small profile)
    python scripts/benchmark_wikitext2.py --steps 2000 --profile small --no-char-tokenizer

Requirements
------------
    pip install datasets
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

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ternair.model.size_profiles import tiny_profile, small_profile
from ternair.model.modeling import TernairForCausalLM

_MAX_LOSS_FOR_PPL = 20.0
_WIKITEXT_CANDIDATES = [
    ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    ("wikitext",            "wikitext-2-raw-v1"),
]


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

class CharTokenizer:
    """Character-level tokenizer, vocab_size=256.

    Ideal for the tiny profile: embedding is 256x256 = 65K params,
    leaving the full model capacity for learning language structure.
    """
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return [min(ord(c), 255) for c in text]


class BPETokenizer:
    """Thin wrapper around a HuggingFace tokenizer."""
    def __init__(self, hf_tokenizer):
        self._tok = hf_tokenizer
        self.vocab_size = hf_tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)


def make_tokenizer(use_char: bool) -> CharTokenizer | BPETokenizer:
    if use_char:
        print("Using character-level tokenizer (vocab_size=256)")
        return CharTokenizer()
    try:
        from transformers import AutoTokenizer
        print("Loading GPT-2 tokenizer...")
        tok = AutoTokenizer.from_pretrained("gpt2")
        return BPETokenizer(tok)
    except Exception as e:
        print(f"[warn] GPT-2 tokenizer failed ({e}), falling back to char-level.")
        return CharTokenizer()


# ---------------------------------------------------------------------------
# FP16 baseline
# ---------------------------------------------------------------------------

class FP16Baseline(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, layers: int, heads: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(2048, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads,
            dim_feedforward=hidden * 4, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        B, L = x.shape
        pos  = torch.arange(L, device=x.device).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h = self.embed(x) + self.pos_emb(pos)
        h = self.transformer(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))

    def num_bytes(self) -> int:
        return sum(p.numel() * 4 for p in self.parameters())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wikitext2_tokens(
    tokenizer: CharTokenizer | BPETokenizer,
    split: str = "train",
    max_tokens: int = 2_000_000,
) -> list[int]:
    if not HAS_DATASETS:
        print(f"[warn] datasets not installed — using random tokens for {split}")
        return list(torch.randint(0, tokenizer.vocab_size, (min(max_tokens, 100_000),)).tolist())

    dataset = None
    for ds_name, ds_config in _WIKITEXT_CANDIDATES:
        try:
            dataset = load_dataset(ds_name, ds_config, split=split)
            print(f"  Loaded WikiText-2 from '{ds_name}'")
            break
        except Exception:
            pass

    if dataset is None:
        print("[warn] Could not load WikiText-2. Using random tokens.")
        return list(torch.randint(0, tokenizer.vocab_size, (min(max_tokens, 100_000),)).tolist())

    tokens: list[int] = []
    for row in dataset:
        text = row["text"].strip()
        if text:
            tokens.extend(tokenizer.encode(text))
        if len(tokens) >= max_tokens:
            break
    return tokens[:max_tokens]


def token_batches(
    tokens: list[int], seq_len: int, batch_size: int, device: str
) -> Iterator[tuple[Tensor, Tensor]]:
    total = (len(tokens) - 1) // seq_len
    for start in range(0, total, batch_size):
        idx = list(range(start, min(start + batch_size, total)))
        if not idx:
            break
        xs = [tokens[i * seq_len : i * seq_len + seq_len] for i in idx]
        ys = [tokens[i * seq_len + 1 : i * seq_len + seq_len + 1] for i in idx]
        yield torch.tensor(xs, device=device), torch.tensor(ys, device=device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


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
) -> None:
    model.train().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8
    )
    warmup = max(1, int(steps * warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, warmup, steps)
    )
    step = 0
    while step < steps:
        for x, y in token_batches(tokens, seq_len, batch_size, device):
            if step >= steps:
                break
            logits = model(x)
            loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if not torch.isfinite(loss):
                step += 1
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 50 == 0 or step == 1:
                ppl = math.exp(min(loss.item(), _MAX_LOSS_FOR_PPL))
                print(f"  [{label}] step {step:>4d}/{steps}  "
                      f"loss={loss.item():.3f}  ppl={ppl:.1f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    tokens: list[int],
    seq_len: int = 256,
    stride: int = 128,
    device: str = "cpu",
    max_tokens: int = 20_000,
) -> tuple[float, int]:
    model.eval().to(device)
    tokens    = tokens[:max_tokens]
    nll_sum   = 0.0
    n_tokens  = 0
    for i in range(0, len(tokens) - 1, stride):
        chunk = tokens[i : i + seq_len + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk[:-1]], device=device)
        y = torch.tensor([chunk[1:]],  device=device)
        logits = model(x)
        logits = torch.clamp(logits, -100.0, 100.0)
        loss   = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        )
        val = loss.item()
        if not math.isfinite(val):
            continue
        nll_sum  += val
        n_tokens += y.numel()
    if n_tokens == 0:
        return float("inf"), 0
    return math.exp(min(nll_sum / n_tokens, _MAX_LOSS_FOR_PPL)), n_tokens


@torch.no_grad()
def benchmark_speed(
    model: nn.Module,
    prompt: list[int],
    seq_len: int = 64,
    num_runs: int = 5,
    device: str = "cpu",
) -> float:
    model.eval().to(device)
    ids = (prompt * ((seq_len // max(len(prompt), 1)) + 1))[:seq_len]
    x   = torch.tensor([ids], device=device)
    for _ in range(2):
        model(x)
    t0 = time.perf_counter()
    for _ in range(num_runs):
        model(x)
    elapsed = time.perf_counter() - t0
    return (num_runs * seq_len) / max(elapsed, 1e-9)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def make_markdown_table(results: dict) -> str:
    rows = [
        "## WikiText-2 Benchmark Results\n",
        f"> Profile: {results['config']['profile']}  ",
        f"> Tokenizer: {results['config'].get('tokenizer', 'char')}  ",
        f"> Training steps: {results['config']['steps']}  ",
        f"> Device: {results['config']['device']}  ",
        f"> Date: {results['config']['date']}\n",
        "| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |",
        "|-------|---------------------|------------|------------|" ,
    ]
    for name, r in results["models"].items():
        if name in ("ternair", "fp16"):
            continue
        ppl   = f"{r['perplexity']:.2f}" if math.isfinite(r["perplexity"]) else "inf"
        speed = f"{r['tokens_per_sec']:.1f}" if r["tokens_per_sec"] > 0 else "N/A"
        rows.append(f"| {name} | {ppl} | {r['size_mib']:.1f} | {speed} |")
    t  = results["models"].get("ternair", {})
    fp = results["models"].get("fp16",    {})
    if t and fp and math.isfinite(t.get("perplexity", float("nan"))):
        ppl_delta  = t["perplexity"] - fp["perplexity"]
        size_ratio = fp["size_mib"]         / max(t["size_mib"],         0.001)
        spd_ratio  = t["tokens_per_sec"]    / max(fp["tokens_per_sec"],  0.001)
        rows += [
            "",
            "**Summary:**",
            f"- PPL overhead vs FP16: {ppl_delta:+.2f} ({ppl_delta / max(fp['perplexity'], 0.001) * 100:+.1f}%)",
            f"- Size compression: {size_ratio:.1f}×",
            f"- Speed ratio: {spd_ratio:.2f}×",
        ]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps",           type=int,   default=500)
    p.add_argument("--device",                      default="cpu")
    p.add_argument("--seq-len",         type=int,   default=128)
    p.add_argument("--batch-size",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--warmup-ratio",    type=float, default=0.1)
    p.add_argument("--eval-max-tokens", type=int,   default=20_000)
    p.add_argument("--speed-seq-len",   type=int,   default=64)
    p.add_argument("--output",                      default="benchmark_results.json")
    p.add_argument("--output-md",                   default="benchmark_results.md")
    p.add_argument("--profile",         choices=["tiny", "small"], default="tiny")
    p.add_argument("--storage",         choices=["int8", "packed", "fastpacked"], default="fastpacked")
    p.add_argument("--ternair-ckpt",    default=None)
    p.add_argument("--fp16-ckpt",       default=None)
    # Tokenizer choice — char-level is the correct default for tiny profile
    p.add_argument("--use-char-tokenizer",  dest="use_char", action="store_true",  default=True)
    p.add_argument("--no-char-tokenizer",   dest="use_char", action="store_false")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = args.device

    import datetime
    print("=" * 60)
    print("  Ternair — WikiText-2 Benchmark")
    print("=" * 60)

    tokenizer  = make_tokenizer(args.use_char)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    print("\nLoading WikiText-2...")
    train_tokens = load_wikitext2_tokens(tokenizer, split="train", max_tokens=500_000)
    test_tokens  = load_wikitext2_tokens(tokenizer, split="test",  max_tokens=100_000)
    print(f"  Train tokens : {len(train_tokens):,}")
    print(f"  Test tokens  : {len(test_tokens):,}")

    profile_fn     = tiny_profile if args.profile == "tiny" else small_profile
    cfg            = profile_fn(storage=args.storage)
    cfg.vocab_size = vocab_size

    print(f"\nBuilding Ternair ({args.profile}, hidden={cfg.hidden_size}, vocab={vocab_size})...")
    ternair_model    = TernairForCausalLM(cfg)
    n_params_ternair = sum(p.numel() for p in ternair_model.parameters())
    print(f"  Parameters: {n_params_ternair:,}")

    print(f"Building FP16 baseline (hidden={cfg.hidden_size}, vocab={vocab_size})...")
    fp16_model  = FP16Baseline(
        vocab_size=vocab_size,
        hidden=cfg.hidden_size,
        layers=cfg.num_hidden_layers,
        heads=cfg.num_attention_heads,
    )
    n_params_fp16 = sum(p.numel() for p in fp16_model.parameters())
    print(f"  Parameters: {n_params_fp16:,}")

    if args.ternair_ckpt and Path(args.ternair_ckpt).exists():
        ternair_model.load_state_dict(torch.load(args.ternair_ckpt, map_location=device))
    elif args.steps > 0:
        print(f"\nTraining Ternair ({args.steps} steps)...")
        train(ternair_model, train_tokens, args.steps,
              args.seq_len, args.batch_size, args.lr, device, "ternair", args.warmup_ratio)
        torch.save(ternair_model.state_dict(), "ternair_checkpoint.pt")

    if args.fp16_ckpt and Path(args.fp16_ckpt).exists():
        fp16_model.load_state_dict(torch.load(args.fp16_ckpt, map_location=device))
    elif args.steps > 0:
        print(f"\nTraining FP16 baseline ({args.steps} steps)...")
        train(fp16_model, train_tokens, args.steps,
              args.seq_len, args.batch_size, args.lr, device, "fp16  ", args.warmup_ratio)
        torch.save(fp16_model.state_dict(), "fp16_checkpoint.pt")

    print("\nFreezing Ternair storage...")
    ternair_model.freeze_storage()
    ternair_model.eval()
    fp16_model.eval()

    print("\nEvaluating perplexity on WikiText-2 test...")
    ternair_ppl, ternair_ntok = evaluate_perplexity(
        ternair_model, test_tokens, seq_len=256, stride=128,
        device=device, max_tokens=args.eval_max_tokens)
    print(f"  Ternair PPL : {ternair_ppl:.2f} ({ternair_ntok:,} tokens)")

    fp16_ppl, fp16_ntok = evaluate_perplexity(
        fp16_model, test_tokens, seq_len=256, stride=128,
        device=device, max_tokens=args.eval_max_tokens)
    print(f"  FP16    PPL : {fp16_ppl:.2f} ({fp16_ntok:,} tokens)")

    def ternair_size(m: TernairForCausalLM) -> int:
        total = sum(mod.state_bytes() for mod in m.modules() if hasattr(mod, "state_bytes"))
        if hasattr(m.model, "embed_tokens"):
            total += m.model.embed_tokens.weight.numel() * 2
        return total

    t_bytes = ternair_size(ternair_model)
    f_bytes = fp16_model.num_bytes()

    print("\nSpeed benchmark...")
    prompt      = train_tokens[:args.speed_seq_len]
    t_speed     = benchmark_speed(ternair_model, prompt, args.speed_seq_len, device=device)
    fp_speed    = benchmark_speed(fp16_model,    prompt, args.speed_seq_len, device=device)
    print(f"  Ternair : {t_speed:.1f} tok/s")
    print(f"  FP16    : {fp_speed:.1f} tok/s")

    tok_label = "char-256" if args.use_char else "gpt2-bpe"
    results = {
        "config": {
            "profile":   args.profile,
            "storage":   args.storage,
            "steps":     args.steps,
            "seq_len":   args.seq_len,
            "device":    device,
            "tokenizer": tok_label,
            "date":      datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "models": {
            "Ternair (ternary)": {
                "perplexity":     round(ternair_ppl, 4),
                "num_tokens":     ternair_ntok,
                "size_mib":       round(t_bytes / 1024**2, 2),
                "tokens_per_sec": round(t_speed, 2),
                "params":         n_params_ternair,
            },
            "FP16 baseline": {
                "perplexity":     round(fp16_ppl, 4),
                "num_tokens":     fp16_ntok,
                "size_mib":       round(f_bytes / 1024**2, 2),
                "tokens_per_sec": round(fp_speed, 2),
                "params":         n_params_fp16,
            },
        },
    }
    results["models"]["ternair"] = results["models"]["Ternair (ternary)"]
    results["models"]["fp16"]    = results["models"]["FP16 baseline"]

    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults → {args.output}")

    md = make_markdown_table(results)
    with open(args.output_md, "w") as fh:
        fh.write(md)
    print(f"Markdown → {args.output_md}")

    ppl_delta  = ternair_ppl - fp16_ppl
    size_ratio = (f_bytes / 1024**2) / max(t_bytes / 1024**2, 0.001)
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  {'Model':<22}  {'PPL':>10}  {'MiB':>7}  {'Tok/s':>8}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*7}  {'-'*8}")
    ppl_t_s = f"{ternair_ppl:.2f}" if math.isfinite(ternair_ppl) else "inf"
    ppl_f_s = f"{fp16_ppl:.2f}"    if math.isfinite(fp16_ppl)    else "inf"
    print(f"  {'Ternair (ternary)':<22}  {ppl_t_s:>10}  {t_bytes/1024**2:>7.1f}  {t_speed:>8.1f}")
    print(f"  {'FP16 baseline':<22}  {ppl_f_s:>10}  {f_bytes/1024**2:>7.1f}  {fp_speed:>8.1f}")
    print("=" * 60)
    if math.isfinite(ppl_delta):
        print(f"  PPL overhead   : {ppl_delta:+.2f} ({ppl_delta/max(fp16_ppl,0.001)*100:+.1f}%)")
    print(f"  Size reduction : {size_ratio:.1f}×")
    print(f"  Speed ratio    : {t_speed/max(fp_speed,0.001):.2f}×")
    print("=" * 60)


if __name__ == "__main__":
    main()
