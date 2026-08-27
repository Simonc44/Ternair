#!/usr/bin/env python3
"""Benchmark frozen Ternair inference vs the HF BitNet bf16 reference on CPU.

Measures prefill and decode tokens/second for:

* ``ternair``  -- converted model (packed ternary + cached dequantisation);
* ``bitnet-hf`` -- the same architecture with master bf16 weights
  (``transformers.BitNetForCausalLM``), which does NOT ternarise on the fly
  in transformers 5.x -- so this is an *upper bound* for what a plain HF
  deployment would do.

Usage
-----
.. code-block:: bash

    python scripts/bench_vs_bitnet.py --source ./bitnet-2b4t --output ./ternair-2b4t

    # Or on a random profile (no checkpoint needed)
    python scripts/bench_vs_bitnet.py --profile tiny --seq-len 128 --decode-tokens 64
"""

from __future__ import annotations

import argparse
import time

import torch


def _timeit(fn, warmup: int = 2, runs: int = 5) -> float:
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _bench_ternair(model, ids, decode_tokens: int, device: str) -> dict:
    model.eval()
    with torch.no_grad():
        prefill_s = _timeit(lambda: model(ids), warmup=1, runs=3)
        # Decode: single-token steps with KV-cache disabled fallback is the
        # fair comparison; use full-reshape decode to measure raw throughput.
        one = ids[:, -1:]
        decode_s = _timeit(
            lambda: model(one), warmup=1, runs=3
        )
    prefill_tok = ids.shape[1] / prefill_s
    decode_tok = 1.0 / decode_s
    return {
        "prefill_s": round(prefill_s, 4),
        "prefill_tok_per_s": round(prefill_tok, 1),
        "decode_tok_per_s": round(decode_tok, 1),
    }


def _bench_hf(model, ids, decode_tokens: int) -> dict:
    model.eval()
    with torch.no_grad():
        prefill_s = _timeit(lambda: model(ids), warmup=1, runs=3)
        one = ids[:, -1:]
        decode_s = _timeit(lambda: model(one), warmup=1, runs=3)
    return {
        "prefill_s": round(prefill_s, 4),
        "prefill_tok_per_s": round(ids.shape[1] / prefill_s, 1),
        "decode_tok_per_s": round(1.0 / decode_s, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Ternair vs HF BitNet on CPU")
    parser.add_argument("--source", default=None, help="BitNet checkpoint dir (optional)")
    parser.add_argument("--output", default=None, help="Converted Ternair package (optional)")
    parser.add_argument("--profile", default="tiny", help="Profile when no checkpoint is used")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device

    if args.output and args.source:
        from ternair.model.bitnet_converter import convert_bitnet_checkpoint, load_converted_model

        print(f"=== Converting {args.source} -> {args.output} ===")
        convert_bitnet_checkpoint(args.source, args.output, storage="packed")
        ternair_model, _ = load_converted_model(args.output, device=device)
        from transformers import AutoModelForCausalLM, AutoConfig

        hf_cfg = AutoConfig.from_pretrained(args.source, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_config(hf_cfg)
        vocab = hf_cfg.vocab_size
    else:
        from ternair import TernairForCausalLM, tiny_profile
        from ternair.model.bitnet_converter import bitnet_config_to_ternair

        # IMPORTANT: use the attention-pure config (no SSM layers) so the
        # comparison is architecture-for-architecture identical to BitNet
        # (BitNet b1.58 is a plain GQA transformer).  The default ``tiny``
        # profile mixes in SSM blocks, which are a different architecture
        # and would make the benchmark unfair.
        cfg = tiny_profile(storage="packed")
        cfg.num_attn_layers = cfg.num_hidden_layers
        cfg.attn_layer_period = 1
        ternair_model = TernairForCausalLM(cfg)
        ternair_model.freeze_storage()
        vocab = ternair_model.config.vocab_size

        # HF BitNet reference from the same config.
        try:
            from transformers.models.bitnet import BitNetConfig, BitNetForCausalLM

            tc = ternair_model.config
            hf_cfg = BitNetConfig(
                vocab_size=tc.vocab_size, hidden_size=tc.hidden_size,
                intermediate_size=tc.intermediate_size,
                num_hidden_layers=tc.num_hidden_layers,
                num_attention_heads=tc.num_attention_heads,
                num_key_value_heads=tc.num_key_value_heads,
                max_position_embeddings=tc.max_position_embeddings,
                rope_theta=tc.rope_theta, rms_norm_eps=tc.rms_norm_eps,
                tie_word_embeddings=tc.tie_word_embeddings,
                torch_dtype=torch.float32,
            )
            hf_model = BitNetForCausalLM(hf_cfg)
            hf_model.eval()
        except Exception as exc:
            print(f"[warn] HF BitNet reference unavailable: {exc}")
            hf_model = None

    torch.manual_seed(0)
    ids = torch.randint(0, vocab, (1, args.seq_len))

    print("=== Ternair (frozen, packed) ===")
    r1 = _bench_ternair(ternair_model, ids, args.decode_tokens, device)
    print(f"  prefill: {r1['prefill_s']}s  ({r1['prefill_tok_per_s']} tok/s)  "
          f"decode: {r1['decode_tok_per_s']} tok/s")

    if hf_model is not None:
        print("=== BitNet HF (bf16 master, transformers 5.x) ===")
        r2 = _bench_hf(hf_model, ids, args.decode_tokens)
        print(f"  prefill: {r2['prefill_s']}s  ({r2['prefill_tok_per_s']} tok/s)  "
              f"decode: {r2['decode_tok_per_s']} tok/s")
        print(
            f"  speedup (ternair/hf): prefill "
            f"{r1['prefill_tok_per_s'] / r2['prefill_tok_per_s']:.2f}x, "
            f"decode {r1['decode_tok_per_s'] / r2['decode_tok_per_s']:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
