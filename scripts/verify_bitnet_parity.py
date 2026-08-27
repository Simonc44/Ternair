#!/usr/bin/env python3
"""Verify a converted BitNet b1.58 checkpoint against the HF reference.

Workflow:
1. (optional) download the checkpoint with ``hf download`` (or
   ``huggingface-cli`` on huggingface_hub < 1.0);
2. convert it with Ternair (``convert_bitnet_checkpoint``);
3. load the HF reference with ``transformers`` (master bf16 weights);
4. compare logits: Pearson correlation + top-1 agreement;
5. measure WikiText-2 perplexity of the converted model (optional).

Usage
-----
.. code-block:: bash

    # Local checkpoint
    python scripts/verify_bitnet_parity.py --source ./bitnet-2b4t --output ./ternair-2b4t

    # With perplexity on a token file (one document per line, optional)
    python scripts/verify_bitnet_parity.py --source ./bitnet-2b4t --output ./ternair-2b4t \\
        --wikitext ./wikitext-2-raw.txt --max-eval-tokens 4096

Exit code is 0 on success, 1 when the parity checks fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _load_ref_logits(model_dir: str, ids: torch.Tensor) -> torch.Tensor:
    """Load the HF reference model and return its logits for ``ids``.

    Two checkpoint formats are supported:

    * master bf16 weights (older BitNet checkpoints): the plain model runs
      the bf16 forward, no quantisation is applied;
    * the official offline U8 format (the current 2B-4T): weights are
      already-ternarised U8 packed projections + one ``weight_scale`` per
      tensor.  ``AutoBitLinear`` is installed so the reference runs the
      official pipeline: 8-bit per-token activation quantisation,
      ternary weights, and the per-tensor scale.
    """
    from types import SimpleNamespace

    # The checkpoint's config.json declares custom code (``auto_map``) but
    # the repo does NOT ship ``configuration_bitnet.py`` -- the official
    # class lives in transformers' native ``bitnet`` module (5.x).  Load
    # the concrete class directly to avoid the remote-code fetch that
    # would fail on the missing file.
    from transformers.models.bitnet import BitNetConfig, BitNetForCausalLM

    config = BitNetConfig.from_pretrained(model_dir)
    # Build the reference directly in bf16 (the checkpoint dtype): a
    # float32 build would need ~8 GB of RAM for the 2B model and can OOM
    # the CI runner.
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = BitNetForCausalLM(config)
    finally:
        torch.set_default_dtype(old)

    qc = getattr(config, "quantization_config", None)
    if qc:
        # Official offline-U8 checkpoint: install AutoBitLinear (8-bit
        # activation quantisation + ternary weights + weight_scale).  Its
        # load_state_dict pre-hook unpacks the U8 projections.
        from transformers.integrations.bitnet import replace_with_bitnet_linear

        qc = dict(qc) if isinstance(qc, dict) else {}
        qc.setdefault("use_rms_norm", False)
        qc.setdefault("rms_norm_eps", 1e-6)
        replace_with_bitnet_linear(
            model,
            quantization_config=SimpleNamespace(**qc),
            modules_to_not_convert=["lm_head"],
        )
        # The replacement runs under torch.device("meta"); materialise.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to_empty(device=device)
        model.to(torch.bfloat16)

    # Stream weights from the checkpoint directory, one shard at a time.
    from safetensors import safe_open

    shards = sorted(
        p for p in Path(model_dir).glob("*.safetensors") if "index" not in p.name
    )
    if not shards:
        shards = [Path(model_dir) / "model.safetensors"]
    with torch.no_grad():
        for shard in shards:
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                loaded = {k: f.get_tensor(k) for k in f.keys()}
            model.load_state_dict(loaded, strict=False)
            del loaded

    # Tie the LM head to the embedding (the checkpoint omits it).
    if model.config.tie_word_embeddings and model.lm_head is not None:
        with torch.no_grad():
            model.lm_head.weight.copy_(model.model.embed_tokens.weight)

    model.eval()
    with torch.no_grad():
        out = model(ids)
    logits = out.logits if hasattr(out, "logits") else out
    return logits.float()


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def _top1_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.argmax(-1) == b.argmax(-1)).float().mean().item())


def main() -> int:
    # The reference's AutoBitLinear/ActQuant are @torch.compile-decorated;
    # fall back to eager everywhere (no C++ compiler needed, and faster on
    # the CI runner than JIT-compiling a 2B model).
    torch._dynamo.config.suppress_errors = True

    parser = argparse.ArgumentParser(description="Verify BitNet -> Ternair parity")
    parser.add_argument("--source", required=True, help="BitNet checkpoint dir (config.json + model.safetensors)")
    parser.add_argument("--output", required=True, help="Where to write the converted Ternair package")
    parser.add_argument("--storage", default="packed", choices=["packed", "fastpacked"])
    parser.add_argument("--prompt", default="The future of artificial intelligence is", help="Prompt for logit comparison")
    parser.add_argument("--min-correlation", type=float, default=0.7, help="Minimum Pearson correlation to pass")
    parser.add_argument("--min-top1", type=float, default=0.3, help="Minimum top-1 agreement to pass")
    parser.add_argument("--wikitext", default=None, help="Optional .txt file (one doc per line) for perplexity")
    parser.add_argument("--max-eval-tokens", type=int, default=4096)
    parser.add_argument("--json", default=None, help="Write results as JSON")
    args = parser.parse_args()

    from ternair.model.bitnet_converter import convert_bitnet_checkpoint, load_converted_model

    print(f"=== Converting {args.source} -> {args.output} (storage={args.storage}) ===")
    report = convert_bitnet_checkpoint(args.source, args.output, storage=args.storage)
    d = report.as_dict()
    print(f"  tensors loaded : {d['n_loaded_tensors']}  ignored: {d['n_ignored_tensors']} {d['ignored_keys']}")
    print(f"  size           : {d['size_mib']:.2f} MiB (FP16 equiv {d['fp16_equivalent_mib']:.2f} MiB)")
    if d["n_ignored_tensors"]:
        print("  [FAIL] ignored tensors present", file=sys.stderr)
        return 1

    print("=== Logit parity vs HF reference ===")
    ids = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.source)
        ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]
    except Exception as exc:
        print(f"  [warn] tokenizer failed ({exc}); using random ids")
    if ids is None:
        vocab = d.get("n_ternary_params", 0)
        # Read vocab from the source config.json.
        import json as _json
        from pathlib import Path as _P
        cfg_path = _P(args.source) / "config.json"
        try:
            vocab = _json.loads(cfg_path.read_text()).get("vocab_size", 32000)
        except Exception:
            vocab = 32000
        ids = torch.randint(0, max(vocab - 1, 2), (1, min(16, max(vocab - 1, 2))))

    ref = _load_ref_logits(args.source, ids)
    model, _ = load_converted_model(args.output, device="cpu", dtype=torch.bfloat16)
    model.eval()
    with torch.no_grad():
        got = model(ids).float()

    corr = _pearson(ref, got)
    top1 = _top1_agreement(ref, got)
    print(f"  Pearson corr   : {corr:.4f}  (threshold {args.min_correlation})")
    print(f"  top-1 agreement: {top1:.4f}  (threshold {args.min_top1})")
    ok = corr >= args.min_correlation and top1 >= args.min_top1
    print(f"  -> {'PASS' if ok else 'FAIL'}")

    results = {
        "conversion": d,
        "parity": {"pearson_correlation": corr, "top1_agreement": top1, "passed": ok},
    }

    if args.wikitext:
        print("=== Perplexity (converted model) ===")
        ppl = _measure_perplexity(model, args.wikitext, args.max_eval_tokens)
        print(f"  perplexity: {ppl:.2f}")
        results["perplexity"] = ppl

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"  results -> {args.json}")

    return 0 if ok else 1


def _measure_perplexity(model, text_file: str, max_tokens: int) -> float:
    """Sliding-window next-token perplexity over a raw text file."""
    import torch.nn.functional as F

    try:
        from transformers import AutoTokenizer
    except ImportError:
        return float("nan")

    tokenizer = AutoTokenizer.from_pretrained(str(Path(text_file).parent) if Path(text_file).suffix else "")
    # Fall back to a char tokenizer when no HF tokenizer is around.
    if tokenizer is None:
        return float("nan")

    text = Path(text_file).read_text(encoding="utf-8", errors="replace")
    ids = tokenizer(text[: 4 * max_tokens])["input_ids"] if hasattr(tokenizer, "__call__") else tokenizer.encode(text[: 4 * max_tokens])
    ids = torch.tensor([ids[:max_tokens]], dtype=torch.long)

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    seq_len = 128
    with torch.no_grad():
        for start in range(0, ids.shape[1] - seq_len - 1, seq_len):
            chunk = ids[:, start : start + seq_len + 1]
            logits = model(chunk[:, :-1]).float()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), chunk[:, 1:].reshape(-1)
            )
            total_loss += loss.item() * (seq_len)
            total_tokens += seq_len
    return float(torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item())


if __name__ == "__main__":
    sys.exit(main())
