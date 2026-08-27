# Ternair

**Ternary language model inference engine based on BitNet b1.58**

Ternair implements ternary weight quantization (`{-1, 0, +1}`) for language models, achieving ~16x compression compared to FP16. It provides a complete inference pipeline: model training, quantization-aware training, export to SafeTensors, and an OpenAI-compatible HTTP server.

## Quick Start

```bash
pip install git+https://github.com/Simonc44/Ternair.git
python -m ternair demo --profile tiny
```

## Features

- **Ternary weights**: `{-1, 0, +1}` with per-channel learned alpha
- **Multiple storage formats**: base-3 (1.6 b/v), 2-bit (2.0 b/v), int8
- **Hybrid architecture**: GQA attention + SSM blocks with configurable ratio
- **KV-cache**: Incremental decoding with correct RoPE positional offsets
- **OpenAI server**: `/v1/completions`, `/v1/chat/completions`, `/v1/batch`
- **Backends**: PyTorch (reference), NumPy, Triton GPU, C++ SIMD
- **Export**: SafeTensors, HuggingFace-compatible packages, GGUF
- **CLI**: `info`, `size`, `demo`, `train-one`, `infer`, `serve`, `benchmark`

## Architecture

```
TernairConfig
  ├── hidden_size, num_hidden_layers, num_attention_heads
  ├── storage: "packed" | "fastpacked" | "int8"
  │
  ├── TernairForCausalLM
  │   ├── TernairEmbedding (tied weights)
  │   ├── TernairHybridBlock × N
  │   │   ├── TernairBlock (GQA + SwiGLU)  ← attention layers
  │   │   └── TernarySSMBlock              ← SSM layers
  │   └── TernairLMHead
  │
  └── TernairDirectInferencer (backend dispatcher)
```

## Links

- [API Reference](api.md)
- [CLI Reference](cli.md)
- [Server Guide](server.md)
- [Benchmarks](benchmarks.md)
- [GitHub](https://github.com/Simonc44/Ternair)
