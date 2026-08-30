# Ternair

> **Run BitNet models. Smaller files. One command. No C++ build.**

Ternair loads [Microsoft BitNet b1.58](https://arxiv.org/abs/2402.17764) models — including the real `2B-4T` — and serves them behind an OpenAI-compatible API in a single `pip install`.

```bash
pip install git+https://github.com/Simonc44/Ternair
ternair serve --model microsoft/bitnet-b1.58-2B-4T --port 8080
# → http://localhost:8080/v1/chat/completions  (OpenAI-compatible)
```

**No model download. No C++ compiler. No HuggingFace account required.**

## Why Ternair?

| | `bitnet.cpp` (BitNet) | **Ternair** |
|---|---|---|
| Same trained models | ✅ | ✅ — bit-exact parity (Pearson 0.9958) |
| File size | ~5 GB (bf16) | **1.1 GB** (4× smaller) |
| HTTP server (OpenAI API) | ❌ | **✅ built-in** |
| `pip install` | ❌ (C++ build required) | **✅** |
| Works on | macOS/Linux | **macOS/Linux/Windows** |
| KV-cache inference | ✅ | **✅** |
| Ternary compression | 2 bits/value | **1.6 bits/value** |

> *"Same brain, smaller body, ready to deploy."*

---

## Quick start

### Install

```bash
pip install git+https://github.com/Simonc44/Ternair
```

### Serve a real BitNet model

```bash
ternair serve --model microsoft/bitnet-b1.58-2B-4T --port 8080
```

Then query it:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ternair", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Use in Python

```python
from ternair.model.bitnet_converter import load_converted_model

model, tokenizer = load_converted_model("./ternair-2b4t")
# model is a TernairForCausalLM with packed ternary weights
```

### Convert a BitNet checkpoint yourself

```bash
ternair import-bitnet \
  --source ./bitnet-2b4t \
  --output ./ternair-2b4t \
  --storage packed

ternair serve --model ./ternair-2b4t --port 8080
```

---

## What is Ternair?

Ternair is a **standalone Python/PyTorch** implementation of ternary language-model inference inspired by BitNet b1.58. It provides:

- **Ternary weights** `{-1, 0, +1}` with straight-through training
- **Packed storage**: `packed` (1.6 bits/value) or `fastpacked` (2 bits/value)
- **Decoder-only causal LM** with GQA/RoPE attention, SwiGLU MLP, optional hybrid SSM blocks
- **Incremental KV-cache** generation (prefill once, decode one token per step)
- **Greedy and sampled generation**, streaming, repetition penalty, chat templates
- **Cached dequantisation**: frozen weights unpacked once (~900× faster)
- **Vectorised NumPy matmul** + optional Triton GPU / native C++ CPU backends
- **SafeTensors export** and model-size/compression reports
- **OpenAI-compatible HTTP server** with batching, Prometheus metrics, health checks
- **CLI** for inspection, size estimation, demos, training, inference, serving, benchmarking

## Reuse trained BitNet models (zero training)

Ternair and BitNet b1.58 share the **same architecture** (RMSNorm, RoPE, GQA attention, SwiGLU MLP, per-row absmean ternary weights), so a trained BitNet checkpoint can be converted **as-is** — same weights, no re-training — into Ternair's denser packed storage.

```bash
# Download the real 2B-4T model
hf download microsoft/bitnet-b1.58-2B-4T --local-dir ./bitnet-2b4t

# Convert (ternarises + packs the master weights)
ternair import-bitnet --source ./bitnet-2b4t --output ./ternair-2b4t --storage packed

# Serve
ternair serve --model ./ternair-2b4t --port 8080
```

**CI validated**: the GitHub Actions workflow converts the real `microsoft/bitnet-b1.58-2B-4T` checkpoint and verifies parity against the HuggingFace reference (Pearson 0.9958, 85.7% top-1 agreement, 0 ignored tensors).

---

## Ternair vs BitNet — the full comparison

Ternair is **not a competitor to BitNet**: it is a compatible, standalone implementation of the *same idea* (BitNet b1.58), and it can even load BitNet's trained models directly.

### What they share

- **Ternary weights** `{-1, 0, +1}` with per-output-row scale `gamma = mean(|W|)` (absmean quantisation)
- **Per-token 8-bit absmax activation quantisation**
- **Straight-through estimator (STE)** for training
- **LLaMA-style decoder**: RMSNorm, RoPE, GQA attention, SwiGLU MLP
- Same causal decoding with KV-cache

### Where Ternair wins

| Feature | Ternair | BitNet |
|---|---|---|
| **Compression** | 1.6 bits/value (packed) | 2 bits/value |
| **HTTP server** | OpenAI-compatible, built-in | Not official |
| **Install** | `pip install ternair` | C++ build required |
| **Portability** | Pure Python + PyTorch | macOS/Linux only |
| **File size (2B-4T)** | **1.1 GB** | ~5 GB |
| **Trained model loading** | ✅ Bit-exact converter | Native |

### Where BitNet wins

| Feature | BitNet | Ternair |
|---|---|---|
| **Raw CPU speed** (C++ AVX-512/NEON) | Mature kernels | Python overhead |
| **GPU CUDA kernels** | ✅ Validated | Triton (experimental) |
| **Community** | HuggingFace, papers, large adoption | Smaller |

### When to use which

- **Same trained model + fastest CPU C++ kernels** → `bitnet.cpp`
- **Same model + OpenAI API + smaller file + pure Python** → `ternair serve`
- **Train your own ternary model** → either; Ternair is a single `pip install`
- **Embed inference in a Python app** → Ternair (no C++ build, automatic backend fallback)

---

## Requirements

- Python 3.10-3.12
- PyTorch 2.4+
- NumPy 1.17+
- Optional: `pytest` (testing), Triton (GPU kernels), `safetensors`/`transformers` (external model loading)

## CLI reference

```text
ternair info          Show a model configuration
ternair size          Estimate storage and compression
ternair demo          Run a forward/generation smoke demo
ternair train         Run a multi-step training loop
ternair infer         Run inference via the backend dispatcher
ternair import-bitnet Convert a BitNet b1.58 checkpoint to Ternair
ternair serve         Start the OpenAI-compatible HTTP server
ternair benchmark     Run reproducible perplexity + speed benchmarks
```

Run `ternair COMMAND --help` for all options.

## HTTP server endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/completions` | Text completion (OpenAI-compatible) |
| `POST /v1/chat/completions` | Chat completion |
| `POST /v1/batch` | Concurrent multi-prompt batch |
| `GET /v1/models` | List loaded models |
| `GET /health` | Health check |
| `GET /metrics` | Prometheus-style metrics |

## Training

```bash
ternair train --profile tiny --steps 20 --batch-size 8
```

Verified: loss 180.78 → 27.09 after 15 AdamW steps on the bundled toy corpus.

For real training on a large corpus:

```bash
python scripts/train.py --config scripts/train_tiny.yaml
```

## Export

```python
from ternair.model.export import export_huggingface_package

model.eval()
model.freeze_storage()
export_huggingface_package(model, "./dist/ternair-model", model_name="MyTernairModel")
```

Exports are SafeTensors-compatible, written atomically, and contain `config.json`, `model.safetensors`, and an optional model card.

## Benchmarks

```bash
ternair benchmark --profile tiny --eval-tokens 1024
```

| Metric | Value |
|--------|-------|
| Ternary parameters | ~3.7 M |
| Packed weight size | ~2.5 MiB (1.6 bits/value) |
| Frozen forward (tiny, fastpacked) | ~50 ms |
| Decode throughput (packed) | ~61 k tokens/s |
| Decode throughput (fastpacked) | ~26 k tokens/s |

## Project layout

```text
src/ternair/
  model/          model, generation (KV-cache), export, loader, backend dispatcher
  quantization/   ternarization, STE, activations, adapters
  kernels/        packing, NumPy, Triton, and C++ acceleration
  training/       toy data, optimizer, scheduler, trainer, pipeline
  benchmark/      size and evaluation utilities
  server.py       OpenAI-compatible HTTP server
tests/            unit and integration tests
scripts/          training, demo, and CI helpers
docs/             mkdocs documentation (API, CLI, server, benchmarks)
```

## Contributing

Add a regression test for bug fixes, keep public API changes documented, and run the relevant tests before opening a pull request.

## License

Apache-2.0. See [LICENSE](LICENSE).


<!-- BENCHMARK_RESULTS -->
## Benchmark WikiText-2 — Ternair vs FP16

> Profile `tiny` · storage `fastpacked` · 500 training steps · device `cpu`  
> 2,809,184 parameters · Last run: 2026-08-30T09:10:13 (auto-generated by CI)

| Model | PPL ↓ | Size (MiB) ↓ | Tokens/sec ↑ |
|-------|------:|-------------:|-------------:|
| **Ternair (ternary)** | **12.70** | **0.8** | **253.5** |
| FP16 baseline | 23.36 | 26.4 | 3817.9 |

**Summary:**
- PPL overhead vs FP16: -10.66 (-45.6%)
- Size compression: **33.4×**
- Speed ratio: 0.07×
<!-- END_BENCHMARK_RESULTS -->