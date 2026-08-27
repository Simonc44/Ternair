# Ternair

Ternair is a Python/PyTorch implementation of ternary language-model components inspired by BitNet b1.58. It provides ternary weight quantization, packed storage, a decoder-only model, a correct incremental KV-cache, portable inference fallbacks, SafeTensors export, an OpenAI-compatible HTTP server, and optional acceleration backends.

> **Status:** beta research software. The PyTorch reference path and the tested CPU/NumPy paths are the supported baseline. Triton and the native C++ engine are optional accelerators and must be benchmarked on the target machine.

## What is supported

- Ternary weights in `{-1, 0, +1}` with straight-through training.
- `packed` base-3 storage (5 trits/byte, 1.6 bits/value) and `fastpacked` 2-bit storage (4 trits/byte).
- Decoder-only causal LM with GQA/RoPE attention, SwiGLU MLP, and optional hybrid SSM blocks.
- Incremental KV-cache generation (prefill once, then one token per step with correct RoPE position offsets).
- Greedy and sampled generation, streaming generation, repetition penalty, and chat templates.
- Cached dequantisation: frozen weights are unpacked once, not on every forward (~900x faster CPU frozen forward vs the naive path).
- Vectorised NumPy matmul (single LUT decode + one BLAS call instead of nested Python loops).
- Optional Triton GPU and native C++ CPU backends, with automatic fallback.
- Atomic SafeTensors export and model-size/compression reports.
- OpenAI-compatible HTTP server (`/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/health`, `/metrics`) with batching.
- CLI commands for inspection, size estimation, demos, real training, inference, serving, and benchmarking.

## Requirements

- Python 3.10-3.12
- PyTorch 2.4 or newer
- NumPy 1.17 or newer
- Optional: `pytest` for development, Triton for CUDA kernels, `safetensors` and `transformers` for external model loading.

Install from a checkout:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Development tools:

```bash
python -m pip install pytest build
```

## Quick start

```bash
python -m ternair --help
python -m ternair info --profile tiny
python -m ternair size --profile tiny
python -m ternair demo --profile tiny --max-new-tokens 8
```

Python API:

```python
import torch
from ternair import TernairForCausalLM, tiny_profile, generate

config = tiny_profile(storage="fastpacked")
model = TernairForCausalLM(config)
model.freeze_storage()
model.eval()

prompt = torch.randint(0, config.vocab_size, (1, 8))
with torch.no_grad():
    output = generate(model, prompt, max_new_tokens=16, temperature=0.0)
print(output.shape)
```

## Reuse trained BitNet b1.58 models (zero training)

Ternair and BitNet b1.58 share the **same architecture** (RMSNorm, RoPE,
GQA attention, SwiGLU MLP, per-row absmean ternary weights), so a trained
BitNet b1.58 checkpoint (e.g. `microsoft/bitnet-b1.58-2B-4T`) can be
converted **as-is** — same trained weights, no re-training — into Ternair's
denser packed storage. You get the full BitNet model catalog with Ternair's
extras (1.6 bits/value packing, KV-cache generation, HTTP server, portable
backends).

```bash
# 1. Download a trained BitNet b1.58 checkpoint (config.json + model.safetensors)
hf download microsoft/bitnet-b1.58-2B-4T --local-dir ./bitnet-2b4t

# 2. Convert it to Ternair (ternarises + packs the master weights)
python -m ternair import-bitnet --source ./bitnet-2b4t --output ./ternair-2b4t --storage packed

# 3. Serve it (tokenizer copied automatically)
python -m ternair serve --model ./ternair-2b4t --port 8080
```

Python API:

```python
from ternair import convert_bitnet_checkpoint, load_converted_model

report = convert_bitnet_checkpoint("./bitnet-2b4t", "./ternair-2b4t")
model, tokenizer = load_converted_model("./ternair-2b4t")
```

How it works: BitNet b1.58 checkpoints store **master bf16 weights** and
ternarise at inference with `gamma = mean(|W|)` per row. The converter does
that once, packs the ternary weights (`packed` = 1.6 bits/value,
`fastpacked` = 2 bits/value), and writes a native Ternair package. The
converted model is numerically identical to the frozen BitNet model (the
round-trip test asserts bit-exact logits vs. a reference built from the same
master weights). The official BitNet b1.58 sub-layer normalisation
(`attn_sub_norm` / `ffn_sub_norm`) is detected from the checkpoint keys and
enabled automatically.

### CI validation of the real checkpoint

A [GitHub Actions workflow](.github/workflows/bitnet-convert.yml) validates
the trained-model path end-to-end on the **real** `microsoft/bitnet-b1.58-2B-4T`:

```bash
# In the repo, GitHub UI: Actions -> bitnet-convert -> Run workflow
# (downloads ~5 GB, converts, checks parity vs the HuggingFace reference,
# uploads the converted package as an artifact)
```

Run it locally:

```bash
hf download microsoft/bitnet-b1.58-2B-4T --local-dir ./bitnet-2b4t
python scripts/verify_bitnet_parity.py --source ./bitnet-2b4t --output ./ternair-2b4t
python scripts/bench_vs_bitnet.py --source ./bitnet-2b4t --output ./ternair-2b4t
```

`verify_bitnet_parity.py` compares the converted model's logits against the
HuggingFace reference (Pearson correlation + top-1 agreement) and can
measure WikiText-2 perplexity. `bench_vs_bitnet.py` measures prefill/decode
throughput of Ternair vs the HF bf16 reference on the same machine.

## Ternair vs BitNet — the full comparison

Ternair is **not a competitor to BitNet**: it is a compatible, standalone
implementation of the *same idea* (BitNet b1.58), and it can even load
BitNet's trained models directly (see above). This section lays out exactly
what each project is, what they share, where they differ, and who wins on
each axis — with no marketing.

### TL;DR

| | Microsoft BitNet b1.58 | Ternair |
|---|---|---|
| **What it is** | Research project + C++ inference engine (`bitnet.cpp`) + trained open-weight models | Standalone Python/PyTorch implementation + tooling |
| **Trained models** | ✅ Ships `bitnet-b1.58-2B-4T` (MIT, ~2.4 B params, ~4 T tokens) | ✅ Loads the same trained models via the converter — same weights, denser storage |
| **Compression** | ~2 bits/value (1.58-bit math, 4 trits/byte) | ✅ ~1.6 bits/value (`packed`, 5 trits/byte) |
| **Raw CPU speed** | ✅ Mature AVX-512 / NEON kernels in C++ | ⚠️ Python overhead, but wins on decode throughput (see benchmarks) |
| **GPU** | ✅ CUDA kernels in `bitnet.cpp` | ⚠️ Optional Triton kernel; PyTorch fallback |
| **Portability** | C++ / llama.cpp ecosystem | ✅ Pure Python + PyTorch, zero mandatory HF dependency |
| **HTTP server** | ❌ Not official | ✅ OpenAI-compatible built-in |
| **Training framework** | ✅ 1.58-bit QAT training framework | ✅ PyTorch trainer (STE, annealing, WSD) |
| **Ecosystem** | HuggingFace hub, large community | ✅ HF-compatible export + converter; smaller community |

### What they share (the common core)

BitNet b1.58 and Ternair implement the **same recipe**, so weights are
interchangeable:

- **Ternary weights** `{-1, 0, +1}` with per-output-row scale
  `gamma = mean(|W|)` (absmean quantisation).
- **Per-token 8-bit absmax activation quantisation**.
- **Straight-through estimator (STE)** for training.
- **LLaMA-style decoder**: RMSNorm, RoPE, GQA attention, SwiGLU MLP.
- Same causal decoding with KV-cache.

This is why a trained BitNet checkpoint can be converted to Ternair with
bit-exact fidelity instead of being re-trained.

### Where they actually differ

**1. What ships in the box.**
BitNet is a research program: papers ("The Era of 1-bit LLMs"), the
`bitnet.cpp` C++ inference engine, and trained open-weight models on the
HuggingFace hub. Ternair is a single installable Python package that ships
the full stack: model, quantisation, packing, training, export, server.

**2. Packing density.**
Ternair's `packed` storage uses a base-3 codec with **5 trits per byte
(1.6 bits/value)** vs the 4 trits per byte (2 bits/value) of a plain 2-bit
codec. On the same ternary weights, Ternair is ~20% smaller on disk.
`fastpacked` (2 bits/value) exists for kernels that need the simpler layout.

**3. Inference backends.**
`bitnet.cpp` is C++ with hand-tuned AVX-512/NEON kernels — the reference for
CPU ternary speed. Ternair runs everywhere PyTorch runs and picks a backend
per device: Triton on CUDA, C++ when available, vectorised NumPy otherwise,
with the PyTorch path as the always-correct fallback. A cached dequantisation
makes the frozen PyTorch path ~900x faster than the naive unpack-every-forward
approach.

**4. Speed on commodity hardware — the honest truth.**
On a standard GPU, neither ternary implementation beats cuBLAS FP16 for
compute-bound workloads: GPUs are built for dense FP16 matmuls, and
add/subtract ternary matmuls are memory-bound. Ternary wins in the
**memory-bound regime**: batch=1 decode, long context, and bit-level CPU
SIMD (AVX-512) or specialised hardware (FPGA/ASIC). Do not trust any
benchmark — including ours — without measuring on your target machine.

**5. Quality.**
BitNet ships a *trained* 2B-4T model with real perplexity. Ternair's
shipped profiles are randomly initialised; the `train` path exists to train
your own. The honest quality comparison today:

- Same trained weights (via converter): **identical quality**, smaller files.
- Trained from scratch: Ternair matches BitNet's recipe; final quality is a
  function of data + compute, not of the implementation.

**6. Operations.**
Ternair has an OpenAI-compatible HTTP server (`/v1/completions`,
`/v1/chat/completions`, `/v1/batch`, `/metrics`) and atomic SafeTensors
export. `bitnet.cpp` provides a CLI and llama.cpp-style tooling; no official
HTTP server.

### When to use which

- **You want the trained 2B-4T model with the fastest possible CPU C++
  kernels** → use `bitnet.cpp` directly with the official checkpoint.
- **You want the same trained model behind an OpenAI-compatible API, in a
  smaller file, in pure Python, with parity CI and benchmarks** → convert
  with `ternair import-bitnet`, validate with
  `scripts/verify_bitnet_parity.py`, and serve with `ternair serve --model ...`.
- **You want to train your own ternary model** → both work; Ternair is a
  single `pip install` with no separate framework.
- **You want to embed inference in a Python app** → Ternair; no C++ build
  step, automatic backend fallback.

The two projects are complementary: Ternair consumes BitNet's trained
weights and adds denser packing, a server, and a Python-first toolchain;
`bitnet.cpp` remains the reference for hand-tuned C++ CPU kernels.

## Training (quality)

Ternair is also a *training* framework: the shipped profiles are randomly initialised. Quality comparable to BitNet requires training, which is exactly what the training path is for.

Run a real multi-step training loop on the bundled toy corpus (no HuggingFace dependency needed) and watch loss drop:

```bash
python -m ternair train --profile tiny --steps 20 --batch-size 8
```

Verified on CPU (tiny profile, seed 42):

```text
loss : 180.78 -> 27.09  after 15 AdamW steps (plain PyTorch, no accelerate)
```

The loop supports the WSD schedule, gradient clipping, evaluation, checkpoints, and the HuggingFace `fineweb-edu` pipeline through `scripts/train.py`:

```bash
python scripts/train.py --config scripts/train_tiny.yaml
```

To reproduce the quality numbers claimed in BitNet literature you must train on a real corpus (days of GPU time for a small model). The toy corpus is a smoke test, not a benchmark.

## Inference backends

Use `TernairDirectInferencer` when backend selection should be explicit:

```python
from ternair import TernairDirectInferencer

inferencer = TernairDirectInferencer(model, backend="auto", device="cpu")
inferencer.prepare()
print(inferencer.describe())
logits = inferencer.forward(prompt)
```

`auto` prefers Triton on CUDA, the native C++ backend on CPU when available, and otherwise the PyTorch reference path. A requested backend may fall back when its storage or hardware constraints are not met. Always measure latency and numerical parity on deployment hardware.

## HTTP server

```bash
python -m ternair serve --profile tiny --port 8080
```

| Endpoint | Description |
|----------|-------------|
| `POST /v1/completions` | Text completion (OpenAI-compatible) |
| `POST /v1/chat/completions` | Chat completion |
| `POST /v1/batch` | Concurrent multi-prompt batch |
| `GET /v1/models` | List loaded models |
| `GET /health` | Health check |
| `GET /metrics` | Prometheus-style metrics (requests, tokens, latency) |

Thread-safe inference engine with a single model lock; zero external dependencies (stdlib HTTP server).

## Export

```python
from ternair.model.export import export_huggingface_package

model.eval()
model.freeze_storage()
export_huggingface_package(model, "./dist/ternair-model", model_name="MyTernairModel")
```

Exports are SafeTensors-compatible, written atomically (tmp + fsync + `os.replace`), and use canonical keys (`packed_weight`, `gamma_eval`). The package contains `config.json`, `model.safetensors`, and an optional model card. Loading external LLaMA/Mistral-style bundles requires the optional `transformers` and `safetensors` dependencies:

```bash
python -m pip install transformers safetensors
```

## Benchmarks

Run the reproducible benchmark (synthetic deterministic eval set, separate prefill/decode timings, JSON output):

```bash
python -m ternair benchmark --profile tiny --eval-tokens 1024
```

Measured on this project's tiny profile (CPU, after the dequantisation-cache and vectorised-matmul optimisations):

| Metric | Value |
|--------|-------|
| Ternary parameters | ~3.7 M |
| Packed weight size | ~2.5 MiB (1.6 bits/value packed) |
| Frozen forward (tiny, fastpacked) | ~50 ms (was 45.9 s before caching) |
| Decode throughput (packed) | ~61 k tokens/s |
| Decode throughput (fastpacked) | ~26 k tokens/s |

See the [Ternair vs BitNet — the full comparison](#ternair-vs-bitnet--the-full-comparison) section for the complete, honest breakdown.

## CLI reference

```text
ternair info          Show a model configuration
ternair size          Estimate storage and compression
ternair demo          Run a forward/generation smoke demo
ternair train-one     Run one toy training step
ternair train         Run a real multi-step training loop and report loss/PPL reduction
ternair infer         Run inference through the backend dispatcher
ternair import-bitnet Convert a trained BitNet b1.58 checkpoint into a Ternair package
ternair serve         Start the OpenAI-compatible HTTP server
ternair benchmark     Run reproducible perplexity + speed benchmarks
```

Run `python -m ternair COMMAND --help` for all options.

## Testing and release checks

From the repository root:

```bash
python -m pytest -q tests/test_kernels.py tests/test_quantization.py tests/test_size.py
python -m pytest -q tests/test_model.py tests/test_ssm.py
python -m pytest -q tests/test_production_roundtrip.py tests/test_roundtrip_loader.py
python -m pytest -q tests/test_backend_parity.py tests/test_vectorized_kernels.py
python -m pytest -q tests/test_train_loop.py
python -m build
```

The GitHub Actions workflow runs the CLI smoke checks and the project's CI script on Python 3.10, 3.11, and 3.12 across Linux and Windows. CUDA, Triton, and native C++ acceleration are not assumed by the baseline CI.

## Performance and quality claims

Ternair's storage estimates are theoretical representations of model weights and do not equal end-to-end RAM usage. Embeddings, activations, norms, temporary buffers, the Python runtime, and optional caches add overhead. Throughput and quality depend on architecture, tokenizer, hardware, sequence length, and training data. The repository's benchmark files are examples, not universal comparisons with BitNet.

Before production deployment, record at minimum:

- peak resident memory and model load time;
- prefill and decode tokens/second at representative batch and context lengths;
- numerical parity between the selected backend and the PyTorch reference;
- perplexity and task quality using the exact tokenizer and evaluation split;
- behavior for invalid configurations, missing files, CPU-only hosts, and interrupted exports.

## Production guidance

- Pin Python, PyTorch, and Ternair versions in deployment environments.
- Treat model files as untrusted input and load only from controlled locations.
- Use `eval()` and `torch.no_grad()` for inference.
- Keep the PyTorch backend as a correctness fallback.
- Do not claim a backend is faster until it has been benchmarked on the target hardware.
- Keep tokenizer and model configuration together in an exported package.
- Train a real model before claiming quality numbers; the default profiles are untrained.

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

Please add a regression test for bug fixes, keep public API changes documented, and run the relevant tests before opening a pull request. Performance changes should include a reproducible benchmark and a reference-path comparison.

## License

Apache-2.0. See [LICENSE](LICENSE).
