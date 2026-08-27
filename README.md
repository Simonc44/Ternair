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

## Training (quality)

Ternair is a *training* framework, not a pre-trained checkpoint: the shipped profiles are randomly initialised. Quality comparable to BitNet requires training, which is exactly what the training path is for.

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

Honest comparison with BitNet:

- **Size/compression:** Ternair wins — 33x smaller than the FP16 equivalent (1.6 bits/value vs 2 bits typical of BitNet).
- **Portability:** Ternair wins — zero mandatory HuggingFace dependency, bundled server, multiple backends with fallback.
- **Raw speed on commodity GPUs:** neither ternary implementation beats cuBLAS FP16 on standard hardware; the ternary advantage only materialises on bit-level CPU SIMD or specialised hardware.
- **Quality:** BitNet wins today — it is trained on trillions of tokens; Ternair ships untrained profiles and a working training path.

## CLI reference

```text
ternair info        Show a model configuration
ternair size        Estimate storage and compression
ternair demo        Run a forward/generation smoke demo
ternair train-one   Run one toy training step
ternair train       Run a real multi-step training loop and report loss/PPL reduction
ternair infer       Run inference through the backend dispatcher
ternair serve       Start the OpenAI-compatible HTTP server
ternair benchmark   Run reproducible perplexity + speed benchmarks
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
