# Ternair

Ternair is a Python/PyTorch implementation of ternary language-model components inspired by BitNet b1.58. It provides ternary weight quantization, packed storage, a decoder-only model, portable inference fallbacks, SafeTensors export, and optional acceleration backends.

> **Status:** beta research software. The PyTorch reference path and the tested CPU/NumPy paths are the supported baseline. Triton and the native C++ engine are optional accelerators and must be benchmarked on the target machine.

## What is supported

- Ternary weights in `{-1, 0, +1}` with straight-through training.
- `packed` base-3 storage (5 trits/byte) and `fastpacked` 2-bit storage.
- Decoder-only causal LM with GQA/RoPE attention, SwiGLU MLP, and optional hybrid SSM blocks.
- Greedy and sampled generation, streaming generation, repetition penalty, and chat templates.
- Portable PyTorch and NumPy inference paths.
- Optional Triton GPU and native C++ CPU backends.
- Atomic SafeTensors export and model-size/compression reports.
- CLI commands for inspection, size estimation, demos, training smoke tests, and inference.

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

## Export

```python
from ternair.model.export import export_huggingface_package

model.eval()
model.freeze_storage()
export_huggingface_package(model, "./dist/ternair-model", model_name="MyTernairModel")
```

Exports are SafeTensors-compatible and written atomically. The package contains `config.json`, `model.safetensors`, and an optional model card. Loading external LLaMA/Mistral-style bundles requires the optional `transformers` and `safetensors` dependencies:

```bash
python -m pip install transformers safetensors
```

The loader returns `(model, tokenizer, report)` and validates the requested file, with `model.safetensors` accepted for native Ternair exports.

## CLI reference

```text
ternair info       Show a model configuration
ternair size       Estimate storage and compression
ternair demo       Run a forward/generation smoke demo
ternair train-one  Run one toy training step
ternair infer      Run inference through the backend dispatcher
```

Run `python -m ternair COMMAND --help` for all options.

## Testing and release checks

From the repository root:

```bash
python -m pytest -q tests/test_kernels.py tests/test_quantization.py tests/test_size.py
python -m pytest -q tests/test_model.py tests/test_ssm.py
python -m build
```

The GitHub Actions workflow runs the CLI smoke checks and the project's CI script on Python 3.10, 3.11, and 3.12. CUDA, Triton, and native C++ acceleration are not assumed by the baseline CI.

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

## Project layout

```text
src/ternair/
  model/          model, generation, export, loader, backend dispatcher
  quantization/   ternarization, STE, activations, adapters
  kernels/        packing, NumPy, Triton, and C++ acceleration
  training/       toy data, optimizer, scheduler, trainer, pipeline
  benchmark/      size and evaluation utilities
tests/            unit and integration tests
scripts/          training, demo, and CI helpers
```

## Contributing

Please add a regression test for bug fixes, keep public API changes documented, and run the relevant tests before opening a pull request. Performance changes should include a reproducible benchmark and a reference-path comparison.

## License

Apache-2.0. See [LICENSE](LICENSE).
