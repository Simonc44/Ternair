[![CI](https://github.com/Simonc44/Ternair/actions/workflows/ci.yml/badge.svg)](https://github.com/Simonc44/Ternair/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.4%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/Simonc44/Ternair?include_prereleases)](https://github.com/Simonc44/Ternair/releases)

# Ternair

> **BitNet b1.58 in pure PyTorch — production-quality ternary LLM framework**

Ternair is a clean, well-tested Python/PyTorch implementation of **BitNet b1.58**: every weight is constrained to `{-1, 0, +1}` (~1.58 bits). The result is:

- **~16× memory compression** vs FP16 (942 MiB for a 4 B-parameter model)
- **Zero floating-point multiplications** at inference — pure additions and subtractions
- **O(1) generation memory** via optional SSM layers (no KV cache)
- A single, coherent Python API covering training, QAT, export, and benchmarking

---

## Why Ternair?

BitNet b1.58 implementations exist in C++ (llama.cpp) and scattered research code. Ternair fills the gap: a **PyTorch-native, batteries-included framework** with:

| What | How |
|------|-----|
| Drop-in `nn.Linear` replacement | `TernairLinear` with STE + learned alpha |
| Hybrid SSM/Attention architecture | Vectorised parallel scan, no Python loop |
| Quantization-Aware Training | Beta annealing (tanh → round), per-model state |
| LoRA-style fine-tuning | T-LoRA / BitDelta adapters |
| Full export pipeline | SafeTensors + HuggingFace-compatible `config.json` |
| Browser inference | WebGPU WGSL shaders + JS runtime |

---

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install git+https://github.com/Simonc44/Ternair.git
```

**Requirements:** Python 3.10+, PyTorch 2.4+

---

## Quick Start

```bash
# Display default config
python -m ternair info

# Size projection for the 1 GiB target
python -m ternair size --profile one_gb

# Run a tiny demo model end-to-end
python -m ternair demo --profile tiny
```

```python
from ternair.model.size_profiles import tiny_profile
from ternair.model.modeling import TernairForCausalLM
from ternair.model.export import print_compression_report, export_to_safetensors

# Build, freeze, export
model = TernairForCausalLM(tiny_profile(storage="fastpacked"))
model.freeze_storage()
model.eval()
print_compression_report(model)
export_to_safetensors(model, "my_model.safetensors")
```

---

## Architecture

```
TernairForCausalLM
├── TernairEmbedding          (vocab × hidden, FP16, tied with LM head)
├── TernairModel
│   └── TernairHybridBlock × num_hidden_layers
│       ├── [Attention layers — 1 in 4]
│       │   TernairAttention (GQA + RoPE + optional 2-bit KV cache)
│       └── [SSM layers — 3 in 4]
│           TernarySSMBlock  (vectorised parallel scan, O(1) memory)
│               ├── TernairLinear × 5   (ternary projections)
│               └── A_log, D            (learned FP32 SSM params)
├── TernairLMHead             (tied with embedding)
│
├── Optional modules
│   ├── ThalamicBottleneck    (K-WTA cross-attention, any seq → 32 latents)
│   ├── TernaryMoEBlock       (8 experts, top-2 active per token)
│   └── AdapterRegistry       (T-LoRA / BitDelta fine-tuning adapters)
│
└── Quantization stack
    ├── TernairLinear         STE + learned alpha + beta annealing
    ├── AnnealingState        per-model thread-safe beta (NEW v0.6)
    ├── Hadamard rotation     activation smoothing before INT8 quant
    └── OmniQuant scales      S-matrix learned at calibration time
```

### Quantization math

**Weight ternarization (per output row):**
```
γ = mean(|W|)
W_t = round(clamp(W / γ, -1, 1))  ∈ {-1, 0, +1}
```

**Training (STE):** `∂L/∂W ← ∂L/∂W_t` — gradient flows through `round` as identity.

**Annealing:** beta increases from 1.0 → 15.0 over training.
```
W_proxy = tanh(β · W / α) · α   # smooth at β≈1, hard at β→∞
```

**2-bit fastpacked storage:**
```python
trit = (bits & 1) - ((bits >> 1) & 1)  # {-1, 0, +1} — no branch, no LUT
```

### Vectorised SSM scan (v0.6 fix)

Previous versions used a Python `for t in range(L)` loop — **30–100× slower** than necessary on GPU. The new implementation uses a fully-vectorised parallel prefix scan in log-space:

```
log_A_cumsum[t] = cumsum_s≤t( delta_s * A )
h[t] = exp(log_A_cumsum[t]) * cumsum_s≤t( exp(-log_A_cumsum[s]) * dBx_s )
y[t] = (C[t] · h[t]).sum(N) + D * x[t]
```

All operations are standard PyTorch — no custom CUDA kernel required.

---

## Features

| Feature | Status |
|---------|--------|
| Ternary quantization (STE, gamma, 3-value) | ✅ |
| `fastpacked` storage (2 bits/value) | ✅ |
| `packed` storage (1.6 bits/value, base-3) | ✅ |
| Triton GPU kernel (JIT loop decode) | ✅ |
| CPU kernel (C++ SIMD, AVX-512 / ARM NEON) | ✅ |
| GQA attention + RoPE | ✅ |
| SwiGLU ternary MLP | ✅ |
| **Vectorised parallel SSM scan** | ✅ **v0.6** |
| **Per-model AnnealingState (thread-safe)** | ✅ **v0.6** |
| **Explicit device tracking in TernairLinear** | ✅ **v0.6** |
| ThalamicBottleneck (K-WTA, K=32) | ✅ |
| 2-bit KV cache (BitAttention) | ✅ |
| Ternary MoE (8 experts, top-2 active) | ✅ |
| T-LoRA / BitDelta adapters | ✅ |
| OmniQuant scale calibration | ✅ |
| Hadamard rotation (QuaRot) | ✅ |
| Quantization annealing (beta schedule) | ✅ |
| KL distillation + feature matching | ✅ |
| WSD scheduler + decoupled optimizer | ✅ |
| Accelerate training pipeline | ✅ |
| SafeTensors export | ✅ |
| HuggingFace package export | ✅ |
| WebGPU / WASM browser backend | ✅ |
| Compression report | ✅ |
| Generation (temperature, top-k, top-p, penalty) | ✅ |
| Streaming token generation | ✅ |
| Chat templates (ChatML, Llama-3) | ✅ |
| Eval suite (perplexity, zero-shot, speed) | ✅ |
| C++ header-only runtime (`inference.h`) | ✅ |
| **WikiText-2 benchmark** | 🔜 v0.7 |

---

## Training

```bash
# Smoke test (20 steps, ~2M params, ~10 seconds)
accelerate launch scripts/train.py --config scripts/train_tiny.yaml

# Full 1 GiB model (4 B params, 60 layers)
accelerate launch scripts/train.py --config scripts/train_one_gb.yaml

# QAT distillation from a HuggingFace teacher
PYTHONPATH=src python scripts/qat_distill.py
```

### Per-model AnnealingState (v0.6)

Use `AnnealingState` to isolate beta across multiple models in the same process:

```python
from ternair.quantization.ternary import create_annealing_state

state = create_annealing_state(beta_start=1.0, beta_end=15.0)

for step in range(total_steps):
    beta = state.step(step, total_steps)  # thread-safe update
    # beta is automatically used by ternary_linear_forward
    # when passed as annealing_state=state
    loss.backward()
    optimizer.step()
```

---

## Fine-tuning with T-LoRA

```python
from ternair.quantization.bitdelta import add_lora_to_model

registry = add_lora_to_model(model, rank=8, alpha=1.0)
optimizer = torch.optim.AdamW(registry.adapter_params(), lr=1e-3)

for step in range(100):
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step(); optimizer.zero_grad()

# Save (a few KB)
torch.save(registry.state_dict(), "adapters_code.pt")
```

---

## Export

```python
from ternair.model.export import export_huggingface_package, print_compression_report

model.freeze_storage(); model.eval()
print_compression_report(model)
export_huggingface_package(model, output_dir="./my-ternary-model", model_name="MyModel")
```

**1 GiB projection (4.07 B params):**

| Component | Size |
|-----------|------|
| Ternary weights (fastpacked) | 776.2 MiB |
| Embedding + LM head (tied, FP16) | 160.0 MiB |
| Gamma scales (FP32) | 5.1 MiB |
| RMSNorm + misc | 0.6 MiB |
| **Total** | **941.9 MiB** |

---

## Evaluation

```python
from ternair.benchmark.eval import run_eval_suite, print_report

report = run_eval_suite(model, tokenizer, run_perplexity=True, run_speed=True)
print_report(report)
```

> **WikiText-2 perplexity benchmark vs FP16 baseline is planned for v0.7.**
> Contributions welcome — see `scripts/benchmark_wikitext2.py` (coming soon).

---

## WebGPU / Browser Inference

```bash
python -c "
from ternair.kernels.webternair import generate_wgsl_ternary_matmul, generate_js_runtime, validate_webgpu_kernels
results = validate_webgpu_kernels()
print(f'Shaders valid: {results}')
js = generate_js_runtime()
open('ternair-web.js', 'w').write(js)
"
```

---

## C++ Runtime

`src/ternair/kernels/inference.h` — zero-dependency C++17 header-only engine:

```cpp
#include "ternair/inference.h"

TernairRuntime rt;
rt.load("model.safetensors");
auto result = rt.generate({1, 234, 567}, 64);
```

Also accessible from C (`ternair_create`, `ternair_load`, `ternair_generate`).

---

## Project Structure

```
src/ternair/
├── quantization/
│   ├── linear.py         TernairLinear — device tracking fix (v0.6)
│   ├── ternary.py        AnnealingState, STE, beta schedule (v0.6)
│   ├── activation.py     Hadamard + INT8 + OmniQuant
│   ├── bitdelta.py       T-LoRA / BitDelta adapters
│   ├── distillation.py   KL + feature-matching loss
│   └── packing.py        base-3 and 2-bit trit packing
├── kernels/
│   ├── inference.h       C++ header-only runtime
│   ├── webternair.py     WebGPU / WASM backend
│   ├── packing_fast.py   NumPy 2-bit pack/unpack
│   └── cpu_matmul.h      AVX-512 / ARM NEON matmul
├── model/
│   ├── config.py         TernairConfig (validated dataclass)
│   ├── attention.py      GQA + RoPE + 2-bit KV cache
│   ├── ssm.py            Vectorised parallel scan (v0.6)
│   ├── hybrid_block.py   SSM/Attention dispatcher
│   ├── thalamus.py       ThalamicBottleneck (K-WTA)
│   ├── moe.py            TernaryMoEBlock
│   ├── modeling.py       TernairForCausalLM
│   ├── export.py         SafeTensors + HuggingFace export
│   ├── generation.py     Sampling, streaming, chat templates
│   └── size_profiles.py  tiny / small / medium / large / one_gb
├── training/             WSD scheduler, optimizer, trainer
├── benchmark/            Perplexity, zero-shot, speed
└── cli.py                python -m ternair entry point

scripts/
├── train.py              accelerate entry point
├── train_tiny.yaml       smoke-test config
├── train_one_gb.yaml     production config
└── qat_distill.py        QAT distillation from HF teacher

tests/
├── test_quantization.py
├── test_model.py
├── test_kernels.py
├── test_ssm.py
├── test_thalamus.py
├── test_size.py
├── test_optimizer.py
├── test_scheduler.py
└── test_trainer.py
```

---

## Changelog

### v0.6.0 (current)
- **fix** `TernairLinear`: replace fragile `next(self.parameters()).device` with explicit `_frozen_device` attribute; add `.to()`/`.cuda()`/`.cpu()` overrides to keep it in sync
- **fix** `ternary.py`: replace module-level `_global_beta` float with thread-safe `AnnealingState` dataclass; expose `create_annealing_state()` for per-model isolation; backward-compatible `set_quant_annealing_beta()` preserved
- **perf** `ssm.py`: replace Python `for t in range(L)` time-loop with fully-vectorised parallel prefix scan — 30–100× faster on GPU for long sequences
- **dx** `TernairLinear.extra_repr()`: human-readable repr showing shape, storage mode and frozen state

### v0.5.0
- Modular `TernairPipeline`; atomic checkpoint saves; memory estimator; `small`/`medium`/`large` profiles; strengthened `TernairConfig` validation

### v0.3.0
- T-LoRA / BitDelta adapters; OmniQuant scale calibration; BitAttention 2-bit KV cache; Ternary MoE; WebGPU/WASM backend

### v0.2.0
- SafeTensors export; HuggingFace package; generation with repetition penalty + streaming; chat templates; eval suite; C++ header-only runtime

---

## Contributing

Pull requests are welcome. Before opening one:

1. Run the test suite: `pytest tests/ -x -q`
2. Check types: `mypy src/ternair --ignore-missing-imports`
3. The most impactful contribution right now: **WikiText-2 perplexity results**. See `scripts/benchmark_wikitext2.py` (planned).

---

## Credits

Built on published research:

- **BitNet b1.58** — Microsoft Research (2024)
- **BitDelta / T-LoRA** — Liu et al. (2024)
- **OmniQuant** — Shao et al. (2024)
- **QuaRot** (Hadamard rotation) — Ashkboos et al. (2024)
- **BitMoE** — (2025)
- **Mamba / SSM** — Gu & Dao (2023)

---

## License

Apache 2.0
