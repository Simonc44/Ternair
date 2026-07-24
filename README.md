<p align="center">
  <h1 align="center">⚡ Ternair</h1>
  <p align="center"><strong>BitNet b1.58 — Ternary Neural Networks at 1 GiB Scale</strong></p>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.4%2B-orange">
  <img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green">
  <img alt="Tests" src="https://img.shields.io/badge/tests-52/52-green">
</p>

---

**Ternair** is a production-grade implementation of **BitNet b1.58** — a neural network architecture where every weight is constrained to `{-1, 0, +1}` (ternary values, ~1.58 bits). This enables:

- **~16× memory compression** vs FP16 (942 MiB for a 4B-parameter model)
- **ADD/SUB-only arithmetic** — zero float multiplications during inference
- **No KV-cache** when using the optional SSM layers (`O(1)` generation memory)
- **K-WTA token compression** via the ThalamicBottleneck (32 fixed latents per sequence)

## ✨ Features

| Feature | Status |
|---------|--------|
| **Ternary quantization** (STE, γ-scaling, 3 values) | ✅ |
| **Packing**: base-3 (1.6 b/v) or 2-bit (2.0 b/v) | ✅ |
| **GPU kernel** (Triton — decode in JIT loop) | ✅ |
| **CPU kernel** (C++ SIMD — AVX-512 / ARM NEON) | ✅ |
| **Causal LM** (GQA attention, RoPE, SquaredReLU MLP) | ✅ |
| **ThalamicBottleneck** (K-WTA compression, K=32) | ✅ |
| **SSM block** (Mamba-style recurrence, O(1) memory) | ✅ |
| **WSD scheduler** (Warmup-Stable-Decay) | ✅ |
| **Decoupled optimizer** (ternary WD=0, emb WD=0.1) | ✅ |
| **Accelerate training pipeline** | ✅ |
| **1 GiB size projection** (942 MiB, 4.07B params) | ✅ |
| **52 unit tests** | ✅ |

## 🚀 Quick start

```bash
# Create environment
uv venv .venv
source .venv/bin/activate

# Install ternair
uv pip install -e src/ternair[torch]

# Show default configuration
python -m ternair info

# Project size for 1 GiB target
python -m ternair size --profile one_gb

# Run a tiny demo model
python -m ternair demo --profile tiny

# Run all tests
python -m pytest tests/ --confcutdir=tests
```

## 🏋️ Training

```bash
# Smoke test (20 steps, tiny model, 2M params)
accelerate launch scripts/train.py --config scripts/train_tiny.yaml

# Full 1 GiB training (60 layers, 2560 hidden, 4B params)
accelerate launch scripts/train.py --config scripts/train_one_gb.yaml
```

## 🧠 Architecture

```
TernairConfig:
  ├── hidden_size=2560, num_hidden_layers=60
  ├── num_attention_heads=32, num_key_value_heads=8  (GQA)
  ├── intermediate_size=5120, max_position_embeddings=2048
  ├── storage: "packed" | "fastpacked" | "int8"
  │
  ├── ThalamicBottleneck (optional)
  │   └── K-WTA: top-32 tokens → cross-attention → 32 latents
  │
  ├── TernairHybridBlock × num_hidden_layers
  │   ├── Attention (GQA + RoPE)          — first `num_attn_layers`
  │   └── TernarySSM (Mamba-style scan)   — remaining layers
  │
  └── TernairForCausalLM
      ├── TernairEmbedding (tied weights)
      ├── TernairModel (hybrid blocks)
      └── TernairLMHead (tied)
```

## ⚡ How it works

### Ternary quantization

Every weight matrix `W` is quantized per-row:

```
γ = mean(|W|)                     # per-row scale
W_norm = W / γ
W_t = round(clamp(W_norm, -1, 1)) # → {-1, 0, +1}
```

**Forward**: `y = (γ · W_t) ⊗ x` → reduces to ADD/SUB of activations.
**Backward**: Straight-Through Estimator (STE) — gradient flows through `round` as identity.

### Fast packing (2-bit)

Each trit `{-1, 0, +1}` is stored as 2 bits in a `uint8` byte (4 trits/byte):

```python
# Decode: no modulo, no lookup, no branching
trit = (bits & 1) - ((bits >> 1) & 1)  # → {-1, 0, +1}
```

### SSD memory projection

| Component | Size (1 GiB profile) |
|-----------|---------------------|
| Ternary weights (packed) | 776.2 MiB |
| Embedding + LM head (tied) | 160.0 MiB |
| γ scales | 5.1 MiB |
| RMSNorm + misc buffers | 0.6 MiB |
| **Total** | **941.9 MiB (< 1 GiB)** |

## 📂 Project structure

```
src/ternair/
├── quantization/     # STE, packing, TernairLinear
├── kernels/          # Triton GPU, C++ SIMD, numpy ref
├── model/            # Config, attention, MLP, SSM, thalamus, generation
├── training/         # WSD scheduler, optimizer, trainer
├── benchmark/        # Size projection
├── cli.py            # CLI entry point
└── README.md         # Full documentation (FR/EN)

scripts/
├── train.py          # accelerate entry point
├── train_tiny.yaml   # Smoke config
└── train_one_gb.yaml # 1 GiB config
```

## 📊 Performance

- **4.07 billion ternary parameters** stored in **942 MiB**
- **ADD/SUB-only matmul** — zero FP multiplications at inference
- **O(1) memory for generation** (SSM mode — no KV-cache)
- **K-WTA compression**: any input length → 32 fixed latents

## 📄 License

Apache 2.0

Built on the BitNet b1.58 research by Microsoft Research (2024).
