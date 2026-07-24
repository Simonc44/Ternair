# ternair — BitNet b1.58-style ternary language model (stand-alone prototype)

> A pure-Python + PyTorch implementation of **BitNet b1.58** — the
> recipe where every linear weight is quantised to ``{-1, 0, +1}`` with
> a single per-output scale ``γ``. The prototype ships ready-made
> decoder-only configs that demonstrably fit **under 1 GiB** of storage
> while holding **~4 B ternary parameters**.

This package lives at `src/ternair/` and is intentionally **stand-alone**:
it only depends on `torch` and `numpy`, so it drops into any Python
project without inheriting the rest of `transformers`.

## Why ternary?

Each linear layer's weight tensor ``W`` of shape ``(M, N)`` is replaced
by:

* ``γ ∈ ℝ^M`` — one FP32 scale per output row, computed as
  ``γ[m] = mean(|W[m, :])``.
* ``W_t ∈ {-1, 0, +1}^{M × N}`` — the (signed) ternary weight.

The forward pass becomes ``y = W_t · γ · x`` (with an 8-bit absmax
quantisation on the activation ``x``). A straight-through estimator
(STE) makes this differentiable during training.

### Compression math

| Storage mode | Bits / value | 1 B weights occupy |
|--------------|-------------:|-------------------:|
| FP32         | 32.00        | 4.00 GiB           |
| FP16         | 16.00        | 2.00 GiB           |
| INT8         | 8.00         | 1.00 GiB           |
| **Ternary (`int8` storage)** | 8.00     | 1.00 GiB           |
| **Ternary (`packed` storage — 5 trits/byte)** | **1.60** | **0.20 GiB** |

`log2(3) ≈ 1.585`. The 5-into-8 base-3 packing used here gets within
~1% of that theoretical optimum and is the scheme used by the
community ports of BitNet.

## What's in the box

```text
src/ternair/
├── __init__.py            # top-level public API
├── _version.py
├── py.typed               # PEP 561 marker
├── pyproject.toml         # standalone install (`pip install -e src/ternair`)
├── README.md
├── quantization/
│   ├── ternary.py         # _compute_gamma, ternarize, ternarize_ste, custom autograd STE
│   ├── activation.py      # 8-bit absmax per-token quant with STE
│   ├── packing.py         # pack/unpack trits in base-3 (1.6 bits/value)
│   └── linear.py          # TernairLinear (drop-in nn.Linear)
├── model/
│   ├── config.py          # TernairConfig
│   ├── attention.py       # GQA-capable attention with ternary linears
│   ├── mlp.py             # squared-ReLU MLP (BitNet b1.58 choice)
│   ├── block.py           # decoder block (RMSNorm + Attn + MLP)
│   ├── modeling.py        # TernairModel / TernairForCausalLM
│   ├── generation.py      # greedy decode (no KV cache)
│   └── size_profiles.py   # tiny / base / one_gb presets
├── benchmark/
│   └── size.py            # describe, auto_fit_to_bytes, fit_one_gb
├── training/
│   ├── data.py            # 6-line corpus + char-level tokenizer (no external download)
│   └── trainer.py         # one-step cross-entropy smoke test
└── cli.py                 # `python -m ternair …`
```

## Install

```bash
# from the repo root
pip install -e src/ternair

# or — without editable install — just put src/ on sys.path
PYTHONPATH=src python -m ternair demo --profile tiny
```

The package only needs `torch>=2.4` and `numpy>=1.17`. PyTorch's CPU
build is enough to run the demo; the model is intentionally small
enough that the entire `tiny` profile (`~25 M ternary params`) fits in
~6 MiB of packed storage and runs end-to-end in a few seconds on a
single CPU.

## CLI

```bash
python -m ternair info   --profile one_gb            # config dump
python -m ternair size   --profile one_gb            # size projection
python -m ternair size   --profile base --fit-one-gb # tune layers to fit
python -m ternair demo   --profile tiny --max-new-tokens 16
python -m ternair train-one --profile tiny --lr 1e-3
```

`--storage` accepts `packed` (1.6 bits/value, ~1.58 in spirit) or
`int8` (8 bits/value, the conservative baseline).

## Quickstart (Python)

```python
import torch
from ternair import TernairForCausalLM, one_gb_profile, generate
from ternair.training.data import tokenise_corpus

cfg = one_gb_profile(storage="packed")
print(f"Target: ~{cfg.num_hidden_layers} layers, {cfg.hidden_size} hidden …")

model = TernairForCausalLM(cfg)
ids, tok = tokenise_corpus()

# Smoke forward + switch to packed storage
model.freeze_storage()
model.eval()
out = generate(model, ids, max_new_tokens=16, eos_token_id=tok.eos_id)
print(tok.decode(out[0].tolist()))
```

## How does it fit in 1 GiB?

`one_gb_profile()` ships with **60 transformer blocks at
hidden 2560**, producing ~4.07 B ternary weights. Packed at 1.6 b/v
that's ~813 MiB; add a 32 768×2560 FP16 embedding (~160 MiB) and a
few MiB of γ scales / RMSNorm, and the projection lands at
**~973 MiB** — comfortably under 1 GiB.

Run the size calculator on any config:

```python
from ternair import one_gb_profile, describe_size
print(describe_size(one_gb_profile()))
```

…or auto-tune the layer count to a custom budget:

```python
from ternair import base_profile, benchmark  # auto_fit_to_bytes / fit_one_gb
cfg = benchmark.fit_one_gb(base_profile())
```

### Caveats on the "1 GB ↔ 1 T" claim

* The model in this prototype ships **untrained**. The 1 GiB ceiling
  is the storage budget, not a quality budget — quality still comes
  from training, which the prototype does *not* automate (it ships a
  one-step smoke test only).
* The BitNet b1.58 paper reports that a 7 B ternary model is roughly
  competitive with a 3.9 B FP16 model on benchmark perplexity and
  matches or beats it on many knowledge/reasoning benchmarks. Going
  from ~3 B to the user's 1 T-param equivalent via ternarisation alone
  is **not realistic** — stacking additional compression (low-rank
  factorisation, distillation, mixture-of-experts sharing, 1-bit KV
  cache) is required to push past the order-of-magnitude gap.

## Tests

```bash
PYTHONPATH=src pytest tests/ -q
```

Tests cover STE flow, 8-bit activation quant, packing round-trip, the
1 GiB projection, and a forward + freeze + generate end-to-end smoke
test on the `tiny` profile.
