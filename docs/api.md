# API Reference

## Core

### `TernairConfig`

```python
from ternair import TernairConfig
```

Model configuration dataclass. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 32000 | Vocabulary size |
| `hidden_size` | 2560 | Hidden dimension |
| `intermediate_size` | 6912 | MLP intermediate dimension |
| `num_hidden_layers` | 24 | Number of transformer layers |
| `num_attention_heads` | 32 | Attention heads |
| `num_key_value_heads` | 4 | GQA KV heads |
| `storage` | "packed" | Weight storage format |
| `kv_cache_bits` | 0 | KV-cache quantization (0/2/4) |
| `num_experts` | 1 | MoE experts (1 = disabled) |

### `TernairForCausalLM`

```python
from ternair import TernairForCausalLM
```

Causal language model with ternary weights.

**Methods:**

- `forward(input_ids, use_cache=False)` → logits tensor
- `freeze_storage()` → switch to packed storage for inference
- `count_parameters(include_embedding=True)` → int
- `num_bytes(embedding_dtype_bytes=2)` → int

### `TernairDirectInferencer`

```python
from ternair import TernairDirectInferencer
```

High-level inference wrapper with automatic backend selection.

```python
inferencer = TernairDirectInferencer(model, backend="auto")
inferencer.prepare()
logits = inferencer.forward(input_ids)
output = inferencer.generate(input_ids, max_new_tokens=64)
```

**Backends:** `auto`, `torch`, `triton`, `cpu_cpp`, `numpy`

## Quantization

### `TernairLinear`

```python
from ternair.quantization.linear import TernairLinear
```

Drop-in `nn.Linear` replacement with ternary weights. Supports training (STE), frozen inference, and multiple storage formats.

### `ternarize`

```python
from ternair import ternarize
W_t, gamma = ternarize(W, dim=-1)  # → (int8 tensor, float32 scale)
```

Ternarize a weight matrix: `{-1, 0, +1}` with per-row scale `gamma = mean(|W|)`.

## Generation

### `generate`

```python
from ternair import generate
output = generate(model, input_ids, max_new_tokens=64, temperature=0.8)
```

Autoregressive generation with KV-cache. Supports temperature, top-k, top-p, repetition penalty.

### `generate_stream`

```python
from ternair.model.generation import generate_stream
for token in generate_stream(model, prompt, max_new_tokens=32):
    print(token.item(), end=" ")
```

Streaming generation that yields one token at a time.

## Export

### `export_to_safetensors`

```python
from ternair.model.export import export_to_safetensors
export_to_safetensors(model, "model.safetensors")
```

Atomic SafeTensors export with canonical key naming (`packed_weight`, `gamma_eval`).

### `export_huggingface_package`

```python
from ternair.model.export import export_huggingface_package
export_huggingface_package(model, "./dist/my-model", model_name="MyModel")
```

Full HuggingFace package: `config.json` + `model.safetensors` + `README.md`.

## Server

### `serve`

```python
from ternair.server import serve
serve(profile_name="tiny", port=8080)
```

OpenAI-compatible HTTP server. Endpoints:

- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/batch` — parallel batch processing
- `GET /v1/models`
- `GET /health`
- `GET /metrics` — Prometheus format

## Benchmarks

### `run_benchmark`

```python
from ternair.benchmark.reproducible import run_benchmark
result = run_benchmark(profile="tiny", storage="packed")
print(result.summary())
```

Measures perplexity and generation speed (prefill/decode).

## Profiles

```python
from ternair import tiny_profile, small_profile, base_profile, one_gb_profile
```

| Profile | Params | Size |
|---------|--------|------|
| `tiny` | ~2.6M | ~2.5 MiB |
| `small` | ~50M | ~10 MiB |
| `medium` | ~150M | ~30 MiB |
| `large` | ~250M | ~55 MiB |
| `base` | ~700M | ~150 MiB |
| `one_gb` | ~4B | ~942 MiB |
