# Using trained BitNet b1.58 models

Ternair and BitNet b1.58 share the same architecture, so **trained BitNet
checkpoints can be converted as-is** — same trained weights, zero
re-training — into Ternair's denser packed storage.

## Why this works

| Component | BitNet b1.58 | Ternair |
|-----------|--------------|---------|
| Weight quantization | `{-1, 0, +1}` × `gamma = mean(|W|)` per row | identical |
| Normalization | RMSNorm | identical |
| Positional encoding | RoPE | identical |
| Attention | GQA | identical |
| MLP | SwiGLU | identical |

BitNet checkpoints store **master bf16 weights** and ternarise at inference
time. The converter performs that ternarisation once and packs the result.

## Conversion

```bash
# Download a trained checkpoint (config.json + model.safetensors)
huggingface-cli download microsoft/bitnet-b1.58-2B-4T --local-dir ./bitnet-2b4t

# Convert
python -m ternair import-bitnet --source ./bitnet-2b4t --output ./ternair-2b4t --storage packed
```

Options:

- `--storage packed` (default, 1.6 bits/value) or `--storage fastpacked` (2 bits/value).
- `--no-tokenizer` skips copying tokenizer files.

The output package contains `config.json`, `model.safetensors` (packed
ternary weights), and the tokenizer files. Packed weights are typically
~16-33x smaller than the FP16 master checkpoint.

## Loading and serving

```bash
python -m ternair serve --model ./ternair-2b4t --port 8080
```

Python:

```python
from ternair import convert_bitnet_checkpoint, load_converted_model

report = convert_bitnet_checkpoint("./bitnet-2b4t", "./ternair-2b4t")
print(report.as_dict())

model, tokenizer = load_converted_model("./ternair-2b4t", device="cuda")
```

`load_converted_model` returns a frozen `TernairForCausalLM` ready for
`generate(...)`, with the original HuggingFace tokenizer attached when
available.

## Fidelity

The conversion is deterministic and lossless for the frozen path: the
round-trip test builds a reference model from the same master weights and
asserts the converted model produces matching logits. Because both paths
apply the same `round(clamp(W/gamma, -1, 1)) × gamma` ternarisation and the
same packing, the converted model behaves exactly like the frozen BitNet
model.
