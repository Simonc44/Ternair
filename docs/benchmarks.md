# Benchmarks

## Running Benchmarks

```bash
# Quick benchmark (tiny profile)
python -m ternair benchmark --profile tiny --eval-tokens 256

# Full benchmark with JSON output
python -m ternair benchmark --profile tiny --eval-tokens 1024 --output results.json

# Speed only (skip perplexity)
python -m ternair benchmark --profile tiny --skip-perplexity
```

## Programmatic API

```python
from ternair.benchmark.reproducible import run_benchmark, run_comparison

# Single profile
result = run_benchmark(profile="tiny", storage="packed", eval_tokens=1024)
print(result.summary())

# Compare multiple profiles
results = run_comparison(
    profiles=["tiny", "small"],
    storage="packed",
    output_path="comparison.json",
)
```

## Metrics Measured

| Metric | Description |
|--------|-------------|
| **Perplexity** | Language modeling perplexity on synthetic dataset |
| **Prefill speed** | Tokens/second for processing the initial prompt |
| **Decode speed** | Tokens/second for autoregressive token generation |
| **E2E speed** | End-to-end tokens/second (prefill + decode) |
| **Model size** | On-disk footprint in MiB |
| **Parameter count** | Total parameters including embedding |

## Example Output

```
=== Ternair Benchmark: tiny (packed) ===
Device        : cpu
Parameters    : 3,686,400
Model size    : 2.54 MiB

Perplexity    : 788680891032401769889793878460026739573381381606054681947743968576100321813586559922975945278857431482368.00
Eval tokens   : 254
Eval loss     : 241.5340

Prefill speed : 175.1 tokens/s (91.4 ms)
Decode speed  : 15.0 tokens/s (533.3 ms)
E2E speed     : 12.8 tokens/s
```

Note: Perplexity is very high because the model is randomly initialized (no training). A trained model would show much lower perplexity.

## Reproducibility

All benchmarks use:
- Fixed random seed (`seed=42` by default)
- Deterministic synthetic dataset
- Same model initialization per seed
- Averaged over multiple runs (3 by default)
