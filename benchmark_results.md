## WikiText-2 Benchmark Results

> Profile: tiny  
> Tokenizer: char-256  
> Training steps: 500  
> Device: cpu  
> Date: 2026-11-01T10:22:30

| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |
|-------|---------------------|------------|------------|
| Ternair (ternary) | 13.47 | 0.8 | 319.2 |
| FP16 baseline | 23.38 | 26.4 | 3750.8 |

**Summary:**
- PPL overhead vs FP16: -9.91 (-42.4%)
- Size compression: 33.4×
- Speed ratio: 0.09×