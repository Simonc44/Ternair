## WikiText-2 Benchmark Results

> Profile: tiny  
> Tokenizer: char-256  
> Training steps: 500  
> Device: cpu  
> Date: 2026-11-15T10:40:01

| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |
|-------|---------------------|------------|------------|
| Ternair (ternary) | 14.81 | 0.8 | 330.9 |
| FP16 baseline | 23.37 | 26.4 | 4873.1 |

**Summary:**
- PPL overhead vs FP16: -8.56 (-36.6%)
- Size compression: 33.4×
- Speed ratio: 0.07×