## WikiText-2 Benchmark Results

> Profile: tiny  
> Tokenizer: char-256  
> Training steps: 500  
> Device: cpu  
> Date: 2026-09-20T08:31:22

| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |
|-------|---------------------|------------|------------|
| Ternair (ternary) | 14.25 | 0.8 | 249.2 |
| FP16 baseline | 23.38 | 26.4 | 8379.8 |

**Summary:**
- PPL overhead vs FP16: -9.12 (-39.0%)
- Size compression: 33.4×
- Speed ratio: 0.03×