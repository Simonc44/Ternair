## WikiText-2 Benchmark Results

> Profile: tiny  
> Tokenizer: char-256  
> Training steps: 500  
> Device: cpu  
> Date: 2026-07-25T20:17:19

| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |
|-------|---------------------|------------|------------|
| Ternair (ternary) | 22.64 | 0.8 | 2059.2 |
| FP16 baseline | 23.39 | 26.4 | 6227.6 |

**Summary:**
- PPL overhead vs FP16: -0.75 (-3.2%)
- Size compression: 33.4×
- Speed ratio: 0.33×