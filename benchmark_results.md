## WikiText-2 Benchmark Results

> Profile: tiny  
> Tokenizer: char-256  
> Training steps: 500  
> Device: cpu  
> Date: 2026-08-30T09:10:13

| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |
|-------|---------------------|------------|------------|
| Ternair (ternary) | 12.70 | 0.8 | 253.5 |
| FP16 baseline | 23.36 | 26.4 | 3817.9 |

**Summary:**
- PPL overhead vs FP16: -10.66 (-45.6%)
- Size compression: 33.4×
- Speed ratio: 0.07×