## WikiText-2 Benchmark Results

> Profile: tiny  
> Tokenizer: char-256  
> Training steps: 500  
> Device: cpu  
> Date: 2026-10-11T09:55:53

| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |
|-------|---------------------|------------|------------|
| Ternair (ternary) | 13.13 | 0.8 | 284.9 |
| FP16 baseline | 23.36 | 26.4 | 4043.4 |

**Summary:**
- PPL overhead vs FP16: -10.23 (-43.8%)
- Size compression: 33.4×
- Speed ratio: 0.07×