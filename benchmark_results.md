## WikiText-2 Benchmark Results

> Profile: tiny  
> Tokenizer: char-256  
> Training steps: 500  
> Device: cpu  
> Date: 2026-09-06T07:52:38

| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |
|-------|---------------------|------------|------------|
| Ternair (ternary) | 13.27 | 0.8 | 515.3 |
| FP16 baseline | 23.39 | 26.4 | 3861.6 |

**Summary:**
- PPL overhead vs FP16: -10.12 (-43.3%)
- Size compression: 33.4×
- Speed ratio: 0.13×