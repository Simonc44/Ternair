## WikiText-2 Benchmark Results

> Profile: tiny  
> Tokenizer: char-256  
> Training steps: 500  
> Device: cpu  
> Date: 2026-10-04T09:36:06

| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |
|-------|---------------------|------------|------------|
| Ternair (ternary) | 12.95 | 0.8 | 418.8 |
| FP16 baseline | 23.38 | 26.4 | 5177.7 |

**Summary:**
- PPL overhead vs FP16: -10.43 (-44.6%)
- Size compression: 33.4×
- Speed ratio: 0.08×