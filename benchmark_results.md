## WikiText-2 Benchmark Results

> Profile: tiny  
> Tokenizer: char-256  
> Training steps: 500  
> Device: cpu  
> Date: 2026-09-27T09:11:39

| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |
|-------|---------------------|------------|------------|
| Ternair (ternary) | 14.74 | 0.8 | 450.1 |
| FP16 baseline | 23.38 | 26.4 | 5176.5 |

**Summary:**
- PPL overhead vs FP16: -8.64 (-37.0%)
- Size compression: 33.4×
- Speed ratio: 0.09×