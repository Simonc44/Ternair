## WikiText-2 Benchmark Results

> Profile: tiny  
> Tokenizer: char-256  
> Training steps: 500  
> Device: cpu  
> Date: 2026-10-25T08:21:11

| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |
|-------|---------------------|------------|------------|
| Ternair (ternary) | 13.30 | 0.8 | 378.2 |
| FP16 baseline | 23.37 | 26.4 | 4044.3 |

**Summary:**
- PPL overhead vs FP16: -10.07 (-43.1%)
- Size compression: 33.4×
- Speed ratio: 0.09×