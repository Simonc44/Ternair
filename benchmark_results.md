## WikiText-2 Benchmark Results

> Profile: tiny  
> Tokenizer: char-256  
> Training steps: 500  
> Device: cpu  
> Date: 2026-10-18T08:00:16

| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |
|-------|---------------------|------------|------------|
| Ternair (ternary) | 13.73 | 0.8 | 386.9 |
| FP16 baseline | 23.38 | 26.4 | 3833.4 |

**Summary:**
- PPL overhead vs FP16: -9.65 (-41.3%)
- Size compression: 33.4×
- Speed ratio: 0.10×