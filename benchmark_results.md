## WikiText-2 Benchmark Results

> Profile: tiny  
> Tokenizer: char-256  
> Training steps: 500  
> Device: cpu  
> Date: 2026-11-08T10:07:14

| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |
|-------|---------------------|------------|------------|
| Ternair (ternary) | 13.14 | 0.8 | 289.6 |
| FP16 baseline | 23.39 | 26.4 | 8874.6 |

**Summary:**
- PPL overhead vs FP16: -10.25 (-43.8%)
- Size compression: 33.4×
- Speed ratio: 0.03×