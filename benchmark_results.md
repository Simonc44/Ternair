## WikiText-2 Benchmark Results

> Profile: tiny  
> Tokenizer: char-256  
> Training steps: 500  
> Device: cpu  
> Date: 2026-11-22T11:02:30

| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |
|-------|---------------------|------------|------------|
| Ternair (ternary) | 12.61 | 0.8 | 485.9 |
| FP16 baseline | 23.38 | 26.4 | 4766.4 |

**Summary:**
- PPL overhead vs FP16: -10.77 (-46.1%)
- Size compression: 33.4×
- Speed ratio: 0.10×