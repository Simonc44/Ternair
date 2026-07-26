## WikiText-2 Benchmark Results

> Profile: tiny  
> Tokenizer: char-256  
> Training steps: 500  
> Device: cpu  
> Date: 2026-07-26T06:17:38

| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |
|-------|---------------------|------------|------------|
| Ternair (ternary) | 24.55 | 0.8 | 2012.5 |
| FP16 baseline | 23.39 | 26.4 | 6508.5 |

**Summary:**
- PPL overhead vs FP16: +1.16 (+5.0%)
- Size compression: 33.4×
- Speed ratio: 0.31×