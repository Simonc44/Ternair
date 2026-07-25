## WikiText-2 Benchmark Results

> Profile: tiny  
> Training steps: 300  
> Device: cpu  
> Date: 2026-07-25T09:59:05

| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |
|-------|---------------------|------------|------------|
| Ternair (ternary) | nan | 0.7 | 1.7 |
| FP16 baseline | 15352762.55 | 36.6 | 52.9 |

**Summary:**
- PPL overhead vs FP16: +nan (+nan%)
- Size compression: 54.6×
- Speed ratio: 0.03×