## WikiText-2 Benchmark Results

> Profile: tiny  
> Training steps: 300  
> Device: cpu  
> Date: 2026-07-25T19:43:28

| Model | PPL (WikiText-2 test) | Size (MiB) | Tokens/sec |
|-------|---------------------|------------|------------|
| Ternair (ternary) | 485165195.41 | 25.2 | 95.6 |
| FP16 baseline | 485165195.41 | 75.2 | 2182.7 |

**Summary:**
- PPL overhead vs FP16: +0.00 (+0.0%)
- Size compression: 3.0×
- Speed ratio: 0.04×