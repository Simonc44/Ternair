# CLI Reference

All commands are invoked via `python -m ternair COMMAND [options]`.

## `info`

Print model configuration as JSON.

```bash
python -m ternair info --profile tiny
python -m ternair info --profile base --storage fastpacked
```

## `size`

Show projected size breakdown.

```bash
python -m ternair size --profile tiny
python -m ternair size --profile one_gb
python -m ternair size --profile base --fit-one-gb
```

## `demo`

Build a model, run forward pass, and generate tokens.

```bash
python -m ternair demo --profile tiny --max-new-tokens 16
python -m ternair demo --profile tiny --storage fastpacked
```

## `train-one`

Run one training step on toy data (smoke test).

```bash
python -m ternair train-one --profile tiny --lr 1e-3
```

## `infer`

Direct inference with backend selection.

```bash
python -m ternair infer --profile tiny --backend auto --prompt "hello"
python -m ternair infer --profile tiny --backend numpy --prompt "world"
python -m ternair infer --profile tiny --backend torch --prompt "demo"
```

**Backends:** `auto`, `torch`, `triton`, `cpu_cpp`, `numpy`

## `serve`

Start the OpenAI-compatible HTTP server.

```bash
python -m ternair serve --profile tiny --port 8080
python -m ternair serve --profile small --host 0.0.0.0 --port 9090
```

## `benchmark`

Run reproducible benchmarks.

```bash
python -m ternair benchmark --profile tiny --eval-tokens 1024
python -m ternair benchmark --profile tiny --skip-perplexity
python -m ternair benchmark --profile tiny --output results.json
```

## Global Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--profile` | tiny, small, medium, large, base, one_gb | tiny | Model profile |
| `--storage` | packed, fastpacked, int8 | packed | Weight storage format |
| `--fit-one-gb` | flag | off | Auto-tune layers for 1 GiB target |
