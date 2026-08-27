# Server Guide

## Starting the Server

```bash
python -m ternair serve --profile tiny --port 8080
```

The server starts an OpenAI-compatible HTTP API with a thread-safe inference engine.

## Endpoints

### `POST /v1/completions`

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 32}'
```

### `POST /v1/chat/completions`

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}'
```

### `POST /v1/batch`

Process multiple prompts concurrently:

```bash
curl -X POST http://localhost:8080/v1/batch \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hello", "World", "Test"], "max_tokens": 16}'
```

### `GET /v1/models`

```bash
curl http://localhost:8080/v1/models
```

### `GET /health`

```bash
curl http://localhost:8080/health
```

### `GET /metrics`

Prometheus-compatible metrics:

```bash
curl http://localhost:8080/metrics
```

## Dynamic Batching

The `/v1/batch` endpoint processes multiple prompts in parallel using a thread pool. Each prompt gets its own generation call, and results are returned as a JSON array.

```json
{
  "results": [
    {"choices": [{"text": "..."}], "usage": {...}},
    {"choices": [{"text": "..."}], "usage": {...}}
  ]
}
```

## Metrics

Prometheus-format metrics are exposed at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `ternair_requests_total` | counter | Total requests served |
| `ternair_tokens_generated_total` | counter | Total tokens generated |
| `ternair_errors_total` | counter | Total errors |
| `ternair_avg_seconds_per_token` | gauge | Average decode time per token |
| `ternair_uptime_seconds` | gauge | Server uptime |

## Configuration

| Environment | Default | Description |
|-------------|---------|-------------|
| `--profile` | tiny | Model profile to load |
| `--host` | 0.0.0.0 | Bind address |
| `--port` | 8080 | Bind port |
| `--storage` | fastpacked | Weight storage format |
