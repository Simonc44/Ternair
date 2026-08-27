"""OpenAI-compatible HTTP inference server for Ternair.

Usage::

    python -m ternair serve --profile tiny --port 8080

Endpoints
---------
* ``POST /v1/chat/completions`` — OpenAI-compatible chat completion.
* ``POST /v1/completions`` — OpenAI-compatible text completion.
* ``GET  /v1/models`` — List available models.
* ``GET  /health`` — Health check.
* ``GET  /metrics`` — Prometheus-compatible metrics (text format).

Dependencies: ``torch``, ``ternair`` (core only; no extra packages required).
"""

from __future__ import annotations

import json
import logging
import time
import threading
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import torch

from ternair.model.modeling import TernairForCausalLM
from ternair.model.size_profiles import tiny_profile
from ternair.model.generation import generate, format_chat_prompt

_LOGGER = logging.getLogger("ternair.server")

# ---------------------------------------------------------------------------
# Metrics (simple in-process counters, Prometheus text exposition)
# ---------------------------------------------------------------------------


class _Metrics:
    """Thread-safe counters for Prometheus exposition."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_total = 0
        self.tokens_generated = 0
        self.errors_total = 0
        self.total_prompt_tokens = 0
        self.total_generation_seconds = 0.0
        self._start = time.time()

    def record(self, prompt_tokens: int, generated_tokens: int, elapsed: float) -> None:
        with self._lock:
            self.requests_total += 1
            self.tokens_generated += generated_tokens
            self.total_prompt_tokens += prompt_tokens
            self.total_generation_seconds += elapsed

    def record_error(self) -> None:
        with self._lock:
            self.errors_total += 1

    def to_prometheus(self) -> str:
        with self._lock:
            uptime = time.time() - self._start
            avg_gen = (
                self.total_generation_seconds / max(self.tokens_generated, 1)
            )
            return (
                "# HELP ternair_requests_total Total requests served.\n"
                f"# TYPE ternair_requests_total counter\n"
                f"ternair_requests_total {self.requests_total}\n"
                "# HELP ternair_tokens_generated_total Total tokens generated.\n"
                f"# TYPE ternair_tokens_generated_total counter\n"
                f"ternair_tokens_generated_total {self.tokens_generated}\n"
                "# HELP ternair_errors_total Total errors.\n"
                f"# TYPE ternair_errors_total counter\n"
                f"ternair_errors_total {self.errors_total}\n"
                "# HELP ternair_avg_seconds_per_token Average decode time per token.\n"
                f"# TYPE ternair_avg_seconds_per_token gauge\n"
                f"ternair_avg_seconds_per_token {avg_gen:.6f}\n"
                "# HELP ternair_uptime_seconds Server uptime.\n"
                f"# TYPE ternair_uptime_seconds gauge\n"
                f"ternair_uptime_seconds {uptime:.1f}\n"
            )


# ---------------------------------------------------------------------------
# Inference wrapper (thread-safe with a model lock)
# ---------------------------------------------------------------------------


class _InferenceEngine:
    """Wraps a Ternair model for threaded serving."""

    def __init__(self, model: TernairForCausalLM, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self.model_name = "ternair"
        self._lock = threading.Lock()

    @torch.no_grad()
    def complete(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> dict[str, Any]:
        """Run text completion and return an OpenAI-compatible response dict."""
        # Tokenise using the internal char tokenizer
        from ternair.training.data import CharTokenizer, DEFAULT_CORPUS

        tok = CharTokenizer(DEFAULT_CORPUS)
        ids = torch.tensor(
            [tok.bos_id] + tok.encode(prompt), dtype=torch.long
        ).unsqueeze(0).to(self.device)
        prompt_len = ids.shape[1]

        t0 = time.time()
        with self._lock:
            out = generate(
                self.model,
                ids,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-6),
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=tok.eos_id,
            )
        elapsed = time.time() - t0

        gen_ids = out[0, prompt_len:].tolist()
        text = tok.decode(gen_ids)
        gen_count = len(gen_ids)

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": gen_count,
                "total_tokens": prompt_len + gen_count,
            },
            "_metrics": (prompt_len, gen_count, elapsed),
        }

    @torch.no_grad()
    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> dict[str, Any]:
        """Run chat completion and return an OpenAI-compatible response dict."""
        prompt = format_chat_prompt(messages, format="chatml")
        comp = self.complete(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": comp["choices"][0]["text"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": comp["usage"],
            "_metrics": comp["_metrics"],
        }


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    """Routes requests to the inference engine."""

    engine: _InferenceEngine  # set by serve()
    metrics: _Metrics  # set by serve()

    def log_message(self, fmt: str, *args: Any) -> None:
        _LOGGER.info(fmt % args)

    # --- helpers ---
    def _json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, data: str, content_type: str = "text/plain") -> None:
        body = data.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    # --- routes ---
    def do_GET(self) -> None:
        if self.path == "/health":
            self._json({"status": "ok"})
        elif self.path == "/v1/models":
            self._json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.engine.model_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "ternair",
                        }
                    ],
                }
            )
        elif self.path == "/metrics":
            self._text(self.metrics.to_prometheus(), "text/plain; version=0.0.4")
        else:
            self._json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        try:
            body = self._read_body()
        except Exception as exc:
            self.metrics.record_error()
            self._json({"error": f"Invalid JSON: {exc}"}, 400)
            return

        try:
            if self.path == "/v1/completions":
                result = self.engine.complete(
                    prompt=body.get("prompt", ""),
                    max_tokens=body.get("max_tokens", 64),
                    temperature=body.get("temperature", 0.8),
                    top_k=body.get("top_k", 40),
                    top_p=body.get("top_p", 0.9),
                    repetition_penalty=body.get("repetition_penalty", 1.1),
                )
                metrics = result.pop("_metrics")
                self.metrics.record(*metrics)
                self._json(result)

            elif self.path == "/v1/chat/completions":
                result = self.engine.chat(
                    messages=body.get("messages", []),
                    max_tokens=body.get("max_tokens", 64),
                    temperature=body.get("temperature", 0.8),
                    top_k=body.get("top_k", 40),
                    top_p=body.get("top_p", 0.9),
                    repetition_penalty=body.get("repetition_penalty", 1.1),
                )
                metrics = result.pop("_metrics")
                self.metrics.record(*metrics)
                self._json(result)

            else:
                self._json({"error": "Not found"}, 404)

        except Exception as exc:
            self.metrics.record_error()
            _LOGGER.exception("Inference error")
            self._json({"error": str(exc)}, 500)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def serve(
    profile_name: str = "tiny",
    host: str = "0.0.0.0",
    port: int = 8080,
    storage: str = "fastpacked",
) -> None:
    """Start the Ternair HTTP server (blocking).

    Parameters
    ----------
    profile_name
        Model profile: ``tiny``, ``small``, ``medium``, ``large``, ``base``.
    host
        Bind address.
    port
        Bind port.
    storage
        Weight storage mode.
    """
    from ternair.model.size_profiles import PROFILE_REGISTRY

    profile_fn = PROFILE_REGISTRY.get(profile_name)
    if profile_fn is None:
        raise ValueError(f"Unknown profile {profile_name!r}; choose from {list(PROFILE_REGISTRY)}")

    _LOGGER.info("Building model with profile=%s storage=%s ...", profile_name, storage)
    config = profile_fn(storage=storage)
    model = TernairForCausalLM(config)
    model.freeze_storage()
    model.eval()
    _LOGGER.info("Model ready. Starting server on %s:%d", host, port)

    engine = _InferenceEngine(model)
    metrics = _Metrics()
    _Handler.engine = engine
    _Handler.metrics = metrics

    server = HTTPServer((host, port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _LOGGER.info("Shutting down.")
        server.shutdown()


__all__ = ["serve"]
