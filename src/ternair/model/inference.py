"""Direct inference mode for Ternair.

This module wires the existing low-level kernels
(:mod:`ternair.kernels.triton_fast`, :mod:`ternair.kernels.cpu_matmul`,
:mod:`ternair.kernels.packed_ops`) into the high-level model so that
the eval-mode forward pass can either:

* stay on the reference path (``F.linear`` over a dequantised weight
  tensor -- always works, default ``"torch"`` backend), **or**
* dispatch the ternary matmul to the Triton GPU kernel, **or**
* dispatch to the C++ SIMD backend (AVX-512 / ARM NEON via cppyy).

Typical usage::

    from ternair.model.inference import TernairDirectInferencer

    model = TernairForCausalLM(tiny_profile(storage="fastpacked"))
    model.load_state_dict(...)

    inferer = TernairDirectInferencer(model, backend="auto")
    inferer.prepare()   # calls freeze_storage() + sets the backend

    ids = torch.tensor([[1, 2, 3]])
    out = inferer.generate(ids, max_new_tokens=16,
                           temperature=0.0, eos_token_id=0)

    # Streaming
    for tok in inferer.generate_stream(ids, max_new_tokens=8):
        print(tok.item(), end=" ", flush=True)

The class does NOT modify the global state of the model: each layer's
``inference_backend`` is set to a per-instance attribute that defaults
back to ``"auto"`` -> ``"torch"`` when restored via ``restore()``.
"""

from __future__ import annotations

from typing import Iterator, Literal, Optional

import torch
from torch import Tensor

from ternair.model.generation import generate, generate_stream
from ternair.model.modeling import TernairForCausalLM
from ternair.quantization.linear import InferenceBackend, TernairLinear


BackendName = Literal["auto", "torch", "triton", "cpu_cpp", "numpy"]


class TernairDirectInferencer:
    """High-level direct inference wrapper around a :class:`TernairForCausalLM`.

    Parameters
    ----------
    model
        A Ternair model (must already be constructed).
    backend
        Inference backend to use for every ``TernairLinear`` layer:

        * ``"auto"``   - resolve at ``prepare()`` time
        * ``"torch"``  - reference path via ``F.linear`` (default)
        * ``"triton"`` - fastpacked weights -> Triton GPU kernel
        * ``"cpu_cpp"``- cppyy + cpu_matmul.h (AVX-512 / NEON)
        * ``"numpy"``  - pure-numpy reference (slow but portable)

    device
        Optional device string forwarded to :meth:`prepare` (e.g.
        ``"cuda"`` or ``"cpu"``).  ``None`` keeps the model's device.
    """

    def __init__(
        self,
        model: TernairForCausalLM,
        backend: BackendName = "auto",
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.requested_backend: BackendName = backend
        self.device = device
        self._resolved_backend: BackendName = "torch"
        self._previous_backends: dict[int, BackendName] = {}
        self._prepared = False

    # ------------------------------------------------------------------
    # Capability introspection
    # ------------------------------------------------------------------
    @staticmethod
    def available_backends() -> dict[str, bool]:
        """Map backend name -> availability on this host.

        Useful for CLI ``--backend auto`` resolution and for tests.
        """
        out: dict[str, bool] = {
            "torch": True,    # always available
            "numpy": True,    # pure-numpy fallback
            "triton": False,
            "cpu_cpp": False,
        }
        try:
            from ternair.kernels.triton_fast import has_triton
            out["triton"] = bool(has_triton())
        except Exception:
            pass
        try:
            from ternair.kernels.cpu_matmul import has_cpu_backend
            out["cpu_cpp"] = bool(has_cpu_backend())
        except Exception:
            pass
        return out

    # ------------------------------------------------------------------
    # Prepare / restore
    # ------------------------------------------------------------------
    def prepare(self) -> "TernairDirectInferencer":
        """Freeze the storage and set the inference backend.

        Idempotent -- calling :meth:`prepare` twice is a no-op the
        second time (we still re-apply the backend flag in case the
        user toggled ``requested_backend``).
        """
        # 1. Optionally move the model.
        if self.device is not None:
            self.model.to(self.device)

        # 2. Freeze storage (skip if already frozen).
        already_frozen = all(
            isinstance(m, TernairLinear) and m.is_frozen()
            for m in self.model.modules()
        )
        if not already_frozen:
            self.model.freeze_storage()

        # 3. Resolve backend.
        avail = self.available_backends()
        if self.requested_backend == "auto":
            self._resolved_backend = self._auto_select(avail)
        else:
            self._resolved_backend = self.requested_backend
        if not avail.get(self._resolved_backend, False) and self._resolved_backend != "torch":
            # Fall back to torch if the requested backend isn't available.
            self._resolved_backend = "torch"

        # 4. Apply per-layer.
        for m in self.model.modules():
            if isinstance(m, TernairLinear):
                m.set_inference_backend(self._resolved_backend)

        # 5. eval() mode for inference.
        self.model.eval()
        self._prepared = True
        return self

    def restore(self) -> None:
        """Reset every ``TernairLinear`` to ``"auto"`` inference backend."""
        for m in self.model.modules():
            if isinstance(m, TernairLinear):
                m.set_inference_backend("auto")
        self._prepared = False

    @staticmethod
    def _auto_select(avail: dict[str, bool]) -> BackendName:
        """Pick the best available backend for the current host."""
        if avail.get("triton", False):
            return "triton"
        if avail.get("cpu_cpp", False):
            return "cpu_cpp"
        return "torch"

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def resolved_backend(self) -> BackendName:
        """Backend actually wired into the layers after :meth:`prepare`."""
        return self._resolved_backend

    def describe(self) -> dict:
        """Return a small dict summarising the inference configuration.

        Used by the CLI ``infer`` subcommand and the test suite.
        """
        avail = self.available_backends()
        n_layers = sum(1 for _ in self.model.modules() if isinstance(_, TernairLinear))
        return {
            "requested_backend": self.requested_backend,
            "resolved_backend": self._resolved_backend,
            "available": avail,
            "n_ternary_layers": n_layers,
            "device": next(self.model.parameters()).device.type,
        }

    # ------------------------------------------------------------------
    # Forward / generate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, input_ids: Tensor) -> Tensor:
        """One forward pass (no sampling)."""
        if not self._prepared:
            self.prepare()
        return self.model(input_ids)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 16,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        pad_token_id: Optional[int] = None,
    ) -> Tensor:
        """Sample tokens with the existing :func:`generate` helper.

        The model used here is the same ``TernairForCausalLM`` but
        with every ``TernairLinear`` switched to the resolved backend.
        """
        if not self._prepared:
            self.prepare()
        return generate(
            self.model,
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_token_id,
        )

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 64,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        pad_token_id: Optional[int] = None,
    ) -> Iterator[Tensor]:
        """Yield one token tensor per generated step."""
        if not self._prepared:
            self.prepare()
        return generate_stream(
            self.model,
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_token_id,
        )


__all__ = [
    "BackendName",
    "TernairDirectInferencer",
    "InferenceBackend",
]