"""Public exception types used by Ternair."""

from __future__ import annotations


class TernairError(Exception):
    """Base class for expected Ternair errors."""


class ArtifactError(TernairError):
    """A model artifact is missing, malformed, or inconsistent."""


class ConfigurationError(TernairError, ValueError):
    """A model configuration is invalid for the requested operation."""


class BackendUnavailableError(TernairError, RuntimeError):
    """A requested optional inference backend is unavailable."""


__all__ = [
    "TernairError",
    "ArtifactError",
    "ConfigurationError",
    "BackendUnavailableError",
]
