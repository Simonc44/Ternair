"""Atomic checkpoint helper -- write to .tmp, then ``os.replace``.

This module addresses a real issue: ``torch.save(obj, "ckpt.pt")`` is
*not* atomic at the OS level -- the Python interpreter can be killed
mid-write, leaving a truncated/corrupt ``ckpt.pt``.  By writing to a
sibling ``.tmp`` file and then performing a single ``os.replace`` call
(which is POSIX-atomic on the same filesystem), we guarantee that the
final file is either the old version or the new version, never a
half-written one.

Usage::

    saver = AtomicCheckpointSaver("/path/to/dir")
    saver.save({"step": 42, "model_state_dict": sd})  # writes training_state.tmp -> rename

The class also keeps the previous generation around as
``training_state.prev.pt`` (one-step rollback), so resume can fall
back to the last known-good checkpoint if the latest rename somehow
left a corrupt file (defence-in-depth).
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any

import torch

_LOGGER = logging.getLogger(__name__)

DEFAULT_FILENAME = "training_state.pt"
PREVIOUS_FILENAME = "training_state.prev.pt"
TMP_FILENAME = "training_state.tmp"


@dataclass
class AtomicCheckpointSaver:
    """Write a state dict atomically via ``write to .tmp -> os.replace``.

    Attributes
    ----------
    save_dir
        Directory in which the checkpoints live.  Created if missing.
    filename
        Final filename (default: ``training_state.pt``).
    keep_previous
        Whether to retain the previous generation as ``.prev.pt`` for
        emergency rollback (one-generation-deep).
    """

    save_dir: str
    filename: str = DEFAULT_FILENAME
    keep_previous: bool = True

    def __post_init__(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    @property
    def final_path(self) -> str:
        return os.path.join(self.save_dir, self.filename)

    @property
    def tmp_path(self) -> str:
        return os.path.join(self.save_dir, TMP_FILENAME)

    @property
    def previous_path(self) -> str:
        return os.path.join(self.save_dir, PREVIOUS_FILENAME)

    def save(self, state: Any, *, tag_suffix: str | None = None) -> str:
        """Atomically persist ``state``.

        Steps:
            1. Move the current checkpoint (if any) to ``.prev.pt`` --
               this preserves the previous good copy.
            2. ``torch.save`` to ``.tmp``.
            3. ``os.replace(.tmp, final)`` -- atomic on POSIX.

        Parameters
        ----------
        state
            Any picklable object (typically a ``dict`` of tensors).
        tag_suffix
            If given, the FINAL file is renamed to
            ``training_state.<tag_suffix>.pt``.  Useful for
            ``step_1000`` / ``best`` style save tags.

        Returns
        -------
        str
            Absolute path to the final file.
        """
        final = self.final_path
        if tag_suffix:
            final = os.path.join(
                self.save_dir, f"training_state.{tag_suffix}.pt"
            )

        # 1. Preserve previous generation (for emergency rollback).
        if self.keep_previous and os.path.exists(final):
            try:
                shutil.copy2(final, self.previous_path)
            except OSError as e:
                _LOGGER.warning("Could not preserve previous checkpoint: %s", e)

        # 2. Write to temp file.  torch.save closes its fd internally,
        #    so by the time we return, the .tmp is fully flushed.
        try:
            torch.save(state, self.tmp_path)
        except Exception:
            # Cleanup partial tmp file before propagating.
            if os.path.exists(self.tmp_path):
                try:
                    os.remove(self.tmp_path)
                except OSError:
                    pass
            raise

        # 3. POSIX-atomic rename (Windows: os.replace is also atomic
        #    for files on the same volume).
        os.replace(self.tmp_path, final)
        _LOGGER.debug("Checkpoint written atomically to %s", final)
        return final

    def load(self, path: str | None = None, prefer_previous: bool = False) -> Any:
        """Load a checkpoint, optionally falling back to the previous generation.

        Parameters
        ----------
        path
            Explicit path to load.  If ``None``, uses
            ``final_path`` (or ``previous_path`` if
            ``prefer_previous=True``).
        prefer_previous
            Skip the latest write and load ``previous_path`` instead.
            Useful for crash-recovery: enable when the latest rename
            left a corrupt file.
        """
        candidate = path if path is not None else self.final_path
        if prefer_previous and os.path.exists(self.previous_path):
            candidate = self.previous_path

        if not os.path.exists(candidate):
            raise FileNotFoundError(f"No checkpoint at {candidate}")

        return torch.load(candidate, map_location="cpu", weights_only=False)

    def resolve_resume_path(self) -> str | None:
        """Return the best candidate path for ``resume()`` if any exists.

        Prefers the latest good write, but verifies it is non-empty.
        If the latest looks corrupt, falls back to ``previous_path``.
        Returns ``None`` if no checkpoint is available.
        """
        if os.path.exists(self.final_path):
            # Sanity: file should be non-empty.
            if os.path.getsize(self.final_path) > 0:
                return self.final_path
            _LOGGER.warning("Final checkpoint is empty; falling back to .prev")
        if os.path.exists(self.previous_path):
            return self.previous_path
        return None


__all__ = ["AtomicCheckpointSaver", "DEFAULT_FILENAME", "PREVIOUS_FILENAME"]
