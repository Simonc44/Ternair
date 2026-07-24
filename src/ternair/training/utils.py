"""Training utilities: logging, checkpointing helpers, version check."""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, log_level.upper(), logging.INFO),
        stream=sys.stderr,
    )
