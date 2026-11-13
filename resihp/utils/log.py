"""Process-tagged logging so interleaved multi-rank output stays readable."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        level = os.environ.get("RESIHP_LOG_LEVEL", "WARNING").upper()
        tag = os.environ.get("RESIHP_WORKER_ID", f"pid{os.getpid()}")
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(f"[%(asctime)s %(levelname)s {tag} %(name)s] %(message)s"))
        root = logging.getLogger("resihp")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("resihp") else f"resihp.{name}")
