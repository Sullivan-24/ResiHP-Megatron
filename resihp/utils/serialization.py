"""JSON helpers for plans and control-plane messages."""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from typing import Any


def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(to_jsonable(v) for v in obj)
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"))


def loads(s: str | bytes) -> Any:
    return json.loads(s)


def tuplify(obj: Any) -> Any:
    """Recursively convert lists to tuples (JSON round-trips lose tuple-ness)."""
    if isinstance(obj, list):
        return tuple(tuplify(v) for v in obj)
    if isinstance(obj, dict):
        return {k: tuplify(v) for k, v in obj.items()}
    return obj
