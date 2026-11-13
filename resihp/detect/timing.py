"""Per-chunk timing with deferred readback.

On CUDA, each F/B/W chunk is bracketed by events and read back one iteration
later, so measurement never synchronizes the training stream. On CPU the
wall-clock fallback is used (also what the CPU test-suite exercises).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChunkRecord:
    iteration: int
    op: str
    mb: int
    seconds: float


class ChunkTimer:
    def __init__(self, use_cuda: bool | None = None, injector=None):
        self.use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
        self.injector = injector
        self._pending: list[tuple[int, str, int, object, object]] = []
        self._done: list[ChunkRecord] = []

    @contextmanager
    def time(self, iteration: int, op: str, mb: int):
        if self.use_cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            self._pending.append((iteration, op, mb, start, end))
        else:
            t0 = time.perf_counter()
            yield
            elapsed = time.perf_counter() - t0
            if self.injector is not None:
                elapsed += self.injector.pad_compute(elapsed)
            self._done.append(ChunkRecord(iteration, op, mb, elapsed))

    def collect(self, upto_iteration: int | None = None) -> list[ChunkRecord]:
        """Records of finished chunks (CUDA events are only queried once they completed)."""
        keep = []
        for it, op, mb, start, end in self._pending:
            if (upto_iteration is None or it <= upto_iteration) and end.query():
                self._done.append(ChunkRecord(it, op, mb, start.elapsed_time(end) / 1e3))
            else:
                keep.append((it, op, mb, start, end))
        self._pending = keep
        out, self._done = self._done, []
        return out
