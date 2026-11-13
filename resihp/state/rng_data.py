"""Placement-independent data and randomness.

A micro-batch is identified by ``(iteration, global_microbatch_id)``. Which
samples it contains, and every random number it consumes, are pure functions of
that identity, so a micro-batch can be re-executed after a failure, or executed
by a different DP replica than originally planned, without changing training
semantics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

_MASK64 = (1 << 64) - 1


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def derive_seed(*parts: int) -> int:
    h = 0
    for p in parts:
        h = splitmix64(h ^ (int(p) & _MASK64))
    return h & ((1 << 63) - 1)


def generator(*parts: int, device: str = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(derive_seed(*parts))
    return g


class IndexAddressableSampler:
    """Global batch ``i`` is a fixed slice of a per-epoch permutation; independent of DP size."""

    def __init__(self, num_samples: int, global_batch_size: int, micro_batch_size: int, seed: int = 0,
                 shuffle: bool = True):
        if global_batch_size % micro_batch_size:
            raise ValueError("global batch must be a multiple of the micro batch")
        self.num_samples = num_samples
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.seed = seed
        self.shuffle = shuffle
        self._perm_cache: dict[int, np.ndarray] = {}

    @property
    def num_microbatches(self) -> int:
        return self.global_batch_size // self.micro_batch_size

    def _perm(self, epoch: int) -> np.ndarray:
        p = self._perm_cache.get(epoch)
        if p is None:
            if self.shuffle:
                p = np.random.default_rng(derive_seed(self.seed, epoch)).permutation(self.num_samples)
            else:
                p = np.arange(self.num_samples)
            if len(self._perm_cache) > 4:
                self._perm_cache.clear()
            self._perm_cache[epoch] = p
        return p

    def sample_index(self, position: int) -> int:
        epoch, offset = divmod(position, self.num_samples)
        return int(self._perm(epoch)[offset])

    def batch_indices(self, iteration: int) -> list[int]:
        start = iteration * self.global_batch_size
        return [self.sample_index(start + i) for i in range(self.global_batch_size)]

    def microbatch_indices(self, iteration: int, mb: int) -> list[int]:
        if not 0 <= mb < self.num_microbatches:
            raise IndexError(mb)
        start = iteration * self.global_batch_size + mb * self.micro_batch_size
        return [self.sample_index(start + i) for i in range(self.micro_batch_size)]

    def consumed_samples(self, iteration: int) -> int:
        return iteration * self.global_batch_size


@dataclass
class DataCursor:
    iteration: int = 0

    def snapshot(self) -> dict:
        return {"iteration": self.iteration}

    def restore(self, snap: dict) -> None:
        self.iteration = snap["iteration"]


def capture_rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
