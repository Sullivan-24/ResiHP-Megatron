"""T2: persistent checkpoints stored as logical slices, so any later plan can load them.

Each rank writes the slices it owns: ``(key, global_shape, partition_dim, interval)``
plus one tensor per state kind (param / main / exp_avg / exp_avg_sq). Loading
assembles any requested interval from whichever files overlap it, i.e. resharding
on load for arbitrary TP/PP/DP changes.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

from resihp.utils.intervals import intersect


@dataclass(frozen=True)
class SliceMeta:
    key: str
    global_shape: tuple[int, ...]
    dim: int | None
    interval: tuple[int, int]

    def to_json(self) -> dict:
        return {"key": self.key, "global_shape": list(self.global_shape), "dim": self.dim,
                "interval": list(self.interval)}

    @staticmethod
    def from_json(d: dict) -> "SliceMeta":
        return SliceMeta(d["key"], tuple(d["global_shape"]), d["dim"], tuple(d["interval"]))


class SliceCheckpointer:
    def __init__(self, directory: str | os.PathLike):
        self.root = Path(directory)
        self._thread: threading.Thread | None = None

    def _dir(self, iteration: int) -> Path:
        return self.root / f"iter_{iteration:08d}"

    def save(self, iteration: int, owner: str, slices: list[tuple[SliceMeta, dict[str, torch.Tensor]]],
             extra: dict | None = None, blocking: bool = False) -> None:
        self.wait()
        payload = [(m, {k: v.detach().to("cpu", copy=True) for k, v in t.items()}) for m, t in slices]

        def work():
            d = self._dir(iteration)
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f".{owner}.pt.tmp"
            torch.save({"slices": [(m.to_json(), t) for m, t in payload], "extra": extra or {}}, tmp)
            os.replace(tmp, d / f"{owner}.pt")
            (d / f"{owner}.done").write_text(json.dumps([m.to_json() for m, _ in payload]))

        if blocking:
            work()
        else:
            self._thread = threading.Thread(target=work, daemon=True, name="ckpt-save")
            self._thread.start()

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _index(self, iteration: int) -> list[tuple[str, SliceMeta]]:
        d = self._dir(iteration)
        out = []
        for done in sorted(d.glob("*.done")):
            owner = done.name[:-len(".done")]
            out.extend((owner, SliceMeta.from_json(m)) for m in json.loads(done.read_text()))
        return out

    def complete(self, iteration: int, required: dict[str, tuple[tuple[int, ...], int | None]]) -> bool:
        """Every required key is fully covered by saved slices."""
        idx = self._index(iteration)
        for key, (shape, dim) in required.items():
            parts = sorted(m.interval for _, m in idx if m.key == key)
            n = shape[dim] if dim is not None else 1
            end = 0
            for s, e in parts:
                if s > end:
                    return False
                end = max(end, e)
            if end < n:
                return False
        return True

    def latest_complete(self, required: dict[str, tuple[tuple[int, ...], int | None]]) -> int | None:
        if not self.root.exists():
            return None
        its = sorted((int(p.name[5:]) for p in self.root.glob("iter_*")), reverse=True)
        return next((it for it in its if self.complete(it, required)), None)

    def load(self, iteration: int, wanted: list[SliceMeta], kinds: tuple[str, ...]) -> dict[str, dict[str, torch.Tensor]]:
        """Assemble each wanted slice (keyed by ``meta.key``) from overlapping saved slices."""
        idx = self._index(iteration)
        cache: dict[str, list] = {}
        out: dict[str, dict[str, torch.Tensor]] = {}
        for want in wanted:
            dim = want.dim
            shape = list(want.global_shape)
            if dim is not None:
                shape[dim] = want.interval[1] - want.interval[0]
            assembled = {k: None for k in kinds}
            covered = 0
            for owner, meta in idx:
                if meta.key != want.key:
                    continue
                o = intersect(meta.interval, want.interval) if dim is not None else (0, 1)
                if o is None:
                    continue
                if owner not in cache:
                    cache[owner] = torch.load(self._dir(iteration) / f"{owner}.pt", weights_only=False)["slices"]
                tensors = next(t for m, t in cache[owner] if SliceMeta.from_json(m) == meta)
                for k in kinds:
                    if assembled[k] is None:
                        assembled[k] = torch.empty(shape, dtype=tensors[k].dtype)
                    if dim is None:
                        assembled[k].copy_(tensors[k])
                    else:
                        src = tensors[k].narrow(dim, o[0] - meta.interval[0], o[1] - o[0])
                        assembled[k].narrow(dim, o[0] - want.interval[0], o[1] - o[0]).copy_(src)
                covered += (o[1] - o[0]) if dim is not None else 1
                if dim is None:
                    break
            need = (want.interval[1] - want.interval[0]) if dim is not None else 1
            if covered < need:
                raise KeyError(f"checkpoint {iteration} does not cover {want}")
            out[want.key] = assembled
        return out
