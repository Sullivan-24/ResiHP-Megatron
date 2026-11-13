"""Reader for the ring written by the ResiHP NCCL profiler plugin (resihp/detect/nccl_profiler)."""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = 0x52455348
HEADER = struct.Struct("<IIII Q")
RECORD = struct.Struct("<QQQQ iiii 24s Q")
COLL, P2P = 1 << 1, 1 << 2


@dataclass(frozen=True)
class CommRecord:
    start_ns: int
    end_ns: int
    comm_id: int
    count: int
    rank: int
    nranks: int
    type: int
    peer: int
    func: str

    @property
    def seconds(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9


class RingReader:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._f = open(self.path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        magic, version, capacity, rsize, _ = HEADER.unpack_from(self._mm, 0)
        if magic != MAGIC or rsize != RECORD.size:
            raise ValueError(f"{path}: not a ResiHP NCCL ring (magic={magic:#x}, record={rsize}B)")
        self.capacity = capacity
        self.cursor = 0
        self.dropped = 0

    def head(self) -> int:
        return HEADER.unpack_from(self._mm, 0)[4]

    def poll(self) -> list[CommRecord]:
        head = self.head()
        if head - self.cursor > self.capacity:
            self.dropped += head - self.cursor - self.capacity
            self.cursor = head - self.capacity
        out = []
        while self.cursor < head:
            off = HEADER.size + (self.cursor % self.capacity) * RECORD.size
            vals = RECORD.unpack_from(self._mm, off)
            if vals[-1] != self.cursor + 1:
                break
            s, e, comm, count, rank, nranks, typ, peer, func, _ = vals
            out.append(CommRecord(s, e, comm, count, rank, nranks, typ, peer, func.split(b"\0", 1)[0].decode()))
            self.cursor += 1
        return out

    def close(self) -> None:
        self._mm.close()
        self._f.close()


def find_rings(directory: str | os.PathLike = "/dev/shm") -> list[Path]:
    return sorted(Path(directory).glob("resihp_nccl_*.ring"))
