"""Builds the NCCL profiler plugin and drives its v5 entry points through ctypes (no NCCL needed)."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest

from resihp.detect.nccl_profile_reader import COLL, P2P, RingReader, find_rings

SRC = Path(__file__).resolve().parents[2] / "resihp" / "detect" / "nccl_profiler"
CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")


class Coll(ctypes.Structure):
    _fields_ = [("seqNumber", ctypes.c_uint64), ("func", ctypes.c_char_p), ("sendBuff", ctypes.c_void_p),
                ("recvBuff", ctypes.c_void_p), ("count", ctypes.c_size_t), ("root", ctypes.c_int),
                ("datatype", ctypes.c_char_p), ("nChannels", ctypes.c_uint8), ("nWarps", ctypes.c_uint8),
                ("algo", ctypes.c_char_p), ("proto", ctypes.c_char_p), ("parentGroup", ctypes.c_void_p)]


class P2p(ctypes.Structure):
    _fields_ = [("func", ctypes.c_char_p), ("buff", ctypes.c_void_p), ("datatype", ctypes.c_char_p),
                ("count", ctypes.c_size_t), ("peer", ctypes.c_int), ("nChannels", ctypes.c_uint8),
                ("parentGroup", ctypes.c_void_p)]


class U(ctypes.Union):
    _fields_ = [("coll", Coll), ("p2p", P2p)]


class Descr(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_uint64), ("parentObj", ctypes.c_void_p), ("rank", ctypes.c_int), ("u", U)]


INIT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64, ctypes.POINTER(ctypes.c_int),
                        ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
START = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(Descr))
STOP = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
STATE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
FIN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


class Profiler(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("init", INIT), ("startEvent", START), ("stopEvent", STOP),
                ("recordEventState", STATE), ("finalize", FIN)]


@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_plugin_records_coll_and_p2p(tmp_path, monkeypatch):
    lib = tmp_path / "libresihp_nccl_profiler.so"
    subprocess.run([CC, "-O2", "-fPIC", "-shared", "-std=c11", "-Wall", "-Werror", "-o", str(lib),
                    str(SRC / "resihp_nccl_profiler.c"), "-lpthread"], check=True)
    monkeypatch.setenv("RESIHP_NCCL_PROF_DIR", str(tmp_path))
    prof = Profiler.in_dll(ctypes.CDLL(str(lib)), "ncclProfiler_v5")
    assert prof.name == b"resihp"
    ctx = ctypes.c_void_p()
    mask = ctypes.c_int(0)
    assert prof.init(ctypes.byref(ctx), 0xABC, ctypes.byref(mask), b"tp", 1, 4, 2, None) == 0
    assert mask.value == COLL | P2P
    reader = RingReader(find_rings(tmp_path)[0])
    d = Descr(type=COLL, rank=2)
    d.coll.func, d.coll.count = b"AllReduce", 1024
    h = ctypes.c_void_p()
    prof.startEvent(ctx, ctypes.byref(h), ctypes.byref(d))
    prof.stopEvent(h)
    d2 = Descr(type=P2P, rank=2)
    d2.p2p.func, d2.p2p.count, d2.p2p.peer = b"Send", 77, 3
    prof.startEvent(ctx, ctypes.byref(h), ctypes.byref(d2))
    prof.stopEvent(h)
    ignored = Descr(type=1 << 3, rank=2)
    prof.startEvent(ctx, ctypes.byref(h), ctypes.byref(ignored))
    assert h.value is None
    recs = reader.poll()
    assert [(r.func, r.count, r.peer, r.type, r.comm_id, r.rank, r.nranks) for r in recs] == [
        ("AllReduce", 1024, -1, COLL, 0xABC, 2, 4), ("Send", 77, 3, P2P, 0xABC, 2, 4)]
    assert all(r.end_ns >= r.start_ns for r in recs)
    assert reader.poll() == []
    reader.close()
    assert prof.finalize(ctx) == 0
