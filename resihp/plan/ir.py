"""Pipeline instruction IR executed by the worker-side interpreter.

One instruction stream per physical (replica, stage) slot; every TP rank of the slot
runs the same stream. Communication instructions name the peer *group* as a
(replica, virtual stage) key; the interpreter expands them into rank-level transfers
for the two TP layouts. With interleaving, an instruction also names the virtual
stage (model chunk) it belongs to.

NCCL safety: ResiHP gives each direction of a communicating rank pair its own
2-rank channel (shared by all chunks), so a schedule is safe if (a) per direction of
a physical pair, messages are received in send order, and (b) it is deadlock-free
when sends never block. ``validate`` checks both, plus completeness and data
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from resihp.plan.hetero_plan import HeteroParallelPlan

Key = tuple[int, int]


class Op(str, Enum):
    F = "F"
    B = "B"
    W = "W"
    SEND_ACT = "SA"
    RECV_ACT = "RA"
    SEND_GRAD = "SG"
    RECV_GRAD = "RG"


COMPUTE = {Op.F, Op.B, Op.W}
SENDS = {Op.SEND_ACT: "act", Op.SEND_GRAD: "grad"}
RECVS = {Op.RECV_ACT: "act", Op.RECV_GRAD: "grad"}


@dataclass(frozen=True)
class Instr:
    op: Op
    mb: int
    peer: Key | None = None
    stage: int | None = None

    def stage_on(self, key: Key) -> int:
        """Virtual stage of this instruction when it sits in the stream of ``key``."""
        return key[1] if self.stage is None else self.stage

    def to_json(self) -> list:
        peer = list(self.peer) if self.peer is not None else []
        if self.stage is None:
            return [self.op.value, self.mb] + peer
        return [self.op.value, self.mb] + (peer or [-1, -1]) + [self.stage]

    @staticmethod
    def from_json(d: list) -> "Instr":
        peer = (d[2], d[3]) if len(d) > 2 and d[2] >= 0 else None
        return Instr(Op(d[0]), d[1], peer, d[4] if len(d) > 4 else None)


Schedule = dict[Key, list[Instr]]


class ScheduleError(ValueError):
    pass


def compute_order_to_ir(plan: HeteroParallelPlan, order: dict[Key, list[tuple]]) -> Schedule:
    """``order`` maps a physical slot to ``(op, mb)`` or, with interleaving, ``(op, mb, stage)``."""
    S = plan.num_stages
    out: Schedule = {}
    for key, ops in order.items():
        stream: list[Instr] = []
        for entry in ops:
            op, mb = entry[0], entry[1]
            s = entry[2] if len(entry) > 2 else key[1]
            tag = s if plan.vpp > 1 else None
            if op == "F":
                if s > 0:
                    stream.append(Instr(Op.RECV_ACT, mb, (plan.executor(s - 1, mb), s - 1), tag))
                stream.append(Instr(Op.F, mb, None, tag))
                if s < S - 1:
                    stream.append(Instr(Op.SEND_ACT, mb, (plan.executor(s + 1, mb), s + 1), tag))
            elif op == "B":
                if s < S - 1:
                    stream.append(Instr(Op.RECV_GRAD, mb, (plan.executor(s + 1, mb), s + 1), tag))
                stream.append(Instr(Op.B, mb, None, tag))
                if s > 0:
                    stream.append(Instr(Op.SEND_GRAD, mb, (plan.executor(s - 1, mb), s - 1), tag))
            elif op == "W":
                stream.append(Instr(Op.W, mb, None, tag))
            else:
                raise ScheduleError(f"unknown compute op {op}")
        out[key] = stream
    return out


def schedule_to_json(schedule: Schedule) -> dict:
    return {f"{r},{s}": [i.to_json() for i in stream] for (r, s), stream in schedule.items()}


def schedule_from_json(d: dict) -> Schedule:
    out: Schedule = {}
    for k, stream in d.items():
        r, s = (int(x) for x in k.split(","))
        out[(r, s)] = [Instr.from_json(i) for i in stream]
    return out


def validate(plan: HeteroParallelPlan, schedule: Schedule, split_backward: bool = False) -> None:
    S, M = plan.num_stages, plan.num_microbatches
    phys = lambda k: plan.physical_key(*k)  # noqa: E731
    for s in range(S):
        for m in range(M):
            key = phys((plan.executor(s, m), s))
            stream = schedule.get(key, [])
            pos = {op: [i for i, ins in enumerate(stream) if ins.op == op and ins.mb == m and ins.stage_on(key) == s]
                   for op in COMPUTE}
            if len(pos[Op.F]) != 1 or len(pos[Op.B]) != 1:
                raise ScheduleError(f"mb {m} stage {s}: expected one F and one B on {key}")
            if pos[Op.F][0] > pos[Op.B][0]:
                raise ScheduleError(f"mb {m} stage {s}: B before F")
            if split_backward:
                if len(pos[Op.W]) != 1 or pos[Op.W][0] < pos[Op.B][0]:
                    raise ScheduleError(f"mb {m} stage {s}: W missing or before B")
            elif pos[Op.W]:
                raise ScheduleError(f"mb {m} stage {s}: unexpected W")
    for key, stream in schedule.items():
        for ins in stream:
            s = ins.stage_on(key)
            if ins.op in COMPUTE and phys((plan.executor(s, ins.mb), s)) != key:
                raise ScheduleError(f"{key} executes mb {ins.mb} of stage {s} assigned to replica "
                                    f"{plan.executor(s, ins.mb)}")

    sends: dict[tuple[Key, Key, str], list[tuple[int, int]]] = {}
    recvs: dict[tuple[Key, Key, str], list[tuple[int, int]]] = {}
    for key, stream in schedule.items():
        for ins in stream:
            if ins.op in SENDS:
                sends.setdefault((key, phys(ins.peer), SENDS[ins.op]), []).append((ins.mb, ins.peer[1]))
            elif ins.op in RECVS:
                recvs.setdefault((phys(ins.peer), key, RECVS[ins.op]), []).append((ins.mb, ins.stage_on(key)))
    for ch in set(sends) | set(recvs):
        if sends.get(ch, []) != recvs.get(ch, []):
            raise ScheduleError(f"channel {ch}: send order {sends.get(ch)} != receive order {recvs.get(ch)}")
    if not stream_simulation(schedule, phys=phys).complete:
        raise ScheduleError("schedule deadlocks")


@dataclass
class StreamSimResult:
    complete: bool
    finish: dict[Key, float]
    makespan: float


def stream_simulation(schedule: Schedule, duration=lambda key, ins: 1.0 if ins.op in COMPUTE else 0.0,
                      transfer=lambda src, dst, ins: 0.0, phys=lambda k: k) -> StreamSimResult:
    """Earliest-start evaluation of fixed streams (the DAG longest path of the paper's Eq. 2).

    Sends are non-blocking; a receive starts once the matching send has completed and
    the transfer time has elapsed. ``phys`` maps a peer's virtual key to its stream.
    """
    ptr = {k: 0 for k in schedule}
    free = {k: 0.0 for k in schedule}
    sent: dict[tuple, float] = {}
    progress = True
    while progress:
        progress = False
        for key, stream in schedule.items():
            while ptr[key] < len(stream):
                ins = stream[ptr[key]]
                start = free[key]
                if ins.op in RECVS:
                    tag = (phys(ins.peer), key, RECVS[ins.op], ins.mb, ins.stage_on(key))
                    if tag not in sent:
                        break
                    start = max(start, sent[tag])
                end = start + duration(key, ins)
                if ins.op in SENDS:
                    sent[(key, phys(ins.peer), SENDS[ins.op], ins.mb, ins.peer[1])] = end + transfer(key, ins.peer, ins)
                free[key] = end
                ptr[key] += 1
                progress = True
    complete = all(ptr[k] == len(s) for k, s in schedule.items())
    return StreamSimResult(complete, dict(free), max(free.values(), default=0.0))
