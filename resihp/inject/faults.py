"""Fault injection.

Two complementary mechanisms:

* ``FaultInjector`` runs inside a worker and fires at deterministic hook points
  (iteration, phase). It is what the CPU test-suite and ResiBench use to make
  failure scenarios exactly reproducible.
* ``ExternalInjector`` produces (and optionally executes) the host-level
  commands that emulate real hardware faults on a GPU cluster: SIGKILL/SIGSTOP,
  SM clock locking, power capping, DCGM XID injection, NIC down, and
  bandwidth contention.

Spec grammar (``RESIHP_FAULTS`` env var or ``parse_faults``)::

    kind@key=value:key=value;kind@...
    e.g.  kill@iter=5:target=n0g1;slow@iter=10:target=n1g0:factor=2.0:duration=30
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from resihp.utils.log import get_logger

log = get_logger(__name__)


class FaultKind(str, Enum):
    KILL = "kill"
    HANG = "hang"
    SLOW_COMPUTE = "slow"
    SLOW_LINK = "slow_link"
    XID = "xid"
    NIC_DOWN = "nic_down"
    POWER_CAP = "power_cap"


@dataclass(frozen=True)
class FaultSpec:
    kind: FaultKind
    target: str
    at_iteration: int | None = None
    at_time: float | None = None
    duration: float | None = None
    factor: float = 1.0
    phase: str = "forward"
    params: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def param(self, key: str, default: str | None = None) -> str | None:
        return dict(self.params).get(key, default)

    def to_spec(self) -> str:
        parts = [f"target={self.target}"]
        if self.at_iteration is not None:
            parts.append(f"iter={self.at_iteration}")
        if self.at_time is not None:
            parts.append(f"time={self.at_time}")
        if self.duration is not None:
            parts.append(f"duration={self.duration}")
        if self.factor != 1.0:
            parts.append(f"factor={self.factor}")
        if self.phase != "forward":
            parts.append(f"phase={self.phase}")
        parts.extend(f"{k}={v}" for k, v in self.params)
        return f"{self.kind.value}@" + ":".join(parts)


_RESERVED = {"target", "iter", "time", "duration", "factor", "phase"}


def parse_faults(spec: str | None) -> list[FaultSpec]:
    if not spec:
        return []
    faults = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        if "@" not in item:
            raise ValueError(f"fault '{item}' must look like kind@key=value:...")
        kind_s, rest = item.split("@", 1)
        kv = {}
        for tok in rest.split(":"):
            if not tok:
                continue
            if "=" not in tok:
                raise ValueError(f"bad token '{tok}' in fault '{item}'")
            k, v = tok.split("=", 1)
            kv[k.strip()] = v.strip()
        if "target" not in kv:
            raise ValueError(f"fault '{item}' has no target")
        faults.append(
            FaultSpec(
                kind=FaultKind(kind_s.strip()),
                target=kv["target"],
                at_iteration=int(kv["iter"]) if "iter" in kv else None,
                at_time=float(kv["time"]) if "time" in kv else None,
                duration=float(kv["duration"]) if "duration" in kv else None,
                factor=float(kv.get("factor", 1.0)),
                phase=kv.get("phase", "forward"),
                params=tuple(sorted((k, v) for k, v in kv.items() if k not in _RESERVED)),
            )
        )
    return faults


class FaultInjector:
    """In-worker injector driven by hook points in the training runtime."""

    def __init__(self, worker_id: str, faults: list[FaultSpec], *, start_time: float | None = None,
                 kill_fn: Callable[[], None] | None = None):
        self.worker_id = worker_id
        self.faults = [f for f in faults if f.target == worker_id or f.target == "*"]
        self.fired: set[int] = set()
        self.start_time = start_time if start_time is not None else time.monotonic()
        self._kill_fn = kill_fn
        self._slow_until: float | None = None
        self._slow_factor = 1.0
        self._link_delay: dict[str, float] = {}

    def on_hook(self, iteration: int, phase: str) -> None:
        now = time.monotonic()
        for idx, f in enumerate(self.faults):
            if idx in self.fired:
                continue
            if f.at_iteration is not None and (iteration < f.at_iteration or phase != f.phase):
                continue
            if f.at_time is not None and now - self.start_time < f.at_time:
                continue
            if f.at_iteration is None and f.at_time is None:
                continue
            self.fired.add(idx)
            self._fire(f, now)

    def _fire(self, f: FaultSpec, now: float) -> None:
        log.warning("injecting %s on %s", f.to_spec(), self.worker_id)
        if f.kind == FaultKind.KILL:
            if self._kill_fn is not None:
                self._kill_fn()
            else:
                os.kill(os.getpid(), signal.SIGKILL)
        elif f.kind == FaultKind.HANG:
            if f.duration is not None:
                subprocess.Popen(["sh", "-c", f"sleep {f.duration}; kill -CONT {os.getpid()}"])
            os.kill(os.getpid(), signal.SIGSTOP)
        elif f.kind == FaultKind.SLOW_COMPUTE:
            self._slow_factor = max(1.0, f.factor)
            self._slow_until = None if f.duration is None else now + f.duration
        elif f.kind == FaultKind.SLOW_LINK:
            peer = f.param("peer", "*")
            self._link_delay[peer] = float(f.param("delay_s", "0.01"))
        else:
            raise ValueError(f"{f.kind} cannot be injected in-process; use ExternalInjector")

    def compute_factor(self) -> float:
        if self._slow_until is not None and time.monotonic() > self._slow_until:
            self._slow_factor, self._slow_until = 1.0, None
        return self._slow_factor

    def pad_compute(self, elapsed_s: float) -> float:
        """Busy-wait so a chunk that took ``elapsed_s`` appears ``factor`` times slower; returns padding."""
        factor = self.compute_factor()
        if factor <= 1.0 or elapsed_s <= 0:
            return 0.0
        pad = (factor - 1.0) * elapsed_s
        deadline = time.perf_counter() + pad
        while time.perf_counter() < deadline:
            pass
        return pad

    def link_delay(self, peer: str) -> float:
        delay = self._link_delay.get(peer, self._link_delay.get("*", 0.0))
        if delay > 0:
            time.sleep(delay)
        return delay


_INJECTOR: FaultInjector | None = None


def get_injector() -> FaultInjector:
    """Process-wide injector configured from ``RESIHP_FAULTS`` and ``RESIHP_WORKER_ID``."""
    global _INJECTOR
    if _INJECTOR is None:
        _INJECTOR = FaultInjector(os.environ.get("RESIHP_WORKER_ID", ""),
                                  parse_faults(os.environ.get("RESIHP_FAULTS")))
    return _INJECTOR


def reset_injector(injector: FaultInjector | None = None) -> None:
    global _INJECTOR
    _INJECTOR = injector


@dataclass
class ExternalCommand:
    apply: list[str]
    revert: list[str] | None = None


class ExternalInjector:
    """Host-level fault emulation for GPU clusters (requires the corresponding privileges)."""

    def __init__(self, dry_run: bool = False, runner: Callable[[list[str]], int] | None = None):
        self.dry_run = dry_run
        self.executed: list[list[str]] = []
        self._runner = runner or (lambda cmd: subprocess.call(cmd))

    @staticmethod
    def commands(f: FaultSpec) -> ExternalCommand:
        gpu = f.param("gpu", "0")
        if f.kind == FaultKind.KILL:
            return ExternalCommand(["kill", "-9", f.param("pid", "0")])
        if f.kind == FaultKind.HANG:
            pid = f.param("pid", "0")
            return ExternalCommand(["kill", "-STOP", pid], ["kill", "-CONT", pid])
        if f.kind == FaultKind.SLOW_COMPUTE:
            base = float(f.param("base_mhz", "1980"))
            mhz = str(int(base / max(f.factor, 1.0)))
            return ExternalCommand(["nvidia-smi", "-i", gpu, "-lgc", f"{mhz},{mhz}"],
                                   ["nvidia-smi", "-i", gpu, "-rgc"])
        if f.kind == FaultKind.POWER_CAP:
            return ExternalCommand(["nvidia-smi", "-i", gpu, "-pl", f.param("watts", "300")],
                                   ["nvidia-smi", "-i", gpu, "-pl", f.param("default_watts", "700")])
        if f.kind == FaultKind.XID:
            # DCGM field 230 is DCGM_FI_DEV_XID_ERRORS; injection exercises the health monitor path.
            return ExternalCommand(["dcgmi", "test", "--inject", "--gpuid", gpu, "-f", "230",
                                    "-v", f.param("xid", "79")])
        if f.kind == FaultKind.NIC_DOWN:
            iface = f.param("iface", "ib0")
            return ExternalCommand(["ip", "link", "set", "dev", iface, "down"],
                                   ["ip", "link", "set", "dev", iface, "up"])
        if f.kind == FaultKind.SLOW_LINK:
            return ExternalCommand(["python", "-m", "resihp.inject.contention", "--host", f.param("peer", "127.0.0.1"),
                                    "--port", f.param("port", "29700"), "--seconds", str(f.duration or 60)])
        raise ValueError(f"unsupported external fault {f.kind}")

    def apply(self, f: FaultSpec) -> ExternalCommand:
        cmd = self.commands(f)
        self._run(cmd.apply)
        return cmd

    def revert(self, f: FaultSpec) -> None:
        cmd = self.commands(f)
        if cmd.revert:
            self._run(cmd.revert)

    def _run(self, argv: list[str]) -> None:
        self.executed.append(argv)
        if self.dry_run:
            log.warning("[dry-run] %s", shlex.join(argv))
            return
        rc = self._runner(argv)
        if rc != 0:
            raise RuntimeError(f"fault command failed ({rc}): {shlex.join(argv)}")


def run_schedule(faults: list[FaultSpec], injector: ExternalInjector, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> list[tuple[float, FaultSpec, str]]:
    """Execute time-triggered external faults (and their reverts) in order; returns an event log."""
    t0 = clock()
    events: list[tuple[float, str, FaultSpec]] = []
    for f in faults:
        if f.at_time is None:
            continue
        events.append((f.at_time, "apply", f))
        if f.duration is not None:
            events.append((f.at_time + f.duration, "revert", f))
    events.sort(key=lambda e: (e[0], e[1] != "revert"))
    out = []
    for at, action, f in events:
        wait = at - (clock() - t0)
        if wait > 0:
            sleep(wait)
        if action == "apply":
            injector.apply(f)
        else:
            injector.revert(f)
        out.append((at, f, action))
    return out
