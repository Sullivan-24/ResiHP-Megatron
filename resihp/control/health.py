"""Node health probes (NVML/XID/ECC/throttling and InfiniBand port state).

Every probe degrades to a no-op when the underlying facility is missing, so the
same agent binary runs on CPU-only CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# XIDs after which the GPU must be considered lost for this job.
FATAL_XIDS = {48, 61, 62, 63, 64, 74, 79, 92, 94, 95, 119, 120, 140}


@dataclass(frozen=True)
class HealthEvent:
    device: str
    kind: str
    fatal: bool
    detail: str = ""

    def matches(self, worker_device: str) -> bool:
        return self.device in ("*", worker_device)

    def describe(self) -> str:
        return f"{self.kind} on {self.device}: {self.detail}"


class NVMLProbe:
    def __init__(self, nvml=None, device_indices: list[int] | None = None):
        self.nvml = nvml
        self._ok = False
        self._handles: dict[int, object] = {}
        self._events = None
        self._last_ecc: dict[int, int] = {}
        try:
            if self.nvml is None:
                import pynvml  # optional dependency (nvidia-ml-py)

                self.nvml = pynvml
            self.nvml.nvmlInit()
            n = self.nvml.nvmlDeviceGetCount()
            for i in device_indices if device_indices is not None else range(n):
                self._handles[i] = self.nvml.nvmlDeviceGetHandleByIndex(i)
            self._events = self.nvml.nvmlEventSetCreate()
            for h in self._handles.values():
                try:
                    self.nvml.nvmlDeviceRegisterEvents(h, self.nvml.nvmlEventTypeXidCriticalError, self._events)
                except Exception:  # noqa: BLE001 - some SKUs do not support XID events
                    pass
            self._ok = True
        except Exception:  # noqa: BLE001 - no NVML (CPU host)
            self._ok = False

    def poll(self) -> list[HealthEvent]:
        if not self._ok:
            return []
        out: list[HealthEvent] = []
        while True:
            try:
                ev = self.nvml.nvmlEventSetWait(self._events, 0)
            except Exception:  # noqa: BLE001 - NVML_ERROR_TIMEOUT means no pending event
                break
            idx = next((i for i, h in self._handles.items() if h == ev.device), -1)
            xid = int(ev.eventData)
            out.append(HealthEvent(f"cuda:{idx}" if idx >= 0 else "*", "xid", xid in FATAL_XIDS, f"XID {xid}"))
        for i, h in self._handles.items():
            try:
                ecc = self.nvml.nvmlDeviceGetTotalEccErrors(h, self.nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                                            self.nvml.NVML_VOLATILE_ECC)
                if ecc > self._last_ecc.get(i, ecc):
                    out.append(HealthEvent(f"cuda:{i}", "ecc", True, f"uncorrected ECC count {ecc}"))
                self._last_ecc[i] = ecc
            except Exception:  # noqa: BLE001
                pass
        return out

    def throttle(self, index: int) -> dict:
        """Signals used by the fail-slow detector as side information (not as a trigger)."""
        if not self._ok:
            return {}
        h = self._handles[index]
        out = {}
        for key, fn in (("sm_clock", lambda: self.nvml.nvmlDeviceGetClockInfo(h, self.nvml.NVML_CLOCK_SM)),
                        ("temp", lambda: self.nvml.nvmlDeviceGetTemperature(h, self.nvml.NVML_TEMPERATURE_GPU)),
                        ("throttle", lambda: self.nvml.nvmlDeviceGetCurrentClocksThrottleReasons(h))):
            try:
                out[key] = fn()
            except Exception:  # noqa: BLE001
                pass
        return out


class IBPortProbe:
    def __init__(self, root: str = "/sys/class/infiniband"):
        self.root = Path(root)

    def poll(self) -> list[HealthEvent]:
        """One rail down degrades the node (a fail-slow candidate); every rail down isolates it."""
        if not self.root.exists():
            return []
        ports = sorted(self.root.glob("*/ports/*/state"))
        down = []
        for state in ports:
            text = state.read_text().strip()
            if "ACTIVE" not in text:
                down.append(f"{state.parts[-4]} port {state.parts[-2]} {text}")
        if not down:
            return []
        fatal = len(down) == len(ports)
        return [HealthEvent("*", "ib_port", fatal, d) for d in down]


class CompositeProbe:
    def __init__(self, *probes):
        self.probes = probes

    def poll(self) -> list[HealthEvent]:
        return [ev for p in self.probes for ev in p.poll()]
