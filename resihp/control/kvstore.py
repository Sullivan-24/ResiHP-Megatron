"""Small KV interface for control-plane state (generation, membership, plans, leader lease).

``InMemoryKV`` backs tests and single-coordinator deployments; ``EtcdKV`` is the
production backend that makes the coordinator stateless and replaceable.
"""

from __future__ import annotations

import itertools
import threading
import time
from typing import Callable, Protocol


class KVStore(Protocol):
    def get(self, key: str) -> str | None: ...
    def get_prefix(self, prefix: str) -> dict[str, str]: ...
    def put(self, key: str, value: str, lease: int | None = None) -> None: ...
    def cas(self, key: str, expected: str | None, value: str, lease: int | None = None) -> bool: ...
    def put_if(self, key: str, value: str, guard_key: str, guard_value: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def grant_lease(self, ttl: float) -> int: ...
    def keepalive(self, lease: int) -> bool: ...
    def revoke(self, lease: int) -> None: ...


class InMemoryKV:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.RLock()
        self._data: dict[str, tuple[str, int | None]] = {}
        self._leases: dict[int, tuple[float, float]] = {}
        self._ids = itertools.count(1)

    def _expire(self) -> None:
        now = self._clock()
        dead = {lid for lid, (expiry, _) in self._leases.items() if expiry <= now}
        if not dead:
            return
        for lid in dead:
            del self._leases[lid]
        for k in [k for k, (_, lid) in self._data.items() if lid in dead]:
            del self._data[k]

    def get(self, key: str) -> str | None:
        with self._lock:
            self._expire()
            v = self._data.get(key)
            return v[0] if v else None

    def get_prefix(self, prefix: str) -> dict[str, str]:
        with self._lock:
            self._expire()
            return {k: v for k, (v, _) in self._data.items() if k.startswith(prefix)}

    def put(self, key: str, value: str, lease: int | None = None) -> None:
        with self._lock:
            self._expire()
            if lease is not None and lease not in self._leases:
                raise KeyError(f"lease {lease} expired")
            self._data[key] = (value, lease)

    def cas(self, key: str, expected: str | None, value: str, lease: int | None = None) -> bool:
        with self._lock:
            self._expire()
            cur = self._data.get(key)
            if (cur[0] if cur else None) != expected:
                return False
            self.put(key, value, lease)
            return True

    def put_if(self, key: str, value: str, guard_key: str, guard_value: str) -> bool:
        with self._lock:
            self._expire()
            cur = self._data.get(guard_key)
            if cur is None or cur[0] != guard_value:
                return False
            self._data[key] = (value, None)
            return True

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def grant_lease(self, ttl: float) -> int:
        with self._lock:
            lid = next(self._ids)
            self._leases[lid] = (self._clock() + ttl, ttl)
            return lid

    def keepalive(self, lease: int) -> bool:
        with self._lock:
            self._expire()
            if lease not in self._leases:
                return False
            _, ttl = self._leases[lease]
            self._leases[lease] = (self._clock() + ttl, ttl)
            return True

    def revoke(self, lease: int) -> None:
        with self._lock:
            self._leases[lease] = (float("-inf"), 0.0)
            self._expire()


class EtcdKV:
    """etcd v3 backend (``pip install resihp[etcd]``)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 2379, **kwargs):
        import etcd3  # optional dependency

        self._client = etcd3.client(host=host, port=port, **kwargs)
        self._leases: dict[int, object] = {}

    def get(self, key: str) -> str | None:
        v, _ = self._client.get(key)
        return v.decode() if v is not None else None

    def get_prefix(self, prefix: str) -> dict[str, str]:
        return {m.key.decode(): v.decode() for v, m in self._client.get_prefix(prefix)}

    def put(self, key: str, value: str, lease: int | None = None) -> None:
        self._client.put(key, value, lease=self._leases.get(lease) if lease else None)

    def cas(self, key: str, expected: str | None, value: str, lease: int | None = None) -> bool:
        t = self._client.transactions
        cond = t.version(key) == 0 if expected is None else t.value(key) == expected
        ok, _ = self._client.transaction(compare=[cond],
                                         success=[t.put(key, value, lease=self._leases.get(lease) if lease else None)],
                                         failure=[])
        return ok

    def put_if(self, key: str, value: str, guard_key: str, guard_value: str) -> bool:
        t = self._client.transactions
        ok, _ = self._client.transaction(compare=[t.value(guard_key) == guard_value],
                                         success=[t.put(key, value)], failure=[])
        return ok

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def grant_lease(self, ttl: float) -> int:
        lease = self._client.lease(max(1, int(round(ttl))))
        self._leases[lease.id] = lease
        return lease.id

    def keepalive(self, lease: int) -> bool:
        obj = self._leases.get(lease)
        if obj is None:
            return False
        try:
            resp = list(obj.refresh())
            return bool(resp) and resp[0].TTL > 0
        except Exception:  # noqa: BLE001 - treat any etcd error as lost lease
            return False

    def revoke(self, lease: int) -> None:
        obj = self._leases.pop(lease, None)
        if obj is not None:
            obj.revoke()
