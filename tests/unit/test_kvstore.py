import pytest

from resihp.control.kvstore import InMemoryKV


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_put_get_cas_delete_prefix():
    kv = InMemoryKV()
    assert kv.get("a") is None
    assert kv.cas("a", None, "1")
    assert not kv.cas("a", None, "2")
    assert kv.cas("a", "1", "2") and kv.get("a") == "2"
    kv.put("p/x", "x")
    kv.put("p/y", "y")
    assert kv.get_prefix("p/") == {"p/x": "x", "p/y": "y"}
    kv.delete("a")
    assert kv.get("a") is None


def test_lease_expiry_and_keepalive():
    clock = FakeClock()
    kv = InMemoryKV(clock=clock)
    lease = kv.grant_lease(2.0)
    assert kv.cas("leader", None, "c0", lease=lease)
    clock.t = 1.5
    assert kv.keepalive(lease)
    clock.t = 3.0
    assert kv.get("leader") == "c0"
    clock.t = 3.6
    assert kv.get("leader") is None
    assert not kv.keepalive(lease)
    with pytest.raises(KeyError):
        kv.put("x", "y", lease=lease)
    other = kv.grant_lease(1.0)
    assert kv.cas("leader", None, "c1", lease=other)
    kv.revoke(other)
    assert kv.get("leader") is None


def test_writes_fenced_on_leadership():
    clock = FakeClock()
    kv = InMemoryKV(clock=clock)
    lease = kv.grant_lease(1.0)
    assert kv.cas("leader", None, "c0", lease=lease)
    assert kv.put_if("state", "from-c0", "leader", "c0")
    clock.t = 2.0
    assert kv.cas("leader", None, "c1", lease=kv.grant_lease(5.0))
    assert not kv.put_if("state", "stale", "leader", "c0")
    assert kv.put_if("state", "from-c1", "leader", "c1") and kv.get("state") == "from-c1"
