from types import SimpleNamespace

from resihp.control.health import FATAL_XIDS, IBPortProbe, NVMLProbe


class FakeNVML:
    NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1
    NVML_VOLATILE_ECC = 0
    nvmlEventTypeXidCriticalError = 8

    def __init__(self):
        self.events = []
        self.ecc = {0: 0, 1: 0}

    def nvmlInit(self):
        pass

    def nvmlDeviceGetCount(self):
        return 2

    def nvmlDeviceGetHandleByIndex(self, i):
        return f"h{i}"

    def nvmlEventSetCreate(self):
        return "set"

    def nvmlDeviceRegisterEvents(self, h, kind, s):
        pass

    def nvmlEventSetWait(self, s, timeout):
        if not self.events:
            raise RuntimeError("NVML_ERROR_TIMEOUT")
        return self.events.pop(0)

    def nvmlDeviceGetTotalEccErrors(self, h, kind, counter):
        return self.ecc[int(h[1:])]


def test_nvml_probe_classifies_xid_and_ecc():
    nv = FakeNVML()
    probe = NVMLProbe(nvml=nv)
    assert probe.poll() == []
    nv.events = [SimpleNamespace(device="h1", eventData=79), SimpleNamespace(device="h0", eventData=13)]
    nv.ecc[0] = 3
    evs = probe.poll()
    assert [(e.device, e.fatal) for e in evs if e.kind == "xid"] == [("cuda:1", True), ("cuda:0", False)]
    assert 79 in FATAL_XIDS and 13 not in FATAL_XIDS
    assert any(e.kind == "ecc" and e.device == "cuda:0" and e.fatal for e in evs)
    assert evs[0].matches("cuda:1") and not evs[0].matches("cuda:0")


def test_nvml_probe_absent_is_noop():
    class Broken:
        def nvmlInit(self):
            raise OSError("no driver")

    assert NVMLProbe(nvml=Broken()).poll() == []


def test_ib_port_probe(tmp_path):
    for hca, state in (("mlx5_0", "4: ACTIVE"), ("mlx5_1", "1: DOWN")):
        d = tmp_path / hca / "ports" / "1"
        d.mkdir(parents=True)
        (d / "state").write_text(state)
    evs = IBPortProbe(str(tmp_path)).poll()
    assert len(evs) == 1 and "mlx5_1" in evs[0].detail and not evs[0].fatal
    (tmp_path / "mlx5_0" / "ports" / "1" / "state").write_text("1: DOWN")
    assert all(e.fatal for e in IBPortProbe(str(tmp_path)).poll())
    assert IBPortProbe(str(tmp_path / "missing")).poll() == []
