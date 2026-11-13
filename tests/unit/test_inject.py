import time

import pytest

from resihp.inject.faults import (
    ExternalInjector,
    FaultInjector,
    FaultKind,
    parse_faults,
    run_schedule,
)


def test_parse_roundtrip():
    spec = "kill@iter=5:target=n0g1;slow@iter=10:target=n1g0:factor=2.0:duration=30:phase=backward"
    faults = parse_faults(spec)
    assert [f.kind for f in faults] == [FaultKind.KILL, FaultKind.SLOW_COMPUTE]
    assert faults[1].factor == 2.0 and faults[1].duration == 30.0 and faults[1].phase == "backward"
    assert parse_faults(";".join(f.to_spec() for f in faults)) == faults
    with pytest.raises(ValueError):
        parse_faults("kill@iter=3")
    with pytest.raises(ValueError):
        parse_faults("nonsense")


def test_in_process_kill_fires_once_at_iteration():
    killed = []
    inj = FaultInjector("w0", parse_faults("kill@iter=3:target=w0;kill@iter=1:target=w1"),
                        kill_fn=lambda: killed.append(True))
    for it in range(3):
        inj.on_hook(it, "forward")
    assert not killed
    inj.on_hook(3, "backward")
    assert not killed
    inj.on_hook(3, "forward")
    inj.on_hook(4, "forward")
    assert killed == [True]


def test_slow_compute_padding_and_expiry():
    inj = FaultInjector("w0", parse_faults("slow@iter=0:target=w0:factor=3.0:duration=0.2"))
    inj.on_hook(0, "forward")
    assert inj.compute_factor() == 3.0
    t0 = time.perf_counter()
    pad = inj.pad_compute(0.01)
    assert pad == pytest.approx(0.02)
    assert time.perf_counter() - t0 >= 0.02
    time.sleep(0.25)
    assert inj.compute_factor() == 1.0


def test_external_commands_and_schedule_dry_run():
    faults = parse_faults(
        "slow@time=0:target=n0:gpu=3:factor=2.0:duration=5;xid@time=1:target=n0:gpu=1:xid=79;"
        "nic_down@time=2:target=n1:iface=ib2:duration=1"
    )
    inj = ExternalInjector(dry_run=True)
    clock = iter([0.0, 0.0, 1.0, 2.0, 3.0, 5.0, 6.0]).__next__
    log = run_schedule(faults, inj, clock=clock, sleep=lambda s: None)
    actions = [(a, f.kind.value) for _, f, a in log]
    assert actions == [("apply", "slow"), ("apply", "xid"), ("apply", "nic_down"), ("revert", "nic_down"),
                       ("revert", "slow")]
    assert inj.executed[0] == ["nvidia-smi", "-i", "3", "-lgc", "990,990"]
    assert inj.executed[1][:4] == ["dcgmi", "test", "--inject", "--gpuid"]
    assert inj.executed[-1] == ["nvidia-smi", "-i", "3", "-rgc"]
