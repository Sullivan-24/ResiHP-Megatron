from resihp.plan.hetero_plan import HeteroParallelPlan, StageGroup
from resihp.runtime.reference_model import RefModelConfig
from resihp.state.layer_registry import FULL, TensorSpec, device_layout, holdings, reference_layer_of, reference_specs
from resihp.state.transfer_plan import plan_transfers


def plan_of(groups, layers=4, M=4):
    world = tuple(d for g in groups for d in g.devices)
    S = max(g.stage for g in groups) + 1
    R = max(g.replica for g in groups) + 1
    rows = tuple(tuple(min(m * R // M, R - 1) for m in range(M)) for _ in range(S))
    return HeteroParallelPlan(1, world, layers, S, R, tuple(groups), M, rows)


def coverage_ok(plan_t, new_h):
    for key, hs in new_h.items():
        for dst, iv in hs:
            pieces = sorted([o for d, k, o in plan_t.reuse if d == dst and k == key] +
                            [t.interval for t in plan_t.transfers if t.dst == dst and t.key == key])
            pos = iv[0]
            for s, e in pieces:
                assert s == pos, (key, dst, pieces, iv)
                pos = e
            assert pos == iv[1], (key, dst, pieces, iv)


def test_tp_shrink_and_layer_move_reuse_and_cover():
    cfg = RefModelConfig()
    specs = reference_specs(cfg)
    old = plan_of([StageGroup(0, 0, ("a", "b"), (0, 2)), StageGroup(0, 1, ("c", "d"), (2, 4)),
                   StageGroup(1, 0, ("e", "f"), (0, 2)), StageGroup(1, 1, ("g", "h"), (2, 4))])
    new = plan_of([StageGroup(0, 0, ("a", "b"), (0, 3)), StageGroup(0, 1, ("c",), (3, 4)),
                   StageGroup(1, 0, ("e", "f"), (0, 3)), StageGroup(1, 1, ("g", "h"), (3, 4))])
    old_h = holdings(old, specs, reference_layer_of)
    new_h = holdings(new, specs, reference_layer_of)
    valid = set(old.world) - {"d"}
    nodes = {"a": "n0", "b": "n0", "c": "n1", "d": "n1", "e": "n2", "f": "n2", "g": "n3", "h": "n3"}
    tp = plan_transfers(old_h, new_h, valid_sources=valid, node_of=nodes)
    assert not tp.lost
    coverage_ok(tp, new_h)
    assert all(t.src != "d" for t in tp.transfers), "a dead/stale device is never a source"
    assert ("a", "layers.0.fc1.weight", (0, 12)) in tp.reuse
    l2 = [t for t in tp.transfers if t.key == "layers.2.fc1.weight" and t.dst == "a"]
    assert l2 and all(t.src in {"c", "e", "g", "h"} for t in l2)
    c3 = [t for t in tp.transfers if t.key == "layers.3.fc1.weight" and t.dst == "c"]
    assert [t.interval for t in c3] == [(12, 24)] and c3[0].src in {"g", "h"}


def test_locality_striping_and_lost():
    spec = {"w": TensorSpec("w", (16, 4), 0)}
    old = {"w": [("r0", (0, 16)), ("r1", (0, 16)), ("r2", (0, 16))]}
    new = {"w": [("x", (0, 16))]}
    nodes = {"r0": "n0", "r1": "n1", "r2": "n1", "x": "n1"}
    tp = plan_transfers(old, new, valid_sources={"r0", "r1", "r2"}, node_of=nodes, chunk=4)
    assert [t.interval for t in tp.transfers] == [(0, 4), (4, 8), (8, 12), (12, 16)]
    assert {t.src for t in tp.transfers} == {"r1", "r2"}, "same-node replicas share the load"
    tp2 = plan_transfers(old, new, valid_sources={"r0"}, node_of=nodes)
    assert [t.src for t in tp2.transfers] == ["r0"]
    tp3 = plan_transfers(old, {"w": [("x", (0, 16))], "v": [("x", FULL)]}, valid_sources=set(), node_of=nodes)
    assert ("w", (0, 16)) in tp3.lost and ("v", FULL) in tp3.lost
    assert tp.bytes_moved(spec, 2, 3) == 16 * 4 * 2 * 3


def test_device_layout():
    cfg = RefModelConfig()
    p = plan_of([StageGroup(0, 0, ("a", "b", "c"), (0, 4))])
    lay = device_layout(holdings(p, reference_specs(cfg), reference_layer_of), "b")
    assert lay["layers.1.fc1.weight"] == (8, 16) and lay["layers.1.ln.weight"] == FULL
