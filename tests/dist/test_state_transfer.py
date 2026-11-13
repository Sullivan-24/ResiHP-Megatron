"""Slice-level migration of parameters + Adam state across a combined TP/PP/DP transition, and warm handoff."""

import pytest
import torch
import torch.distributed as dist

from tests.dist_utils import spawn

pytestmark = pytest.mark.dist

KINDS = ("param", "exp_avg", "exp_avg_sq")


@pytest.fixture
def store():
    srv = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    yield ("127.0.0.1", srv.port)
    del srv


def _full(cfg, specs):
    from resihp.runtime.reference_model import full_layer_params

    out = {}
    for j, kind in enumerate(KINDS):
        for key in specs:
            layer, name = int(key.split(".")[1]), key.split(".", 2)[2]
            out.setdefault(kind, {})[key] = full_layer_params(cfg, layer)[name] * (j + 1) + j
    return out


def _slice(t, spec, iv):
    return t if spec.dim is None else t.narrow(spec.dim, iv[0], iv[1] - iv[0])


def _transition(rank, world, port, store_addr):
    from resihp.comm import ResilientCommManager
    from resihp.plan.hetero_plan import HeteroParallelPlan, StageGroup
    from resihp.runtime.reference_model import RefModelConfig
    from resihp.state.layer_registry import device_layout, holdings, reference_layer_of, reference_specs
    from resihp.state.transfer_engine import execute_transfers
    from resihp.state.transfer_plan import plan_transfers

    cfg = RefModelConfig()
    specs = reference_specs(cfg)
    ids = [f"w{i}" for i in range(world)]
    me = ids[rank]
    old = HeteroParallelPlan(1, ("w0", "w1", "w2", "w3", "w4", "w5"), 4, 2, 2,
                             (StageGroup(0, 0, ("w0", "w1"), (0, 2)), StageGroup(1, 0, ("w2",), (0, 2)),
                              StageGroup(0, 1, ("w3",), (2, 4)), StageGroup(1, 1, ("w4", "w5"), (2, 4))),
                             4, ((0, 0, 1, 1), (0, 0, 1, 1)))
    new_world = ("w0", "w1", "w6", "w2", "w3", "w4")
    new = HeteroParallelPlan(2, new_world, 4, 2, 2,
                             (StageGroup(0, 0, ("w0", "w1", "w6"), (0, 1)), StageGroup(1, 0, ("w2",), (0, 1)),
                              StageGroup(0, 1, ("w3",), (1, 4)), StageGroup(1, 1, ("w4",), (1, 4))),
                             4, ((0, 0, 1, 1), (0, 0, 1, 1)))
    if me == "w5":
        return {"dead": True}
    old.validate()
    new.validate()
    mgr = ResilientCommManager("gloo", timeout_s=30)
    comms = mgr.build(2, list(new_world), me, [], store_addr=store_addr, store_prefix="xfer/gen2/")
    full = _full(cfg, specs)
    old_h = holdings(old, specs, reference_layer_of)
    new_h = holdings(new, specs, reference_layer_of)
    old_layout = device_layout(old_h, me)
    old_state = {k: {key: _slice(full[k][key], specs[key], iv).clone() for key, iv in old_layout.items()} for k in KINDS}
    nodes = {"w0": "n0", "w1": "n0", "w6": "n0", "w2": "n1", "w3": "n2", "w4": "n3", "w5": "n3"}
    tp = plan_transfers(old_h, new_h, valid_sources=set(old.world) - {"w5"}, node_of=nodes, chunk=3)
    assert not tp.lost
    layout = device_layout(new_h, me)
    got = execute_transfers(tp, me, comms, mgr, specs=specs, old_layout=old_layout, old_state=old_state,
                            new_layout=layout, kinds=KINDS, dtype=cfg.dtype)
    err = 0.0
    for k in KINDS:
        for key, iv in layout.items():
            err = max(err, float((got[k][key] - _slice(full[k][key], specs[key], iv)).abs().max()))
    mgr.destroy()
    return {"err": err, "keys": len(layout), "received": sum(1 for t in tp.transfers if t.dst == me)}


def test_combined_tp_pp_dp_transition_is_bitwise_exact(store):
    out = spawn(_transition, 7, store, timeout=120)
    live = {r: v for r, v in out.items() if not v.get("dead")}
    assert len(live) == 6
    assert all(v["err"] == 0.0 for v in live.values()), live
    assert live[6]["received"] > 0 and live[6]["keys"] == 6, "the new device receives its whole shard"


def _warm(rank, world, port, store_addr):
    from resihp.comm import ResilientCommManager
    from resihp.runtime.optim import Adam
    from resihp.state.layer_registry import TensorSpec
    from resihp.state.rng_data import generator
    from resihp.state.transfer_plan import plan_transfers
    from resihp.state.warm_handoff import WarmHandoff

    ids = ["s", "d"]
    me = ids[rank]
    mgr = ResilientCommManager("gloo", timeout_s=30)
    comms = mgr.build(1, ids, me, [], store_addr=store_addr, store_prefix="warm/gen1/")
    specs = {"w": TensorSpec("w", (6, 4), 0)}
    tp = plan_transfers({"w": [("s", (0, 6))]}, {"w": [("d", (0, 6))]}, valid_sources={"s"}, node_of={})
    wh = WarmHandoff(tp, me, comms, mgr, specs, old_layout={"w": (0, 6)} if me == "s" else {},
                     incoming_layout={"w": (0, 6)} if me == "d" else {})
    p = torch.nn.Parameter(torch.randn(6, 4, generator=generator(1), dtype=torch.float64))
    opt = Adam({"w": p}, lr=0.05, weight_decay=0.01)
    if me == "s":
        for i in range(3):
            p.grad = torch.randn(6, 4, generator=generator(2, i), dtype=torch.float64)
            opt.step()
        state = {"param": {"w": p.data}, "exp_avg": {"w": opt.state["w"]["exp_avg"]},
                 "exp_avg_sq": {"w": opt.state["w"]["exp_avg_sq"]}}
    else:
        state = {}
    wh.prestage(state, torch.float64)
    g = torch.randn(6, 4, generator=generator(3), dtype=torch.float64)
    if me == "s":
        p.grad = g.clone()
        opt.step()
        wh.forward_grads({"w": g}, torch.float64)
        result = {k: v["w"].clone() for k, v in {"param": {"w": p.data}, "exp_avg": {"w": opt.state["w"]["exp_avg"]},
                                                   "exp_avg_sq": {"w": opt.state["w"]["exp_avg_sq"]}}.items()}
    else:
        got = wh.forward_grads({}, torch.float64)
        wh.lockstep_update(got, step_count=4, lr=0.05, weight_decay=0.01)
        result = {k: v["w"].clone() for k, v in wh.staged.items()}
    mgr.destroy()
    return result


def test_warm_handoff_keeps_destination_bitwise_in_lockstep(store):
    out = spawn(_warm, 2, store, timeout=60)
    for k in KINDS:
        assert torch.equal(out[0][k], out[1][k]), k
