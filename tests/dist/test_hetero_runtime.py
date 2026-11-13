"""Heterogeneous-TP boundary transfers, shard-aligned gradient sync, and a full hetero pipeline
with stage-level migration, all checked against exact references on gloo."""

import pytest
import torch
import torch.distributed as dist

from tests.dist_utils import spawn

pytestmark = pytest.mark.dist


@pytest.fixture
def store():
    srv = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    yield ("127.0.0.1", srv.port)
    del srv


COMBOS = [(1, 1), (2, 1), (1, 3), (4, 2), (2, 3), (3, 4), (4, 3)]


def _p2p(rank, world, port, store_addr):
    from resihp.comm import GroupSpec, ResilientCommManager
    from resihp.comm.hetero_p2p import P2PEngine, boundary_routes, channel_name, local_rows

    ids = [f"w{i}" for i in range(world)]
    specs = []
    for ka, kb in COMBOS:
        if kb > 1:
            specs.append(GroupSpec(f"tp/{ka}-{kb}", tuple(range(ka, ka + kb)), "tp"))
    pairs = sorted({(r.src, r.dst) for r in boundary_routes((0, 1, 2, 3), (4, 5, 6), 24)})
    specs += [GroupSpec(channel_name(a, b), tuple(sorted((a, b))), "p2p") for a, b in pairs]
    mgr = ResilientCommManager("gloo", timeout_s=30)
    comms = mgr.build(1, ids, ids[rank], specs, store_addr=store_addr, store_prefix="p2p/gen1/")
    full = torch.arange(24 * 5, dtype=torch.float64).reshape(24, 5)
    ok = []
    for ka, kb in COMBOS:
        src, dst = tuple(range(ka)), tuple(range(ka, ka + kb))
        tp = comms.group(f"tp/{ka}-{kb}") if kb > 1 else None
        for sp in (False, True):
            eng = P2PEngine(rank, mgr)
            tag = ka * 10 + kb + (100 if sp else 0)
            works = []
            if rank in src:
                local = full[slice(*local_rows(24, src, rank))] if sp else full
                works = eng.send(local, src, dst, 24, sp, tag)
            if rank in dst:
                got = eng.recv(24, (5,), torch.float64, src, dst, sp, tag, tp_group=tp)
                want = full[slice(*local_rows(24, dst, rank))] if sp else full
                ok.append(bool(torch.equal(got, want)))
            eng.wait_sends(works)
    chans = {(a, b): comms.group(channel_name(a, b)) for a, b in pairs if rank in (a, b)}
    eng = P2PEngine(rank, mgr, channels=chans)
    works = eng.send(full, (0, 1, 2, 3), (4, 5, 6), 24, False, 7) if rank < 4 else []
    if 4 <= rank < 7:
        got = eng.recv(24, (5,), torch.float64, (0, 1, 2, 3), (4, 5, 6), False, 7, tp_group=comms.group("tp/4-3"))
        ok.append(bool(torch.equal(got, full)))
    eng.wait_sends(works)
    mgr.destroy()
    return ok


def test_boundary_routing_all_tp_pairs(store):
    out = spawn(_p2p, 7, store, timeout=120)
    assert all(all(v) for v in out.values())
    assert sum(len(v) for v in out.values()) > 20


def _grad_sync(rank, world, port, store_addr, sp):
    from resihp.comm import ResilientCommManager
    from resihp.plan.hetero_plan import HeteroParallelPlan, StageGroup
    from resihp.runtime.hetero_grad_sync import HeteroGradSync
    from resihp.runtime.hetero_runtime import reference_partition_sizes, reference_shards
    from resihp.runtime.reference_model import PARAM_DIMS, RefModelConfig, RefStage, TPContext, full_layer_params, shard
    from resihp.state.rng_data import generator

    cfg = RefModelConfig()
    ids = [f"w{i}" for i in range(world)]
    groups = (StageGroup(0, 0, tuple(ids[0:4]), (0, 1)), StageGroup(1, 0, tuple(ids[4:7]), (0, 1)))
    plan = HeteroParallelPlan(1, tuple(ids), 1, 1, 2, groups, 2, ((0, 1),), sequence_parallel=sp)
    mgr = ResilientCommManager("gloo", timeout_s=30)
    comms = mgr.build(1, ids, ids[rank], plan.group_specs(reference_partition_sizes(cfg)), store_addr=store_addr,
                      store_prefix=f"gs{int(sp)}/gen1/")
    r = 0 if rank < 4 else 1
    g = groups[r]
    tp_rank = g.devices.index(ids[rank])
    tp = TPContext(group=comms.group(f"tp/r{r}/s0"), rank=tp_rank, size=g.tp, sequence_parallel=sp,
                   waiter=lambda w: mgr.guarded_wait(w))
    module = RefStage(cfg, 0, 1, tp)
    import zlib

    params = module.named_state()
    full_grads = {}
    for key, p in params.items():
        name = key.split(".", 2)[2]
        crc = zlib.crc32(name.encode())
        total = torch.zeros_like(full_layer_params(cfg, 0)[name])
        for rep in (0, 1):
            total += torch.randn(total.shape, generator=generator(rep, crc), dtype=cfg.dtype)
        mine = torch.randn(total.shape, generator=generator(r, crc), dtype=cfg.dtype)
        if module.sp_partial(key):
            others = [torch.randn(total.shape, generator=generator(r, t, 5, crc), dtype=cfg.dtype) for t in range(1, g.tp)]
            p.grad = (mine - sum(others)) if tp_rank == 0 else others[tp_rank - 1].clone()
        else:
            p.grad = shard(mine, PARAM_DIMS[name], tp_rank, g.tp)
        full_grads[key] = shard(total, PARAM_DIMS[name], tp_rank, g.tp)
    HeteroGradSync(plan, comms, mgr, 0, r, params, reference_shards(cfg, module), tp_group=tp.group).sync()
    errs = {k: float((p.grad - full_grads[k]).abs().max()) for k, p in params.items()}
    mgr.destroy()
    return errs


@pytest.mark.parametrize("sp", [False, True])
def test_shard_aligned_grad_sync_tp4_vs_tp3(store, sp):
    out = spawn(_grad_sync, 7, store, sp, timeout=120)
    for rank, errs in out.items():
        assert max(errs.values()) < 1e-12, (rank, errs)


def _pipeline(rank, world, port, store_addr, sp, iterations, zb=False):
    from resihp.comm import ResilientCommManager
    from resihp.plan.hetero_plan import HeteroParallelPlan, StageGroup
    from resihp.plan.ir import compute_order_to_ir, validate
    from resihp.plan.simulator import list_schedule
    from resihp.runtime.hetero_runtime import (HeteroRuntime, build_stage_module, channel_specs,
                                               reference_partition_sizes)
    from resihp.runtime.optim import Adam
    from resihp.runtime.reference_model import RefModelConfig
    from resihp.state.rng_data import IndexAddressableSampler

    cfg = RefModelConfig()
    sampler = IndexAddressableSampler(256, 16, 2, seed=7)
    ids = [f"w{i}" for i in range(world)]
    groups = (StageGroup(0, 0, ("w0", "w1"), (0, 3)), StageGroup(1, 0, ("w2",), (0, 3)),
              StageGroup(0, 1, ("w3",), (3, 4)), StageGroup(1, 1, ("w4", "w5", "w6"), (3, 4)))
    assign = ((0, 0, 0, 0, 0, 1, 1, 1), (0, 0, 1, 1, 1, 1, 1, 1))
    plan = HeteroParallelPlan(1, tuple(ids), 4, 2, 2, groups, 8, assign, sequence_parallel=sp)
    plan.validate()
    res = list_schedule(plan, lambda k, op, mb: 1.0 if op == "F" else 2.0, lambda a, b, kind: 0.1, split_backward=zb)
    schedule = compute_order_to_ir(plan, res.order)
    validate(plan, schedule, split_backward=zb)
    specs = plan.group_specs(reference_partition_sizes(cfg)) + channel_specs(plan, schedule, 48)
    mgr = ResilientCommManager("gloo", timeout_s=60)
    comms = mgr.build(1, ids, ids[rank], specs, store_addr=store_addr, store_prefix=f"pipe{int(sp)}{int(zb)}/gen1/")
    module = build_stage_module(cfg, plan, comms, mgr)
    opt = Adam(module.named_state(), lr=1e-2)
    rt = HeteroRuntime(cfg, sampler, plan, schedule, comms, mgr, module, use_channels=True)
    losses = []
    for it in range(iterations):
        losses.append(rt.run_iteration(it))
        opt.step()
    g = plan.group_of_device(ids[rank])
    shards = {k: v.detach().clone() for k, v in module.named_state().items()}
    mgr.destroy()
    return {"losses": losses, "replica": g.replica, "stage": g.stage, "tp_rank": g.devices.index(ids[rank]),
            "tp": g.tp, "shards": shards}


def _vpp_pipeline(rank, world, port, store_addr, sp, zb, iterations):
    from resihp.comm import ResilientCommManager
    from resihp.plan.hetero_plan import HeteroParallelPlan, StageGroup, interleave
    from resihp.plan.ir import compute_order_to_ir, validate
    from resihp.plan.simulator import schedule_with_fallback
    from resihp.runtime.hetero_runtime import (HeteroRuntime, build_stage_module, channel_specs,
                                               reference_partition_sizes)
    from resihp.runtime.optim import Adam
    from resihp.runtime.reference_model import RefModelConfig
    from resihp.state.rng_data import IndexAddressableSampler

    cfg = RefModelConfig(num_layers=8)
    sampler = IndexAddressableSampler(256, 16, 2, seed=7)
    ids = [f"w{i}" for i in range(world)]
    groups = (StageGroup(0, 0, ("w0", "w1"), (0, 5)), StageGroup(1, 0, ("w2",), (0, 5)),
              StageGroup(0, 1, ("w3",), (5, 8)), StageGroup(1, 1, ("w4", "w5", "w6"), (5, 8)))
    assign = ((0, 0, 0, 0, 0, 1, 1, 1), (0, 0, 1, 1, 1, 1, 1, 1))
    plan = interleave(HeteroParallelPlan(1, tuple(ids), 8, 2, 2, groups, 8, assign, sequence_parallel=sp), 2)
    plan.validate()
    res = schedule_with_fallback(plan, lambda k, op, mb: 1.0 if op == "F" else 2.0, lambda a, b, kind: 0.1,
                                 split_backward=zb)
    schedule = compute_order_to_ir(plan, res.order)
    validate(plan, schedule, split_backward=zb)
    specs = plan.group_specs(reference_partition_sizes(cfg)) + channel_specs(plan, schedule, 48)
    mgr = ResilientCommManager("gloo", timeout_s=60)
    comms = mgr.build(1, ids, ids[rank], specs, store_addr=store_addr, store_prefix=f"vpp{int(sp)}{int(zb)}/gen1/")
    module = build_stage_module(cfg, plan, comms, mgr)
    opt = Adam(module.named_state(), lr=1e-2)
    rt = HeteroRuntime(cfg, sampler, plan, schedule, comms, mgr, module, use_channels=True)
    losses = []
    for it in range(iterations):
        losses.append(rt.run_iteration(it))
        opt.step()
    g = plan.group_of_device(ids[rank])
    shards = {k: v.detach().clone() for k, v in module.named_state().items()}
    mgr.destroy()
    return {"losses": losses, "replica": g.replica, "stage": g.stage, "tp_rank": g.devices.index(ids[rank]),
            "chunks": len(plan.groups_of_device(ids[rank])), "shards": shards}


def _cp_pipeline(rank, world, port, store_addr, sp, iterations):
    from resihp.comm import ResilientCommManager
    from resihp.plan.hetero_plan import HeteroParallelPlan, StageGroup
    from resihp.plan.ir import compute_order_to_ir, validate
    from resihp.plan.simulator import schedule_with_fallback
    from resihp.runtime.hetero_runtime import (HeteroRuntime, build_stage_module, channel_specs,
                                               reference_partition_sizes)
    from resihp.runtime.optim import Adam
    from resihp.runtime.reference_model import RefModelConfig
    from resihp.state.rng_data import IndexAddressableSampler

    cfg = RefModelConfig()
    sampler = IndexAddressableSampler(256, 16, 2, seed=7)
    ids = tuple(f"w{i}" for i in range(world))
    # cp=2 everywhere; stage 0 has TP2 and TP1 replicas, replica 1 lost stage 1 (its work runs on replica 0).
    groups = (StageGroup(0, 0, ids[0:4], (0, 3), 2), StageGroup(1, 0, ids[4:6], (0, 3), 2),
              StageGroup(0, 1, ids[6:8], (3, 4), 2))
    plan = HeteroParallelPlan(1, ids, 4, 2, 2, groups, 8, ((0, 0, 0, 0, 0, 1, 1, 1), (0,) * 8), sequence_parallel=sp)
    plan.validate()
    res = schedule_with_fallback(plan, lambda k, op, mb: 1.0 if op == "F" else 2.0, lambda a, b, kind: 0.1)
    schedule = compute_order_to_ir(plan, res.order)
    validate(plan, schedule)
    specs = plan.group_specs(reference_partition_sizes(cfg)) + channel_specs(plan, schedule, 48)
    mgr = ResilientCommManager("gloo", timeout_s=60)
    comms = mgr.build(1, list(ids), ids[rank], specs, store_addr=store_addr, store_prefix=f"cp{int(sp)}/gen1/")
    module = build_stage_module(cfg, plan, comms, mgr)
    opt = Adam(module.named_state(), lr=1e-2)
    rt = HeteroRuntime(cfg, sampler, plan, schedule, comms, mgr, module, use_channels=True)
    losses = []
    for it in range(iterations):
        losses.append(rt.run_iteration(it))
        opt.step()
    g = plan.group_of_device(ids[rank])
    me = ids[rank]
    shards = {k: v.detach().clone() for k, v in module.named_state().items()}
    mgr.destroy()
    return {"losses": losses, "replica": g.replica, "stage": g.stage, "slice": g.slice_of(me),
            "tp_rank": g.slice(g.slice_of(me)).index(me), "shards": shards}


@pytest.mark.parametrize("sp", [False, True])
def test_context_parallel_hetero_pipeline_matches_reference(store, sp):
    from resihp.runtime.reference_model import PARAM_DIMS, RefModelConfig, reference_train
    from resihp.state.rng_data import IndexAddressableSampler

    iterations = 3
    out = spawn(_cp_pipeline, 8, store, sp, iterations, timeout=240)
    ref_losses, ref_params = reference_train(RefModelConfig(), IndexAddressableSampler(256, 16, 2, seed=7), iterations)
    for res in out.values():
        assert res["losses"] == pytest.approx(ref_losses, rel=1e-10)
    by_slice = {}
    for res in out.values():
        by_slice.setdefault((res["replica"], res["stage"], res["slice"]), []).append(res)
    covered = set()
    for members in by_slice.values():
        members.sort(key=lambda r: r["tp_rank"])
        for key in members[0]["shards"]:
            dim = PARAM_DIMS[key.split(".", 2)[2]]
            full = members[0]["shards"][key] if dim is None else torch.cat([m["shards"][key] for m in members], dim=dim)
            torch.testing.assert_close(full, ref_params[key], rtol=1e-10, atol=1e-12)
            covered.add(key)
    assert covered == set(ref_params)


@pytest.mark.parametrize("sp,zb", [(False, False), (True, True)])
def test_interleaved_hetero_pipeline_matches_reference(store, sp, zb):
    from resihp.runtime.reference_model import PARAM_DIMS, RefModelConfig, reference_train
    from resihp.state.rng_data import IndexAddressableSampler

    iterations = 3
    cfg = RefModelConfig(num_layers=8)
    out = spawn(_vpp_pipeline, 7, store, sp, zb, iterations, timeout=240)
    ref_losses, ref_params = reference_train(cfg, IndexAddressableSampler(256, 16, 2, seed=7), iterations)
    assert {r["chunks"] for r in out.values()} == {2}
    for res in out.values():
        assert res["losses"] == pytest.approx(ref_losses, rel=1e-10)
    by_group = {}
    for res in out.values():
        by_group.setdefault((res["replica"], res["stage"]), []).append(res)
    covered = set()
    for members in by_group.values():
        members.sort(key=lambda r: r["tp_rank"])
        for key in members[0]["shards"]:
            dim = PARAM_DIMS[key.split(".", 2)[2]]
            full = members[0]["shards"][key] if dim is None else torch.cat([m["shards"][key] for m in members], dim=dim)
            torch.testing.assert_close(full, ref_params[key], rtol=1e-10, atol=1e-12)
            covered.add(key)
    assert covered == set(ref_params)


@pytest.mark.parametrize("sp,zb", [(False, False), (True, False), (False, True), (True, True)])
def test_hetero_pipeline_with_migration_matches_reference(store, sp, zb):
    from resihp.runtime.reference_model import PARAM_DIMS, RefModelConfig, reference_train
    from resihp.state.rng_data import IndexAddressableSampler

    iterations = 3
    out = spawn(_pipeline, 7, store, sp, iterations, zb, timeout=180)
    ref_losses, ref_params = reference_train(RefModelConfig(), IndexAddressableSampler(256, 16, 2, seed=7), iterations)
    for res in out.values():
        assert res["losses"] == pytest.approx(ref_losses, rel=1e-10)
    by_group = {}
    for res in out.values():
        by_group.setdefault((res["replica"], res["stage"]), []).append(res)
    for (rep, stage), members in by_group.items():
        members.sort(key=lambda r: r["tp_rank"])
        for key in members[0]["shards"]:
            dim = PARAM_DIMS[key.split(".", 2)[2]]
            if dim is None:
                for m in members:
                    torch.testing.assert_close(m["shards"][key], ref_params[key], rtol=1e-10, atol=1e-12)
            else:
                full = torch.cat([m["shards"][key] for m in members], dim=dim)
                torch.testing.assert_close(full, ref_params[key], rtol=1e-10, atol=1e-12)
