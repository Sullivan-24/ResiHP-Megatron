"""Resilient expert parallelism: an EP rank dies, experts are re-placed (with replicas of hot
experts) over the survivors, expert parameters and Adam state migrate, and training is
indistinguishable from the single-process reference."""

import pytest
import torch
import torch.distributed as dist

from tests.dist_utils import spawn

pytestmark = pytest.mark.dist

KINDS = ("param", "exp_avg", "exp_avg_sq")
BEFORE, AFTER = 4, 4


@pytest.fixture
def store():
    srv = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    yield ("127.0.0.1", srv.port)
    del srv


def _train(cfg, rt, opt, it):
    from resihp.runtime.moe import batch, token_loss

    opt.zero_grad()
    x, y = batch(cfg, it)
    s, e = rt.token_range()
    loss = token_loss(rt.forward(x[s:e]), y[s:e])
    (loss / cfg.tokens).backward()
    rt.sync_grads()
    rt.finish_iteration()
    opt.step()
    return loss.item()


def _ep(rank, world, port, store_addr, bias_rate):
    from resihp.comm import ResilientCommManager
    from resihp.plan.expert_placement import moved_experts, plan_expert_placement
    from resihp.runtime.moe import (EPRuntime, MoEConfig, init_param, local_keys, moe_group_specs, moe_holdings,
                                    moe_specs)
    from resihp.runtime.optim import Adam
    from resihp.state.layer_registry import FULL
    from resihp.state.transfer_engine import execute_transfers
    from resihp.state.transfer_plan import plan_transfers

    cfg = MoEConfig(bias_update_rate=bias_rate)
    specs = moe_specs(cfg)
    ids = ["n0g0", "n0g1", "n1g0", "n1g1"]
    me = ids[rank]
    mgr = ResilientCommManager("gloo", timeout_s=30)
    p1 = plan_expert_placement([ids[:2], ids[2:]], cfg.num_experts)
    comms = mgr.build(1, ids, me, moe_group_specs(p1, ids), store_addr=store_addr,
                      store_prefix=f"moe{bias_rate}/gen1/")
    mine = local_keys(p1, me, specs)
    params = {k: torch.nn.Parameter(init_param(cfg, specs[k])) for k in mine if specs[k].trainable}
    buffers = {k: init_param(cfg, specs[k]) for k in mine if not specs[k].trainable}
    opt = Adam(params, lr=1e-2)
    rt = EPRuntime(cfg, p1, me, ids, comms, mgr, params, buffers=buffers)
    losses = {it: _train(cfg, rt, opt, it) for it in range(BEFORE)}
    loads = rt.global_loads()
    if me == "n1g1":
        return {"dead": True}

    survivors = ids[:3]
    p2 = plan_expert_placement([survivors], cfg.num_experts, loads=loads, prev=p1, slots_per_device=3)
    comms = mgr.build(2, survivors, me, moe_group_specs(p2, survivors), store_addr=store_addr,
                      store_prefix=f"moe{bias_rate}/gen2/")
    old_state = {"param": {**{k: p.data for k, p in params.items()}, **buffers},
                 "exp_avg": {k: opt.state[k]["exp_avg"] for k in params},
                 "exp_avg_sq": {k: opt.state[k]["exp_avg_sq"] for k in params}}
    xfer = plan_transfers(moe_holdings(p1, specs), moe_holdings(p2, specs), valid_sources=set(survivors),
                          node_of={d: d[:2] for d in ids})
    assert not xfer.lost
    keys = local_keys(p2, me, specs)
    got = execute_transfers(xfer, me, comms, mgr, specs=specs, old_layout={k: FULL for k in mine},
                            old_state=old_state, new_layout={k: FULL for k in keys}, kinds=KINDS, dtype=cfg.dtype)
    trainable = [k for k in keys if specs[k].trainable]
    params = {k: torch.nn.Parameter(got["param"][k].clone()) for k in trainable}
    buffers = {k: got["param"][k].clone() for k in keys if not specs[k].trainable}
    step = opt.step_count
    opt = Adam(params, lr=1e-2)
    opt.load_state_tensors({f"{kind}/{k}": got[kind][k] for kind in KINDS for k in trainable}, step)
    rt = EPRuntime(cfg, p2, me, survivors, comms, mgr, params, buffers=buffers)
    losses.update({it: _train(cfg, rt, opt, it) for it in range(BEFORE, BEFORE + AFTER)})
    mgr.destroy()
    return {"losses": losses, "params": {**{k: v.detach() for k, v in params.items()}, **buffers},
            "replicated": [e for e in range(cfg.num_experts) if len(p2.holders(e)) > 1],
            "moved": moved_experts(p1, p2), "received": sum(1 for t in xfer.transfers if t.dst == me),
            "local_experts": list(p2.slots[me])}


@pytest.mark.parametrize("bias_rate", [0.0, 0.01])
def test_ep_rank_failure_replaces_experts_and_preserves_training(store, bias_rate):
    from resihp.runtime.moe import MoEConfig, reference_train

    out = spawn(_ep, 4, store, bias_rate, timeout=180)
    cfg = MoEConfig(bias_update_rate=bias_rate)
    ref_losses, ref_params = reference_train(cfg, BEFORE + AFTER)
    alive = [r for r in out.values() if not r.get("dead")]
    assert len(alive) == 3
    for it in range(BEFORE + AFTER):
        contributors = alive if it >= BEFORE else list(out.values())
        if it < BEFORE:
            continue
        total = sum(r["losses"][it] for r in contributors) / cfg.tokens
        assert total == pytest.approx(ref_losses[it], rel=1e-10)
    for r in alive:
        for k, v in r["params"].items():
            torch.testing.assert_close(v, ref_params[k], rtol=1e-10, atol=1e-12)
    if bias_rate:
        assert ref_params["router.expert_bias"].abs().sum() > 0
        assert all("router.expert_bias" in r["params"] for r in alive)
    assert alive[0]["replicated"], "spare slots must replicate hot experts"
    assert alive[0]["moved"] > 0 and sum(r["received"] for r in alive) > 0
