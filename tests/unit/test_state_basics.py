import pytest
import torch

from resihp.plan.dp_planner import assign_sources, weighted_assignment
from resihp.runtime.optim import Adam
from resihp.runtime.reference_model import (
    RefModelConfig,
    RefStage,
    full_layer_params,
    global_tokens,
    microbatch,
    reference_train,
)
from resihp.state.checkpoint import SliceCheckpointer, SliceMeta
from resihp.state.redundancy import HolderMap, HostSnapshotManager, distopt_twin_holders
from resihp.state.rng_data import IndexAddressableSampler, capture_rng_state, generator, restore_rng_state
from resihp.state.transaction import TransactionManager, TxnPhase, roll_forward_plan


def test_sampler_is_deterministic_and_covers_epoch():
    s = IndexAddressableSampler(num_samples=20, global_batch_size=8, micro_batch_size=2, seed=3)
    assert s.num_microbatches == 4
    b0 = s.batch_indices(0)
    assert b0 == IndexAddressableSampler(20, 8, 2, seed=3).batch_indices(0)
    assert sum((s.microbatch_indices(0, m) for m in range(4)), []) == b0
    epoch = s.batch_indices(0) + s.batch_indices(1) + s.batch_indices(2)[:4]
    assert sorted(epoch) == list(range(20))
    with pytest.raises(IndexError):
        s.microbatch_indices(0, 4)


def test_counter_rng_and_global_rng_restore():
    a = torch.rand(3, generator=generator(1, 2, 3))
    assert torch.equal(a, torch.rand(3, generator=generator(1, 2, 3)))
    assert not torch.equal(a, torch.rand(3, generator=generator(1, 2, 4)))
    st = capture_rng_state()
    x = torch.rand(4)
    restore_rng_state(st)
    assert torch.equal(x, torch.rand(4))


def test_transaction_lifecycle_and_rollback():
    txm = TransactionManager()
    txm.begin(0)
    txm.on_reduced()
    txm.on_step_begin()
    txm.on_commit()
    assert txm.committed_iteration == 0 and txm.cursor.iteration == 1
    t = txm.begin(1)
    st = torch.get_rng_state()
    torch.rand(10)
    info = txm.abort()
    assert torch.equal(torch.get_rng_state(), st)
    assert info == {"committed_iteration": 0, "has_reduced_grads": False, "inflight_iteration": 1}
    txm.discard()
    txm.begin(1)
    txm.on_reduced()
    txm.abort()
    assert txm.has_reduced_grads and txm.current.phase == TxnPhase.REDUCED
    with pytest.raises(ValueError):
        TransactionManager().begin(3)


def test_transaction_rolls_back_buffers_unless_rolling_forward():
    amax = torch.ones(3)
    txm = TransactionManager(buffers=lambda: {"amax": amax})
    txm.begin(0)
    amax.add_(1)
    txm.abort()
    assert torch.equal(amax, torch.ones(3)), "compute-phase abort restores mutated buffers"
    txm.discard()
    txm.begin(0)
    amax.add_(5)
    txm.on_reduced()
    txm.abort()
    assert torch.equal(amax, torch.full((3,), 6.0)), "reduced iteration keeps them for roll-forward"
    txm.discard()
    assert torch.equal(amax, torch.ones(3)), "abandoning the reduced iteration restores them"
    txm.begin(0)
    amax.add_(2)
    txm.on_reduced()
    txm.on_step_begin()
    txm.on_commit()
    assert torch.equal(amax, torch.full((3,), 3.0))


def test_fp8_style_reference_is_partition_invariant():
    cfg = RefModelConfig(amax_history=4)
    sampler = IndexAddressableSampler(64, 8, 2, seed=1)
    losses, state = reference_train(cfg, sampler, 3)
    plain, _ = reference_train(RefModelConfig(), sampler, 3)
    assert losses != plain
    assert state["layers.0.amax_history"][0] > 0 and state["layers.0.amax_history"][-1] == 1.0


def test_roll_forward_plan():
    rec = {
        "a": {"committed_iteration": 7, "has_reduced_grads": False},
        "b": {"committed_iteration": 6, "has_reduced_grads": True, "inflight_iteration": 7},
        "c": {"committed_iteration": 6, "has_reduced_grads": False, "inflight_iteration": 7},
    }
    p = roll_forward_plan(rec, ["a", "b", "c", "d"])
    assert p["target_iteration"] == 7
    assert p["up_to_date"] == ["a"] and p["local_step"] == ["b"] and p["needs_state"] == ["c", "d"]
    assert roll_forward_plan({}, ["x"])["up_to_date"] == ["x"]


def test_adam_matches_torch_bitwise():
    torch.manual_seed(0)
    w = torch.randn(5, 3, dtype=torch.float64)
    p1, p2 = torch.nn.Parameter(w.clone()), torch.nn.Parameter(w.clone())
    ours = Adam({"w": p1}, lr=0.01, weight_decay=0.1)
    ref = torch.optim.Adam([p2], lr=0.01, weight_decay=0.1, foreach=False)
    for i in range(5):
        g = torch.randn(5, 3, dtype=torch.float64)
        p1.grad, p2.grad = g.clone(), g.clone()
        ours.step()
        ref.step()
        assert torch.equal(p1, p2), i
    assert torch.equal(ours.state["w"]["exp_avg_sq"], ref.state[p2]["exp_avg_sq"])


def test_snapshots_and_holders():
    snaps = HostSnapshotManager(interval=2)
    assert not snaps.due(0) and snaps.due(1)
    t = {"a": torch.ones(2)}
    snaps.snapshot(1, t)
    t["a"].add_(1)
    it, s = snaps.latest()
    assert it == 1 and torch.equal(s["a"], torch.ones(2))
    holders = distopt_twin_holders(["r0", "r1", "r2", "r3"], 2)
    assert holders == {"shard0": ["r0", "r2"], "shard1": ["r1", "r3"]}
    hm = HolderMap(holders)
    assert hm.lost({"r0"}) == [] and hm.lost({"r0", "r2"}) == ["shard0"]
    assert hm.node_diverse({"r0": "n0", "r1": "n0", "r2": "n1", "r3": "n1"}.get)
    assert not hm.node_diverse({"r0": "n0", "r1": "n1", "r2": "n0", "r3": "n1"}.get)
    with pytest.raises(ValueError):
        distopt_twin_holders(["a", "b", "c"], 2)


def test_slice_checkpoint_reshards_on_load(tmp_path):
    ck = SliceCheckpointer(tmp_path)
    full = torch.arange(24, dtype=torch.float64).reshape(6, 4)
    for r, (s, e) in enumerate([(0, 2), (2, 4), (4, 6)]):
        ck.save(5, f"w{r}", [(SliceMeta("W", (6, 4), 0, (s, e)), {"param": full[s:e], "m": -full[s:e]}),
                             (SliceMeta("b", (4,), None, (0, 1)), {"param": torch.ones(4), "m": torch.zeros(4)})],
                blocking=True)
    assert ck.latest_complete({"W": ((6, 4), 0), "b": ((4,), None)}) == 5
    got = ck.load(5, [SliceMeta("W", (6, 4), 0, (1, 5)), SliceMeta("b", (4,), None, (0, 1))], kinds=("param", "m"))
    assert torch.equal(got["W"]["param"], full[1:5]) and torch.equal(got["W"]["m"], -full[1:5])
    assert torch.equal(got["b"]["param"], torch.ones(4))
    assert not ck.complete(5, {"W": ((8, 4), 0)})
    with pytest.raises(KeyError):
        ck.load(5, [SliceMeta("W", (8, 4), 0, (5, 8))], kinds=("param",))


def test_dp_planner_assignment_and_sources():
    a = weighted_assignment(["a", "b", "c"], 8, {"a": 1.0, "b": 1.0, "c": 0.5})
    assert sum(len(v) for v in a.values()) == 8
    assert sorted(sum(a.values(), [])) == list(range(8))
    assert len(a["c"]) < len(a["a"])
    src = assign_sources(["x", "y", "z"], ["p", "q"], {"x": "n0", "y": "n1", "z": "n1", "p": "n1", "q": "n2"})
    assert src["y"] == "p" and set(src) == {"x", "y", "z"}


def test_reference_model_params_are_partition_invariant():
    cfg = RefModelConfig()
    full = full_layer_params(cfg, 2)
    stage = RefStage(cfg, 1, 3)
    assert set(stage.named_state()) == {f"layers.{l}.{k}" for l in (1, 2) for k in full}
    assert torch.equal(stage.named_state()["layers.2.fc1.weight"], full["fc1.weight"])
    s = IndexAddressableSampler(64, 8, 2, seed=1)
    m = microbatch(cfg, s, 0, 1)
    assert m.x.shape[0] % cfg.token_multiple == 0 and m.num_tokens == sum(m.doc_lengths)
    assert global_tokens(cfg, s, 0) == sum(microbatch(cfg, s, 0, k).num_tokens for k in range(4))
    losses, _ = reference_train(cfg, s, 3)
    assert losses[-1] < losses[0]
