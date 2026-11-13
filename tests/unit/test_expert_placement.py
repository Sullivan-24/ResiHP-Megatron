import pytest

from resihp.plan.expert_placement import (expert_dp_groups, group_time, moved_experts, place_group,
                                          plan_expert_placement, replica_counts, smooth_weighted_round_robin)
from resihp.plan.planner import PlanInfeasible


def test_replica_counts_give_spare_slots_to_hot_experts():
    assert replica_counts([1, 1, 1], 3) == [1, 1, 1]
    assert replica_counts([8, 1, 1, 2], 6) == [3, 1, 1, 1]
    with pytest.raises(PlanInfeasible):
        replica_counts([1] * 5, 4)


def test_group_placement_replicates_hot_experts_and_unloads_slow_devices():
    loads = [12.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    speeds = {"c": 0.5}
    slots = place_group(["a", "b", "c"], speeds, loads, 3)
    assert {e for v in slots.values() for e in v} == set(range(6))
    assert all(len(set(v)) == len(v) <= 3 for v in slots.values())
    assert sum(1 for v in slots.values() if 0 in v) >= 2, "the hot expert is replicated"
    singles = {d: tuple(e for e in range(6) if e % 3 == i) for i, d in enumerate("abc")}
    assert group_time(slots, loads, speeds) < group_time(singles, loads, speeds)

    def tokens(d):
        speed_sum = {e: sum(speeds.get(h, 1.0) for h, v in slots.items() if e in v) for e in range(6)}
        return sum(loads[e] * speeds.get(d, 1.0) / speed_sum[e] for e in slots[d])

    assert tokens("c") < min(tokens("a"), tokens("b"))


def test_replan_after_failure_keeps_incumbents_and_reports_moves():
    p1 = plan_expert_placement([["a", "b"], ["c", "d"]], 6)
    assert {len(p1.holders(e)) for e in range(6)} == {2}
    p2 = plan_expert_placement([["a", "b", "c"]], 6, prev=p1, slots_per_device=2)
    p2.validate()
    # a and c held the same three experts, b the other three: one of b's must move.
    assert set(p1.slots["a"]) == set(p1.slots["c"]) and not set(p1.slots["a"]) & set(p1.slots["b"])
    assert moved_experts(p1, p2) == 1
    assert set(p2.slots["a"]) <= set(p1.slots["a"]) and set(p2.slots["b"]) <= set(p1.slots["b"])
    cold = plan_expert_placement([["a", "b", "c"]], 6, slots_per_device=2)
    assert moved_experts(p1, cold) > moved_experts(p1, p2)
    with pytest.raises(PlanInfeasible):
        plan_expert_placement([["a", "b"]], 6, max_slots=2)


def test_expert_dp_groups_and_token_split():
    p = plan_expert_placement([["a", "b"], ["c", "d"]], 4)
    groups = expert_dp_groups(p)
    assert sorted(e for es in groups.values() for e in es) == [0, 1, 2, 3]
    assert all(len(h) == 2 and len({p.group_of(d) for d in h}) == 2 for h in groups)
    split = smooth_weighted_round_robin(["x", "y"], {"y": 0.5}, 30)
    assert split.count("x") == 20 and split.count("y") == 10
    assert smooth_weighted_round_robin(["x", "y"], {"y": 0.5}, 30) == split
