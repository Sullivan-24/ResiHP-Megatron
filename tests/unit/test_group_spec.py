import pytest

from resihp.comm.group_spec import GroupSpec, canonical_order, split_layers


def test_spec_requires_ascending_unique_ranks():
    with pytest.raises(ValueError):
        GroupSpec("x", (2, 1))
    with pytest.raises(ValueError):
        GroupSpec("x", (1, 1))


def test_canonical_order_is_input_order_independent():
    a = [GroupSpec("tp/r1", (2, 3), "tp"), GroupSpec("dp/0", (0, 2), "dp"), GroupSpec("tp/r0", (0, 1), "tp")]
    assert canonical_order(a) == canonical_order(list(reversed(a)))
    with pytest.raises(ValueError):
        canonical_order(a + [GroupSpec("tp/r0", (4, 5), "tp")])


def test_split_layers_disjoint_and_dedup():
    specs = [
        GroupSpec("tp/r0", (0, 1), "tp"), GroupSpec("tp/r1", (2, 3), "tp"),
        GroupSpec("dpshard/a", (0, 2), "dpshard"), GroupSpec("dpshard/b", (0, 3), "dpshard"),
        GroupSpec("dpshard/c", (1, 3), "dpshard"), GroupSpec("alias", (0, 1), "zz"), GroupSpec("solo", (4,), "x"),
    ]
    layers = split_layers(specs)
    for layer in layers:
        seen = set()
        for s in layer:
            assert not seen & set(s.ranks)
            seen |= set(s.ranks)
    created = [s.ranks for layer in layers for s in layer]
    assert len(created) == len(set(created)) == 5
    assert (4,) not in created
    assert split_layers(list(reversed(specs))) == layers
