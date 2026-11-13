import pytest
import torch

from resihp.utils.intervals import (
    contains,
    even_split,
    intersect,
    is_partition,
    locate,
    narrow,
    overlaps,
    refine,
)


def test_even_split_balanced_and_uneven():
    assert even_split(12, 4) == [(0, 3), (3, 6), (6, 9), (9, 12)]
    assert even_split(10, 3) == [(0, 4), (4, 7), (7, 10)]
    assert even_split(0, 2) == [(0, 0), (0, 0)]
    with pytest.raises(ValueError):
        even_split(4, 0)


@pytest.mark.parametrize("n,k", [(1, 1), (7, 3), (64, 8), (100, 7)])
def test_even_split_is_partition(n, k):
    parts = even_split(n, k)
    assert is_partition(parts, n)
    sizes = [e - s for s, e in parts]
    assert max(sizes) - min(sizes) <= 1


def test_overlap_routing_power_of_two_and_odd():
    four, two = even_split(16, 4), even_split(16, 2)
    routes = overlaps(four, two)
    assert routes == [(0, 0, (0, 4)), (1, 0, (4, 8)), (2, 1, (8, 12)), (3, 1, (12, 16))]
    routes = overlaps(even_split(12, 4), even_split(12, 3))
    covered = sorted(o for _, _, o in routes)
    assert covered == [(0, 3), (3, 4), (4, 6), (6, 8), (8, 9), (9, 12)]


def test_refine_and_locate():
    r = refine([even_split(12, 4), even_split(12, 3)])
    assert r == [(0, 3), (3, 4), (4, 6), (6, 8), (8, 9), (9, 12)]
    for iv in r:
        assert 0 <= locate(even_split(12, 4), iv) < 4
        assert 0 <= locate(even_split(12, 3), iv) < 3
    with pytest.raises(ValueError):
        refine([even_split(12, 4), even_split(10, 2)])
    with pytest.raises(ValueError):
        locate(even_split(12, 2), (5, 7))


def test_intersect_contains_narrow():
    assert intersect((0, 5), (5, 9)) is None
    assert intersect((0, 6), (5, 9)) == (5, 6)
    assert contains((0, 10), (3, 4)) and not contains((3, 4), (0, 10))
    t = torch.arange(20).reshape(4, 5)
    assert torch.equal(narrow(t, 1, (2, 4), offset=1), t[:, 1:3])
