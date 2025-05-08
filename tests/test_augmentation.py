import pytest

from openscvx.augmentation import sort_ctcs_constraints

class DummyCTCS:
    def __init__(self, nodes=None, idx=None):
        self.nodes = nodes
        self.idx = idx

def test_empty_list():
    constraints = []
    sorted_constraints, intervals, count = sort_ctcs_constraints(constraints, N=5)
    assert sorted_constraints == []
    assert intervals == []
    assert count == 0

def test_single_constraint_defaults_to_full_horizon():
    c = DummyCTCS()
    sorted_constraints, intervals, count = sort_ctcs_constraints([c], N=10)
    assert count == 1
    assert intervals == [(0, 10)]
    assert c.idx == 0

def test_multiple_constraints_same_interval():
    c1 = DummyCTCS()
    c2 = DummyCTCS()
    sorted_constraints, intervals, count = sort_ctcs_constraints([c1, c2], N=6)
    assert count == 1
    assert intervals == [(0, 6)]
    assert c1.idx == 0 and c2.idx == 0

def test_distinct_intervals_auto_assignment():
    c1 = DummyCTCS(nodes=(0, 3))
    c2 = DummyCTCS(nodes=(3, 7))
    sorted_constraints, intervals, count = sort_ctcs_constraints([c1, c2], N=7)
    assert count == 2
    # intervals sorted by idx: first idx=0→(0,3), then idx=1→(3,7)
    assert intervals == [(0, 3), (3, 7)]
    assert c1.idx == 0
    assert c2.idx == 1

def test_explicit_idx_consistency():
    c1 = DummyCTCS(nodes=(1, 4), idx=5)
    c2 = DummyCTCS(nodes=(1, 4), idx=5)
    sorted_constraints, intervals, count = sort_ctcs_constraints([c1, c2], N=10)
    assert count == 1
    assert intervals == [(1, 4)]
    assert c1.idx == 5 and c2.idx == 5

def test_explicit_idx_inconsistency_raises():
    c1 = DummyCTCS(nodes=(1, 4), idx=2)
    c2 = DummyCTCS(nodes=(2, 5), idx=2)
    with pytest.raises(ValueError):
        sort_ctcs_constraints([c1, c2], N=10)

def test_mixed_manual_and_auto_assignment():
    c1 = DummyCTCS(nodes=(0, 3), idx=1)
    c2 = DummyCTCS(nodes=(3, 8))
    c3 = DummyCTCS(nodes=(0, 3))
    sorted_constraints, intervals, count = sort_ctcs_constraints([c1, c2, c3], N=8)
    assert count == 2
    # idx_to_nodes keys are {1:(0,3), 0:(3,8)}, sorted → [0,1]
    assert intervals == [(3, 8), (0, 3)]
    assert c1.idx == 1
    assert c3.idx == 1
    assert c2.idx == 0
