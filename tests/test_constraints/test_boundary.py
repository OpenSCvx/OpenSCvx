import pytest
import jax.numpy as jnp
from openscvx.constraints.boundary import BoundaryConstraint, boundary


@pytest.fixture
def arr():
    return jnp.array([1.0, 2.0, 3.0])


def test_default_types_and_value_get_set(arr):
    bc = BoundaryConstraint(arr)
    # default types
    assert bc.types == ["Fix", "Fix", "Fix"]
    # __getitem__
    assert float(bc[1]) == 2.0
    # __setitem__
    bc[0] = 5.5
    assert float(bc[0]) == 5.5


@pytest.mark.parametrize(
    "key,val,expected",
    [
        # single-index assignment
        (1, "Free", ["Fix", "Free", "Fix"]),
        # slice with list
        (slice(0, 2), ["Minimize", "Maximize"], ["Minimize", "Maximize", "Fix"]),
        # slice with scalar
        (slice(1, 3), "Free", ["Fix", "Free", "Free"]),
    ],
)
def test_type_set_valid(arr, key, val, expected):
    bc = BoundaryConstraint(arr)
    tp = bc.type
    tp[key] = val
    assert bc.types == expected


@pytest.mark.parametrize(
    "key,val,msg",
    [
        # mismatched lengths
        (slice(0, 2), ["Free"], "Mismatch between"),
        # invalid type name
        (1, "NotAValidType", "Invalid type"),
    ],
)
def test_type_set_invalid(arr, key, val, msg):
    bc = BoundaryConstraint(arr)
    tp = bc.type
    with pytest.raises(ValueError) as exc:
        tp[key] = val
    assert msg in str(exc.value)


def test_type_get_slice_len_and_repr(arr):
    bc = BoundaryConstraint(arr)
    tp = bc.type
    # single get and slice get
    assert tp[0] == "Fix"
    assert tp[1:3] == ["Fix", "Fix"]
    # len and repr
    assert len(tp) == 3
    assert repr(tp) == repr(bc.types)


def test_boundary_factory_preserves_array_and_types(arr):
    bc = boundary(arr)
    assert isinstance(bc, BoundaryConstraint)
    assert jnp.all(bc.value == arr)
    assert bc.types == ["Fix", "Fix", "Fix"]
