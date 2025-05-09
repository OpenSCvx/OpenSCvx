# tests/test_boundary_constraint.py

import pytest
import jax.numpy as jnp
from openscvx.constraints.boundary import BoundaryConstraint, boundary, ALLOWED_TYPES


def test_initial_types_are_fix():
    arr = jnp.array([1.0, 2.0, 3.0])
    bc = BoundaryConstraint(arr)
    # default types should all be "Fix"
    assert bc.types == ["Fix", "Fix", "Fix"]


def test_getitem_returns_value():
    arr = jnp.array([10.0, 20.0, 30.0])
    bc = BoundaryConstraint(arr)
    assert float(bc[1]) == 20.0
    # slicing
    sub = bc[0:2]
    assert isinstance(sub, jnp.ndarray)
    assert jnp.all(sub == jnp.array([10.0, 20.0]))


def test_setitem_updates_value():
    arr = jnp.array([0.0, 0.0, 0.0])
    bc = BoundaryConstraint(arr)
    bc[2] = 5.5
    # new array at index 2 should be 5.5
    assert float(bc[2]) == 5.5
    # other entries unchanged
    assert float(bc[0]) == 0.0


def test_type_get_single_and_slice():
    arr = jnp.array([1.0, 2.0, 3.0, 4.0])
    bc = BoundaryConstraint(arr)
    tp = bc.type
    # single index
    assert tp[0] == "Fix"
    # slice returns list
    assert tp[1:3] == ["Fix", "Fix"]


def test_type_set_single():
    arr = jnp.array([0.0, 0.0])
    bc = BoundaryConstraint(arr)
    tp = bc.type
    tp[1] = "Free"
    assert bc.types == ["Fix", "Free"]
    # ensure allowed set
    assert tp[1] == "Free"


def test_type_set_slice_with_list():
    arr = jnp.array([0.0, 0.0, 0.0])
    bc = BoundaryConstraint(arr)
    tp = bc.type
    # set indices 0..2 to ["Free","Minimize"]
    tp[0:2] = ["Free", "Minimize"]
    assert bc.types == ["Free", "Minimize", "Fix"]


def test_type_set_slice_with_scalar():
    arr = jnp.array([0.0, 0.0, 0.0])
    bc = BoundaryConstraint(arr)
    tp = bc.type
    # set first two entries to "Minimize"
    tp[0:2] = "Minimize"
    assert bc.types == ["Minimize", "Minimize", "Fix"]


def test_type_set_mismatch_length():
    arr = jnp.array([0.0, 0.0, 0.0])
    bc = BoundaryConstraint(arr)
    tp = bc.type
    # slice of length 2 but list of length 1 → error
    with pytest.raises(ValueError) as exc:
        tp[0:2] = ["Free"]
    assert "Mismatch between indices and values length" in str(exc.value)


def test_type_set_invalid_value():
    arr = jnp.array([0.0, 0.0])
    bc = BoundaryConstraint(arr)
    tp = bc.type
    with pytest.raises(ValueError) as exc:
        tp[1] = "InvalidType"
    assert "Invalid type: InvalidType" in str(exc.value)


def test_type_len_and_repr():
    arr = jnp.array([1.0, 2.0, 3.0])
    bc = BoundaryConstraint(arr)
    tp = bc.type
    assert len(tp) == 3
    # repr should match list repr
    assert repr(tp) == repr(bc.types)


def test_boundary_factory_function():
    arr = jnp.array([9.0, 8.0])
    bc = boundary(arr)
    assert isinstance(bc, BoundaryConstraint)
    assert jnp.all(bc.value == arr)
    # types initialized correctly
    assert bc.types == ["Fix", "Fix"]
