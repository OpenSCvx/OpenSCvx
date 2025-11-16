"""Tests for array operation nodes.

This module tests array operation nodes: Index, Concat, Stack, Hstack, Vstack.
Tests cover:
- Node creation and indexing/slicing operations
- Concatenation and stacking operations
- Shape inference
- Lowering to JAX
- Lowering to CVXPY
- Canonicalization patterns
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import Concat, Constant, Index

# =============================================================================
# Canonicalization Tests
# =============================================================================


def test_concat_and_index_recurse():
    """Test that Concat and Index canonicalize their children recursively."""
    # Concat should simply rebuild with canonical children
    x = Constant([1, 2])
    y = Constant([3, 4])
    concat = Concat(x, y)
    result = concat.canonicalize()
    assert isinstance(result, Concat)
    # both children are still Constant
    assert all(isinstance(c, Constant) for c in result.exprs)

    # Index should also rebuild
    idx = Index(Constant([5, 6, 7]), 1)
    result = idx.canonicalize()
    assert isinstance(result, Index)
    assert result.index == 1
    assert isinstance(result.base, Constant)
