"""Tests for core AST functionality.

This module tests the fundamental AST infrastructure of the symbolic expression
system, including:
- to_expr() conversion function
- traverse() tree traversal function
- Tree structure and pretty printing
- Base Expr/Leaf behavior

Note: Tests for specific node types (Add, Mul, Constraint, etc.) are in their
respective test_*_nodes.py files.
"""

import numpy as np

from openscvx.symbolic.expr import (
    Add,
    Constant,
    Control,
    Mul,
    State,
    Variable,
    to_expr,
    traverse,
)

# =============================================================================
# to_expr() Conversion Tests
# =============================================================================


def test_to_expr_wraps_numbers_and_arrays():
    # scalars
    c1 = to_expr(5)
    assert isinstance(c1, Constant)
    assert c1.value.shape == () and c1.value == 5

    # 1-D arrays become Constant
    arr = [1, 2, 3]
    c2 = to_expr(arr)
    assert isinstance(c2, Constant)
    assert np.array_equal(c2.value, np.array(arr))

    # passing through an Expr unchanged
    a = Constant(np.array([1.0, 2.0]))
    assert to_expr(a) is a


def test_to_expr_passes_variables_through():
    v = Variable("v", (1,))
    x = State("x", (1,))
    u = Control("u", (1,))
    assert to_expr(v) is v
    assert to_expr(x) is x
    assert to_expr(u) is u


# =============================================================================
# Tree Structure and Pretty Printing
# =============================================================================


def test_pretty_print_tree_structure():
    # build a nested tree: -( (a + b) * c )
    a, b, c = Constant(1), Constant(2), Constant(3)
    tree = -((a + b) * c)
    p = tree.pretty()
    # Should indent like:
    # Neg
    #   Mul
    #     Add
    #       Const
    #       Const
    #     Const
    lines = p.splitlines()
    assert lines[0].strip() == "Neg"
    assert lines[1].strip() == "Mul"
    # deeper indent for Add's children:
    assert "Add" in lines[2]
    assert "Const" in lines[3]  # one of the leaves


# =============================================================================
# traverse() Function Tests
# =============================================================================


def test_traverse_visits_all_nodes_in_preorder():
    # build a small graph: (a + (b * c))
    a, b, c = Constant(1), Constant(2), Constant(3)
    expr = Add(a, Mul(b, c))
    visited = []

    def visit(node):
        visited.append(type(node).__name__)

    traverse(expr, visit)

    # preorder: Add → a → Mul → b → c
    assert visited == ["Add", "Constant", "Mul", "Constant", "Constant"]


# =============================================================================
# Expr.canonicalize() Method Tests
# =============================================================================


def test_constants_are_unchanged_by_canonicalization():
    """Test that constants are already normalized and unchanged by canonicalization.

    Since canonicalization is now a method on Expr, this tests the base behavior
    that Constants return themselves unchanged.
    """
    # Constants are now normalized at construction time, so canonicalization should be a no-op
    const_scalar = Constant(5.0)
    const_vector = Constant([1.0, 2.0, 3.0])
    const_matrix = Constant([[1.0, 2.0], [3.0, 4.0]])

    # Canonicalization should return the same object (no changes needed)
    canon_scalar = const_scalar.canonicalize()
    canon_vector = const_vector.canonicalize()
    canon_matrix = const_matrix.canonicalize()

    assert canon_scalar is const_scalar  # Should be same object
    assert canon_vector is const_vector
    assert canon_matrix is const_matrix

    # Values should be already normalized
    assert const_scalar.value.shape == ()
    assert const_vector.value.shape == (3,)
    assert const_matrix.value.shape == (2, 2)


def test_vector_constraint_equivalence_after_canonicalization():
    """Test that different ways of creating vector constraints become equivalent after
    canonicalization.

    This tests that canonicalize() properly normalizes expressions across the tree.
    """
    x = State("x", shape=(3,))
    bounds = np.array([1.0, 2.0, 3.0])

    # Two ways to create the same constraint - constants are normalized at construction now
    constraint1 = x <= Constant(bounds)
    constraint2 = x <= Constant(np.array([bounds]))  # Extra dimension gets squeezed at construction

    # Constants should already be equivalent at construction time
    assert np.array_equal(constraint1.rhs.value, constraint2.rhs.value)
    assert constraint1.rhs.value.shape == constraint2.rhs.value.shape

    canon1 = constraint1.canonicalize()
    canon2 = constraint2.canonicalize()

    # After canonicalization, they should remain equivalent (and in canonical form)
    assert isinstance(canon1.rhs, Constant)
    assert isinstance(canon2.rhs, Constant)
    assert np.array_equal(canon1.rhs.value, canon2.rhs.value)
    assert canon1.rhs.value.shape == canon2.rhs.value.shape
