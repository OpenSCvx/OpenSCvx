"""Tests for mathematical function nodes.

This module tests mathematical function nodes:
- Trigonometric: Sin, Cos
- Exponential: Exp, Log, Sqrt
- Nonlinear: Square, PositivePart, Huber, SmoothReLU, Max

Tests cover:
- Node creation and properties
- Behavior with constants and variables
- Lowering to JAX
- Lowering to CVXPY (where applicable)
- Special properties (differentiability, convexity)
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Huber,
    PositivePart,
    SmoothReLU,
    Square,
    Variable,
)

# =============================================================================
# Penalty/Nonlinear Function Node Creation
# =============================================================================


def test_penalty_expressions():
    """Test the penalty expression building blocks."""
    x = Variable("x", shape=(1,))

    # PositivePart
    pos = PositivePart(x)
    assert repr(pos) == "pos(Var('x'))"
    assert pos.children() == [x]

    # Square
    sq = Square(x)
    assert repr(sq) == "(Var('x'))^2"
    assert sq.children() == [x]

    # Huber
    hub = Huber(x, delta=0.5)
    assert repr(hub) == "huber(Var('x'), delta=0.5)"
    assert hub.delta == 0.5
    assert hub.children() == [x]

    # SmoothReLU
    smooth = SmoothReLU(x, c=1e-6)
    assert repr(smooth) == "smooth_relu(Var('x'), c=1e-06)"
    assert smooth.c == 1e-6
    assert smooth.children() == [x]
