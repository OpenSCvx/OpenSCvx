"""Logical and control flow operations for symbolic expressions.

This module provides logical and control flow operations used in optimization problems,
enabling conditional logic in dynamics and constraints. These operations are
JAX-only and not supported in CVXPy lowering.

Operations:
    - **Conditional:** `Cond` - Conditional expression using jax.lax.cond for
        JAX-traceable branching

Example:
    Using conditional logic in dynamics::

        import openscvx as ox

        x = ox.State("x", shape=(3,))
        u = ox.Control("u", shape=(2,))

        # Conditional dynamics based on state
        dynamics = {
            "x": ox.Cond(
                ox.Norm(x) > 1.0,  # predicate
                x / ox.Norm(x),    # true branch: normalize if norm > 1
                x                   # false branch: keep as is
            )
        }
"""

from typing import Tuple

import numpy as np

from .expr import Expr, to_expr


class Cond(Expr):
    """Conditional expression for JAX-traceable branching.

    Implements a conditional expression that selects between two branches based
    on a predicate. This wraps `jax.lax.cond` to enable conditional logic in
    symbolic expressions for dynamics and constraints.

    The predicate must evaluate to a scalar boolean value. The true and false
    branches must have broadcastable shapes (following JAX/NumPy broadcasting rules).

    Attributes:
        pred: Predicate expression that evaluates to a scalar boolean
        true_branch: Expression to evaluate when predicate is True
        false_branch: Expression to evaluate when predicate is False

    Example:
        Define a conditional expression::

            x = ox.State("x", shape=(3,))
            expr = ox.Cond(
                ox.Norm(x) > 1.0,  # predicate
                x / ox.Norm(x),    # true branch
                x                   # false branch
            )

    Note:
        This operation is only supported for JAX lowering. CVXPy lowering will
        raise NotImplementedError since conditional logic is not DCP-compliant.
    """

    def __init__(self, pred, true_branch, false_branch):
        """Initialize a conditional expression.

        Args:
            pred: Predicate expression that evaluates to a scalar boolean
            true_branch: Expression to evaluate when predicate is True
            false_branch: Expression to evaluate when predicate is False
        """
        self.pred = to_expr(pred)
        self.true_branch = to_expr(true_branch)
        self.false_branch = to_expr(false_branch)

    def children(self):
        """Return the child expressions: predicate, true branch, and false branch."""
        return [self.pred, self.true_branch, self.false_branch]

    def canonicalize(self) -> "Expr":
        """Canonicalize by canonicalizing all three children."""
        pred = self.pred.canonicalize()
        true_branch = self.true_branch.canonicalize()
        false_branch = self.false_branch.canonicalize()
        return Cond(pred, true_branch, false_branch)

    def check_shape(self) -> Tuple[int, ...]:
        """Check and return the output shape of the conditional.

        The predicate must be scalar, and the true and false branches must have
        broadcastable shapes. The output shape is the broadcasted shape of the
        two branches.

        Returns:
            tuple: The broadcasted shape of true_branch and false_branch

        Raises:
            ValueError: If predicate is not scalar or branches have incompatible shapes
        """
        pred_shape = self.pred.check_shape()
        true_shape = self.true_branch.check_shape()
        false_shape = self.false_branch.check_shape()

        # Predicate must be scalar
        if pred_shape != ():
            raise ValueError(
                f"Cond predicate must be scalar, got shape {pred_shape}"
            )

        # True and false branches must be broadcastable
        try:
            return np.broadcast_shapes(true_shape, false_shape)
        except ValueError as e:
            raise ValueError(
                f"Cond branches have incompatible shapes: {true_shape} and {false_shape}"
            ) from e

    def __repr__(self):
        """Return string representation of the conditional."""
        return f"cond({self.pred!r}, {self.true_branch!r}, {self.false_branch!r})"

