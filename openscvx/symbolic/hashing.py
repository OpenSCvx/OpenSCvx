"""Structural hashing for symbolic problems.

This module provides name-invariant hashing for symbolic optimization problems.
Two problems with the same mathematical structure will produce the same hash,
regardless of the variable names used.

This enables efficient caching: if a problem has already been compiled with
the same structure, the cached compiled artifacts can be reused.
"""

import hashlib
from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State
    from openscvx.symbolic.problem import SymbolicProblem


class HashContext:
    """Context for name-invariant structural hashing of expressions.

    This class provides context for hashing leaf nodes in a name-invariant way.
    Instead of maintaining a mapping, we use the `_slice` attribute that is set
    on Variable subclasses (State, Control) during preprocessing. This slice
    represents their canonical position in the unified state/control vector.

    Example:
        Two equivalent problems with different names::

            # Problem A
            x = State("x", (3,))  # x._slice = slice(0, 3) after preprocessing
            dynamics_a = {x: x * 2}

            # Problem B (same structure, different name)
            position = State("position", (3,))  # position._slice = slice(0, 3)
            dynamics_b = {position: position * 2}

            # Both hash identically because _slice values are the same
    """

    def __init__(
        self,
        states: List["State"],
        controls: List["Control"],
        parameters: Dict[str, any],
    ):
        """Initialize HashContext.

        Args:
            states: Ordered list of State objects (for reference)
            controls: Ordered list of Control objects (for reference)
            parameters: Dictionary of parameters (name -> value)
        """
        self._parameters = parameters


def hash_symbolic_problem(problem: "SymbolicProblem") -> str:
    """Compute a structural hash of a symbolic optimization problem.

    This function computes a hash that depends only on the mathematical structure
    of the problem, not on variable names or runtime values. Two problems with the same:
    - Dynamics expressions (using _slice for canonical variable positions)
    - Constraints
    - State/control shapes and boundary condition types
    - Parameter shapes
    - Configuration (N, etc.)

    will produce the same hash, regardless of what names are used for variables.

    Notably, the following are NOT included in the hash (allowing solver reuse):
    - Boundary condition values (initial/final state values)
    - Bound values (min/max for states and controls)
    - Parameter values (only shapes are hashed)

    Args:
        problem: A SymbolicProblem (should be preprocessed for best results,
                 so that _slice attributes are set on states/controls)

    Returns:
        A hex string representing the SHA-256 hash of the problem structure
    """
    # Create the hash context (lightweight, just stores parameters reference)
    ctx = HashContext(
        states=problem.states,
        controls=problem.controls,
        parameters=problem.parameters,
    )

    hasher = hashlib.sha256()

    # Hash the dynamics
    hasher.update(b"dynamics:")
    problem.dynamics._hash_into(hasher, ctx)

    # Hash propagation dynamics if present
    if problem.dynamics_prop is not None:
        hasher.update(b"dynamics_prop:")
        problem.dynamics_prop._hash_into(hasher, ctx)

    # Hash all constraints
    hasher.update(b"constraints:")
    for constraint_list in [
        problem.constraints.ctcs,
        problem.constraints.nodal,
        problem.constraints.nodal_convex,
        problem.constraints.cross_node,
        problem.constraints.cross_node_convex,
    ]:
        for constraint in constraint_list:
            constraint._hash_into(hasher, ctx)

    # Hash all states and controls explicitly to capture metadata (boundary
    # condition types) that may not appear in expressions. For example, a state
    # with dynamics dx/dt = 1.0 doesn't appear in the expression tree, but its
    # boundary condition types still affect the compiled problem structure.
    hasher.update(b"states:")
    for state in problem.states:
        state._hash_into(hasher, ctx)

    hasher.update(b"controls:")
    for control in problem.controls:
        control._hash_into(hasher, ctx)

    # Hash parameter shapes (not values) from the problem's parameter dict.
    # This allows the same compiled solver to be reused across parameter sweeps -
    # only the structure matters for compilation, not the actual values.
    hasher.update(b"parameters:")
    for name in sorted(problem.parameters.keys()):
        value = problem.parameters[name]
        hasher.update(name.encode())
        if isinstance(value, np.ndarray):
            hasher.update(str(value.shape).encode())
        else:
            hasher.update(b"scalar")

    # Hash configuration
    hasher.update(f"N:{problem.N}".encode())

    # Hash node intervals for CTCS
    hasher.update(b"node_intervals:")
    for interval in problem.node_intervals:
        hasher.update(f"{interval}".encode())

    return hasher.hexdigest()
