from typing import List, Tuple

from openscvx.backend.expr import (
    CTCS,
    Concat,
    Constraint,
    Expr,
)
from openscvx.backend.state import State


def augment_dynamics_with_ctcs(
    xdot: Expr, states: List[State], constraints: List[Expr]
) -> Tuple[Expr, List[State], List[Constraint]]:
    """
    Augment dynamics with continuous-time constraint satisfaction (CTCS).

    Args:
        xdot: The original dynamics expression
        states: The list of state variables (will be modified in-place)
        constraints: List of constraints (mix of CTCS and regular Constraints)

    Returns:
        Tuple of:
        - Augmented dynamics expression
        - Updated list of states (including augmented states)
        - List of constraints to check at nodes
    """
    constraints_ctcs: List[CTCS] = []
    constraints_nodal: List[Constraint] = []

    # Separate CTCS from regular constraints
    for c in constraints:
        if isinstance(c, CTCS):
            constraints_ctcs.append(c)
        elif isinstance(c, Constraint):
            constraints_nodal.append(c)
        else:
            raise ValueError(f"Constraints must be `Constraint` or `CTCS`, got {type(c).__name__}")

    # Build augmented dynamics and states
    penalty_exprs: List[Expr] = []
    augmented_states: List[State] = []

    for i, ctcs in enumerate(constraints_ctcs):
        # Create the penalty expression for this constraint
        penalty_expr = ctcs.penalty_expr()
        penalty_exprs.append(penalty_expr)

        # Create a corresponding augmented state variable
        # Use a reserved prefix to avoid name collisions
        aug_state = State(f"_ctcs_aug_{i}", shape=(1,))
        augmented_states.append(aug_state)

    # Update the states list (in-place to maintain references)
    states.extend(augmented_states)

    # Build augmented dynamics
    if penalty_exprs:
        xdot_aug = Concat(xdot, *penalty_exprs)
    else:
        xdot_aug = xdot

    # All regular constraints are checked at nodes
    # CTCS constraints can optionally also be checked at nodes (future feature)
    node_constraints = constraints_nodal.copy()

    return xdot_aug, states, node_constraints
