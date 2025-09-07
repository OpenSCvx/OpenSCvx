from typing import List, Tuple

import numpy as np

from openscvx.backend.control import Control
from openscvx.backend.expr import (
    CTCS,
    Add,
    Concat,
    Constraint,
    Expr,
)
from openscvx.backend.state import Free, State


def augment_dynamics_with_ctcs(
    xdot: Expr, states: List[State], controls: List[Control], constraints: List[Expr]
) -> Tuple[Expr, List[State], List[Control]]:
    """
    Augment dynamics with continuous-time constraint satisfaction (CTCS).

    Args:
        xdot: The original dynamics expression
        states: The list of state variables
        controls: The list of control variables
        constraints: List of constraints (mix of CTCS and regular Constraints)

    Returns:
        Tuple of:
        - Augmented dynamics expression
        - Updated list of states (including augmented states)
        - Updated list of controls (including time dilation)
    """
    constraints_ctcs: List[CTCS] = []
    # constraints_nodal: List[Constraint] = []

    # Separate CTCS from regular constraints
    for c in constraints:
        if isinstance(c, CTCS):
            constraints_ctcs.append(c)
        elif isinstance(c, Constraint):
            pass  # constraints_nodal.append(c)
        else:
            raise ValueError(f"Constraints must be `Constraint` or `CTCS`, got {type(c).__name__}")

    # Copy the original states and controls lists
    states_augmented = list(states)
    controls_augmented = list(controls)

    # Build penalty expressions for all CTCS constraints
    penalty_terms: List[Expr] = []

    for ctcs in constraints_ctcs:
        # Get the penalty expression for this CTCS constraint
        penalty_expr = ctcs.penalty_expr()

        # TODO: In the future, apply scaling here if ctcs has a scaling attribute
        # if hasattr(ctcs, 'scaling') and ctcs.scaling != 1.0:
        #     penalty_expr = Mul(Constant(np.array(ctcs.scaling)), penalty_expr)

        penalty_terms.append(penalty_expr)

    # Sum all penalty terms into a single augmented state (default behavior)
    if penalty_terms:
        # Add all penalty terms together
        if len(penalty_terms) == 1:
            augmented_state_expr = penalty_terms[0]
        else:
            augmented_state_expr = Add(*penalty_terms)

        # Create a new Variable for the augmented state
        # TODO: In the future, create multiple variables based on idx grouping
        aug_var = State(f"_ctcs_aug_{0}", shape=(1,))
        aug_var.initial = np.array([0])
        aug_var.final = np.array([Free(0)])
        aug_var.min = np.array([0])
        # TODO: (norrisg) take `LICQ_max`, `N` as inputs to this function
        aug_var.max = np.array([1e-8])  # LICQ_MAX
        aug_var.guess = np.zeros([2, 1])  # N x num augmented states,
        states_augmented.append(aug_var)

        # Concatenate with original dynamics
        xdot_aug = Concat(xdot, augmented_state_expr)

        # TODO: Future implementation for index-based grouping
        # When idx is implemented, we would:
        # 1. Group penalty_terms by ctcs.idx
        # 2. Sum penalties within each group
        # 3. Create a Variable for each group
        # 4. Concatenate each group's sum as a separate augmented state
    else:
        xdot_aug = xdot

    time_dilation = Control("_time_dilation", shape=(1,))
    # TODO: (norrisg) Construct the min, max, initial, final, guess for time_dilation here
    controls_augmented.append(time_dilation)

    # # Collect all constraints that should be checked at nodes
    # node_checks: List[Constraint] = []

    # # Add the underlying constraints from CTCS (if they should be checked at nodes)
    # for ctcs in constraints_ctcs:
    #     # TODO: In the future, check ctcs.check_at_nodes attribute
    #     # if getattr(ctcs, 'check_at_nodes', True):
    #     node_checks.append(ctcs.constraint)

    # # Regular constraints are always checked at nodes
    # node_checks.extend(constraints_nodal)

    return xdot_aug, states_augmented, controls_augmented
