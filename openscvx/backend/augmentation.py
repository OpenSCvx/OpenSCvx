from typing import Dict, List, Tuple

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


def sort_ctcs_constraints(
    constraints_ctcs: List[CTCS], N: int
) -> Tuple[List[CTCS], List[Tuple[int, int]], int]:
    """
    Sort and group CTCS constraints by their idx, ensuring proper grouping rules.

    Args:
        constraints_ctcs: List of CTCS constraints to sort and group
        N: Number of discretization nodes (for normalizing None nodes to (0, N))

    Returns:
        Tuple of:
        - List of CTCS constraints with idx assigned
        - List of node intervals in ascending idx order
        - Number of augmented states (number of unique idx values)

    Rules:
        - All CTCS constraints with the same nodes can go into the same idx
        - CTCS constraints with different node intervals cannot go into the same idx
        - idx values must form a contiguous block starting from 0
        - Unspecified idx values are auto-assigned
    """
    idx_to_nodes: Dict[int, Tuple[int, int]] = {}
    next_idx = 0

    for c in constraints_ctcs:
        # Normalize None to full horizon
        c.nodes = c.nodes or (0, N)
        key = c.nodes

        if c.idx is not None:
            # User supplied an identifier: ensure it always points to the same interval
            if c.idx in idx_to_nodes:
                if idx_to_nodes[c.idx] != key:
                    raise ValueError(
                        f"idx={c.idx} was first used with interval={idx_to_nodes[c.idx]}, "
                        f"but now you gave it interval={key}"
                    )
            else:
                idx_to_nodes[c.idx] = key
        else:
            # No identifier: see if this interval already has one
            for existing_id, nodes in idx_to_nodes.items():
                if nodes == key:
                    c.idx = existing_id
                    break
            else:
                # Brand-new interval: pick the next free auto-id
                while next_idx in idx_to_nodes:
                    next_idx += 1
                c.idx = next_idx
                idx_to_nodes[next_idx] = key
                next_idx += 1

    # Validate that idx values form a contiguous block starting from 0
    ordered_ids = sorted(idx_to_nodes.keys())
    expected_ids = list(range(len(ordered_ids)))
    if ordered_ids != expected_ids:
        raise ValueError(
            f"CTCS constraint idx values must form a contiguous block starting from 0. "
            f"Got {ordered_ids}, expected {expected_ids}"
        )

    # Extract intervals in ascending idx order
    node_intervals = [idx_to_nodes[i] for i in ordered_ids]
    num_augmented_states = len(ordered_ids)

    return constraints_ctcs, node_intervals, num_augmented_states


def separate_constraints(
    constraints: List[Expr],
) -> Tuple[List[CTCS], List[Constraint]]:
    """
    Separate CTCS constraints from regular constraints.
    
    Args:
        constraints: List of constraints (mix of CTCS and regular Constraints)
        
    Returns:
        Tuple of:
        - List of CTCS constraints
        - List of regular constraints (non-CTCS)
    """
    constraints_ctcs: List[CTCS] = []
    constraints_nodal: List[Constraint] = []

    for c in constraints:
        if isinstance(c, CTCS):
            constraints_ctcs.append(c)
        elif isinstance(c, Constraint):
            constraints_nodal.append(c)
        else:
            raise ValueError(f"Constraints must be `Constraint` or `CTCS`, got {type(c).__name__}")
    
    return constraints_ctcs, constraints_nodal


def get_nodal_constraints_from_ctcs(constraints_ctcs: List[CTCS]) -> List[Constraint]:
    """
    Extract underlying constraints from CTCS constraints that should also be checked nodally.
    
    Args:
        constraints_ctcs: List of CTCS constraint wrappers
        
    Returns:
        List of underlying Constraint objects from CTCS constraints with check_nodally=True
    """
    nodal_ctcs = []
    for ctcs in constraints_ctcs:
        if ctcs.check_nodally:
            nodal_ctcs.append(ctcs.constraint)
    return nodal_ctcs


def augment_dynamics_with_ctcs(
    xdot: Expr,
    states: List[State],
    controls: List[Control],
    constraints: List[Expr],
    N,
    idx_time,
    licq_min=0.0,
    licq_max=1e-4,
    time_dilation_factor_min=0.3,
    time_dilation_factor_max=3.0,
) -> Tuple[Expr, List[State], List[Control], List[Constraint]]:
    """
    Augment dynamics with continuous-time constraint satisfaction (CTCS).

    Args:
        xdot: The original dynamics expression
        states: The list of state variables
        controls: The list of control variables
        constraints: List of constraints (mix of CTCS and regular Constraints)
        N: Number of discretization nodes
        licq_min: Minimum value for LICQ augmented state
        licq_max: Maximum value for LICQ augmented state
        time_dilation_factor_min: Minimum time dilation factor
        time_dilation_factor_max: Maximum time dilation factor
        idx_time: Index of time variable in the state vector for time dilation setup

    Returns:
        Tuple of:
        - Augmented dynamics expression
        - Updated list of states (including augmented states)
        - Updated list of controls (including time dilation)
        - All nodal constraints (regular constraints + CTCS constraints with check_nodally=True)
    """
    # Separate CTCS from regular constraints
    constraints_ctcs, constraints_nodal = separate_constraints(constraints)
    
    # Get CTCS constraints that should also be checked nodally
    nodal_from_ctcs = get_nodal_constraints_from_ctcs(constraints_ctcs)
    
    # Combine all nodal constraints
    all_nodal_constraints = constraints_nodal + nodal_from_ctcs

    # Copy the original states and controls lists
    states_augmented = list(states)
    controls_augmented = list(controls)

    if constraints_ctcs:
        # Sort and group CTCS constraints by their idx
        constraints_ctcs, node_intervals, num_augmented_states = sort_ctcs_constraints(
            constraints_ctcs, N
        )

        # Group penalty expressions by idx
        penalty_groups: Dict[int, List[Expr]] = {}

        for ctcs in constraints_ctcs:
            # Get the penalty expression for this CTCS constraint
            penalty_expr = ctcs.penalty_expr()

            # TODO: In the future, apply scaling here if ctcs has a scaling attribute
            # if hasattr(ctcs, 'scaling') and ctcs.scaling != 1.0:
            #     penalty_expr = Mul(Constant(np.array(ctcs.scaling)), penalty_expr)

            if ctcs.idx not in penalty_groups:
                penalty_groups[ctcs.idx] = []
            penalty_groups[ctcs.idx].append(penalty_expr)

        # Create augmented state expressions for each group
        augmented_state_exprs = []
        for idx in sorted(penalty_groups.keys()):
            penalty_terms = penalty_groups[idx]
            if len(penalty_terms) == 1:
                augmented_state_expr = penalty_terms[0]
            else:
                augmented_state_expr = Add(*penalty_terms)
            augmented_state_exprs.append(augmented_state_expr)

        # Create augmented state variables
        for idx in range(num_augmented_states):
            aug_var = State(f"_ctcs_aug_{idx}", shape=(1,))
            aug_var.initial = np.array([licq_min])  # Set initial to respect bounds
            aug_var.final = np.array([Free(0)])
            aug_var.min = np.array([licq_min])
            aug_var.max = np.array([licq_max])
            # Set guess to licq_min as well
            aug_var.guess = np.full([N, 1], licq_min)  # N x num augmented states
            states_augmented.append(aug_var)

        # Concatenate with original dynamics
        xdot_aug = Concat(xdot, *augmented_state_exprs)
    else:
        xdot_aug = xdot

    time_dilation = Control("_time_dilation", shape=(1,))

    # Set up time dilation bounds and initial guess
    # Get the time value from the main state (assuming states[0] is the main state)
    time_final = states[0].final[idx_time]
    time_dilation.min = np.array([time_dilation_factor_min * time_final])
    time_dilation.max = np.array([time_dilation_factor_max * time_final])
    time_dilation.guess = np.ones([N, 1]) * time_final

    controls_augmented.append(time_dilation)

    return xdot_aug, states_augmented, controls_augmented, all_nodal_constraints
