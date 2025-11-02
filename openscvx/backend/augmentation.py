from typing import Dict, List, Tuple

import numpy as np

from openscvx.backend.expr import (
    CTCS,
    Add,
    Concat,
    Constraint,
    Expr,
    Index,
    NodalConstraint,
)
from openscvx.backend.expr.control import Control
from openscvx.backend.expr.state import State


def sort_ctcs_constraints(
    constraints_ctcs: List[CTCS],
) -> Tuple[List[CTCS], List[Tuple[int, int]], int]:
    """
    Sort and group CTCS constraints by their idx, ensuring proper grouping rules.

    Args:
        constraints_ctcs: List of CTCS constraints to sort and group

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
    constraints: List[Expr], n_nodes: int
) -> Tuple[List[CTCS], List[NodalConstraint], List[NodalConstraint]]:
    """
    Separate CTCS constraints from regular constraints, converting bare constraints
    to NodalConstraints that apply at all nodes. Also separate convex constraints.

    Args:
        constraints: List of constraints (mix of CTCS, NodalConstraint, and bare Constraints)
        n_nodes: Total number of nodes in the trajectory

    Returns:
        Tuple of:
        - List of CTCS constraints
        - List of non-convex NodalConstraint objects (including converted bare constraints)
        - List of convex NodalConstraint objects
    """
    constraints_ctcs: List[CTCS] = []
    constraints_nodal: List[NodalConstraint] = []
    constraints_nodal_convex: List[NodalConstraint] = []

    for c in constraints:
        if isinstance(c, CTCS):
            # Normalize None to full horizon
            c.nodes = c.nodes or (0, n_nodes)
            constraints_ctcs.append(c)
        elif isinstance(c, NodalConstraint):
            # Check if the underlying constraint is convex
            if c.constraint.is_convex:
                constraints_nodal_convex.append(c)
            else:
                constraints_nodal.append(c)
        elif isinstance(c, Constraint):
            # Convert bare constraint to NodalConstraint that applies at all nodes
            all_nodes = list(range(n_nodes))
            nodal_constraint = NodalConstraint(c, all_nodes)

            # Check if the constraint is convex
            if c.is_convex:
                constraints_nodal_convex.append(nodal_constraint)
            else:
                constraints_nodal.append(nodal_constraint)
        else:
            raise ValueError(
                f"Constraints must be `Constraint`, `NodalConstraint`, or `CTCS`, got {type(c).__name__}"
            )

    # Add nodal constraints from CTCS constraints that have check_nodally=True
    ctcs_nodal_constraints = get_nodal_constraints_from_ctcs(constraints_ctcs)
    for constraint in ctcs_nodal_constraints:
        # These also need to be converted to NodalConstraint (apply at all nodes)
        all_nodes = list(range(n_nodes))
        nodal_constraint = NodalConstraint(constraint, all_nodes)

        # Check if the underlying constraint is convex
        if constraint.is_convex:
            constraints_nodal_convex.append(nodal_constraint)
        else:
            constraints_nodal.append(nodal_constraint)

    return constraints_ctcs, constraints_nodal, constraints_nodal_convex


def decompose_vector_nodal_constraints(
    constraints_nodal: List[NodalConstraint],
) -> List[NodalConstraint]:
    """
    Decompose vector-valued nodal constraints into multiple scalar constraints.

    This is necessary for nonconvex nodal constraints that get lowered to JAX functions,
    because the JAX->CVXPY interface expects scalar constraint values per node.

    CTCS constraints and future convex nodal constraints can handle vector values,
    so they don't need decomposition.

    Args:
        constraints_nodal: List of NodalConstraint objects (already canonicalized)

    Returns:
        List of NodalConstraint objects with vector constraints decomposed into scalars
    """
    decomposed_constraints = []

    for nodal_constraint in constraints_nodal:
        constraint = nodal_constraint.constraint
        nodes = nodal_constraint.nodes

        try:
            # Get the shape of the constraint residual
            # Canonicalized constraints are in form: residual <= 0 or residual == 0
            residual_shape = constraint.lhs.check_shape()

            # Check if this is a vector constraint
            if len(residual_shape) > 0 and np.prod(residual_shape) > 1:
                # Vector constraint - decompose into scalar constraints
                total_elements = int(np.prod(residual_shape))

                for i in range(total_elements):
                    # Create indexed version: residual[i] <= 0 or residual[i] == 0
                    indexed_lhs = Index(constraint.lhs, i)
                    indexed_rhs = constraint.rhs  # Should be Constant(0)
                    indexed_constraint = constraint.__class__(indexed_lhs, indexed_rhs)
                    decomposed_constraints.append(NodalConstraint(indexed_constraint, nodes))
            else:
                # Scalar constraint - keep as is
                decomposed_constraints.append(nodal_constraint)

        except Exception:
            # If shape analysis fails, keep original constraint for backward compatibility
            decomposed_constraints.append(nodal_constraint)

    return decomposed_constraints


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


def augment_with_time_state(
    states: List[State],
    constraints: List[Constraint],
    time_initial: float | tuple,
    time_final: float | tuple,
    time_min: float,
    time_max: float,
    N: int,
) -> Tuple[List[State], List[Constraint], int]:
    """
    Augment the states list with a time state and add corresponding CTCS constraints.

    Args:
        states: List of State objects (will not be modified)
        constraints: List of constraints (will not be modified)
        time_initial: Initial time boundary condition (float or tuple like ("free", value))
        time_final: Final time boundary condition (float or tuple like ("free", value))
        time_min: Minimum bound for time variable
        time_max: Maximum bound for time variable
        N: Number of discretization nodes

    Returns:
        Tuple of:
        - Updated list of states (including time state appended at end)
        - Updated list of constraints (including time CTCS constraints)
        - Index where time state starts in the concatenated state vector
    """
    # Create copies to avoid mutating inputs
    states_aug = list(states)
    constraints_aug = list(constraints)

    # Calculate index where time will be inserted
    idx_time = sum(state.shape[0] for state in states)

    # Create time State
    time_state = State("time", shape=(1,))
    time_state.min = np.array([time_min])
    time_state.max = np.array([time_max])

    # Set time boundary conditions
    time_state.initial = [time_initial]
    time_state.final = [time_final]

    # Create initial guess for time (linear interpolation)
    time_guess_start = (
        time_state.initial[0]
        if isinstance(time_state.initial[0], (int, float))
        else time_state.initial[0][1]
    )
    time_guess_end = (
        time_state.final[0]
        if isinstance(time_state.final[0], (int, float))
        else time_state.final[0][1]
    )
    time_state.guess = np.linspace(time_guess_start, time_guess_end, N).reshape(-1, 1)

    # Add time state to the list
    states_aug.append(time_state)

    # Add CTCS constraints for time bounds
    constraints_aug.append(CTCS(time_state <= time_state.max))
    constraints_aug.append(CTCS(time_state.min <= time_state))

    return states_aug, constraints_aug, idx_time


def augment_dynamics_with_ctcs(
    xdot: Expr,
    states: List[State],
    controls: List[Control],
    constraints_ctcs: List[CTCS],
    N: int,
    licq_min=0.0,
    licq_max=1e-4,
    time_dilation_factor_min=0.3,
    time_dilation_factor_max=3.0,
) -> Tuple[Expr, List[State], List[Control]]:
    """
    Augment dynamics with continuous-time constraint satisfaction (CTCS).

    Args:
        xdot: The original dynamics expression
        states: The list of state variables (must include a state named "time")
        controls: The list of control variables
        constraints_ctcs: List of CTCS constraints
        N: Number of discretization nodes
        licq_min: Minimum value for LICQ augmented state
        licq_max: Maximum value for LICQ augmented state
        time_dilation_factor_min: Minimum time dilation factor
        time_dilation_factor_max: Maximum time dilation factor

    Returns:
        Tuple of:
        - Augmented dynamics expression
        - Updated list of states (including augmented states)
        - Updated list of controls (including time dilation)
    """
    # Copy the original states and controls lists
    states_augmented = list(states)
    controls_augmented = list(controls)

    if constraints_ctcs:
        # Group penalty expressions by idx (constraints should already be sorted)
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

        # Calculate number of augmented states from the penalty groups
        num_augmented_states = len(penalty_groups)

        # Create augmented state variables
        for idx in range(num_augmented_states):
            aug_var = State(f"_ctcs_aug_{idx}", shape=(1,))
            aug_var.initial = np.array([licq_min])  # Set initial to respect bounds
            aug_var.final = [("free", 0)]
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
    # Find the time state by name
    time_state = None
    for state in states:
        if state.name == "time":
            time_state = state
            break

    if time_state is None:
        raise ValueError("No state named 'time' found in states list")

    time_final = time_state.final[0]
    time_dilation.min = np.array([time_dilation_factor_min * time_final])
    time_dilation.max = np.array([time_dilation_factor_max * time_final])
    time_dilation.guess = np.ones([N, 1]) * time_final

    controls_augmented.append(time_dilation)

    return xdot_aug, states_augmented, controls_augmented
