from typing import Callable, Dict, Iterable, List, Set, Tuple, Union

import numpy as np

from openscvx.backend.expr import (
    CTCS,
    Concat,
    Constant,
    Constraint,
    Control,
    Expr,
    NodalConstraint,
    State,
    Variable,
    traverse,
)


def validate_shapes(exprs: Union[Expr, list[Expr]]) -> None:
    """Validate shapes for a single expression or list of expressions.

    Args:
        exprs: Single expression or list of expressions to validate

    Raises:
        ValueError: If any expression has invalid shapes
    """
    exprs = exprs if isinstance(exprs, (list, tuple)) else [exprs]
    for e in exprs:
        e.check_shape()  # will raise ValueError if anything's wrong


# TODO: (norrisg) allow `traverse` to take a list of visitors, that way we can combine steps
def validate_variable_names(
    exprs: Iterable[Expr], *, reserved_prefix: str = "_", reserved_names: Set[str] = None
) -> None:
    """
    1) Ensure all State/Control names are unique *across distinct variables*.
    2) Ensure no user‐supplied name starts with `reserved_prefix`.
    3) Ensure no name collides with `reserved_names` if given.
    Raises ValueError on any violation.
    """
    seen_names = set()
    seen_ids = set()
    reserved = set(reserved_names or ())

    def visitor(node):
        if not isinstance(node, (State, Control)):
            return

        node_id = id(node)
        if node_id in seen_ids:
            # we already checked this exact object
            return

        name = node.name

        # 1) uniqueness across *different* variables
        if name in seen_names:
            raise ValueError(f"Duplicate variable name: {name!r}")

        # 2) no leading underscore
        if name.startswith(reserved_prefix):
            raise ValueError(
                f"Variable name {name!r} is reserved (cannot start with {reserved_prefix!r})"
            )

        # 3) no collision with explicit reserved set
        if name in reserved:
            raise ValueError(f"Variable name {name!r} collides with reserved name")

        seen_names.add(name)
        seen_ids.add(node_id)

    for e in exprs:
        traverse(e, visitor)


def collect_and_assign_slices(
    states: List[State], controls: List[Control], *, start_index: int = 0
) -> Tuple[list[State], list[Control]]:
    """Assign slices to states and controls in the provided order.

    Args:
        states: List of State objects in canonical order
        controls: List of Control objects in canonical order
        start_index: Starting index for slice assignment (default 0)

    Returns:
        Tuple of (states, controls) with slices assigned
    """

    def assign(vars_list, start_index):
        # split into manual vs auto
        manual = [v for v in vars_list if v._slice is not None]
        auto = [v for v in vars_list if v._slice is None]

        if manual:
            # 1) shape‐match check
            for v in manual:
                dim = int(np.prod(v.shape))
                sl = v._slice
                if (sl.stop - sl.start) != dim:
                    raise ValueError(
                        f"Manual slice for {v.name!r} is length {sl.stop - sl.start}, "
                        f"but variable has shape {v.shape} (dim {dim})"
                    )
            # sort by the start of their slices
            manual.sort(key=lambda v: v._slice.start)
            # 2a) must start at 0
            if manual[0]._slice.start != start_index:
                raise ValueError("User-defined slices must start at index 0")
            # 2b) check contiguity & no overlaps
            cursor = start_index
            for v in manual:
                sl = v._slice
                dim = int(np.prod(v.shape))
                if sl.start != cursor or sl.stop != cursor + dim:
                    raise ValueError(
                        f"Manual slice for {v.name!r} must be contiguous and non-overlapping"
                    )
                cursor += dim
            offset = cursor
        else:
            offset = start_index

        # 3) auto-assign the rest
        for v in auto:
            dim = int(np.prod(v.shape))
            v._slice = slice(offset, offset + dim)
            offset += dim

    # run separately on states (x) and controls (u)
    assign(states, start_index)
    assign(controls, start_index)

    # Return the collected variables
    return states, controls


def _traverse_with_depth(expr: Expr, visit: Callable[[Expr, int], None], depth: int = 0):
    visit(expr, depth)
    for child in expr.children():
        _traverse_with_depth(child, visit, depth + 1)


def validate_constraints_at_root(exprs: Union[Expr, list[Expr]]):
    """
    Raise ValueError if any Constraint or constraint wrapper is found at depth>0.
    Both raw constraints and constraint wrappers (like CTCS, NodalConstraint) must only appear
    at the root level. However, constraints inside constraint wrappers are allowed
    (e.g., the constraint inside CTCS(x <= 5) is valid).

    Accepts a single Expr or a list of Exprs.
    """

    # Define constraint wrappers that must also be at root level
    CONSTRAINT_WRAPPERS = (CTCS, NodalConstraint)

    # normalize to list
    expr_list = exprs if isinstance(exprs, (list, tuple)) else [exprs]

    for expr in expr_list:

        def visit(node: Expr, depth: int):
            if depth > 0:
                if isinstance(node, CONSTRAINT_WRAPPERS):
                    raise ValueError(
                        f"Nested constraint wrapper found at depth {depth!r}: {node!r}; "
                        "constraint wrappers must only appear as top-level roots"
                    )
                elif isinstance(node, Constraint):
                    raise ValueError(
                        f"Nested Constraint found at depth {depth!r}: {node!r}; "
                        "constraints must only appear as top-level roots"
                    )

            # If this is a constraint wrapper, don't validate its children
            # (we allow constraints inside constraint wrappers)
            if isinstance(node, CONSTRAINT_WRAPPERS):
                return  # Skip traversing children

            # Otherwise, continue traversing children
            for child in node.children():
                visit(child, depth + 1)

        # Start traversal
        visit(expr, 0)


def validate_and_normalize_constraint_nodes(exprs: Union[Expr, list[Expr]], n_nodes: int):
    """
    Validate and normalize constraint nodes specifications.

    For NodalConstraint:
    - nodes should be a list of specific node indices: [2, 4, 6, 8]
    - Validates all nodes are within range

    For CTCS constraints:
    - nodes should be a tuple of (start, end): (0, 10)
    - None is replaced with (0, n_nodes)
    - Validation ensures tuple has exactly 2 elements and start < end

    Args:
        exprs: Single expression or list of expressions to validate
        n_nodes: Total number of nodes in the trajectory

    Raises:
        ValueError: If node specifications are invalid
    """

    # Normalize to list
    expr_list = exprs if isinstance(exprs, (list, tuple)) else [exprs]

    for expr in expr_list:
        if isinstance(expr, CTCS):
            # CTCS constraint validation (already done in __init__, but normalize None)
            if expr.nodes is None:
                expr.nodes = (0, n_nodes)
            elif expr.nodes[0] >= n_nodes or expr.nodes[1] > n_nodes:
                raise ValueError(
                    f"CTCS node range {expr.nodes} exceeds trajectory length {n_nodes}"
                )

        elif isinstance(expr, NodalConstraint):
            # NodalConstraint validation - nodes are already validated in __init__
            # Just need to check they're within trajectory range
            for node in expr.nodes:
                if node < 0 or node >= n_nodes:
                    raise ValueError(f"NodalConstraint node {node} is out of range [0, {n_nodes})")


def validate_dynamics_dimension(
    dynamics_expr: Union[Expr, list[Expr]], states: Union[State, list[State]]
) -> None:
    """
    Validate that dynamics expressions dimension(s) match the total dimension of the given states.

    Args:
        dynamics_expr: Single dynamics expression or list of dynamics expressions.
                      Combined, they should represent x_dot = f(x, u, t) for all states.
        states: Single state variable or list of state variables that the dynamics describe.

    Raises:
        ValueError: If dimensions don't match or if any dynamics is not a vector
    """
    # Normalize inputs to lists
    dynamics_list = dynamics_expr if isinstance(dynamics_expr, (list, tuple)) else [dynamics_expr]
    states_list = states if isinstance(states, (list, tuple)) else [states]

    # Calculate total state dimension
    total_state_dim = sum(int(np.prod(state.shape)) for state in states_list)

    # Validate each dynamics expression and calculate total dynamics dimension
    total_dynamics_dim = 0

    for i, dyn_expr in enumerate(dynamics_list):
        # Get the shape of this dynamics expression
        dynamics_shape = dyn_expr.check_shape()

        # Dynamics should be a 1D vector
        if len(dynamics_shape) != 1:
            prefix = f"Dynamics expression {i}" if len(dynamics_list) > 1 else "Dynamics expression"
            raise ValueError(
                f"{prefix} must be 1-dimensional (vector), but got shape {dynamics_shape}"
            )

        total_dynamics_dim += dynamics_shape[0]

    # Check that total dynamics dimension matches total state dimension
    if total_dynamics_dim != total_state_dim:
        if len(dynamics_list) == 1:
            raise ValueError(
                f"Dynamics dimension mismatch: dynamics has dimension {total_dynamics_dim}, "
                f"but total state dimension is {total_state_dim}. "
                f"States: {[(s.name, s.shape) for s in states_list]}"
            )
        else:
            dynamics_dims = [dyn.check_shape()[0] for dyn in dynamics_list]
            raise ValueError(
                f"Dynamics dimension mismatch: {len(dynamics_list)} dynamics expressions "
                f"have combined dimension {total_dynamics_dim} {dynamics_dims}, "
                f"but total state dimension is {total_state_dim}. "
                f"States: {[(s.name, s.shape) for s in states_list]}"
            )


def validate_dynamics_dict(dynamics: Dict[str, Expr], states: List[State]) -> None:
    """
    Validate that the dynamics dictionary keys match the state names exactly.

    Args:
        dynamics: Dictionary mapping state names to their dynamics expressions
        states: List of State objects

    Raises:
        ValueError: If there's a mismatch between state names and dynamics keys
    """
    state_names_set = set(state.name for state in states)
    dynamics_names_set = set(dynamics.keys())

    if dynamics_names_set != state_names_set:
        missing_in_dynamics = state_names_set - dynamics_names_set
        extra_in_dynamics = dynamics_names_set - state_names_set
        error_msg = "Mismatch between state names and dynamics keys.\n"
        if missing_in_dynamics:
            error_msg += f"  States missing from dynamics: {missing_in_dynamics}\n"
        if extra_in_dynamics:
            error_msg += f"  Extra keys in dynamics: {extra_in_dynamics}\n"
        raise ValueError(error_msg)


def validate_dynamics_dict_dimensions(dynamics: Dict[str, Expr], states: List[State]) -> None:
    """
    Validate that each dynamics expression dimension matches the corresponding state shape.

    Args:
        dynamics: Dictionary mapping state names to their dynamics expressions
        states: List of State objects

    Raises:
        ValueError: If any dynamics expression dimension doesn't match its state shape
    """
    for state in states:
        dyn_expr = dynamics[state.name]
        expected_shape = state.shape

        # Actually compute the shape of the dynamics expression
        actual_shape = dyn_expr.check_shape()

        if actual_shape != expected_shape:
            raise ValueError(
                f"Dynamics for state '{state.name}' has shape {actual_shape}, "
                f"but state has shape {expected_shape}"
            )


def convert_dynamics_dict_to_expr(
    dynamics: Dict[str, Expr], states: List[State]
) -> Tuple[Dict[str, Expr], Expr]:
    """
    Convert a dynamics dictionary to a concatenated Expr, ordered by the states list.
    Also converts scalar values to Constant expressions.

    Args:
        dynamics: Dictionary mapping state names to their dynamics expressions
        states: List of State objects defining the canonical order

    Returns:
        Tuple of:
        - Updated dynamics dictionary (with scalars converted to Constant)
        - Concatenated dynamics expression ordered by states list
    """
    # Create a copy to avoid mutating the input
    dynamics_converted = dict(dynamics)

    # Convert scalar values to Constant expressions
    for state_name, dyn_expr in dynamics_converted.items():
        if isinstance(dyn_expr, (int, float)):
            dynamics_converted[state_name] = Constant(dyn_expr)

    # Create concatenated expression ordered by states list
    dynamics_exprs = [dynamics_converted[state.name] for state in states]
    dynamics_concat = Concat(*dynamics_exprs)

    return dynamics_converted, dynamics_concat
