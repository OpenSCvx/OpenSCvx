"""Symbolic problem preprocessing and augmentation.

This module contains functions for processing and augmenting symbolic problem specifications
before lowering to executable code (JAX/CVXPy). The key function is `preprocess_symbolic_problem`,
which performs all symbolic manipulation without any code generation.
"""

from typing import Dict, List, Tuple, Union

from openscvx.symbolic.augmentation import (
    augment_dynamics_with_ctcs,
    augment_with_time_state,
    decompose_vector_nodal_constraints,
    separate_constraints,
    sort_ctcs_constraints,
)
from openscvx.symbolic.expr import CTCS, Constraint, Parameter, traverse
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.expr.state import State
from openscvx.symbolic.preprocessing import (
    collect_and_assign_slices,
    convert_dynamics_dict_to_expr,
    validate_and_normalize_constraint_nodes,
    validate_constraints_at_root,
    validate_dynamics_dict,
    validate_dynamics_dict_dimensions,
    validate_dynamics_dimension,
    validate_propagation_compatibility,
    validate_shapes,
    validate_time_parameters,
    validate_variable_names,
)
from openscvx.time import Time


def preprocess_symbolic_problem(
    dynamics: dict,
    constraints: List[Union[Constraint, CTCS]],
    states: List[State],
    controls: List[Control],
    N: int,
    time: Time,
    licq_min: float = 0.0,
    licq_max: float = 1e-4,
    time_dilation_factor_min: float = 0.3,
    time_dilation_factor_max: float = 3.0,
) -> Tuple:
    """Preprocess and augment symbolic problem specification.

    This function performs all symbolic preprocessing and augmentation steps:
    1. Time handling & validation
    2. Expression validation
    3. Canonicalization & parameter collection
    4. Constraint separation & CTCS augmentation

    The output is purely symbolic - no code generation occurs here.

    Args:
        dynamics: Dictionary mapping state names to their dynamics expressions
        constraints: List of constraints (CTCS or nodal)
        states: List of State objects
        controls: List of Control objects
        N: Number of segments in the trajectory
        time: Time configuration object
        licq_min: Minimum LICQ constraint value
        licq_max: Maximum LICQ constraint value
        time_dilation_factor_min: Minimum time dilation factor
        time_dilation_factor_max: Maximum time dilation factor

    Returns:
        Tuple containing:
            - dynamics_aug: Augmented dynamics expression (includes CTCS penalties)
            - states_aug: Augmented states list (user + time + CTCS augmented states)
            - controls_aug: Augmented controls list (user + time dilation)
            - constraints_ctcs: CTCS constraints (remain symbolic)
            - constraints_nodal: Non-convex nodal constraints (will be lowered to JAX)
            - constraints_nodal_convex: Convex nodal constraints (will be lowered to CVXPy)
            - parameters: Dictionary of parameter values from symbolic expressions
            - node_intervals: CTCS node intervals metadata

    Example:
        >>> result = preprocess_symbolic_problem(
        ...     dynamics={"x": u, "v": a},
        ...     constraints=[constraint1, constraint2],
        ...     states=[x_state, v_state],
        ...     controls=[a_control],
        ...     N=50,
        ...     time=Time(t_initial=0.0, t_final=10.0)
        ... )
        >>> dynamics_aug, states_aug, controls_aug, *_ = result
        >>> # Inspect augmented states
        >>> print(states_aug)  # [x, v, time, ctcs_aug_0, ...]
    """

    # ==================== PHASE 1: Time Handling & Validation ====================

    # Validate time handling approach and get processed parameters
    (
        has_time_state,
        time_initial,
        time_final,
        time_derivative,
        time_min,
        time_max,
    ) = validate_time_parameters(states, time)

    # Augment states with time state if needed (auto-create approach)
    if not has_time_state:
        states, constraints = augment_with_time_state(
            states, constraints, time_initial, time_final, time_min, time_max, N
        )

    # Add time derivative to dynamics dict (if not already present)
    # Time derivative is always 1.0 when using Time object
    dynamics = dict(dynamics)  # Make a copy to avoid mutating the input
    if "time" not in dynamics:
        dynamics["time"] = 1.0

    # Validate dynamics dict matches state names and dimensions
    validate_dynamics_dict(dynamics, states)
    validate_dynamics_dict_dimensions(dynamics, states)

    # Convert dynamics dict to concatenated expression
    dynamics, dynamics_concat = convert_dynamics_dict_to_expr(dynamics, states)

    # ==================== PHASE 2: Expression Validation ====================

    # Validate all expressions
    all_exprs = [dynamics_concat] + constraints
    validate_variable_names(all_exprs)
    collect_and_assign_slices(states, controls)
    validate_shapes(all_exprs)
    validate_constraints_at_root(constraints)
    validate_and_normalize_constraint_nodes(constraints, N)
    validate_dynamics_dimension(dynamics_concat, states)

    # ==================== PHASE 3: Canonicalization & Parameter Collection ====================

    # Canonicalize all expressions after validation
    dynamics_concat = dynamics_concat.canonicalize()
    constraints = [expr.canonicalize() for expr in constraints]

    # Collect parameter values from all constraints and dynamics
    parameters = {}

    def collect_param_values(expr):
        if isinstance(expr, Parameter):
            if expr.name not in parameters:
                parameters[expr.name] = expr.value

    # Collect from dynamics
    traverse(dynamics_concat, collect_param_values)

    # Collect from constraints
    for constraint in constraints:
        traverse(constraint, collect_param_values)

    # ==================== PHASE 4: Constraint Separation & Augmentation ====================

    # Sort and separate constraints by type
    (
        constraints_ctcs,
        constraints_nodal,
        constraints_nodal_convex,
    ) = separate_constraints(constraints, N)

    # Decompose vector-valued nodal constraints into scalar constraints
    # This is necessary for non-convex nodal constraints that get lowered to JAX
    constraints_nodal = decompose_vector_nodal_constraints(constraints_nodal)

    # Sort CTCS constraints by their idx to get node_intervals
    constraints_ctcs, node_intervals, _ = sort_ctcs_constraints(constraints_ctcs)

    # Augment dynamics, states, and controls with CTCS constraints, time dilation
    dynamics_aug, states_aug, controls_aug = augment_dynamics_with_ctcs(
        dynamics_concat,
        states,
        controls,
        constraints_ctcs,
        N,
        licq_min=licq_min,
        licq_max=licq_max,
        time_dilation_factor_min=time_dilation_factor_min,
        time_dilation_factor_max=time_dilation_factor_max,
    )

    # Assign slices to augmented states and controls in canonical order
    collect_and_assign_slices(states_aug, controls_aug)

    # ==================== Return Symbolic Outputs ====================

    return (
        dynamics_aug,
        states_aug,
        controls_aug,
        constraints_ctcs,
        constraints_nodal,
        constraints_nodal_convex,
        parameters,
        node_intervals,
    )


def preprocess_propagation_dynamics(
    dynamics_prop: dict,
    states_prop: List[State],
    states_opt: List[State],
    controls: List[Control],
    time_state: State,
    constraints_ctcs: List[CTCS],
    parameters: Dict[str, any],
    N: int,
    licq_min: float = 0.0,
    licq_max: float = 1e-4,
    time_dilation_factor_min: float = 0.3,
    time_dilation_factor_max: float = 3.0,
) -> Tuple:
    """Preprocess propagation dynamics with CTCS augmentation.

    Propagation dynamics are used for post-solution trajectory propagation.
    They allow tracking additional states beyond optimization states while
    maintaining the same augmentation structure (CTCS states) for consistency.

    State ordering in propagation:
    1. True states (optimization states + extra propagation states)
    2. Time state (if not already included)
    3. CTCS augmented states (same as optimization)
    4. Time dilation control (same as optimization)

    Args:
        dynamics_prop: Dictionary mapping state names to their dynamics expressions
        states_prop: List of State objects for propagation (superset of optimization states)
        controls: List of Control objects (same as optimization controls)
        time_state: The time State object from the optimization problem
        constraints_ctcs: List of CTCS constraints from optimization (for augmentation)
        parameters: Dictionary of parameter values (from optimization preprocessing)
        N: Number of segments in the trajectory
        licq_min: Minimum LICQ constraint value
        licq_max: Maximum LICQ constraint value
        time_dilation_factor_min: Minimum time dilation factor
        time_dilation_factor_max: Maximum time dilation factor

    Returns:
        Tuple containing:
            - dynamics_prop_aug: Augmented propagation dynamics expression
            - states_prop_aug: Augmented propagation states (true + time + ctcs aug)
            - controls_aug: Augmented controls (includes time dilation)
            - parameters_updated: Updated parameters dict

    Raises:
        ValueError: If validation fails

    Example:
        >>> dyn_prop, states_prop_aug, controls_aug, params = preprocess_propagation_dynamics(
        ...     dynamics_prop={"x": u, "v": a, "distance": speed},
        ...     states_prop=[x_state, v_state, distance_state],
        ...     controls=[a_control, speed_control],
        ...     time_state=time_state,
        ...     constraints_ctcs=constraints_ctcs,
        ...     parameters=parameters,
        ...     N=50
        ... )
    """

    # ==================== PHASE 1: State Reordering & Validation ====================

    # Make copies to avoid mutating inputs
    states_prop_input = list(states_prop)
    dynamics_prop = dict(dynamics_prop)
    parameters = dict(parameters)

    # Separate optimization states from extra propagation states
    # Extra propagation states will be added AFTER augmentation
    opt_state_names = {
        s.name for s in states_opt if s.name != "time"
    }  # Original opt states without time
    opt_state_names.add("time")  # Add time to optimization states

    # Partition states: optimization states vs extra propagation-only states
    states_opt_overlap = []
    states_prop_extra = []
    for state in states_prop_input:
        if state.name in opt_state_names:
            states_opt_overlap.append(state)
        else:
            states_prop_extra.append(state)

    # Ensure time state is included
    has_time = any(state.name == "time" for state in states_opt_overlap)
    if not has_time:
        states_opt_overlap.append(time_state)

    # Build dynamics for optimization states only (extra prop states added after augmentation)
    dynamics_opt_overlap = {}
    for s in states_opt_overlap:
        if s.name in dynamics_prop:
            dynamics_opt_overlap[s.name] = dynamics_prop[s.name]

    # Add time derivative if not present
    if "time" not in dynamics_opt_overlap:
        dynamics_opt_overlap["time"] = 1.0

    # Validate dynamics dict (for optimization overlap part)
    validate_dynamics_dict(dynamics_opt_overlap, states_opt_overlap)
    validate_dynamics_dict_dimensions(dynamics_opt_overlap, states_opt_overlap)

    # Convert dynamics dict to concatenated expression (optimization overlap only)
    dynamics_opt_overlap, dynamics_opt_concat = convert_dynamics_dict_to_expr(
        dynamics_opt_overlap, states_opt_overlap
    )

    # ==================== PHASE 2: Expression Validation (Optimization Overlap) ====================

    # Validate variable names and shapes for optimization overlap
    validate_variable_names([dynamics_opt_concat])
    collect_and_assign_slices(states_opt_overlap, controls)
    validate_shapes([dynamics_opt_concat])
    validate_dynamics_dimension(dynamics_opt_concat, states_opt_overlap)

    # ==================== PHASE 3: Canonicalization & Parameter Collection ====================

    # Canonicalize dynamics expression
    dynamics_opt_concat = dynamics_opt_concat.canonicalize()

    # Collect any new parameter values from propagation dynamics
    def collect_param_values(expr):
        if isinstance(expr, Parameter):
            if expr.name not in parameters:
                parameters[expr.name] = expr.value

    traverse(dynamics_opt_concat, collect_param_values)

    # ==================== PHASE 4: CTCS Augmentation ====================

    # Apply CTCS augmentation to optimization overlap states
    # Ordering after this: {opt states, time, ctcs aug states}
    dynamics_opt_aug, states_opt_aug, controls_prop_aug = augment_dynamics_with_ctcs(
        dynamics_opt_concat,
        states_opt_overlap,
        controls,
        constraints_ctcs,
        N,
        licq_min=licq_min,
        licq_max=licq_max,
        time_dilation_factor_min=time_dilation_factor_min,
        time_dilation_factor_max=time_dilation_factor_max,
    )

    # ==================== PHASE 5: Add Extra Propagation States ====================

    # Now add extra propagation-only states AFTER augmentation
    # Final ordering: {opt states, time, ctcs aug states, extra prop states}
    if states_prop_extra:
        # Process extra propagation state dynamics
        dynamics_extra = {s.name: dynamics_prop[s.name] for s in states_prop_extra}
        validate_dynamics_dict(dynamics_extra, states_prop_extra)
        validate_dynamics_dict_dimensions(dynamics_extra, states_prop_extra)

        # Convert extra dynamics to expression
        _, dynamics_extra_concat = convert_dynamics_dict_to_expr(dynamics_extra, states_prop_extra)

        # Validate and canonicalize
        validate_variable_names([dynamics_extra_concat])
        collect_and_assign_slices(states_prop_extra, controls)
        validate_shapes([dynamics_extra_concat])
        validate_dynamics_dimension(dynamics_extra_concat, states_prop_extra)
        dynamics_extra_concat = dynamics_extra_concat.canonicalize()

        # Collect parameters from extra dynamics
        traverse(dynamics_extra_concat, collect_param_values)

        # Concatenate: {aug dynamics, extra dynamics}
        from openscvx.symbolic.expr import Concat

        dynamics_prop_aug = Concat(dynamics_opt_aug, dynamics_extra_concat)

        # Clear slices from extra states before appending (they were assigned earlier)
        for state in states_prop_extra:
            state._slice = None

        # Append extra states to augmented states
        states_prop_aug = states_opt_aug + states_prop_extra
    else:
        # No extra states, just use augmented optimization states
        dynamics_prop_aug = dynamics_opt_aug
        states_prop_aug = states_opt_aug

    # Assign slices to final propagation states and controls
    collect_and_assign_slices(states_prop_aug, controls_prop_aug)

    # ==================== Return Symbolic Outputs ====================

    return (
        dynamics_prop_aug,
        states_prop_aug,
        controls_prop_aug,
        parameters,
    )
