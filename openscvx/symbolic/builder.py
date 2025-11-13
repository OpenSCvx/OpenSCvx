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
    dynamics_prop_extra: dict = None,
    states_prop_extra: List[State] = None,
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

    # ==================== PHASE 5: Create Propagation Dynamics ====================

    # By default, propagation dynamics are the same as optimization dynamics
    # Use deepcopy to avoid reference issues when lowering
    from copy import deepcopy

    dynamics_prop = deepcopy(dynamics_aug)
    states_prop = list(states_aug)  # Shallow copy of list is fine for states
    controls_prop = list(controls_aug)

    # If user provided extra propagation states, extend propagation dynamics
    if dynamics_prop_extra is not None and states_prop_extra is not None:
        (
            dynamics_prop,
            states_prop,
            controls_prop,
            parameters,
        ) = add_propagation_states(
            dynamics_extra=dynamics_prop_extra,
            states_extra=states_prop_extra,
            dynamics_opt=dynamics_prop,
            states_opt=states_prop,
            controls_opt=controls_prop,
            parameters=parameters,
        )

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
        dynamics_prop,
        states_prop,
        controls_prop,
    )


def add_propagation_states(
    dynamics_extra: dict,
    states_extra: List[State],
    dynamics_opt: any,
    states_opt: List[State],
    controls_opt: List[Control],
    parameters: Dict[str, any],
) -> Tuple:
    """Add extra propagation-only states to optimization dynamics.

    This function extends optimization dynamics with additional states that are only
    needed for post-solution trajectory propagation (e.g., distance traveled, energy used).

    The user only specifies the ADDITIONAL states and their dynamics. These are appended
    after the optimization states (including augmented CTCS states).

    Final state ordering:
    [opt_states, time, ctcs_aug_states, extra_prop_states]

    Args:
        dynamics_extra: Dictionary mapping extra state names to their dynamics expressions
            (only the new states, not the optimization states)
        states_extra: List of extra State objects for propagation only
        dynamics_opt: Augmented optimization dynamics expression
        states_opt: Augmented optimization states (includes time + CTCS aug states)
        controls_opt: Augmented optimization controls (includes time dilation)
        parameters: Dictionary of parameter values (from optimization preprocessing)

    Returns:
        Tuple containing:
            - dynamics_prop: Extended propagation dynamics expression
            - states_prop: Extended propagation states
            - controls_prop: Propagation controls (same as optimization)
            - parameters_updated: Updated parameters dict

    Raises:
        ValueError: If validation fails

    Example:
        >>> # User only specifies extra states
        >>> dynamics_extra = {"distance": speed[0]}
        >>> states_extra = [distance_state]
        >>> dyn_prop, states_prop, controls_prop, params = add_propagation_states(
        ...     dynamics_extra=dynamics_extra,
        ...     states_extra=states_extra,
        ...     dynamics_opt=dynamics_aug,
        ...     states_opt=states_aug,
        ...     controls_opt=controls_aug,
        ...     parameters=parameters
        ... )
    """

    # Make copies to avoid mutating inputs
    states_extra = list(states_extra)
    dynamics_extra = dict(dynamics_extra)
    parameters = dict(parameters)

    # ==================== PHASE 1: Validate Extra States ====================

    # Validate that extra states don't conflict with optimization state names
    opt_state_names = {s.name for s in states_opt}
    extra_state_names = {s.name for s in states_extra}
    conflicts = opt_state_names & extra_state_names
    if conflicts:
        raise ValueError(
            f"Extra propagation states conflict with optimization states: {conflicts}. "
            f"Only specify additional states, not optimization states."
        )

    # Validate dynamics dict for extra states
    validate_dynamics_dict(dynamics_extra, states_extra)
    validate_dynamics_dict_dimensions(dynamics_extra, states_extra)

    # ==================== PHASE 2: Process Extra Dynamics ====================

    # Convert extra dynamics to expression
    _, dynamics_extra_concat = convert_dynamics_dict_to_expr(dynamics_extra, states_extra)

    # Validate and canonicalize
    validate_variable_names([dynamics_extra_concat])

    # Temporarily assign slices for validation (will be recalculated below)
    collect_and_assign_slices(states_extra, controls_opt)
    validate_shapes([dynamics_extra_concat])
    validate_dynamics_dimension(dynamics_extra_concat, states_extra)
    dynamics_extra_concat = dynamics_extra_concat.canonicalize()

    # Collect any new parameter values from extra dynamics
    def collect_param_values(expr):
        if isinstance(expr, Parameter):
            if expr.name not in parameters:
                parameters[expr.name] = expr.value

    traverse(dynamics_extra_concat, collect_param_values)

    # ==================== PHASE 3: Concatenate with Optimization Dynamics ====================

    # Concatenate: {opt dynamics, extra dynamics}
    from openscvx.symbolic.expr import Concat

    dynamics_prop = Concat(dynamics_opt, dynamics_extra_concat)

    # Manually assign slices to extra states ONLY (don't modify optimization state slices)
    # Extra states are appended after all optimization states
    n_opt_states = states_opt[-1]._slice.stop if states_opt else 0
    start_idx = n_opt_states
    for state in states_extra:
        end_idx = start_idx + state.shape[0]
        state._slice = slice(start_idx, end_idx)
        start_idx = end_idx

    # Append extra states to optimization states
    states_prop = states_opt + states_extra

    # Propagation uses same controls as optimization
    controls_prop = controls_opt

    # ==================== Return Symbolic Outputs ====================

    return (
        dynamics_prop,
        states_prop,
        controls_prop,
        parameters,
    )
