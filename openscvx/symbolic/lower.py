"""Symbolic expression lowering to executable code.

This module provides the main entry point for converting symbolic expressions
(AST nodes) into executable code for different backends (JAX, CVXPy, etc.).
The lowering process translates the symbolic expression tree into functions
that can be executed during optimization.

Architecture:
    The lowering process follows a visitor pattern where each backend implements
    a lowerer class (e.g., JaxLowerer, CVXPyLowerer) with visitor methods for
    each expression type. The `lower()` function dispatches expression nodes
    to the appropriate backend.

    Lowering Flow:

    1. Symbolic expressions are built during problem specification
    2. lower_symbolic_expressions() coordinates the full lowering process
    3. Backend-specific lowerers convert each expression node to executable code
    4. Automatic differentiation creates Jacobians for dynamics and constraints
    5. Result is a set of executable functions ready for numerical optimization

Backends:
    - JAX: For dynamics and non-convex constraints (with automatic differentiation)
    - CVXPy: For convex constraints (with disciplined convex programming)

Example:
    Basic lowering to JAX::

        import openscvx as ox
        from openscvx.symbolic.lower import lower_to_jax

        # Define symbolic expression
        x = ox.State("x", shape=(3,))
        u = ox.Control("u", shape=(2,))
        expr = ox.Norm(x)**2 + 0.1 * ox.Norm(u)**2

        # Lower to JAX function
        f = lower_to_jax(expr)
        # f is now a callable: f(x_val, u_val, node, params) -> scalar

    Full problem lowering::

        # After building symbolic problem...
        result = lower_symbolic_expressions(
            dynamics_aug, states_aug, controls_aug,
            constraints_nodal, constraints_nodal_convex,
            parameters, dynamics_prop, states_prop, controls_prop
        )
        dynamics, constraints, _, x, u, dynamics_prop, x_prop = result
        # Now have executable JAX functions with Jacobians
"""

from typing import Any, List, Sequence, Tuple, Union

import jax
from jax import jacfwd

from openscvx.constraints import ConstraintSet, CrossNodeConstraintLowered, LoweredNodalConstraint
from openscvx.dynamics import Dynamics
from openscvx.symbolic.expr import Expr, NodeReference
from openscvx.symbolic.unified import UnifiedControl, UnifiedState, unify_controls, unify_states


def lower(expr: Expr, lowerer: Any):
    """Dispatch an expression node to the appropriate lowerer backend.

    This is the main entry point for lowering a single symbolic expression to
    executable code. It delegates to the lowerer's `lower()` method, which
    uses the visitor pattern to dispatch based on expression type.

    Args:
        expr: Symbolic expression to lower (any Expr subclass)
        lowerer: Backend lowerer instance (e.g., JaxLowerer, CVXPyLowerer)

    Returns:
        Backend-specific representation of the expression. For JaxLowerer,
        returns a callable with signature (x, u, node, params) -> result.
        For CVXPyLowerer, returns a CVXPy expression object.

    Raises:
        NotImplementedError: If the lowerer doesn't support the expression type

    Example:
        Lower an expression to the appropriate backend (here JAX):

            from openscvx.symbolic.lowerers.jax import JaxLowerer
            x = ox.State("x", shape=(3,))
            expr = ox.Norm(x)
            lowerer = JaxLowerer()
            f = lower(expr, lowerer)

        f is now callable: f(x_val, u_val, node, params) -> scalar
    """
    return lowerer.lower(expr)


# --- Convenience wrappers for common backends ---


def lower_to_jax(exprs: Union[Expr, Sequence[Expr]]) -> Union[callable, list[callable]]:
    """Lower symbolic expression(s) to JAX callable(s).

    Convenience wrapper that creates a JaxLowerer and lowers one or more
    symbolic expressions to JAX functions. The resulting functions can be
    JIT-compiled and automatically differentiated.

    Args:
        exprs: Single expression or sequence of expressions to lower

    Returns:
        - If exprs is a single Expr: Returns a single callable with signature
          (x, u, node, params) -> array
        - If exprs is a sequence: Returns a list of callables with the same signature

    Example:
        Single expression::

            x = ox.State("x", shape=(3,))
            expr = ox.Norm(x)**2
            f = lower_to_jax(expr)
            # f(x_val, u_val, node_idx, params_dict) -> scalar

        Multiple expressions::

            exprs = [ox.Norm(x), ox.Norm(u), x @ A @ x]
            fns = lower_to_jax(exprs)
            # fns is [f1, f2, f3], each with same signature

    Note:
        All returned JAX functions have a uniform signature
        (x, u, node, params) regardless of whether they use all arguments.
        This standardization simplifies vectorization and differentiation.
    """
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    jl = JaxLowerer()
    if isinstance(exprs, Expr):
        return lower(exprs, jl)
    fns = [lower(e, jl) for e in exprs]
    return fns


def lower_cvxpy_constraints(
    constraints: ConstraintSet,
    x_cvxpy: List,
    u_cvxpy: List,
    parameters: dict = None,
) -> Tuple[List, dict]:
    """Lower symbolic convex constraints to CVXPy constraints.

    Converts symbolic convex constraint expressions to CVXPy constraint objects
    that can be used in the optimal control problem. This function handles both
    nodal constraints (applied at specific trajectory nodes) and cross-node
    constraints (relating multiple nodes).

    Args:
        constraints: ConstraintSet containing nodal_convex and cross_node_convex
        x_cvxpy: List of CVXPy expressions for state at each node (length N).
            Typically the x_nonscaled list from create_cvxpy_variables().
        u_cvxpy: List of CVXPy expressions for control at each node (length N).
            Typically the u_nonscaled list from create_cvxpy_variables().
        parameters: Optional dict of parameter values to use for any Parameter
            expressions in the constraints. If None, uses Parameter default values.

    Returns:
        Tuple of:
        - List of CVXPy constraint objects ready for the OCP
        - Dict mapping parameter names to their CVXPy Parameter objects

    Example:
        After creating CVXPy variables::

            ocp_vars = create_cvxpy_variables(settings)
            cvxpy_constraints, cvxpy_params = lower_cvxpy_constraints(
                constraint_set,
                ocp_vars["x_nonscaled"],
                ocp_vars["u_nonscaled"],
                parameters,
            )

    Note:
        This function only processes convex constraints (nodal_convex and
        cross_node_convex). Non-convex constraints are lowered to JAX in
        lower_symbolic_expressions() and handled via linearization in the SCP.
    """
    import cvxpy as cp

    from openscvx.symbolic.expr import Parameter, traverse
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State
    from openscvx.symbolic.lowerers.cvxpy import lower_to_cvxpy

    all_constraints = list(constraints.nodal_convex) + list(constraints.cross_node_convex)

    if not all_constraints:
        return [], {}

    # Collect all unique Parameters across all constraints and create cp.Parameter objects
    all_params = {}

    def collect_params(expr):
        if isinstance(expr, Parameter):
            if expr.name not in all_params:
                # Use value from params dict if provided, otherwise use Parameter's initial value
                if parameters and expr.name in parameters:
                    param_value = parameters[expr.name]
                else:
                    param_value = expr.value

                cvx_param = cp.Parameter(expr.shape, value=param_value, name=expr.name)
                all_params[expr.name] = cvx_param

    # Collect all parameters from all constraints
    for constraint in all_constraints:
        traverse(constraint.constraint, collect_params)

    cvxpy_constraints = []

    # Process nodal constraints
    for constraint in constraints.nodal_convex:
        # nodes should already be validated and normalized in preprocessing
        nodes = constraint.nodes

        # Collect all State and Control variables referenced in the constraint
        state_vars = {}
        control_vars = {}

        def collect_vars(expr):
            if isinstance(expr, State):
                state_vars[expr.name] = expr
            elif isinstance(expr, Control):
                control_vars[expr.name] = expr

        traverse(constraint.constraint, collect_vars)

        # Regular nodal constraint: apply at each specified node
        for node in nodes:
            # Create variable map for this specific node
            variable_map = {}

            if state_vars:
                variable_map["x"] = x_cvxpy[node]

            if control_vars:
                variable_map["u"] = u_cvxpy[node]

            # Add all CVXPy Parameter objects to the variable map
            variable_map.update(all_params)

            # Verify all variables have slices (should be guaranteed by preprocessing)
            for state_name, state_var in state_vars.items():
                if state_var._slice is None:
                    raise ValueError(
                        f"State variable '{state_name}' has no slice assigned. "
                        f"This indicates a bug in the preprocessing pipeline."
                    )

            for control_name, control_var in control_vars.items():
                if control_var._slice is None:
                    raise ValueError(
                        f"Control variable '{control_name}' has no slice assigned. "
                        f"This indicates a bug in the preprocessing pipeline."
                    )

            # Lower the constraint to CVXPy
            cvxpy_constraint = lower_to_cvxpy(constraint.constraint, variable_map)
            cvxpy_constraints.append(cvxpy_constraint)

    # Process cross-node constraints
    for constraint in constraints.cross_node_convex:
        # Collect all State and Control variables referenced in the constraint
        state_vars = {}
        control_vars = {}

        def collect_vars(expr):
            if isinstance(expr, State):
                state_vars[expr.name] = expr
            elif isinstance(expr, Control):
                control_vars[expr.name] = expr

        traverse(constraint.constraint, collect_vars)

        # Cross-node constraint: provide full trajectory
        variable_map = {}

        # Stack all nodes into (N, n_x) and (N, n_u) matrices
        if state_vars:
            variable_map["x"] = cp.vstack(x_cvxpy)

        if control_vars:
            variable_map["u"] = cp.vstack(u_cvxpy)

        # Add all CVXPy Parameter objects to the variable map
        variable_map.update(all_params)

        # Verify all variables have slices
        for state_name, state_var in state_vars.items():
            if state_var._slice is None:
                raise ValueError(
                    f"State variable '{state_name}' has no slice assigned. "
                    f"This indicates a bug in the preprocessing pipeline."
                )

        for control_name, control_var in control_vars.items():
            if control_var._slice is None:
                raise ValueError(
                    f"Control variable '{control_name}' has no slice assigned. "
                    f"This indicates a bug in the preprocessing pipeline."
                )

        # Lower the constraint once - NodeReference handles node indexing internally
        cvxpy_constraint = lower_to_cvxpy(constraint.constraint, variable_map)
        cvxpy_constraints.append(cvxpy_constraint)

    return cvxpy_constraints, all_params


def _create_cvxpy_trajectory_variables(
    N: int,
    x_unified: UnifiedState,
    u_unified: UnifiedControl,
) -> dict:
    """Create CVXPy trajectory variables for constraint lowering.

    Creates the minimal CVXPy variables needed to lower convex constraints:
    x, u and their non-scaled versions. This is called during lowering before
    the full OCP variable setup.

    Args:
        N: Number of discretization nodes
        x_unified: Unified state with bounds for scaling
        u_unified: Unified control with bounds for scaling

    Returns:
        Dict with keys: x, u, x_nonscaled, u_nonscaled, S_x, c_x, S_u, c_u
    """
    import cvxpy as cp
    import numpy as np

    from openscvx.config import get_affine_scaling_matrices

    n_states = len(x_unified.max)
    n_controls = len(u_unified.max)

    # Compute scaling matrices from unified object bounds
    # Use scaling_min/max if provided, otherwise use regular min/max
    if x_unified.scaling_min is not None:
        lower_x = np.array(x_unified.scaling_min, dtype=float)
    else:
        lower_x = np.array(x_unified.min, dtype=float)

    if x_unified.scaling_max is not None:
        upper_x = np.array(x_unified.scaling_max, dtype=float)
    else:
        upper_x = np.array(x_unified.max, dtype=float)

    S_x, c_x = get_affine_scaling_matrices(n_states, lower_x, upper_x)

    if u_unified.scaling_min is not None:
        lower_u = np.array(u_unified.scaling_min, dtype=float)
    else:
        lower_u = np.array(u_unified.min, dtype=float)

    if u_unified.scaling_max is not None:
        upper_u = np.array(u_unified.scaling_max, dtype=float)
    else:
        upper_u = np.array(u_unified.max, dtype=float)

    S_u, c_u = get_affine_scaling_matrices(n_controls, lower_u, upper_u)

    # Create CVXPy variables
    x = cp.Variable((N, n_states), name="x")
    u = cp.Variable((N, n_controls), name="u")

    # Create non-scaled versions (affine transformation)
    x_nonscaled = [S_x @ x[k] + c_x for k in range(N)]
    u_nonscaled = [S_u @ u[k] + c_u for k in range(N)]

    return {
        "x": x,
        "u": u,
        "x_nonscaled": x_nonscaled,
        "u_nonscaled": u_nonscaled,
        "S_x": S_x,
        "c_x": c_x,
        "S_u": S_u,
        "c_u": c_u,
        "n_states": n_states,
        "n_controls": n_controls,
    }


def _contains_node_reference(expr: Expr) -> bool:
    """Check if an expression contains any NodeReference nodes.

    Internal helper for routing constraints during lowering.

    Recursively traverses the expression tree to detect the presence of
    NodeReference nodes, which indicate cross-node constraints.

    Args:
        expr: Expression to check for NodeReference nodes

    Returns:
        True if the expression contains at least one NodeReference, False otherwise

    Example:
        position = State("pos", shape=(3,))

        # Regular expression - no NodeReference
        _contains_node_reference(position)  # False

        # Cross-node expression - has NodeReference
        _contains_node_reference(position.at(10) - position.at(9))  # True
    """
    if isinstance(expr, NodeReference):
        return True

    # Recursively check all children
    for child in expr.children():
        if _contains_node_reference(child):
            return True

    return False


def lower_symbolic_expressions(
    dynamics_aug,
    states_aug: List,
    controls_aug: List,
    constraints: ConstraintSet,
    parameters: dict,
    N: int,
    dynamics_prop=None,
    states_prop: List = None,
    controls_prop: List = None,
) -> Tuple:
    """Lower symbolic expressions to executable JAX and CVXPy code.

    This is the main orchestrator for converting symbolic problem specifications
    into executable numerical code. It coordinates the lowering of dynamics,
    constraints, and state/control interfaces from symbolic AST representations
    to JAX functions (with automatic differentiation) and CVXPy constraints.

    The function handles two separate dynamics systems:
        1. Optimization dynamics: Used in the SCP subproblem for trajectory optimization
        2. Propagation dynamics: Used for forward simulation and validation

    Lowering Process:
        1. **Unification**: Creates UnifiedState/UnifiedControl objects that aggregate
           multiple state/control variables into single vectors for efficient computation
        2. **Dynamics Lowering**: Converts symbolic dynamics to JAX functions and
           computes Jacobians A = df/dx and B = df/du using automatic differentiation
        3. **Non-Convex Constraint Lowering**: Lowers non-convex constraints to JAX
           with gradients for penalty-based handling in SCP
        4. **Propagation Setup**: Also lowers propagation dynamics (may differ from
           optimization dynamics if using different augmentation strategies)
        5. **CVXPy Lowering**: Creates CVXPy trajectory variables and lowers convex
           constraints to CVXPy constraint objects for the OCP

    This is pure translation - no validation, shape checking, or augmentation occurs
    here. Those steps happen earlier during problem construction.

    Args:
        dynamics_aug: Symbolic dynamics expression representing dx/dt = f(x, u).
            Should be augmented with any virtual controls or extra states needed
            for optimization (e.g., CTCS augmentation states).
        states_aug: List of State objects used in optimization. Includes original
            states plus any augmentation states (e.g., from CTCS).
        controls_aug: List of Control objects used in optimization. Includes original
            controls plus any virtual controls (e.g., from CTCS).
        constraints: ConstraintSet containing all constraint categories:
            - ctcs: CTCS (continuous-time) constraints
            - nodal: Non-convex nodal constraints (lowered to JAX)
            - nodal_convex: Convex nodal constraints (kept symbolic for CVXPy)
            - cross_node: Non-convex cross-node constraints (lowered to JAX)
            - cross_node_convex: Convex cross-node constraints (kept symbolic)
        parameters: Dictionary mapping parameter names to numpy arrays. Used to
            provide parameter values during function evaluation.
        N: Number of discretization nodes. Used for creating CVXPy variables.
        dynamics_prop: Symbolic propagation dynamics expression. May be the same as
            dynamics_aug or may include additional states for error tracking.
        states_prop: List of State objects for propagation. May include extras beyond
            states_aug (e.g., error states for monitoring).
        controls_prop: List of Control objects for propagation. Typically same as
            controls_aug.

    Returns:
        Tuple containing 9 elements:
            - dynamics_augmented (Dynamics): Optimization dynamics with fields:
                - f: JAX function (x, u, node, params) -> dx/dt
                - A: JAX function (x, u, node, params) -> df/dx Jacobian
                - B: JAX function (x, u, node, params) -> df/du Jacobian
            - lowered_constraints (ConstraintSet): Constraints with non-convex lowered to JAX:
                - ctcs: Unchanged (CTCS constraints)
                - nodal: LoweredNodalConstraint objects with JAX functions and gradients
                - nodal_convex: Lowered CVXPy constraints (list of cp.Constraint)
                - cross_node: CrossNodeConstraintLowered objects with JAX functions
                - cross_node_convex: Lowered CVXPy constraints (included in nodal_convex)
            - x_unified (UnifiedState): Aggregated optimization state interface
            - u_unified (UnifiedControl): Aggregated optimization control interface
            - dynamics_augmented_prop (Dynamics): Propagation dynamics with f, A, B
            - x_prop_unified (UnifiedState): Aggregated propagation state interface
            - cvxpy_trajectory_vars (dict): CVXPy variables for OCP construction:
                - x, u: CVXPy Variable objects
                - x_nonscaled, u_nonscaled: Affine-transformed trajectory lists
                - S_x, c_x, S_u, c_u: Scaling matrices
            - cvxpy_params (dict): CVXPy Parameter objects for user parameters

    Example:
        Basic usage after problem construction::

            # After building symbolic problem and augmentation...
            result = lower_symbolic_expressions(
                dynamics_aug=augmented_dynamics,
                states_aug=[x, y, z, ctcs_state],
                controls_aug=[u, ctcs_virtual],
                constraints=constraint_set,  # ConstraintSet from separate_constraints
                parameters={"obs_center": np.array([1.0, 0.0, 0.0])},
                dynamics_prop=augmented_dynamics_prop,
                states_prop=[x, y, z, ctcs_state],
                controls_prop=[u, ctcs_virtual],
            )

            # Unpack the results
            (dynamics_opt, lowered_constraints,
             x_unified, u_unified, dynamics_prop, x_prop) = result

            # Access lowered constraints via ConstraintSet
            for c in lowered_constraints.nodal:
                residual = c.func(x_batch, u_batch, node, params)

            # Now can evaluate dynamics at a specific point
            dx = dynamics_opt.f(x_val, u_val, node=0, params={...})
            A_jac = dynamics_opt.A(x_val, u_val, node=0, params={...})

    Note:
        **JAX Function Signature**: All JAX functions use a standardized signature
        (x, u, node, params) for uniformity, even if some arguments are unused.
        The node parameter allows for time-varying behavior (e.g., nodal constraints).
        The params dictionary provides runtime parameter updates without recompilation.

        **CVXPy Trajectory Variables**: This function creates CVXPy variables (x, u)
        and their affine-scaled versions (x_nonscaled, u_nonscaled) which are used
        both for constraint lowering and later for OCP construction. The scaling
        matrices are computed from the unified state/control bounds.

    See Also:
        - lower_to_jax(): The underlying lowering function for individual expressions
        - JaxLowerer: The visitor-pattern backend that implements JAX lowering
        - lower_cvxpy_constraints(): CVXPy constraint lowering helper
        - UnifiedState/UnifiedControl: Aggregation containers in symbolic/unified.py
        - Dynamics: Container for dynamics functions in dynamics.py
        - LoweredNodalConstraint: Container for constraint functions in constraints/lowered.py
    """

    # ==================== CREATE UNIFIED AGGREGATES ====================

    # Create unified state/control objects for optimization interface
    x_unified: UnifiedState = unify_states(states_aug)
    u_unified: UnifiedControl = unify_controls(controls_aug)

    # ==================== LOWER OPTIMIZATION DYNAMICS TO JAX ====================

    # Convert symbolic dynamics expression to JAX function
    dyn_fn = lower_to_jax(dynamics_aug)

    # Create Dynamics object with Jacobians computed via automatic differentiation
    dynamics_augmented = Dynamics(
        f=dyn_fn,
        A=jacfwd(dyn_fn, argnums=0),  # df/dx
        B=jacfwd(dyn_fn, argnums=1),  # df/du
    )

    # ==================== LOWER PROPAGATION DYNAMICS TO JAX ====================

    # Convert propagation dynamics (same as opt or with extras)
    dyn_fn_prop = lower_to_jax(dynamics_prop)

    # Create propagation Dynamics object
    dynamics_augmented_prop = Dynamics(
        f=dyn_fn_prop,
        A=jacfwd(dyn_fn_prop, argnums=0),
        B=jacfwd(dyn_fn_prop, argnums=1),
    )

    # Create unified propagation state object
    x_prop_unified: UnifiedState = unify_states(states_prop, name="x_prop")

    # ==================== LOWER NON-CONVEX CONSTRAINTS TO JAX ====================

    # Create result ConstraintSet - will hold lowered non-convex and original convex
    lowered_constraints = ConstraintSet()

    # Copy CTCS constraints unchanged
    lowered_constraints.ctcs = constraints.ctcs

    # Copy convex constraints unchanged (will be lowered to CVXPy later)
    lowered_constraints.nodal_convex = constraints.nodal_convex
    lowered_constraints.cross_node_convex = constraints.cross_node_convex

    # Lower regular nodal constraints (standard path)
    if len(constraints.nodal) > 0:
        # Convert symbolic constraint expressions to JAX functions
        constraints_nodal_fns = lower_to_jax(constraints.nodal)

        # Create LoweredConstraint objects with Jacobians computed automatically
        for i, fn in enumerate(constraints_nodal_fns):
            # Apply vectorization to handle (N, n_x) and (N, n_u) inputs
            # The lowered functions have signature (x, u, node, **kwargs)
            # node parameter is broadcast (same for all)
            constraint = LoweredNodalConstraint(
                func=jax.vmap(fn, in_axes=(0, 0, None, None)),
                grad_g_x=jax.vmap(jacfwd(fn, argnums=0), in_axes=(0, 0, None, None)),
                grad_g_u=jax.vmap(jacfwd(fn, argnums=1), in_axes=(0, 0, None, None)),
                nodes=constraints.nodal[i].nodes,
            )
            lowered_constraints.nodal.append(constraint)

    # Lower cross-node constraints (trajectory-level path)
    # The CrossNodeConstraint visitor in lowerers/jax.py handles wrapping the
    # inner constraint to provide trajectory-level signature (X, U, params) -> scalar
    for cross_node_constraint in constraints.cross_node:
        # Lower the CrossNodeConstraint directly - visitor handles wrapping
        # Returns function with signature (X, U, params) -> scalar
        constraint_fn = lower_to_jax(cross_node_constraint)

        # Compute Jacobians for the trajectory-level function
        grad_g_X = jacfwd(constraint_fn, argnums=0)  # dg/dX - shape (N, n_x)
        grad_g_U = jacfwd(constraint_fn, argnums=1)  # dg/dU - shape (N, n_u)

        # Create CrossNodeConstraintLowered object
        cross_node_lowered = CrossNodeConstraintLowered(
            func=constraint_fn,
            grad_g_X=grad_g_X,
            grad_g_U=grad_g_U,
        )
        lowered_constraints.cross_node.append(cross_node_lowered)

    # ==================== LOWER CONVEX CONSTRAINTS TO CVXPY ====================

    # Create CVXPy trajectory variables from unified objects
    cvxpy_traj_vars = _create_cvxpy_trajectory_variables(N, x_unified, u_unified)

    # Lower convex constraints to CVXPy
    lowered_cvxpy_constraints, cvxpy_params = lower_cvxpy_constraints(
        lowered_constraints,
        cvxpy_traj_vars["x_nonscaled"],
        cvxpy_traj_vars["u_nonscaled"],
        parameters,
    )

    # Store lowered CVXPy constraints in the ConstraintSet
    lowered_constraints.nodal_convex = lowered_cvxpy_constraints

    # ==================== RETURN LOWERED OUTPUTS ====================

    return (
        dynamics_augmented,
        lowered_constraints,
        x_unified,
        u_unified,
        dynamics_augmented_prop,
        x_prop_unified,
        cvxpy_traj_vars,
        cvxpy_params,
    )
