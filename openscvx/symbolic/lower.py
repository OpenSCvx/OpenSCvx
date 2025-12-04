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
        lowered = lower_symbolic_problem(
            dynamics_aug, states_aug, controls_aug,
            constraints, parameters, N,
            dynamics_prop, states_prop, controls_prop
        )
        # Access via LoweredProblem dataclass
        dynamics = lowered.dynamics
        jax_constraints = lowered.jax_constraints
        # Now have executable JAX functions with Jacobians
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple, Union

import jax
from jax import jacfwd

from openscvx.constraints import ConstraintSet, CrossNodeConstraintLowered, LoweredNodalConstraint
from openscvx.dynamics import Dynamics
from openscvx.symbolic.expr import Expr, NodeReference
from openscvx.symbolic.unified import UnifiedControl, UnifiedState, unify_controls, unify_states

if TYPE_CHECKING:
    import cvxpy as cp

    from openscvx.symbolic.expr import CTCS


# ==================== LOWERED OUTPUT DATACLASSES ====================


@dataclass
class LoweredJaxConstraints:
    """JAX-lowered non-convex constraints with gradient functions.

    Contains constraints that have been lowered to JAX callable functions
    with automatically computed gradients. These are used for linearization
    in the SCP (Sequential Convex Programming) loop.

    Attributes:
        nodal: List of LoweredNodalConstraint objects. Each has `func`,
            `grad_g_x`, `grad_g_u` callables and `nodes` list.
        cross_node: List of CrossNodeConstraintLowered objects. Each has
            `func`, `grad_g_X`, `grad_g_U` for trajectory-level constraints.
        ctcs: CTCS constraints (unchanged from input, not lowered here).
    """

    nodal: List[LoweredNodalConstraint] = field(default_factory=list)
    cross_node: List[CrossNodeConstraintLowered] = field(default_factory=list)
    ctcs: List["CTCS"] = field(default_factory=list)


@dataclass
class LoweredCvxpyConstraints:
    """CVXPy-lowered convex constraints.

    Contains constraints that have been lowered to CVXPy constraint objects.
    These are added directly to the optimal control problem without
    linearization.

    Attributes:
        constraints: List of CVXPy constraint objects (cp.Constraint).
            Includes both nodal and cross-node convex constraints.
    """

    constraints: List["cp.Constraint"] = field(default_factory=list)


@dataclass
class LoweredProblem:
    """Container for all outputs from symbolic problem lowering.

    This dataclass holds all the results of lowering symbolic expressions
    to executable JAX and CVXPy code. It provides a clean, typed interface
    for accessing the various components needed for optimization.

    Attributes:
        dynamics: Optimization dynamics with fields f, A, B (JAX functions)
        dynamics_prop: Propagation dynamics with fields f, A, B
        jax_constraints: Non-convex constraints lowered to JAX with gradients
        cvxpy_constraints: Convex constraints lowered to CVXPy
        x_unified: Aggregated optimization state interface
        u_unified: Aggregated optimization control interface
        x_prop_unified: Aggregated propagation state interface
        ocp_vars: Dict of CVXPy variables and parameters for OCP construction
        cvxpy_params: Dict mapping user parameter names to CVXPy Parameter objects

    Example:
        After lowering a symbolic problem::

            lowered = lower_symbolic_problem(
                dynamics_aug=dynamics,
                states_aug=states,
                controls_aug=controls,
                constraints=constraint_set,
                parameters=params,
                N=50,
            )

            # Access components
            dx_dt = lowered.dynamics.f(x, u, node, params)
            jacobian_A = lowered.dynamics.A(x, u, node, params)

            # Use CVXPy objects
            ocp = OptimalControlProblem(settings, lowered.ocp_vars)
    """

    # JAX dynamics
    dynamics: Dynamics
    dynamics_prop: Dynamics

    # Lowered constraints (separate types for JAX vs CVXPy)
    jax_constraints: LoweredJaxConstraints
    cvxpy_constraints: LoweredCvxpyConstraints

    # Unified interfaces
    x_unified: UnifiedState
    u_unified: UnifiedControl
    x_prop_unified: UnifiedState

    # CVXPy objects
    ocp_vars: Dict[str, Any]
    cvxpy_params: Dict[str, "cp.Parameter"]


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


def lower_symbolic_problem(
    dynamics_aug,
    states_aug: List,
    controls_aug: List,
    constraints: ConstraintSet,
    parameters: dict,
    N: int,
    dynamics_prop=None,
    states_prop: List = None,
    controls_prop: List = None,
) -> LoweredProblem:
    """Lower symbolic problem specification to executable JAX and CVXPy code.

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

    Important:
        This function does NOT mutate the input `constraints` ConstraintSet. It returns
        new `LoweredJaxConstraints` and `LoweredCvxpyConstraints` objects containing the
        lowered versions.

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
            - nodal_convex: Convex nodal constraints (lowered to CVXPy)
            - cross_node: Non-convex cross-node constraints (lowered to JAX)
            - cross_node_convex: Convex cross-node constraints (lowered to CVXPy)
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
        LoweredProblem dataclass containing:
            - dynamics: Optimization dynamics (Dynamics with f, A, B)
            - dynamics_prop: Propagation dynamics (Dynamics with f, A, B)
            - jax_constraints: LoweredJaxConstraints with nodal, cross_node, ctcs
            - cvxpy_constraints: LoweredCvxpyConstraints with CVXPy constraint objects
            - x_unified: Aggregated optimization state interface
            - u_unified: Aggregated optimization control interface
            - x_prop_unified: Aggregated propagation state interface
            - ocp_vars: Dict of CVXPy variables for OCP construction
            - cvxpy_params: Dict of CVXPy Parameter objects for user parameters

    Example:
        Basic usage after problem construction::

            # After building symbolic problem and augmentation...
            lowered = lower_symbolic_problem(
                dynamics_aug=augmented_dynamics,
                states_aug=[x, y, z, ctcs_state],
                controls_aug=[u, ctcs_virtual],
                constraints=constraint_set,
                parameters={"obs_center": np.array([1.0, 0.0, 0.0])},
                N=50,
                dynamics_prop=augmented_dynamics_prop,
                states_prop=[x, y, z, ctcs_state],
                controls_prop=[u, ctcs_virtual],
            )

            # Access JAX-lowered constraints
            for c in lowered.jax_constraints.nodal:
                residual = c.func(x_batch, u_batch, node, params)

            # Access dynamics
            dx = lowered.dynamics.f(x_val, u_val, node=0, params={...})
            A_jac = lowered.dynamics.A(x_val, u_val, node=0, params={...})

            # Use CVXPy objects for OCP
            ocp = OptimalControlProblem(settings, lowered.ocp_vars)

    Note:
        **JAX Function Signature**: All JAX functions use a standardized signature
        (x, u, node, params) for uniformity, even if some arguments are unused.
        The node parameter allows for time-varying behavior (e.g., nodal constraints).
        The params dictionary provides runtime parameter updates without recompilation.

        **Immutability**: The input ConstraintSet is not modified. Lowered constraints
        are returned in separate LoweredJaxConstraints and LoweredCvxpyConstraints
        objects to maintain clear type separation.

    See Also:
        - LoweredProblem: The return type dataclass
        - LoweredJaxConstraints: Container for JAX-lowered non-convex constraints
        - LoweredCvxpyConstraints: Container for CVXPy-lowered convex constraints
        - lower_to_jax(): The underlying lowering function for individual expressions
        - lower_cvxpy_constraints(): CVXPy constraint lowering helper
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

    # Build lists for LoweredJaxConstraints (no mutation of input)
    lowered_nodal: List[LoweredNodalConstraint] = []
    lowered_cross_node: List[CrossNodeConstraintLowered] = []

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
            lowered_nodal.append(constraint)

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
        lowered_cross_node.append(cross_node_lowered)

    # Create LoweredJaxConstraints (immutable output)
    jax_constraints = LoweredJaxConstraints(
        nodal=lowered_nodal,
        cross_node=lowered_cross_node,
        ctcs=list(constraints.ctcs),  # Copy the list, don't reference original
    )

    # ==================== CREATE CVXPY VARIABLES AND LOWER CONVEX CONSTRAINTS ====================

    import numpy as np

    from openscvx.config import get_affine_scaling_matrices
    from openscvx.ocp import create_cvxpy_variables

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

    # Create all CVXPy variables for the OCP
    # Pass original constraints (for counting nodal/cross_node to create parameters)
    ocp_vars = create_cvxpy_variables(
        N=N,
        n_states=n_states,
        n_controls=n_controls,
        S_x=S_x,
        c_x=c_x,
        S_u=S_u,
        c_u=c_u,
        constraints=constraints,
    )

    # Lower convex constraints to CVXPy (from original symbolic constraints)
    lowered_cvxpy_constraint_list, cvxpy_params = lower_cvxpy_constraints(
        constraints,
        ocp_vars["x_nonscaled"],
        ocp_vars["u_nonscaled"],
        parameters,
    )

    # Create LoweredCvxpyConstraints (immutable output)
    cvxpy_constraints = LoweredCvxpyConstraints(
        constraints=lowered_cvxpy_constraint_list,
    )

    # ==================== RETURN LOWERED OUTPUTS ====================

    return LoweredProblem(
        dynamics=dynamics_augmented,
        dynamics_prop=dynamics_augmented_prop,
        jax_constraints=jax_constraints,
        cvxpy_constraints=cvxpy_constraints,
        x_unified=x_unified,
        u_unified=u_unified,
        x_prop_unified=x_prop_unified,
        ocp_vars=ocp_vars,
        cvxpy_params=cvxpy_params,
    )
