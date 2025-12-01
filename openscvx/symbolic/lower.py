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

from openscvx.constraints import CrossNodeConstraintLowered, LoweredNodalConstraint
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
        _contains_node_reference(position.node(10) - position.node(9))  # True
    """
    if isinstance(expr, NodeReference):
        return True

    # Recursively check all children
    for child in expr.children():
        if _contains_node_reference(child):
            return True

    return False


def _collect_node_references(expr: Expr) -> List[int]:
    """Collect all unique node indices referenced in a cross-node expression.

    Internal helper for analyzing which nodes are coupled by a constraint.

    This function analyzes the expression tree to identify all NodeReference nodes
    and extract their node indices. The results indicate the sparsity pattern
    (which nodes are coupled), though this is currently not exploited in Jacobian storage.

    Args:
        expr: Expression to analyze for NodeReference nodes

    Returns:
        Sorted list of unique node indices referenced in the expression

    Example:
        For expression `position.node(5) - position.node(4)`:
        Returns [4, 5]

        For expression `state.node(k) + state.node(k-1) + state.node(k-2)`:
        Returns [k-2, k-1, k] (whatever the actual integer values are)

    Note:
        The returned indices indicate which nodes are coupled by the constraint.
        This sparsity pattern could be used for sparse Jacobian storage in the future.
    """
    references = []

    def traverse(e: Expr):
        if isinstance(e, NodeReference):
            references.append(e.node_idx)
        for child in e.children():
            traverse(child)

    traverse(expr)
    return sorted(set(references))


def _create_cross_node_wrapper(constraint_fn, references: List[int], eval_nodes: List[int]):
    """Create a trajectory-level wrapper for cross-node constraint evaluation.

    Internal helper for generating trajectory-level constraint functions during lowering.

    Takes a constraint function lowered with JaxLowerer and wraps it to evaluate
    the constraint at multiple nodes along the trajectory using vmap.

    Args:
        constraint_fn: Lowered constraint function with signature (X, U, node, params)
        references: List of node indices referenced (for documentation/future sparsity)
        eval_nodes: Nodes where the constraint should be evaluated

    Returns:
        Function with signature (X, U, params) -> residuals (M,)
            where X is (N, n_x), U is (N, n_u), M = len(eval_nodes)

    Example:
        For constraint `position.node(k) - position.node(k-1) <= 0.1`:
        - At eval_nodes=[1,2,3,...,N-1]
        - Evaluates position[k] - position[k-1] for each k in eval_nodes
        - Returns array of shape (N-1,) with one residual per evaluation

    Note:
        The returned function will have Jacobians computed via jax.jacfwd, producing
        dense (M, N, n_x) and (M, N, n_u) arrays. See CrossNodeConstraintLowered for
        performance implications.
    """
    import jax
    import jax.numpy as jnp

    # Convert eval_nodes to JAX array for vmapping
    eval_nodes_array = jnp.array(eval_nodes)

    # Create a vmapped version that batches over evaluation nodes
    # in_axes: (None, None, 0, None) means:
    #   - X: broadcast (same trajectory for all evaluations)
    #   - U: broadcast (same trajectory for all evaluations)
    #   - node: batched (different evaluation node for each)
    #   - params: broadcast (same parameters for all evaluations)
    vmapped_constraint = jax.vmap(constraint_fn, in_axes=(None, None, 0, None))

    def trajectory_constraint(X, U, params):
        # Evaluate constraint at all specified nodes simultaneously
        return vmapped_constraint(X, U, eval_nodes_array, params)

    return trajectory_constraint


def lower_symbolic_expressions(
    dynamics_aug,
    states_aug: List,
    controls_aug: List,
    constraints_nodal: List,
    constraints_nodal_convex: List,
    parameters: dict,
    dynamics_prop=None,
    states_prop: List = None,
    controls_prop: List = None,
) -> Tuple:
    """Lower symbolic expressions to executable JAX functions.

    This is the main orchestrator for converting symbolic problem specifications
    into executable numerical code. It coordinates the lowering of dynamics,
    constraints, and state/control interfaces from symbolic AST representations
    to JAX functions with automatic differentiation.

    The function handles two separate dynamics systems:
        1. Optimization dynamics: Used in the SCP subproblem for trajectory optimization
        2. Propagation dynamics: Used for forward simulation and validation

    Lowering Process:
        1. **Unification**: Creates UnifiedState/UnifiedControl objects that aggregate
           multiple state/control variables into single vectors for efficient computation
        2. **Dynamics Lowering**: Converts symbolic dynamics to JAX functions and
           computes Jacobians A = df/dx and B = df/du using automatic differentiation
        3. **Constraint Lowering**: Separates convex and non-convex constraints:
           - Non-convex: Lowered to JAX with gradients for penalty-based handling
           - Convex: Kept symbolic for later CVXPy lowering (see Note below)
        4. **Propagation Setup**: Also lowers propagation dynamics (may differ from
           optimization dynamics if using different augmentation strategies)

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
        constraints_nodal: List of NodalConstraint objects with non-convex constraint
            expressions. These will be lowered to JAX with gradients for SCP.
        constraints_nodal_convex: List of NodalConstraint objects with convex
            constraint expressions. These remain symbolic for CVXPy lowering.
        parameters: Dictionary mapping parameter names to numpy arrays. Used to
            provide parameter values during function evaluation.
        dynamics_prop: Symbolic propagation dynamics expression. May be the same as
            dynamics_aug or may include additional states for error tracking.
        states_prop: List of State objects for propagation. May include extras beyond
            states_aug (e.g., error states for monitoring).
        controls_prop: List of Control objects for propagation. Typically same as
            controls_aug.

    Returns:
        Tuple containing 8 elements:
            - dynamics_augmented (Dynamics): Optimization dynamics with fields:
                - f: JAX function (x, u, node, params) -> dx/dt
                - A: JAX function (x, u, node, params) -> df/dx Jacobian
                - B: JAX function (x, u, node, params) -> df/du Jacobian
            - lowered_constraints_nodal (List[LoweredNodalConstraint]): Regular non-convex
              constraints as JAX functions with gradients:
                - func: Vectorized constraint evaluation (x_batch, u_batch, node, params)
                - grad_g_x: Jacobian wrt state
                - grad_g_u: Jacobian wrt control
                - nodes: List of node indices where constraint applies
            - lowered_cross_node_constraints (List[CrossNodeConstraintLowered]): Cross-node
              constraints that relate multiple trajectory nodes:
                - func: Trajectory-level function (X, U, params) -> residuals
                - grad_g_X: Jacobian wrt full state trajectory (M, N, n_x)
                - grad_g_U: Jacobian wrt full control trajectory (M, N, n_u)
                - eval_nodes: List of node indices where constraint is evaluated
            - constraints_nodal_convex (List[NodalConstraint]): Convex constraints
              (unchanged, still symbolic for later CVXPy lowering)
            - x_unified (UnifiedState): Aggregated optimization state interface
            - u_unified (UnifiedControl): Aggregated optimization control interface
            - dynamics_augmented_prop (Dynamics): Propagation dynamics with f, A, B
            - x_prop_unified (UnifiedState): Aggregated propagation state interface

    Example:
        Basic usage after problem construction::

            # After building symbolic problem and augmentation...
            result = lower_symbolic_expressions(
                dynamics_aug=augmented_dynamics,
                states_aug=[x, y, z, ctcs_state],
                controls_aug=[u, ctcs_virtual],
                constraints_nodal=nonconvex_constraints,
                constraints_nodal_convex=convex_constraints,
                parameters={"obs_center": np.array([1.0, 0.0, 0.0])},
                dynamics_prop=augmented_dynamics_prop,
                states_prop=[x, y, z, ctcs_state],
                controls_prop=[u, ctcs_virtual],
            )

            # Unpack the results
            (dynamics_opt, constraints_lowered, cross_node_constraints, constraints_cvx,
             x_unified, u_unified, dynamics_prop, x_prop) = result

            # Now can evaluate dynamics at a specific point
            dx = dynamics_opt.f(x_val, u_val, node=0, params={...})
            A_jac = dynamics_opt.A(x_val, u_val, node=0, params={...})

    Note:
        **JAX Function Signature**: All JAX functions use a standardized signature
        (x, u, node, params) for uniformity, even if some arguments are unused.
        The node parameter allows for time-varying behavior (e.g., nodal constraints).
        The params dictionary provides runtime parameter updates without recompilation.

        **CVXPy Lowering Deferred**: Convex constraints are NOT lowered to CVXPy in
        this function. They remain symbolic and are lowered later in the pipeline
        during TrajOptProblem.initialize() (see trajoptproblem.py:378-393). This is
        necessary because:
            1. CVXPy lowering requires CVXPy variables (x, u) which are created during
               initialize() by create_cvxpy_variables()
            2. Some SCP weights (lam_vc, lam_vb) are currently baked into the OCP cost
               at creation time rather than being CVXPy Parameters
            3. The OCP must be fully constructed before constraints can be lowered

        This architectural split means:
            - JAX lowering happens early (in __init__ via this function)
            - CVXPy lowering happens late (in initialize() via lower_convex_constraints())

        Future work will move CVXPy lowering here once all SCP weights become CVXPy
        Parameters instead of being baked into the cost function.

    See Also:
        - lower_to_jax(): The underlying lowering function for individual expressions
        - JaxLowerer: The visitor-pattern backend that implements JAX lowering
        - lower_convex_constraints(): CVXPy lowering in ocp.py (called during initialize())
        - TrajOptProblem.initialize(): Where CVXPy lowering actually occurs (trajoptproblem.py)
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

    # Separate cross-node constraints from regular nodal constraints
    regular_constraints = []
    cross_node_constraints = []

    for constraint in constraints_nodal:
        if _contains_node_reference(constraint.constraint):
            cross_node_constraints.append(constraint)
        else:
            regular_constraints.append(constraint)

    # Lower regular nodal constraints (standard path)
    lowered_constraints_nodal = []

    if len(regular_constraints) > 0:
        # Convert symbolic constraint expressions to JAX functions
        constraints_nodal_fns = lower_to_jax(regular_constraints)

        # Create LoweredConstraint objects with Jacobians computed automatically
        for i, fn in enumerate(constraints_nodal_fns):
            # Apply vectorization to handle (N, n_x) and (N, n_u) inputs
            # The lowered functions have signature (x, u, node, **kwargs)
            # node parameter is broadcast (same for all)
            constraint = LoweredNodalConstraint(
                func=jax.vmap(fn, in_axes=(0, 0, None, None)),
                grad_g_x=jax.vmap(jacfwd(fn, argnums=0), in_axes=(0, 0, None, None)),
                grad_g_u=jax.vmap(jacfwd(fn, argnums=1), in_axes=(0, 0, None, None)),
                nodes=regular_constraints[i].nodes,
            )
            lowered_constraints_nodal.append(constraint)

    # Lower cross-node constraints (trajectory-level path)
    lowered_cross_node_constraints = []

    for constraint_nodal in cross_node_constraints:
        # Lower the constraint expression to JAX
        constraint_expr = constraint_nodal.constraint
        constraint_fn = lower_to_jax(constraint_expr)

        # Collect node references (for documentation/future sparsity analysis)
        references = _collect_node_references(constraint_expr)

        # Create trajectory-level wrapper
        wrapped_fn = _create_cross_node_wrapper(constraint_fn, references, constraint_nodal.nodes)

        # Compute Jacobians for the wrapped trajectory-level function
        grad_g_X = jacfwd(wrapped_fn, argnums=0)  # dg/dX - shape (M, N, n_x)
        grad_g_U = jacfwd(wrapped_fn, argnums=1)  # dg/dU - shape (M, N, n_u)

        # Create CrossNodeConstraintLowered object
        cross_node_lowered = CrossNodeConstraintLowered(
            func=wrapped_fn,
            grad_g_X=grad_g_X,
            grad_g_U=grad_g_U,
            eval_nodes=constraint_nodal.nodes,
        )
        lowered_cross_node_constraints.append(cross_node_lowered)

    # ==================== KEEP CONVEX CONSTRAINTS SYMBOLIC ====================

    # TODO: Add CVXPy lowering here once SCP weights become CVXPy Parameters
    # Convex constraints remain symbolic and will be lowered to CVXPy
    # later in initialize() when CVXPy variables are available.
    # Once all SCP weights (lam_vc, lam_vb) are CVXPy Parameters instead of
    # being baked into the OCP cost, this function should handle CVXPy lowering
    # alongside JAX lowering for architectural consistency.
    # See docs/trajoptproblem_preprocessing_analysis.md for full analysis.

    # ==================== RETURN LOWERED OUTPUTS ====================

    return (
        dynamics_augmented,
        lowered_constraints_nodal,
        lowered_cross_node_constraints,
        constraints_nodal_convex,
        x_unified,
        u_unified,
        dynamics_augmented_prop,
        x_prop_unified,
    )
