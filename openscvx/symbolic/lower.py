from typing import Any, List, Sequence, Tuple, Union

import jax
from jax import jacfwd

from openscvx.constraints.lowered import LoweredNodalConstraint
from openscvx.dynamics import Dynamics
from openscvx.symbolic.expr import Expr
from openscvx.symbolic.unified import UnifiedControl, UnifiedState, unify_controls, unify_states


def lower(expr: Expr, lowerer: Any):
    """
    Dispatch an Expr node to the appropriate visit_* method on the lowerer.
    """
    return lowerer.lower(expr)


# --- Convenience wrappers for common backends ---


def lower_to_jax(exprs: Union[Expr, Sequence[Expr]]) -> Union[callable, list[callable]]:
    """
    If `exprs` is a single Expr, returns a single callable.
    Otherwise (a list/tuple of Exprs), returns a list of callables.
    """
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    jl = JaxLowerer()
    if isinstance(exprs, Expr):
        return lower(exprs, jl)
    fns = [lower(e, jl) for e in exprs]
    return fns


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

    This function converts symbolic expressions to executable code:
    1. Creates unified state/control objects
    2. Lowers dynamics to JAX with Jacobians
    3. Lowers non-convex nodal constraints to JAX with Jacobians
    4. Keeps convex constraints symbolic (for later CVXPy lowering)
    5. Sets up parameter storage
    6. Also lowers propagation dynamics

    This is pure translation - no validation or augmentation occurs here.

    Args:
        dynamics_aug: Augmented optimization dynamics expression (symbolic)
        states_aug: Augmented optimization states list
        controls_aug: Augmented optimization controls list
        constraints_nodal: Non-convex nodal constraints (symbolic)
        constraints_nodal_convex: Convex nodal constraints (symbolic, pass-through)
        parameters: Dictionary of parameter values
        dynamics_prop: Augmented propagation dynamics expression (symbolic)
        states_prop: Augmented propagation states list (may include extras)
        controls_prop: Augmented propagation controls list

    Returns:
        Tuple containing:
            - dynamics_augmented: Dynamics object with f, A, B Jacobians (opt)
            - lowered_constraints_nodal: List of LoweredNodalConstraint objects
            - constraints_nodal_convex: Convex constraints (unchanged, symbolic)
            - x_unified: Unified optimization state object
            - u_unified: Unified optimization control object
            - dynamics_augmented_prop: Propagation Dynamics object with f, A, B
            - x_prop_unified: Unified propagation state object

    Example:
        >>> result = lower_symbolic_expressions(
        ...     dynamics_aug, states_aug, controls_aug,
        ...     constraints_nodal, constraints_nodal_convex, parameters,
        ...     dynamics_prop, states_prop, controls_prop
        ... )
        >>> dynamics, constraints, _, x, u, dynamics_prop, x_prop = result
        >>> # Now have Dynamics objects with embedded JAX functions ready to use
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

    # Convert symbolic constraint expressions to JAX functions
    constraints_nodal_fns = lower_to_jax(constraints_nodal)

    # Create LoweredConstraint objects with Jacobians computed automatically
    lowered_constraints_nodal = []
    for i, fn in enumerate(constraints_nodal_fns):
        # Apply vectorization to handle (N, n_x) and (N, n_u) inputs
        # The lowered functions have signature (x, u, node, **kwargs)
        # node parameter is broadcast (same for all)
        constraint = LoweredNodalConstraint(
            func=jax.vmap(fn, in_axes=(0, 0, None, None)),
            grad_g_x=jax.vmap(jacfwd(fn, argnums=0), in_axes=(0, 0, None, None)),
            grad_g_u=jax.vmap(jacfwd(fn, argnums=1), in_axes=(0, 0, None, None)),
            nodes=constraints_nodal[i].nodes,
        )
        lowered_constraints_nodal.append(constraint)

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
        constraints_nodal_convex,
        x_unified,
        u_unified,
        dynamics_augmented_prop,
        x_prop_unified,
    )
