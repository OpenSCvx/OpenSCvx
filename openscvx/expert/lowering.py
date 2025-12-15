"""Lowering logic for bring-your-own-functions (byof).

This module handles integration of user-provided JAX functions into the
lowered problem representation, including dynamics splicing and constraint
addition.
"""

from typing import TYPE_CHECKING, List, Tuple

import jax
from jax import jacfwd

if TYPE_CHECKING:
    from openscvx.lowered.unified import UnifiedState
    from openscvx.symbolic.expr.state import State

from openscvx.lowered import (
    Dynamics,
    LoweredCrossNodeConstraint,
    LoweredJaxConstraints,
    LoweredNodalConstraint,
)

__all__ = ["apply_byof"]


def apply_byof(
    byof: dict,
    dynamics: Dynamics,
    dynamics_prop: Dynamics,
    jax_constraints: LoweredJaxConstraints,
    x_unified: "UnifiedState",
    x_prop_unified: "UnifiedState",
    u_unified: "UnifiedState",
    states: List["State"],
    states_prop: List["State"],
    N: int,
) -> Tuple[Dynamics, Dynamics, LoweredJaxConstraints, "UnifiedState", "UnifiedState"]:
    """Apply bring-your-own-functions (byof) to augment lowered problem.

    Handles raw JAX functions provided by expert users, including:
    - dynamics: Raw JAX functions for specific state derivatives
    - nodal_constraints: Point-wise constraints at each node
    - cross_nodal_constraints: Constraints coupling multiple nodes
    - ctcs_constraints: Continuous-time constraint satisfaction via dynamics augmentation

    Args:
        byof: Dict with keys "dynamics", "nodal_constraints", "cross_nodal_constraints",
            "ctcs_constraints"
        dynamics: Lowered optimization dynamics to potentially augment
        dynamics_prop: Lowered propagation dynamics to potentially augment
        jax_constraints: Lowered JAX constraints to append to
        x_unified: Unified optimization state interface to potentially augment
        x_prop_unified: Unified propagation state interface to potentially augment
        u_unified: Unified control interface for validation
        states: List of State objects for optimization (with _slice attributes)
        states_prop: List of State objects for propagation (with _slice attributes)
        N: Number of nodes in the trajectory

    Returns:
        Tuple of (dynamics, dynamics_prop, jax_constraints, x_unified, x_prop_unified)

    Example:
        >>> dynamics, dynamics_prop, constraints, x_unified, x_prop_unified = apply_byof(
        ...     byof, dynamics, dynamics_prop, jax_constraints,
        ...     x_unified, x_prop_unified, u_unified, states, states_prop, N
        ... )
    """
    import jax.numpy as jnp

    # Note: byof validation happens earlier in Problem.__init__ to fail fast
    # Handle byof dynamics by splicing in raw JAX functions at the correct slices
    byof_dynamics = byof.get("dynamics", {})
    if byof_dynamics:
        # Build mapping from state name to slice for optimization states
        state_slices = {state.name: state._slice for state in states}
        state_slices_prop = {state.name: state._slice for state in states_prop}

        def _make_composite_dynamics(orig_f, byof_fns, slices_map):
            """Create composite dynamics combining symbolic and byof state derivatives.

            This factory splices user-provided byof dynamics into the unified dynamics
            function at the appropriate slice indices, replacing the symbolic dynamics
            for specific states while preserving the rest.

            Args:
                orig_f: Original unified dynamics (x, u, node, params) -> xdot
                byof_fns: Dict mapping state names to byof dynamics functions
                slices_map: Dict mapping state names to slice objects for indexing

            Returns:
                Composite dynamics function with byof derivatives spliced in
            """
            def composite_f(x, u, node, params):
                # Start with symbolic/default dynamics for all states
                xdot = orig_f(x, u, node, params)

                # Splice in byof dynamics for specific states
                for state_name, byof_fn in byof_fns.items():
                    sl = slices_map[state_name]
                    # Replace the derivative for this state with the byof result
                    xdot = xdot.at[sl].set(byof_fn(x, u, node, params))

                return xdot

            return composite_f

        # Create composite optimization dynamics
        composite_f = _make_composite_dynamics(dynamics.f, byof_dynamics, state_slices)
        dynamics = Dynamics(
            f=composite_f,
            A=jacfwd(composite_f, argnums=0),
            B=jacfwd(composite_f, argnums=1),
        )

        # Create composite propagation dynamics
        composite_f_prop = _make_composite_dynamics(
            dynamics_prop.f, byof_dynamics, state_slices_prop
        )
        dynamics_prop = Dynamics(
            f=composite_f_prop,
            A=jacfwd(composite_f_prop, argnums=0),
            B=jacfwd(composite_f_prop, argnums=1),
        )

    # Handle nodal constraints
    for fn in byof.get("nodal_constraints", []):
        constraint = LoweredNodalConstraint(
            func=jax.vmap(fn, in_axes=(0, 0, None, None)),
            grad_g_x=jax.vmap(jacfwd(fn, argnums=0), in_axes=(0, 0, None, None)),
            grad_g_u=jax.vmap(jacfwd(fn, argnums=1), in_axes=(0, 0, None, None)),
            nodes=list(range(N)),  # Apply to all nodes
        )
        jax_constraints.nodal.append(constraint)

    # Handle cross-nodal constraints
    for fn in byof.get("cross_nodal_constraints", []):
        constraint = LoweredCrossNodeConstraint(
            func=fn,
            grad_g_X=jacfwd(fn, argnums=0),
            grad_g_U=jacfwd(fn, argnums=1),
        )
        jax_constraints.cross_node.append(constraint)

    # Handle CTCS constraints by augmenting dynamics
    # Built-in penalty functions
    def _penalty_square(r):
        return jnp.maximum(r, 0.0) ** 2

    def _penalty_l1(r):
        return jnp.maximum(r, 0.0)

    def _penalty_huber(r, delta=1.0):
        abs_r = jnp.maximum(r, 0.0)
        return jnp.where(abs_r <= delta, 0.5 * abs_r**2, delta * (abs_r - 0.5 * delta))

    _PENALTY_FUNCTIONS = {
        "square": _penalty_square,
        "l1": _penalty_l1,
        "huber": _penalty_huber,
    }

    for i, ctcs_spec in enumerate(byof.get("ctcs_constraints", [])):
        constraint_fn = ctcs_spec["constraint_fn"]
        bounds = ctcs_spec.get("bounds", (0.0, 1e-4))
        initial = ctcs_spec.get("initial", bounds[0])

        # Get penalty function (default: squared positive part)
        penalty_spec = ctcs_spec.get("penalty", "square")
        if callable(penalty_spec):
            penalty_func = penalty_spec
        else:
            penalty_func = _PENALTY_FUNCTIONS[penalty_spec]

        # IMPORTANT: We use a factory function here to avoid late-binding closure issues.
        # Without the factory, all augmented dynamics would share references to the same
        # constraint_fn and penalty_func variables, which would be overwritten on each
        # loop iteration. The factory creates a new closure for each iteration, capturing
        # the current values by passing them as function arguments.
        # See: https://docs.python-guide.org/writing/gotchas/#late-binding-closures
        def _make_ctcs_augmented_dynamics(orig_f, cons_fn, pen_func):
            """Create dynamics augmented with a CTCS constraint penalty accumulator.

            Args:
                orig_f: Original dynamics function (x, u, node, params) -> xdot
                cons_fn: Constraint function (x, u, node, params) -> scalar residual
                pen_func: Penalty function (residual) -> penalty value

            Returns:
                Augmented dynamics function that appends penalty derivative to xdot
            """
            def augmented_f(x, u, node, params):
                # Compute original state derivatives
                xdot = orig_f(x, u, node, params)

                # Compute constraint residual and apply penalty
                residual = cons_fn(x, u, node, params)
                penalty = pen_func(residual)

                # Append penalty accumulation rate to state derivatives
                # The penalty value becomes the derivative of the augmented state
                return jnp.concatenate([xdot, jnp.atleast_1d(penalty)])

            return augmented_f

        # Augment optimization dynamics with CTCS constraint
        aug_f = _make_ctcs_augmented_dynamics(dynamics.f, constraint_fn, penalty_func)
        dynamics = Dynamics(
            f=aug_f,
            A=jacfwd(aug_f, argnums=0),
            B=jacfwd(aug_f, argnums=1),
        )

        # Augment propagation dynamics with CTCS constraint
        aug_f_prop = _make_ctcs_augmented_dynamics(dynamics_prop.f, constraint_fn, penalty_func)
        dynamics_prop = Dynamics(
            f=aug_f_prop,
            A=jacfwd(aug_f_prop, argnums=0),
            B=jacfwd(aug_f_prop, argnums=1),
        )

        # Update both unified states with new augmented state
        x_unified.append(
            min=bounds[0],
            max=bounds[1],
            guess=initial,
            initial=initial,
            final=0.0,
            augmented=True,
        )
        x_prop_unified.append(
            min=bounds[0],
            max=bounds[1],
            guess=initial,
            initial=initial,
            final=0.0,
            augmented=True,
        )

    return dynamics, dynamics_prop, jax_constraints, x_unified, x_prop_unified
