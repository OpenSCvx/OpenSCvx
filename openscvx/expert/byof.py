"""Bring-Your-Own-Functions (BYOF) - Expert User Mode.

This module provides type definitions and documentation for expert users who want
to bypass the symbolic layer and directly provide raw JAX functions.

Important:
    The unified state/control vectors include ALL states/controls in the order
    they were provided, plus any augmented states from CTCS constraints. You are
    responsible for correct indexing. Consider inspecting the symbolic problem
    to understand the layout.

Function Signatures:
    All byof functions must be JAX-compatible (use jax.numpy, avoid side effects).

    - dynamics: ``(x, u, node, params) -> xdot_component``
        - x: Full unified state vector (1D array)
        - u: Full unified control vector (1D array)
        - node: Integer node index
        - params: Dict of parameters
        - Returns: State derivative component (array matching state shape)

    - nodal_constraints: ``(x, u, node, params) -> residual``
        - Same arguments as dynamics
        - Returns: Constraint residual. Follows g(x,u) <= 0 convention

    - cross_nodal_constraints: ``(X, U, params) -> residual``
        - X: State trajectory (N, n_x) where N is number of trajectory nodes,
            n_x is unified state dimension
        - U: Control trajectory (N, n_u) where N is number of trajectory nodes,
            n_u is unified control dimension
        - params: Dict of parameters
        - Returns: Constraint residual. Follows g(X,U) <= 0 convention

    - ctcs constraint_fn: ``(x, u, node, params) -> scalar``
        - Same as nodal_constraints but MUST return scalar
        - Follows g(x,u) <= 0 convention

    - ctcs penalty: ``(residual) -> penalty_value``
        - residual: Scalar constraint residual
        - Returns: Non-negative penalty value

Example:
    Basic usage mixing symbolic and byof::

        import jax.numpy as jnp
        import openscvx as ox
        from openscvx import ByofSpec

        # Define states
        position = ox.State("position", shape=(2,))
        velocity = ox.State("velocity", shape=(1,))
        theta = ox.Control("theta", shape=(1,))

        # Unified state: [position[0], position[1], velocity[0], time, augmented...]
        # Unified control: [theta[0], time_dilation]

        byof: ByofSpec = {
            "nodal_constraints": [
                # Velocity must be positive: -velocity <= 0
                lambda x, u, node, params: -x[2],
            ],
            "ctcs_constraints": [
                {
                    "constraint_fn": lambda x, u, node, params: x[0] - 10.0,
                    "penalty": "square",
                    "bounds": (0.0, 1e-4),
                }
            ],
        }

        problem = ox.Problem(..., byof=byof)
"""

from typing import TYPE_CHECKING, Any, Callable, List, Literal, Tuple, TypedDict, Union

if TYPE_CHECKING:
    from jax import Array as JaxArray
else:
    JaxArray = Any

__all__ = ["ByofSpec", "CtcsConstraintSpec", "PenaltyFunction"]


# Type aliases for clarity
DynamicsFunction = Callable[[JaxArray, JaxArray, int, dict], JaxArray]
NodalConstraintFunction = Callable[[JaxArray, JaxArray, int, dict], JaxArray]
CrossNodalConstraintFunction = Callable[[JaxArray, JaxArray, dict], JaxArray]
CtcsConstraintFunction = Callable[[JaxArray, JaxArray, int, dict], float]
PenaltyFunction = Union[Literal["square", "l1", "huber"], Callable[[float], float]]


class CtcsConstraintSpec(TypedDict, total=False):
    """Specification for CTCS (Continuous-Time Constraint Satisfaction) constraint.

    CTCS constraints are enforced by augmenting the dynamics with a penalty term that
    accumulates violations over time. Useful for path constraints that must be satisfied
    continuously, not just at discrete nodes.

    Attributes:
        constraint_fn: Function computing constraint residual with signature
            ``(x, u, node, params) -> scalar``. Must return scalar.
            Follows g(x,u) <= 0 convention (negative = satisfied). Required field.
        penalty: Penalty function for positive residuals (violations).
            Built-in options: "square" (max(r,0)^2, default), "l1" (max(r,0)),
            "huber" (Huber loss). Custom: Callable ``(r) -> penalty`` (non-negative,
            differentiable).
        bounds: (min, max) bounds for augmented state accumulating penalties.
            Default: (0.0, 1e-4). Max acts as soft constraint on total violation.
        initial: Initial value for augmented state. Default: bounds[0] (usually 0.0).

    Example:
        Enforce position[0] <= 10.0 continuously::

            ctcs_spec: CtcsConstraintSpec = {
                "constraint_fn": lambda x, u, node, params: x[0] - 10.0,
                "penalty": "square",
                "bounds": (0.0, 1e-4),
                "initial": 0.0,
            }
    """

    constraint_fn: CtcsConstraintFunction  # Required
    penalty: PenaltyFunction
    bounds: Tuple[float, float]
    initial: float


class ByofSpec(TypedDict, total=False):
    """Bring-Your-Own-Functions specification for expert users.

    Allows bypassing the symbolic layer and directly providing raw JAX functions.
    All fields are optional - you can mix symbolic and byof as needed.

    Warning:
        You are responsible for:

        - Correct indexing into unified state/control vectors
        - Ensuring functions are JAX-compatible (use jax.numpy, no side effects)
        - Ensuring functions are differentiable
        - Following g(x,u) <= 0 convention for constraints

    Attributes:
        dynamics: Raw JAX functions for state derivatives. Maps state names to functions
            with signature ``(x, u, node, params) -> xdot_component``. States here should
            NOT appear in symbolic dynamics dict. You can mix: some states symbolic,
            some in byof.
        nodal_constraints: Point-wise constraints applied at each node independently.
            Signature: ``(x, u, node, params) -> residual``. Follows g(x,u) <= 0 convention.
            Applied to all nodes.
        cross_nodal_constraints: Constraints coupling multiple nodes (smoothness, rate limits).
            Signature: ``(X, U, params) -> residual`` where X is (N, n_x) and U is (N, n_u).
            N is the number of trajectory nodes, n_x is state dimension, n_u is control dimension.
            Follows g(X,U) <= 0 convention.
        ctcs_constraints: Continuous-time constraint satisfaction via dynamics augmentation.
            Each adds an augmented state accumulating violation penalties.
            See :class:`CtcsConstraintSpec` for details.

    Example:
        Custom dynamics and constraints::

            import jax.numpy as jnp
            from openscvx import ByofSpec

            # Custom dynamics for one state
            def custom_velocity_dynamics(x, u, node, params):
                # x[2] is velocity, u[0] is theta, params["g"] is gravity
                return params["g"] * jnp.cos(u[0])

            byof: ByofSpec = {
                "dynamics": {
                    "velocity": custom_velocity_dynamics,
                },
                "nodal_constraints": [
                    lambda x, u, node, params: x[2] - 10.0,  # velocity <= 10
                    lambda x, u, node, params: -x[2],         # velocity >= 0
                ],
                "cross_nodal_constraints": [
                    # Constrain total velocity across trajectory: sum(velocities) >= 5
                    # X.shape = (N, n_x), extract velocity column and sum
                    lambda X, U, params: 5.0 - jnp.sum(X[:, 2]),
                ],
                "ctcs_constraints": [
                    {
                        "constraint_fn": lambda x, u, node, params: x[0] - 5.0,
                        "penalty": "square",
                    }
                ],
            }
    """

    dynamics: dict[str, DynamicsFunction]
    nodal_constraints: List[NodalConstraintFunction]
    cross_nodal_constraints: List[CrossNodalConstraintFunction]
    ctcs_constraints: List[CtcsConstraintSpec]
