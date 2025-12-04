"""LoweredProblem dataclass - container for all lowering outputs."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

from openscvx.lowered.cvxpy_constraints import LoweredCvxpyConstraints
from openscvx.lowered.cvxpy_variables import CVXPyVariables
from openscvx.lowered.dynamics import Dynamics
from openscvx.lowered.jax_constraints import LoweredJaxConstraints
from openscvx.lowered.unified import UnifiedControl, UnifiedState

if TYPE_CHECKING:
    import cvxpy as cp


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
        ocp_vars: Typed CVXPy variables and parameters for OCP construction
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
            ocp = OptimalControlProblem(settings, lowered)
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
    ocp_vars: CVXPyVariables
    cvxpy_params: Dict[str, "cp.Parameter"]
