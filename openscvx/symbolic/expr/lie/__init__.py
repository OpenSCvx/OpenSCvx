"""Lie algebra operations for rigid body dynamics.

This module provides symbolic expression nodes for Lie algebra operations
commonly used in 6-DOF rigid body dynamics, robotics, and geometric mechanics.
These operations enable elegant formulations of Newton-Euler dynamics using
spatial vectors (twists and wrenches).

The module provides two tiers of functionality:

**Tier 1: Built-in operators (no additional dependencies)**

These operators are implemented using simple cross products and work out of the box:

- ``AdjointDual``: Coadjoint operator ad*(ξ, μ) for Coriolis/centrifugal forces
- ``Adjoint``: Lie bracket [ξ₁, ξ₂] for twist-on-twist action

**Tier 2: jaxlie-backed operators (requires ``pip install openscvx[lie]``)**

These operators wrap `jaxlie <https://github.com/brentyi/jaxlie>`_ for robust
Lie group exponential/logarithm maps:

- ``SO3Exp``: so(3) → SO(3) exponential map (3D vector → 3×3 rotation)
- ``SO3Log``: SO(3) → so(3) logarithm map (3×3 rotation → 3D vector)
- ``SE3Exp``: se(3) → SE(3) exponential map (6D vector → 4×4 transform)
- ``SE3Log``: SE(3) → se(3) logarithm map (4×4 transform → 6D vector)

Conventions:
    - Twist (spatial velocity): ξ = [v; ω] where v ∈ ℝ³ is linear velocity
      and ω ∈ ℝ³ is angular velocity (both in body frame)
    - Wrench (spatial force): F = [f; τ] where f ∈ ℝ³ is force and τ ∈ ℝ³
      is torque (both in body frame)

Note:
    The twist convention [v; ω] (linear first, angular second) matches jaxlie's
    SE3 tangent parameterization, so no reordering is needed during lowering.

Example:
    Newton-Euler dynamics for a rigid body using the coadjoint operator::

        import openscvx as ox

        twist = ox.State("twist", shape=(6,))
        M = ox.Parameter("M", shape=(6, 6), value=spatial_inertia)
        wrench = ox.Control("wrench", shape=(6,))

        momentum = M @ twist
        bias_force = ox.lie.AdjointDual(twist, momentum)
        twist_dot = M_inv @ (wrench - bias_force)

    Product of Exponentials forward kinematics (requires jaxlie)::

        screw_axis = ox.Constant(np.array([0, 0, 0, 0, 0, 1]))
        theta = ox.State("theta", shape=(1,))
        T_joint = ox.lie.SE3Exp(screw_axis * theta)  # 4×4 matrix

References:
    - Murray, Li, Sastry: "A Mathematical Introduction to Robotic Manipulation"
    - Featherstone: "Rigid Body Dynamics Algorithms"
    - Sola et al.: "A micro Lie theory for state estimation in robotics"
"""

# Core operators - no dependencies
from .adjoint import Adjoint, AdjointDual

# jaxlie-backed operators - optional dependency
try:
    from .se3 import SE3Exp, SE3Log
    from .so3 import SO3Exp, SO3Log

    _JAXLIE_AVAILABLE = True
except ImportError:
    _JAXLIE_AVAILABLE = False

    def _make_stub(name: str):
        """Create a stub class that raises ImportError on instantiation."""

        def __init__(self, *args, **kwargs):
            raise ImportError(f"{name} requires jaxlie. Install with: pip install openscvx[lie]")

        return type(name, (), {"__init__": __init__})

    SO3Exp = _make_stub("SO3Exp")
    SO3Log = _make_stub("SO3Log")
    SE3Exp = _make_stub("SE3Exp")
    SE3Log = _make_stub("SE3Log")

__all__ = [
    "AdjointDual",
    "Adjoint",
    "SO3Exp",
    "SO3Log",
    "SE3Exp",
    "SE3Log",
]
