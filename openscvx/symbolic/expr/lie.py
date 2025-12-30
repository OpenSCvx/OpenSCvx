"""Lie algebra operations for rigid body dynamics.

This module provides symbolic expression nodes for Lie algebra operations
commonly used in 6-DOF rigid body dynamics, robotics, and geometric mechanics.
These operations enable elegant formulations of Newton-Euler dynamics using
spatial vectors (twists and wrenches).

The module uses the following conventions:
    - Twist (spatial velocity): ξ = [v; ω] where v ∈ ℝ³ is linear velocity
      and ω ∈ ℝ³ is angular velocity (both in body frame)
    - Wrench (spatial force): F = [f; τ] where f ∈ ℝ³ is force and τ ∈ ℝ³
      is torque (both in body frame)
    - Momentum: μ = [p; L] where p ∈ ℝ³ is linear momentum and L ∈ ℝ³
      is angular momentum

Example:
    Newton-Euler dynamics for a rigid body using the coadjoint operator::

        import openscvx as ox

        # Body twist (6D velocity in body frame)
        twist = ox.State("twist", shape=(6,))  # [v; ω]

        # Spatial inertia matrix (6x6)
        M = ox.Parameter("M", shape=(6, 6), value=spatial_inertia)

        # Control wrench
        wrench = ox.Control("wrench", shape=(6,))  # [f; τ]

        # Newton-Euler equation: M @ twist_dot = wrench - ad*(twist, M @ twist)
        momentum = M @ twist
        bias_force = ox.lie.AdjointDual(twist, momentum)
        twist_dot = M_inv @ (wrench - bias_force)

References:
    - Murray, Li, Sastry: "A Mathematical Introduction to Robotic Manipulation"
    - Featherstone: "Rigid Body Dynamics Algorithms"
    - Sola et al.: "A micro Lie theory for state estimation in robotics"
"""

from typing import Tuple

from .expr import Expr, to_expr


class AdjointDual(Expr):
    """Coadjoint operator ad* for computing Coriolis and centrifugal forces.

    Computes the coadjoint action ad*_ξ(μ) which represents the rate of change
    of momentum due to body rotation. This is the key term in Newton-Euler
    dynamics that captures Coriolis and centrifugal effects.

    For se(3), given twist ξ = [v; ω] and momentum μ = [f; τ]:

        ad*_ξ(μ) = [ ω × f + v × τ ]
                   [     ω × τ     ]

    This appears in the Newton-Euler equations as:

        M @ ξ_dot = F_ext - ad*_ξ(M @ ξ)

    where M is the spatial inertia matrix and F_ext is the external wrench.

    Attributes:
        twist: 6D twist vector [v; ω] (linear velocity, angular velocity)
        momentum: 6D momentum vector [p; L] or [f; τ] (linear, angular)

    Example:
        Compute the bias force (Coriolis + centrifugal) for rigid body dynamics::

            import openscvx as ox

            twist = ox.State("twist", shape=(6,))
            M = ox.Parameter("M", shape=(6, 6), value=inertia_matrix)

            momentum = M @ twist
            bias_force = ox.lie.AdjointDual(twist, momentum)

            # In dynamics: twist_dot = M_inv @ (wrench - bias_force)

    Note:
        The coadjoint is related to the adjoint by: ad*_ξ = -(ad_ξ)^T

        For the special case of pure rotation (v=0) with diagonal inertia,
        the angular part reduces to the familiar ω × (J @ ω) term.

    See Also:
        Adjoint: The adjoint operator for twist-on-twist action
        SSM: 3x3 skew-symmetric matrix for cross products
    """

    def __init__(self, twist, momentum):
        """Initialize a coadjoint operator.

        Args:
            twist: 6D twist vector [v; ω] with shape (6,)
            momentum: 6D momentum vector [p; L] with shape (6,)
        """
        self.twist = to_expr(twist)
        self.momentum = to_expr(momentum)

    def children(self):
        return [self.twist, self.momentum]

    def canonicalize(self) -> "Expr":
        twist = self.twist.canonicalize()
        momentum = self.momentum.canonicalize()
        return AdjointDual(twist, momentum)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that inputs are 6D vectors and return output shape.

        Returns:
            tuple: Shape (6,) for the resulting coadjoint vector

        Raises:
            ValueError: If twist or momentum do not have shape (6,)
        """
        twist_shape = self.twist.check_shape()
        momentum_shape = self.momentum.check_shape()

        if twist_shape != (6,):
            raise ValueError(f"AdjointDual expects twist with shape (6,), got {twist_shape}")
        if momentum_shape != (6,):
            raise ValueError(f"AdjointDual expects momentum with shape (6,), got {momentum_shape}")

        return (6,)

    def __repr__(self):
        return f"ad_dual({self.twist!r}, {self.momentum!r})"


class Adjoint(Expr):
    """Adjoint operator ad (Lie bracket) for twist-on-twist action.

    Computes the adjoint action ad_ξ₁(ξ₂) which represents the Lie bracket
    [ξ₁, ξ₂] of two twists. This is used for velocity propagation in
    kinematic chains and acceleration computations.

    For se(3), given twists ξ₁ = [v₁; ω₁] and ξ₂ = [v₂; ω₂]:

        ad_ξ₁(ξ₂) = [ξ₁, ξ₂] = [ ω₁ × v₂ - ω₂ × v₁ ]
                                [     ω₁ × ω₂       ]

    Equivalently using the adjoint matrix:

        ad_ξ = [ [ω]×    0   ]
               [ [v]×   [ω]× ]

    where [·]× denotes the 3x3 skew-symmetric (cross product) matrix.

    Attributes:
        twist1: First 6D twist vector [v₁; ω₁]
        twist2: Second 6D twist vector [v₂; ω₂]

    Example:
        Compute the Lie bracket of two twists::

            import openscvx as ox

            twist1 = ox.State("twist1", shape=(6,))
            twist2 = ox.State("twist2", shape=(6,))

            bracket = ox.lie.Adjoint(twist1, twist2)

        Velocity propagation in a kinematic chain::

            # Child link velocity includes parent velocity plus relative motion
            # V_child = Ad_T @ V_parent + joint_twist * q_dot

    Note:
        The adjoint satisfies the Jacobi identity and is antisymmetric:
        ad_ξ₁(ξ₂) = -ad_ξ₂(ξ₁)

    See Also:
        AdjointDual: The coadjoint operator for momentum dynamics
    """

    def __init__(self, twist1, twist2):
        """Initialize an adjoint operator.

        Args:
            twist1: First 6D twist vector [v; ω] with shape (6,)
            twist2: Second 6D twist vector [v; ω] with shape (6,)
        """
        self.twist1 = to_expr(twist1)
        self.twist2 = to_expr(twist2)

    def children(self):
        return [self.twist1, self.twist2]

    def canonicalize(self) -> "Expr":
        twist1 = self.twist1.canonicalize()
        twist2 = self.twist2.canonicalize()
        return Adjoint(twist1, twist2)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that inputs are 6D vectors and return output shape.

        Returns:
            tuple: Shape (6,) for the resulting Lie bracket

        Raises:
            ValueError: If either twist does not have shape (6,)
        """
        twist1_shape = self.twist1.check_shape()
        twist2_shape = self.twist2.check_shape()

        if twist1_shape != (6,):
            raise ValueError(f"Adjoint expects twist1 with shape (6,), got {twist1_shape}")
        if twist2_shape != (6,):
            raise ValueError(f"Adjoint expects twist2 with shape (6,), got {twist2_shape}")

        return (6,)

    def __repr__(self):
        return f"ad({self.twist1!r}, {self.twist2!r})"
