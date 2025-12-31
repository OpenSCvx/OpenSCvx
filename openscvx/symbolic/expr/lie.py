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

Note:
    The twist convention [v; ω] (linear first, angular second) matches jaxlie's
    SE3 tangent parameterization, so no reordering is needed during lowering.

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

    Product of Exponentials forward kinematics (requires jaxlie)::

        import openscvx as ox

        # Joint angle
        theta = ox.State("theta", shape=(1,))

        # Screw axis for a revolute joint about z-axis
        screw_axis = ox.Constant(np.array([0, 0, 0, 0, 0, 1]))

        # Joint transformation
        T_joint = ox.lie.SE3Exp(screw_axis * theta)  # 4×4 matrix

References:
    - Murray, Li, Sastry: "A Mathematical Introduction to Robotic Manipulation"
    - Featherstone: "Rigid Body Dynamics Algorithms"
    - Sola et al.: "A micro Lie theory for state estimation in robotics"
"""

from typing import Tuple

from .expr import Expr, to_expr

# Check for jaxlie availability
try:
    import jaxlie  # noqa: F401

    _JAXLIE_AVAILABLE = True
except ImportError:
    _JAXLIE_AVAILABLE = False


def _require_jaxlie(operator_name: str):
    """Raise ImportError if jaxlie is not available."""
    if not _JAXLIE_AVAILABLE:
        raise ImportError(
            f"{operator_name} requires jaxlie. Install with: pip install openscvx[lie]"
        )


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


class SO3Exp(Expr):
    """Exponential map from so(3) to SO(3) rotation matrix.

    Maps a 3D rotation vector (axis-angle representation) to a 3×3 rotation
    matrix using the Rodrigues formula. Uses jaxlie for numerically robust
    implementation with proper handling of small angles.

    The rotation vector ω has direction equal to the rotation axis and
    magnitude equal to the rotation angle in radians.

    Attributes:
        omega: 3D rotation vector with shape (3,)

    Example:
        Create a rotation about the z-axis::

            import openscvx as ox
            import numpy as np

            # 90 degree rotation about z
            omega = ox.Constant(np.array([0, 0, np.pi/2]))
            R = ox.lie.SO3Exp(omega)  # 3×3 rotation matrix

        Parameterized rotation for optimization::

            theta = ox.State("theta", shape=(1,))
            axis = ox.Constant(np.array([0, 0, 1]))  # z-axis
            R = ox.lie.SO3Exp(axis * theta)

    Note:
        Requires jaxlie: `pip install openscvx[lie]`

    See Also:
        SO3Log: Inverse operation (rotation matrix to rotation vector)
        SE3Exp: Full rigid body transformation including translation
    """

    def __init__(self, omega):
        """Initialize SO3 exponential map.

        Args:
            omega: 3D rotation vector (axis × angle) with shape (3,)

        Raises:
            ImportError: If jaxlie is not installed
        """
        _require_jaxlie("SO3Exp")
        self.omega = to_expr(omega)

    def children(self):
        return [self.omega]

    def canonicalize(self) -> "Expr":
        omega = self.omega.canonicalize()
        return SO3Exp(omega)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that input is a 3D vector and return output shape.

        Returns:
            tuple: Shape (3, 3) for the rotation matrix

        Raises:
            ValueError: If omega does not have shape (3,)
        """
        omega_shape = self.omega.check_shape()
        if omega_shape != (3,):
            raise ValueError(f"SO3Exp expects omega with shape (3,), got {omega_shape}")
        return (3, 3)

    def __repr__(self):
        return f"SO3Exp({self.omega!r})"


class SO3Log(Expr):
    """Logarithm map from SO(3) rotation matrix to so(3) rotation vector.

    Maps a 3×3 rotation matrix to a 3D rotation vector (axis-angle
    representation). Uses jaxlie for numerically robust implementation.

    The output rotation vector ω has direction equal to the rotation axis
    and magnitude equal to the rotation angle in radians.

    Attributes:
        rotation: 3×3 rotation matrix with shape (3, 3)

    Example:
        Extract rotation vector from a rotation matrix::

            import openscvx as ox

            R = ox.State("R", shape=(3, 3))  # Rotation matrix state
            omega = ox.lie.SO3Log(R)  # 3D rotation vector

    Note:
        Requires jaxlie: `pip install openscvx[lie]`

    See Also:
        SO3Exp: Inverse operation (rotation vector to rotation matrix)
        SE3Log: Full rigid body transformation logarithm
    """

    def __init__(self, rotation):
        """Initialize SO3 logarithm map.

        Args:
            rotation: 3×3 rotation matrix with shape (3, 3)

        Raises:
            ImportError: If jaxlie is not installed
        """
        _require_jaxlie("SO3Log")
        self.rotation = to_expr(rotation)

    def children(self):
        return [self.rotation]

    def canonicalize(self) -> "Expr":
        rotation = self.rotation.canonicalize()
        return SO3Log(rotation)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that input is a 3×3 matrix and return output shape.

        Returns:
            tuple: Shape (3,) for the rotation vector

        Raises:
            ValueError: If rotation does not have shape (3, 3)
        """
        rotation_shape = self.rotation.check_shape()
        if rotation_shape != (3, 3):
            raise ValueError(f"SO3Log expects rotation with shape (3, 3), got {rotation_shape}")
        return (3,)

    def __repr__(self):
        return f"SO3Log({self.rotation!r})"


class SE3Exp(Expr):
    """Exponential map from se(3) twist to SE(3) transformation matrix.

    Maps a 6D twist vector to a 4×4 homogeneous transformation matrix.
    Uses jaxlie for numerically robust implementation with proper handling
    of small angles and translations.

    The twist ξ = [v; ω] follows the convention:
        - v: 3D linear velocity component
        - ω: 3D angular velocity component

    This is the key operation for Product of Exponentials (PoE) forward
    kinematics in robotic manipulators.

    Attributes:
        twist: 6D twist vector [v; ω] with shape (6,)

    Example:
        Product of Exponentials forward kinematics::

            import openscvx as ox
            import numpy as np

            # Screw axis for revolute joint about z-axis at origin
            screw_axis = np.array([0, 0, 0, 0, 0, 1])  # [v; ω]
            theta = ox.State("theta", shape=(1,))

            # Joint transformation
            T = ox.lie.SE3Exp(ox.Constant(screw_axis) * theta)  # 4×4 matrix

            # Chain multiple joints
            T_01 = ox.lie.SE3Exp(screw1 * q1)
            T_12 = ox.lie.SE3Exp(screw2 * q2)
            T_02 = T_01 @ T_12

        Extract position from transformation::

            T_ee = forward_kinematics(joint_angles)
            p_ee = T_ee[:3, 3]  # End-effector position

    Note:
        Requires jaxlie: `pip install openscvx[lie]`

        The twist convention [v; ω] matches jaxlie's SE3 tangent
        parameterization, so no reordering is performed.

    See Also:
        SE3Log: Inverse operation (transformation matrix to twist)
        SO3Exp: Rotation-only exponential map
        AdjointDual: For dynamics computations with twists
    """

    def __init__(self, twist):
        """Initialize SE3 exponential map.

        Args:
            twist: 6D twist vector [v; ω] with shape (6,)

        Raises:
            ImportError: If jaxlie is not installed
        """
        _require_jaxlie("SE3Exp")
        self.twist = to_expr(twist)

    def children(self):
        return [self.twist]

    def canonicalize(self) -> "Expr":
        twist = self.twist.canonicalize()
        return SE3Exp(twist)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that input is a 6D vector and return output shape.

        Returns:
            tuple: Shape (4, 4) for the homogeneous transformation matrix

        Raises:
            ValueError: If twist does not have shape (6,)
        """
        twist_shape = self.twist.check_shape()
        if twist_shape != (6,):
            raise ValueError(f"SE3Exp expects twist with shape (6,), got {twist_shape}")
        return (4, 4)

    def __repr__(self):
        return f"SE3Exp({self.twist!r})"


class SE3Log(Expr):
    """Logarithm map from SE(3) transformation matrix to se(3) twist.

    Maps a 4×4 homogeneous transformation matrix to a 6D twist vector.
    Uses jaxlie for numerically robust implementation.

    The output twist ξ = [v; ω] follows the convention:
        - v: 3D linear component
        - ω: 3D angular component (rotation vector)

    This is useful for computing error metrics between poses in optimization.

    Attributes:
        transform: 4×4 homogeneous transformation matrix with shape (4, 4)

    Example:
        Compute pose error for trajectory optimization::

            import openscvx as ox

            T_current = forward_kinematics(q)
            T_target = ox.Parameter("T_target", shape=(4, 4), value=goal_pose)

            # Relative transformation
            T_error = ox.linalg.inv(T_target) @ T_current

            # Convert to twist for error metric
            twist_error = ox.lie.SE3Log(T_error)
            pose_cost = ox.linalg.Norm(twist_error) ** 2

    Note:
        Requires jaxlie: `pip install openscvx[lie]`

    See Also:
        SE3Exp: Inverse operation (twist to transformation matrix)
        SO3Log: Rotation-only logarithm map
    """

    def __init__(self, transform):
        """Initialize SE3 logarithm map.

        Args:
            transform: 4×4 homogeneous transformation matrix with shape (4, 4)

        Raises:
            ImportError: If jaxlie is not installed
        """
        _require_jaxlie("SE3Log")
        self.transform = to_expr(transform)

    def children(self):
        return [self.transform]

    def canonicalize(self) -> "Expr":
        transform = self.transform.canonicalize()
        return SE3Log(transform)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that input is a 4×4 matrix and return output shape.

        Returns:
            tuple: Shape (6,) for the twist vector

        Raises:
            ValueError: If transform does not have shape (4, 4)
        """
        transform_shape = self.transform.check_shape()
        if transform_shape != (4, 4):
            raise ValueError(f"SE3Log expects transform with shape (4, 4), got {transform_shape}")
        return (6,)

    def __repr__(self):
        return f"SE3Log({self.transform!r})"
