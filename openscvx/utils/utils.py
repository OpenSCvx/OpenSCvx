import jax
import jax.numpy as jnp
import numpy as np


def generate_orthogonal_unit_vectors(vectors=None):
    """
    Generates 3 orthogonal unit vectors to model the axis of the ellipsoid via QR decomposition

    Parameters:
    vectors (np.ndarray): Optional, axes of the ellipsoid to be orthonormalized.
                            If none specified generates randomly.

    Returns:
    np.ndarray: A 3x3 matrix where each column is a unit vector.
    """
    if vectors is None:
        # Create a random key
        key = jax.random.PRNGKey(0)

        # Generate a 3x3 array of random numbers uniformly distributed between 0 and 1
        vectors = jax.random.uniform(key, (3, 3))
    Q, _ = jnp.linalg.qr(vectors)
    return Q


rot = np.array(
    [
        [np.cos(np.pi / 2), np.sin(np.pi / 2), 0],
        [-np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
        [0, 0, 1],
    ]
)


def gen_vertices(center, radii):
    """
    Obtains the vertices of the gate.
    """
    vertices = []
    vertices.append(center + rot @ [radii[0], 0, radii[2]])
    vertices.append(center + rot @ [-radii[0], 0, radii[2]])
    vertices.append(center + rot @ [-radii[0], 0, -radii[2]])
    vertices.append(center + rot @ [radii[0], 0, -radii[2]])
    return vertices


# TODO (haynec): make this less hardcoded
def get_kp_pose(t, init_pose):
    loop_time = 40.0
    loop_radius = 20.0

    t_angle = t / loop_time * (2 * jnp.pi)
    x = loop_radius * jnp.sin(t_angle)
    y = x * jnp.cos(t_angle)
    z = 0.5 * x * jnp.sin(t_angle)
    return jnp.array([x, y, z]).T + init_pose


def calculate_cost_from_boundaries(
    x: np.ndarray, initial_type: np.ndarray, final_type: np.ndarray
) -> float:
    """Calculate cost from boundary condition objectives.

    This function computes the total cost contribution from state boundary conditions
    marked as "Minimize" or "Maximize" at initial and final times.

    Args:
        x: State trajectory array of shape (N, n_states)
        initial_type: Array of boundary condition types for initial states
        final_type: Array of boundary condition types for final states

    Returns:
        Total cost from minimize/maximize boundary conditions

    Example:
        >>> # State with final time to minimize
        >>> x = np.array([[0.0, 5.0], [10.0, 20.0]])  # 2 nodes, 2 states
        >>> initial_type = np.array(["Fix", "Free"])
        >>> final_type = np.array(["Minimize", "Free"])
        >>> cost = calculate_cost_from_boundaries(x, initial_type, final_type)
        >>> cost  # Returns x[-1, 0] = 10.0
    """
    cost = 0.0

    # Add costs from initial boundary conditions
    for i, bc_type in enumerate(initial_type):
        if bc_type == "Minimize":
            cost += x[0, i]
        elif bc_type == "Maximize":
            cost -= x[0, i]

    # Add costs from final boundary conditions
    for i, bc_type in enumerate(final_type):
        if bc_type == "Minimize":
            cost += x[-1, i]
        elif bc_type == "Maximize":
            cost -= x[-1, i]

    return cost
