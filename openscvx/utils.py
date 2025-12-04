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
