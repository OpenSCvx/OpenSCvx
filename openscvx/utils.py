from datetime import datetime
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    import cProfile


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


def profiling_start(profiling_enabled: bool) -> "Optional[cProfile.Profile]":
    """Start profiling if enabled.

    Args:
        profiling_enabled: Whether to enable profiling.

    Returns:
        Profile object if enabled, None otherwise.
    """
    if profiling_enabled:
        import cProfile

        pr = cProfile.Profile()
        pr.enable()
        return pr
    return None


def profiling_end(pr: "Optional[cProfile.Profile]", identifier: str):
    """Stop profiling and save results with timestamp.

    Args:
        pr: Profile object from profiling_start, or None.
        identifier: Identifier for the profiling session (e.g., "solve", "initialize").
    """
    if pr is not None:
        pr.disable()
        # Save results so it can be visualized with snakeviz
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pr.dump_stats(f"profiling/{timestamp}_{identifier}.prof")
