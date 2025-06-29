import jax
import jax.numpy as jnp
import numpy as np
import hashlib
import ast
import types
import inspect
import textwrap

def stable_function_hash(funcs, n_discretization_nodes=None, dt=None, total_time=None, state_max=None, state_min=None, control_max=None, control_min=None):
    hasher = hashlib.sha256()

    for func in funcs:
        try:
            src = inspect.getsource(func)
            src = textwrap.dedent(src)  # <<< Fix: remove extra indent
            parsed = ast.parse(src)

            # Remove docstrings from the AST
            for node in ast.walk(parsed):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node):
                    if isinstance(node.body[0], ast.Expr):
                        node.body = node.body[1:]

            normalized = ast.dump(parsed, annotate_fields=True, include_attributes=False)
            hasher.update(normalized.encode())

        except Exception as e:
            raise ValueError(f"Could not hash function {func}: {e}")

    # Add additional parameters to the hash
    if n_discretization_nodes is not None:
        hasher.update(f"n_nodes:{n_discretization_nodes}".encode())
    
    if dt is not None:
        hasher.update(f"dt:{dt}".encode())
    
    if total_time is not None:
        hasher.update(f"total_time:{total_time}".encode())
    
    if state_max is not None:
        hasher.update(f"state_max:{state_max.tobytes()}".encode())
    
    if state_min is not None:
        hasher.update(f"state_min:{state_min.tobytes()}".encode())
    
    if control_max is not None:
        hasher.update(f"control_max:{control_max.tobytes()}".encode())
    
    if control_min is not None:
        hasher.update(f"control_min:{control_min.tobytes()}".encode())

    return hasher.hexdigest()


def qdcm(q: jnp.ndarray) -> jnp.ndarray:
    # Convert a quaternion to a direction cosine matrix
    q_norm = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
    w, x, y, z = q / q_norm
    return jnp.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def SSMP(w: jnp.ndarray):
    # Convert an angular rate to a 4 x 4 skew symetric matrix
    x, y, z = w
    return jnp.array([[0, -x, -y, -z], [x, 0, z, -y], [y, -z, 0, x], [z, y, -x, 0]])


def SSM(w: jnp.ndarray):
    # Convert an angular rate to a 3 x 3 skew symetric matrix
    x, y, z = w
    return jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


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
