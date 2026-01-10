"""Trajectory initialization utilities for generating initial guesses.

This module provides interpolation methods for constructing initial trajectory
guesses from keyframe values at specified nodes. Keyframes define values at
specific nodes, and the interpolation fills in values for all intermediate nodes.
"""

from typing import List, Sequence, Tuple, Union

import numpy as np

# Type aliases for clarity
Quaternion = Union[np.ndarray, Tuple[float, float, float, float], List[float]]
Keyframe = Union[np.ndarray, Sequence[float], float]


def linspace(
    keyframes: Sequence[Keyframe],
    nodes: Sequence[int],
) -> np.ndarray:
    """Generate trajectory guess via piecewise linear interpolation between keyframes.

    Keyframes specify values at particular nodes, and linear interpolation fills
    in the values for all nodes in between. This generalizes np.linspace to support
    multiple waypoints.

    Args:
        keyframes: Sequence of values at keyframe nodes. Each keyframe should be
            array-like with the same shape, or a scalar.
        nodes: Sequence of node indices where keyframes occur. Must be sorted in
            ascending order and have the same length as keyframes. The last node
            determines the output size (N = nodes[-1] + 1).

    Returns:
        np.ndarray of shape (N, *keyframe_shape) containing interpolated values
        for all nodes, where N = nodes[-1] + 1.

    Raises:
        ValueError: If keyframes and nodes have different lengths, or nodes are
            not sorted.

    Note:
        The first node should typically be 0 and the last node should be N-1
        (where N is your trajectory length). Nodes before the first keyframe
        or after the last will be left as zeros.

    Example:
        Interpolate position through three waypoints::

            import openscvx as ox

            position_guess = ox.init.linspace(
                keyframes=[[0, 0, 0], [5, 10, 5], [10, 0, 0]],
                nodes=[0, 25, 49],
            )  # Returns shape (50, 3)
            position.guess = position_guess

        Simple start-to-end interpolation (equivalent to np.linspace)::

            position_guess = ox.init.linspace(
                keyframes=[[0, 0, 0], [10, 0, 0]],
                nodes=[0, 49],
            )  # Returns shape (50, 3)
    """
    _validate_keyframe_inputs(keyframes, nodes)
    N = nodes[-1] + 1

    # Convert keyframes to numpy arrays
    keyframes_arr = [np.atleast_1d(np.asarray(kf, dtype=np.float64)) for kf in keyframes]
    keyframe_shape = keyframes_arr[0].shape

    # Validate all keyframes have the same shape
    for i, kf in enumerate(keyframes_arr):
        if kf.shape != keyframe_shape:
            raise ValueError(f"Keyframe {i} has shape {kf.shape}, expected {keyframe_shape}")

    # Initialize output array
    result = np.zeros((N, *keyframe_shape), dtype=np.float64)

    # Interpolate between consecutive keyframe pairs
    for i in range(len(nodes) - 1):
        start_node = nodes[i]
        end_node = nodes[i + 1]
        start_val = keyframes_arr[i]
        end_val = keyframes_arr[i + 1]

        # Number of nodes in this segment (inclusive of start, exclusive of end)
        n_segment = end_node - start_node

        for j in range(n_segment):
            t = j / n_segment
            result[start_node + j] = (1 - t) * start_val + t * end_val

    # Set the final keyframe value
    result[nodes[-1]] = keyframes_arr[-1]

    return result


def nlerp(
    keyframes: Sequence[Quaternion],
    nodes: Sequence[int],
) -> np.ndarray:
    """Generate quaternion trajectory guess via normalized linear interpolation.

    NLERP performs linear interpolation between quaternions followed by
    normalization. This is faster than SLERP but produces non-constant angular
    velocity. For initial guesses this is typically sufficient.

    Quaternion convention: [w, x, y, z] (scalar-first).

    Args:
        keyframes: Sequence of unit quaternions at keyframe nodes. Each quaternion
            should be array-like with shape (4,) in [w, x, y, z] order.
        nodes: Sequence of node indices where keyframes occur. Must be sorted in
            ascending order and have the same length as keyframes.

    Returns:
        np.ndarray of shape (N, 4) containing normalized interpolated quaternions
        for all nodes, where N = nodes[-1] + 1.

    Raises:
        ValueError: If keyframes and nodes have different lengths, nodes are not
            sorted, or quaternions don't have shape (4,).

    Note:
        The first node should typically be 0 and the last node should be N-1.
        Nodes outside the keyframe range will be left as zeros.

    Example:
        Interpolate attitude from identity to 180-degree rotation about z-axis::

            import openscvx as ox

            attitude_guess = ox.init.nlerp(
                keyframes=[
                    [1, 0, 0, 0],  # Identity quaternion
                    [0, 0, 0, 1],  # 180 deg about z
                ],
                nodes=[0, 49],
            )  # Returns shape (50, 4)
            attitude.guess = attitude_guess

        Interpolate through intermediate attitudes::

            attitude_guess = ox.init.nlerp(
                keyframes=[
                    [1, 0, 0, 0],           # Identity
                    [0.707, 0, 0, 0.707],   # 90 deg about z
                    [0, 0, 0, 1],           # 180 deg about z
                ],
                nodes=[0, 25, 49],
            )
    """
    _validate_keyframe_inputs(keyframes, nodes)
    N = nodes[-1] + 1

    # Convert keyframes to numpy arrays and validate shape
    keyframes_arr = []
    for i, kf in enumerate(keyframes):
        q = np.asarray(kf, dtype=np.float64)
        if q.shape != (4,):
            raise ValueError(f"Keyframe {i} has shape {q.shape}, expected (4,) for quaternion")
        # Normalize input quaternions
        q = q / np.linalg.norm(q)
        keyframes_arr.append(q)

    # Initialize output array
    result = np.zeros((N, 4), dtype=np.float64)

    # Interpolate between consecutive keyframe pairs
    for i in range(len(nodes) - 1):
        start_node = nodes[i]
        end_node = nodes[i + 1]
        q0 = keyframes_arr[i]
        q1 = keyframes_arr[i + 1]

        # Ensure shortest path interpolation
        if np.dot(q0, q1) < 0:
            q1 = -q1

        n_segment = end_node - start_node

        for j in range(n_segment):
            t = j / n_segment
            # Linear interpolation
            q_interp = (1 - t) * q0 + t * q1
            # Normalize
            result[start_node + j] = q_interp / np.linalg.norm(q_interp)

    # Set the final keyframe value (normalized)
    result[nodes[-1]] = keyframes_arr[-1]

    return result


def slerp(
    keyframes: Sequence[Quaternion],
    nodes: Sequence[int],
) -> np.ndarray:
    """Generate quaternion trajectory guess via spherical linear interpolation.

    SLERP interpolates along the great arc on the unit quaternion sphere,
    producing constant angular velocity between keyframes. This is more accurate
    than NLERP but slightly more expensive to compute.

    Quaternion convention: [w, x, y, z] (scalar-first).

    Args:
        keyframes: Sequence of unit quaternions at keyframe nodes. Each quaternion
            should be array-like with shape (4,) in [w, x, y, z] order.
        nodes: Sequence of node indices where keyframes occur. Must be sorted in
            ascending order and have the same length as keyframes.

    Returns:
        np.ndarray of shape (N, 4) containing interpolated quaternions for all nodes,
        where N = nodes[-1] + 1.

    Raises:
        ValueError: If keyframes and nodes have different lengths, nodes are not
            sorted, or quaternions don't have shape (4,).

    Note:
        The first node should typically be 0 and the last node should be N-1.
        Nodes outside the keyframe range will be left as zeros.

    Example:
        Interpolate attitude with constant angular velocity::

            import openscvx as ox

            attitude_guess = ox.init.slerp(
                keyframes=[
                    [1, 0, 0, 0],  # Identity quaternion
                    [0, 0, 0, 1],  # 180 deg about z
                ],
                nodes=[0, 49],
            )  # Returns shape (50, 4)
            attitude.guess = attitude_guess
    """
    _validate_keyframe_inputs(keyframes, nodes)
    N = nodes[-1] + 1

    # Convert keyframes to numpy arrays and validate shape
    keyframes_arr = []
    for i, kf in enumerate(keyframes):
        q = np.asarray(kf, dtype=np.float64)
        if q.shape != (4,):
            raise ValueError(f"Keyframe {i} has shape {q.shape}, expected (4,) for quaternion")
        # Normalize input quaternions
        q = q / np.linalg.norm(q)
        keyframes_arr.append(q)

    # Initialize output array
    result = np.zeros((N, 4), dtype=np.float64)

    # Interpolate between consecutive keyframe pairs
    for i in range(len(nodes) - 1):
        start_node = nodes[i]
        end_node = nodes[i + 1]
        q0 = keyframes_arr[i]
        q1 = keyframes_arr[i + 1]

        # Ensure shortest path interpolation
        dot = np.dot(q0, q1)
        if dot < 0:
            q1 = -q1
            dot = -dot

        n_segment = end_node - start_node

        # Handle near-identical quaternions (fall back to NLERP)
        if dot > 0.9995:
            for j in range(n_segment):
                t = j / n_segment
                q_interp = (1 - t) * q0 + t * q1
                result[start_node + j] = q_interp / np.linalg.norm(q_interp)
        else:
            # SLERP formula
            theta = np.arccos(dot)
            sin_theta = np.sin(theta)

            for j in range(n_segment):
                t = j / n_segment
                s0 = np.sin((1 - t) * theta) / sin_theta
                s1 = np.sin(t * theta) / sin_theta
                result[start_node + j] = s0 * q0 + s1 * q1

    # Set the final keyframe value (normalized)
    result[nodes[-1]] = keyframes_arr[-1]

    return result


def _validate_keyframe_inputs(
    keyframes: Sequence,
    nodes: Sequence[int],
) -> None:
    """Validate common inputs for keyframe interpolation functions.

    Args:
        keyframes: Sequence of keyframe values
        nodes: Sequence of node indices

    Raises:
        ValueError: If inputs are invalid
    """
    if len(keyframes) != len(nodes):
        raise ValueError(
            f"keyframes and nodes must have the same length, "
            f"got {len(keyframes)} keyframes and {len(nodes)} nodes"
        )

    if len(keyframes) < 2:
        raise ValueError("At least 2 keyframes are required for interpolation")

    # Check nodes are sorted
    for i in range(len(nodes) - 1):
        if nodes[i] >= nodes[i + 1]:
            raise ValueError(
                f"nodes must be strictly increasing, "
                f"but nodes[{i}]={nodes[i]} >= nodes[{i + 1}]={nodes[i + 1]}"
            )

    # Check first node is non-negative
    if nodes[0] < 0:
        raise ValueError(f"Node indices must be >= 0, got nodes[0]={nodes[0]}")
