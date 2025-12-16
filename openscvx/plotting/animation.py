"""Viser-based trajectory visualization primitives.

This module provides modular components for visualizing trajectory optimization results.
Components can be composed together for custom visualizations.

For convenience functions that compose these primitives, see examples/plotting_viser.py.

Example:
    server = create_server(pos)
    add_gates(server, vertices)
    add_ellipsoid_obstacles(server, centers, radii, axes)
    add_ghost_trajectory(server, pos, colors)

    _, update_trail = add_animated_trail(server, pos, colors)
    _, update_marker = add_position_marker(server, pos)
    _, update_thrust = add_thrust_vector(server, pos, thrust)

    add_animation_controls(server, traj_time, [update_trail, update_marker, update_thrust])
    server.sleep_forever()
"""

import threading
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import viser

# Type alias for update callbacks: fn(frame_idx: int) -> None
UpdateCallback = Callable[[int], None]


# =============================================================================
# Helper Functions
# =============================================================================


def compute_velocity_colors(vel: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """Compute RGB colors based on velocity magnitude.

    Args:
        vel: Velocity array of shape (N, 3)
        cmap_name: Matplotlib colormap name

    Returns:
        Array of RGB colors with shape (N, 3), values in [0, 255]
    """
    vel_norms = np.linalg.norm(vel, axis=1)
    vel_range = vel_norms.max() - vel_norms.min()
    if vel_range < 1e-8:
        vel_normalized = np.zeros_like(vel_norms)
    else:
        vel_normalized = (vel_norms - vel_norms.min()) / vel_range

    cmap = plt.get_cmap(cmap_name)
    colors = np.array([[int(c * 255) for c in cmap(v)[:3]] for v in vel_normalized])
    return colors


def compute_grid_size(pos: np.ndarray, padding: float = 1.2) -> float:
    """Compute grid size based on trajectory extent.

    Args:
        pos: Position array of shape (N, 3)
        padding: Padding factor (1.2 = 20% padding)

    Returns:
        Grid size (width and height)
    """
    max_x = np.abs(pos[:, 0]).max()
    max_y = np.abs(pos[:, 1]).max()
    return max(max_x, max_y) * 2 * padding


# =============================================================================
# Server Setup
# =============================================================================


def create_server(
    pos: np.ndarray,
    dark_mode: bool = True,
) -> viser.ViserServer:
    """Create a viser server with basic scene setup.

    Args:
        pos: Position array for computing grid size
        dark_mode: Whether to use dark theme

    Returns:
        ViserServer instance with grid and origin frame
    """
    server = viser.ViserServer()
    if dark_mode:
        server.gui.configure_theme(dark_mode=True)

    grid_size = compute_grid_size(pos)
    server.scene.add_grid(
        "/grid",
        width=grid_size,
        height=grid_size,
        position=np.array([0.0, 0.0, 0.0]),
    )
    server.scene.add_frame(
        "/origin",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
    )

    return server


# =============================================================================
# Static Visualization Components
# =============================================================================


def _generate_ellipsoid_mesh(
    center: np.ndarray,
    radii: np.ndarray,
    axes: np.ndarray | None = None,
    subdivisions: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ellipsoid mesh vertices and faces via icosphere subdivision.

    Args:
        center: Center position (3,)
        radii: Radii along each principal axis (3,)
        axes: Rotation matrix (3, 3) defining principal axes. If None, uses identity.
        subdivisions: Number of icosphere subdivisions (higher = smoother)

    Returns:
        Tuple of (vertices, faces) where vertices is (V, 3) and faces is (F, 3)
    """
    # Start with icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio
    icosahedron_verts = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    # Normalize to unit sphere
    icosahedron_verts /= np.linalg.norm(icosahedron_verts[0])

    icosahedron_faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    vertices = icosahedron_verts
    faces = icosahedron_faces

    # Subdivide faces
    for _ in range(subdivisions):
        new_faces = []
        midpoint_cache = {}

        def get_midpoint(i1: int, i2: int) -> int:
            """Get or create midpoint vertex between two vertices."""
            key = (min(i1, i2), max(i1, i2))
            if key in midpoint_cache:
                return midpoint_cache[key]

            nonlocal vertices
            p1, p2 = vertices[i1], vertices[i2]
            mid = (p1 + p2) / 2.0
            mid = mid / np.linalg.norm(mid)  # Project onto unit sphere

            idx = len(vertices)
            vertices = np.vstack([vertices, mid])
            midpoint_cache[key] = idx
            return idx

        for tri in faces:
            v0, v1, v2 = tri
            a = get_midpoint(v0, v1)
            b = get_midpoint(v1, v2)
            c = get_midpoint(v2, v0)
            new_faces.extend([[v0, a, c], [v1, b, a], [v2, c, b], [a, b, c]])

        faces = np.array(new_faces, dtype=np.int32)

    # Scale by radii to create ellipsoid
    vertices = vertices / radii

    # Rotate by principal axes if provided
    if axes is not None:
        vertices = vertices @ axes.T

    # Translate to center
    vertices = vertices + center

    return vertices.astype(np.float32), faces


def add_ellipsoid_obstacles(
    server: viser.ViserServer,
    centers: list[np.ndarray],
    radii: list[np.ndarray],
    axes: list[np.ndarray] | None = None,
    color: tuple[int, int, int] = (255, 100, 100),
    opacity: float = 0.6,
    wireframe: bool = False,
    subdivisions: int = 2,
) -> list:
    """Add ellipsoidal obstacles to the scene.

    Args:
        server: ViserServer instance
        centers: List of center positions, each shape (3,)
        radii: List of radii along principal axes, each shape (3,)
        axes: List of rotation matrices (3, 3) defining principal axes.
            If None, ellipsoids are axis-aligned.
        color: RGB color tuple
        opacity: Opacity (0-1), only used when wireframe=False
        wireframe: If True, render as wireframe instead of solid
        subdivisions: Icosphere subdivisions (higher = smoother, 2 is usually good)

    Returns:
        List of mesh handles
    """
    handles = []

    if axes is None:
        axes = [None] * len(centers)

    for i, (center, rad, ax) in enumerate(zip(centers, radii, axes)):
        # Convert JAX arrays to numpy if needed
        center = np.asarray(center, dtype=np.float64)
        rad = np.asarray(rad, dtype=np.float64)
        if ax is not None:
            ax = np.asarray(ax, dtype=np.float64)

        vertices, faces = _generate_ellipsoid_mesh(center, rad, ax, subdivisions)

        handle = server.scene.add_mesh_simple(
            f"/obstacles/ellipsoid_{i}",
            vertices=vertices,
            faces=faces,
            color=color,
            wireframe=wireframe,
            opacity=opacity if not wireframe else 1.0,
        )
        handles.append(handle)

    return handles


def add_gates(
    server: viser.ViserServer,
    vertices: list,
    color: tuple[int, int, int] = (255, 165, 0),
    line_width: float = 3.0,
) -> None:
    """Add gate/obstacle wireframes to the scene.

    Args:
        server: ViserServer instance
        vertices: List of vertex arrays (4 vertices for planar gate, 8 for box)
        color: RGB color tuple
        line_width: Line width for wireframe
    """
    for i, verts in enumerate(vertices):
        verts = np.array(verts)
        n_verts = len(verts)

        if n_verts == 4:
            # Planar gate: 4 vertices forming a closed loop
            edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
        elif n_verts == 8:
            # 3D box: 8 vertices
            edges = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],  # front face
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],  # back face
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],  # connecting edges
            ]
        else:
            # Unknown format, skip
            continue

        # Shape (N, 2, 3) for N line segments
        points = np.array([[verts[e[0]], verts[e[1]]] for e in edges])
        server.scene.add_line_segments(
            f"/gates/gate_{i}",
            points=points,
            colors=color,
            line_width=line_width,
        )


def _generate_cone_mesh(
    apex: np.ndarray,
    height: float,
    half_angle_deg: float,
    n_segments: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a cone mesh with apex at given position, opening upward.

    Args:
        apex: Apex position (3,) - the tip of the cone
        height: Height of the cone (extends in +Z direction from apex)
        half_angle_deg: Half-angle of the cone from the vertical axis in degrees
        n_segments: Number of segments around the circumference

    Returns:
        Tuple of (vertices, faces) where vertices is (V, 3) and faces is (F, 3)
    """
    half_angle_rad = np.radians(half_angle_deg)
    base_radius = height * np.tan(half_angle_rad)

    # Vertices: apex + base circle points
    vertices = [apex.copy()]  # Apex at index 0

    # Base circle vertices
    for i in range(n_segments):
        angle = 2 * np.pi * i / n_segments
        x = apex[0] + base_radius * np.cos(angle)
        y = apex[1] + base_radius * np.sin(angle)
        z = apex[2] + height
        vertices.append([x, y, z])

    # Center of base for closing the bottom
    base_center = apex.copy()
    base_center[2] += height
    vertices.append(base_center)  # Index n_segments + 1

    vertices = np.array(vertices, dtype=np.float32)

    # Faces: triangles from apex to base edge pairs, plus base cap
    faces = []

    # Side faces (apex to each edge of base)
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        # Triangle: apex, base[i], base[next_i]
        faces.append([0, i + 1, next_i + 1])

    # Base cap faces (center to each edge)
    base_center_idx = n_segments + 1
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        # Triangle: center, base[next_i], base[i] (reverse winding for outward normal)
        faces.append([base_center_idx, next_i + 1, i + 1])

    faces = np.array(faces, dtype=np.int32)

    return vertices, faces


def add_glideslope_cone(
    server: viser.ViserServer,
    apex: np.ndarray | tuple = (0.0, 0.0, 0.0),
    height: float = 2000.0,
    glideslope_angle_deg: float = 86.0,
    color: tuple[int, int, int] = (100, 200, 100),
    opacity: float = 0.2,
    wireframe: bool = False,
    n_segments: int = 32,
) -> viser.MeshHandle:
    """Add a glideslope constraint cone to the scene.

    The glideslope constraint typically has the form:
        ||position_xy|| <= tan(angle) * position_z

    This creates a cone with apex at the landing site, opening upward.

    Args:
        server: ViserServer instance
        apex: Apex position (landing site), default is origin
        height: Height of the cone visualization
        glideslope_angle_deg: Glideslope angle in degrees (measured from vertical).
            For constraint ||r_xy|| <= tan(θ) * z, pass θ here.
            Common values: 86° (very wide), 70° (moderate), 45° (steep)
        color: RGB color tuple
        opacity: Opacity (0-1)
        wireframe: If True, render as wireframe
        n_segments: Number of segments for cone smoothness

    Returns:
        Mesh handle for the cone
    """
    apex = np.asarray(apex, dtype=np.float32)

    vertices, faces = _generate_cone_mesh(apex, height, glideslope_angle_deg, n_segments)

    handle = server.scene.add_mesh_simple(
        "/constraints/glideslope_cone",
        vertices=vertices,
        faces=faces,
        color=color,
        wireframe=wireframe,
        opacity=opacity if not wireframe else 1.0,
    )

    return handle


def add_ghost_trajectory(
    server: viser.ViserServer,
    pos: np.ndarray,
    colors: np.ndarray,
    opacity: float = 0.3,
    point_size: float = 0.05,
) -> None:
    """Add a faint ghost trajectory showing the full path.

    Args:
        server: ViserServer instance
        pos: Position array of shape (N, 3)
        colors: RGB color array of shape (N, 3)
        opacity: Opacity factor (0-1) applied to colors
        point_size: Size of trajectory points
    """
    ghost_colors = (colors * opacity).astype(np.uint8)
    server.scene.add_point_cloud(
        "/ghost_traj",
        points=pos,
        colors=ghost_colors,
        point_size=point_size,
    )


# =============================================================================
# Animated Visualization Components
# Each returns (handle, update_callback) where update_callback(frame_idx) updates the visual
# =============================================================================


def add_animated_trail(
    server: viser.ViserServer,
    pos: np.ndarray,
    colors: np.ndarray,
    point_size: float = 0.15,
) -> tuple[viser.PointCloudHandle, UpdateCallback]:
    """Add an animated trail that grows with the animation.

    Args:
        server: ViserServer instance
        pos: Position array of shape (N, 3)
        colors: RGB color array of shape (N, 3)
        point_size: Size of trail points

    Returns:
        Tuple of (handle, update_callback)
    """
    handle = server.scene.add_point_cloud(
        "/trail",
        points=pos[:1],
        colors=colors[:1],
        point_size=point_size,
    )

    def update(frame_idx: int) -> None:
        idx = frame_idx + 1  # Include current frame
        handle.points = pos[:idx]
        handle.colors = colors[:idx]

    return handle, update


def add_position_marker(
    server: viser.ViserServer,
    pos: np.ndarray,
    radius: float = 0.5,
    color: tuple[int, int, int] = (100, 200, 255),
) -> tuple[viser.IcosphereHandle, UpdateCallback]:
    """Add an animated position marker (sphere at current position).

    Args:
        server: ViserServer instance
        pos: Position array of shape (N, 3)
        radius: Marker radius
        color: RGB color tuple

    Returns:
        Tuple of (handle, update_callback)
    """
    handle = server.scene.add_icosphere(
        "/current_pos",
        radius=radius,
        color=color,
        position=pos[0],
    )

    def update(frame_idx: int) -> None:
        handle.position = pos[frame_idx]

    return handle, update


def add_target_marker(
    server: viser.ViserServer,
    target_pos: np.ndarray,
    name: str = "target",
    radius: float = 0.8,
    color: tuple[int, int, int] = (255, 50, 50),
    show_trail: bool = True,
    trail_color: tuple[int, int, int] | None = None,
) -> tuple[viser.IcosphereHandle, UpdateCallback | None]:
    """Add a viewplanning target marker (static or moving).

    Args:
        server: ViserServer instance
        target_pos: Target position - either shape (3,) for static or (N, 3) for moving
        name: Unique name for this target (used in scene path)
        radius: Marker radius
        color: RGB color tuple for marker
        show_trail: If True and target is moving, show trajectory trail
        trail_color: RGB color for trail (defaults to dimmed marker color)

    Returns:
        Tuple of (handle, update_callback). update_callback is None for static targets.
    """
    target_pos = np.asarray(target_pos)

    # Check if static (single position) or moving (trajectory)
    is_moving = target_pos.ndim == 2 and target_pos.shape[0] > 1

    initial_pos = target_pos[0] if is_moving else target_pos

    # Add marker
    handle = server.scene.add_icosphere(
        f"/targets/{name}/marker",
        radius=radius,
        color=color,
        position=initial_pos,
    )

    # For moving targets, optionally show trail
    if is_moving and show_trail:
        if trail_color is None:
            trail_color = tuple(int(c * 0.5) for c in color)
        server.scene.add_point_cloud(
            f"/targets/{name}/trail",
            points=target_pos,
            colors=trail_color,
            point_size=0.1,
        )

    if not is_moving:
        # Static target - no update needed
        return handle, None

    def update(frame_idx: int) -> None:
        # Clamp to valid range for target trajectory
        idx = min(frame_idx, len(target_pos) - 1)
        handle.position = target_pos[idx]

    return handle, update


def add_target_markers(
    server: viser.ViserServer,
    target_positions: list[np.ndarray],
    colors: list[tuple[int, int, int]] | None = None,
    radius: float = 0.8,
    show_trails: bool = True,
) -> list[tuple[viser.IcosphereHandle, UpdateCallback | None]]:
    """Add multiple viewplanning target markers.

    Args:
        server: ViserServer instance
        target_positions: List of target positions, each either (3,) or (N, 3)
        colors: List of RGB colors, one per target. Defaults to distinct colors.
        radius: Marker radius
        show_trails: If True, show trails for moving targets

    Returns:
        List of (handle, update_callback) tuples
    """
    # Default colors if not provided
    if colors is None:
        default_colors = [
            (255, 50, 50),  # Red
            (50, 255, 50),  # Green
            (50, 50, 255),  # Blue
            (255, 255, 50),  # Yellow
            (255, 50, 255),  # Magenta
            (50, 255, 255),  # Cyan
        ]
        colors = [default_colors[i % len(default_colors)] for i in range(len(target_positions))]

    results = []
    for i, (pos, color) in enumerate(zip(target_positions, colors)):
        handle, update = add_target_marker(
            server,
            pos,
            name=f"target_{i}",
            radius=radius,
            color=color,
            show_trail=show_trails,
        )
        results.append((handle, update))

    return results


def _rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (wxyz format).

    Args:
        v: Vector of shape (3,)
        q: Quaternion of shape (4,) in [w, x, y, z] format

    Returns:
        Rotated vector of shape (3,)
    """
    w, x, y, z = q
    # Quaternion rotation: v' = q * v * q_conj
    # Using the formula for rotating a vector by a quaternion
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


def add_thrust_vector(
    server: viser.ViserServer,
    pos: np.ndarray,
    thrust: np.ndarray | None,
    attitude: np.ndarray | None = None,
    scale: float = 0.3,
    color: tuple[int, int, int] = (255, 100, 100),
    line_width: float = 4.0,
) -> tuple[viser.LineSegmentsHandle | None, UpdateCallback | None]:
    """Add an animated thrust/force vector visualization.

    Args:
        server: ViserServer instance
        pos: Position array of shape (N, 3)
        thrust: Thrust/force array of shape (N, 3), or None to skip
        attitude: Quaternion array of shape (N, 4) in [w, x, y, z] format.
            If provided, thrust is assumed to be in body frame and will be
            rotated to world frame using the attitude.
        scale: Scale factor for thrust vector length
        color: RGB color tuple
        line_width: Line width

    Returns:
        Tuple of (handle, update_callback), or (None, None) if thrust is None
    """
    if thrust is None:
        return None, None

    def get_thrust_world(frame_idx: int) -> np.ndarray:
        """Get thrust vector in world frame."""
        thrust_body = thrust[frame_idx]
        if attitude is not None:
            return _rotate_vector_by_quaternion(thrust_body, attitude[frame_idx])
        return thrust_body

    thrust_world = get_thrust_world(0)
    thrust_end = pos[0] + thrust_world * scale
    handle = server.scene.add_line_segments(
        "/thrust_vector",
        points=np.array([[pos[0], thrust_end]]),  # Shape (1, 2, 3)
        colors=color,
        line_width=line_width,
    )

    def update(frame_idx: int) -> None:
        thrust_world = get_thrust_world(frame_idx)
        thrust_end = pos[frame_idx] + thrust_world * scale
        handle.points = np.array([[pos[frame_idx], thrust_end]])

    return handle, update


def add_attitude_frame(
    server: viser.ViserServer,
    pos: np.ndarray,
    attitude: np.ndarray | None,
    axes_length: float = 2.0,
    axes_radius: float = 0.05,
) -> tuple[viser.FrameHandle | None, UpdateCallback | None]:
    """Add an animated body coordinate frame showing attitude.

    Args:
        server: ViserServer instance
        pos: Position array of shape (N, 3)
        attitude: Quaternion array of shape (N, 4) in [w, x, y, z] format, or None to skip
        axes_length: Length of the coordinate axes
        axes_radius: Radius of the axes cylinders

    Returns:
        Tuple of (handle, update_callback), or (None, None) if attitude is None
    """
    if attitude is None:
        return None, None

    # Viser uses wxyz quaternion format
    handle = server.scene.add_frame(
        "/body_frame",
        wxyz=attitude[0],
        position=pos[0],
        axes_length=axes_length,
        axes_radius=axes_radius,
    )

    def update(frame_idx: int) -> None:
        handle.wxyz = attitude[frame_idx]
        handle.position = pos[frame_idx]

    return handle, update


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (wxyz format).

    Args:
        R: Rotation matrix of shape (3, 3)

    Returns:
        Quaternion array [w, x, y, z]
    """
    # Using Shepperd's method for numerical stability
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (wxyz format).

    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]

    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def add_viewcone(
    server: viser.ViserServer,
    pos: np.ndarray,
    attitude: np.ndarray | None,
    fov: float = 60.0,
    aspect: float = 1.0,
    scale: float = 5.0,
    R_sb: np.ndarray | None = None,
    color: tuple[int, int, int] = (255, 200, 100),
) -> tuple[viser.CameraFrustumHandle | None, UpdateCallback | None]:
    """Add an animated camera viewcone/frustum visualization.

    The sensor is assumed to look along +Z in its own frame (boresight = [0,0,1]).

    Args:
        server: ViserServer instance
        pos: Position array of shape (N, 3)
        attitude: Quaternion array of shape (N, 4) in [w, x, y, z] format, or None to skip
        fov: Field of view in degrees (full angle, not half-angle)
        aspect: Aspect ratio (width/height)
        scale: Size/depth of the frustum
        R_sb: Body-to-sensor rotation matrix (3x3). If None, sensor is aligned with body z-axis.
            This matrix transforms vectors FROM body frame TO sensor frame.
        color: RGB color tuple

    Returns:
        Tuple of (handle, update_callback), or (None, None) if attitude is None
    """
    if attitude is None:
        return None, None

    # Sensor-to-body is the inverse (transpose) of body-to-sensor
    R_sb = R_sb.T if R_sb is not None else np.eye(3)
    q_sb = _rotation_matrix_to_quaternion(R_sb)

    def get_sensor_quaternion(frame_idx: int) -> np.ndarray:
        """Get frustum orientation in world frame."""
        q_body = attitude[frame_idx]
        # Chain: body attitude * sensor-to-body * flip for viser convention
        q_sensor = _quaternion_multiply(q_body, q_sb)
        return q_sensor

    initial_wxyz = get_sensor_quaternion(0)
    handle = server.scene.add_camera_frustum(
        "/viewcone",
        fov=fov,
        aspect=aspect,
        scale=scale,
        wxyz=initial_wxyz,
        position=pos[0],
        color=color,
    )

    def update(frame_idx: int) -> None:
        handle.wxyz = get_sensor_quaternion(frame_idx)
        handle.position = pos[frame_idx]

    return handle, update


# =============================================================================
# SCP Iteration Visualization Components
# =============================================================================


def add_scp_iteration_nodes(
    server: viser.ViserServer,
    positions: list[np.ndarray],
    colors: list[tuple[int, int, int]] | None = None,
    point_size: float = 0.3,
    cmap_name: str = "viridis",
) -> tuple[list[viser.PointCloudHandle], UpdateCallback]:
    """Add animated optimization nodes that update per SCP iteration.

    Pre-buffers point clouds for all iterations and toggles visibility for performance.
    This avoids transmitting point data on every frame update.

    Args:
        server: ViserServer instance
        positions: List of position arrays per iteration, each shape (N, 3)
        colors: Optional list of RGB colors per iteration. If None, uses viridis colormap.
        point_size: Size of node markers
        cmap_name: Matplotlib colormap name (default: "viridis")

    Returns:
        Tuple of (list of point_handles, update_callback)
    """
    n_iterations = len(positions)

    # Default: use viridis colormap
    if colors is None:
        cmap = plt.get_cmap(cmap_name)
        colors = []
        for i in range(n_iterations):
            t = i / max(n_iterations - 1, 1)
            rgb = cmap(t)[:3]
            colors.append(tuple(int(c * 255) for c in rgb))

    # Convert colors to numpy arrays for viser compatibility
    colors_np = [np.array([c[0], c[1], c[2]], dtype=np.uint8) for c in colors]

    # Pre-create point clouds for all iterations (only first visible initially)
    handles = []
    for i in range(n_iterations):
        pos = np.asarray(positions[i], dtype=np.float32)
        handle = server.scene.add_point_cloud(
            f"/scp/nodes/iter_{i}",
            points=pos,
            colors=colors_np[i],
            point_size=point_size,
            visible=(i == 0),
        )
        handles.append(handle)

    # Track current visible iteration to minimize visibility toggles
    state = {"current_idx": 0}

    def update(iter_idx: int) -> None:
        idx = min(iter_idx, n_iterations - 1)
        if idx != state["current_idx"]:
            handles[state["current_idx"]].visible = False
            handles[idx].visible = True
            state["current_idx"] = idx

    return handles, update


def add_scp_iteration_attitudes(
    server: viser.ViserServer,
    positions: list[np.ndarray],
    attitudes: list[np.ndarray] | None,
    axes_length: float = 1.5,
    axes_radius: float = 0.03,
    stride: int = 1,
) -> tuple[list[viser.FrameHandle], UpdateCallback | None]:
    """Add animated attitude frames at each node that update per SCP iteration.

    Args:
        server: ViserServer instance
        positions: List of position arrays per iteration, each shape (N, 3)
        attitudes: List of quaternion arrays per iteration, each shape (N, 4) in wxyz format.
            If None, returns empty list and None callback.
        axes_length: Length of coordinate frame axes
        axes_radius: Radius of axes cylinders
        stride: Show attitude frame every `stride` nodes (1 = all nodes)

    Returns:
        Tuple of (list of frame handles, update_callback)
    """
    if attitudes is None:
        return [], None

    n_iterations = len(positions)
    n_nodes = len(positions[0])

    # Create frame handles for nodes at stride intervals
    node_indices = list(range(0, n_nodes, stride))
    handles = []

    for i, node_idx in enumerate(node_indices):
        handle = server.scene.add_frame(
            f"/scp/attitudes/frame_{i}",
            wxyz=attitudes[0][node_idx],
            position=positions[0][node_idx],
            axes_length=axes_length,
            axes_radius=axes_radius,
        )
        handles.append(handle)

    def update(iter_idx: int) -> None:
        idx = min(iter_idx, n_iterations - 1)
        pos = positions[idx]
        att = attitudes[idx]

        for i, node_idx in enumerate(node_indices):
            # Handle case where number of nodes changes between iterations
            if node_idx < len(pos) and node_idx < len(att):
                handles[i].position = pos[node_idx]
                handles[i].wxyz = att[node_idx]

    return handles, update


def add_scp_ghost_iterations(
    server: viser.ViserServer,
    positions: list[np.ndarray],
    point_size: float = 0.15,
    cmap_name: str = "viridis",
) -> tuple[list[viser.PointCloudHandle], UpdateCallback]:
    """Add ghost trails showing all previous SCP iterations.

    Pre-buffers point clouds for all iterations and toggles visibility for performance.
    Shows all previous iterations with viridis coloring to visualize convergence.

    Args:
        server: ViserServer instance
        positions: List of position arrays per iteration, each shape (N, 3)
        point_size: Size of ghost points
        cmap_name: Matplotlib colormap name for ghost colors

    Returns:
        Tuple of (list of handles, update_callback)
    """
    n_iterations = len(positions)
    cmap = plt.get_cmap(cmap_name)

    # Pre-create point clouds for all iterations with their colors
    # (all initially hidden, shown progressively as ghosts)
    handles = []
    for i in range(n_iterations):
        t = i / max(n_iterations - 1, 1)
        rgb = cmap(t)[:3]
        color = np.array([int(c * 255) for c in rgb], dtype=np.uint8)
        pos = np.asarray(positions[i], dtype=np.float32)

        handle = server.scene.add_point_cloud(
            f"/scp/ghosts/iter_{i}",
            points=pos,
            colors=color,
            point_size=point_size,
            visible=False,  # All start hidden
        )
        handles.append(handle)

    # Track which iterations are currently visible as ghosts
    state = {"visible_up_to": -1}

    def update(iter_idx: int) -> None:
        idx = min(iter_idx, n_iterations - 1)
        # Ghosts are iterations 0 through idx-1 (everything before current)
        new_visible_up_to = idx - 1

        if new_visible_up_to != state["visible_up_to"]:
            # Show/hide only the iterations that changed
            if new_visible_up_to > state["visible_up_to"]:
                # Show newly visible ghosts
                for i in range(state["visible_up_to"] + 1, new_visible_up_to + 1):
                    handles[i].visible = True
            else:
                # Hide ghosts that should no longer be visible
                for i in range(new_visible_up_to + 1, state["visible_up_to"] + 1):
                    handles[i].visible = False
            state["visible_up_to"] = new_visible_up_to

    return handles, update


def extract_propagation_positions(
    discretization_history: list[np.ndarray],
    n_x: int,
    n_u: int,
    position_slice: slice,
    scene_scale: float = 1.0,
) -> list[list[np.ndarray]]:
    """Extract 3D position trajectories from discretization history.

    The discretization history contains the multi-shot integration results.
    Each V matrix has shape (flattened_size, n_timesteps) where:
    - flattened_size = (N-1) * i4
    - i4 = n_x + n_x*n_x + 2*n_x*n_u (state + STM + control influence matrices)
    - n_timesteps = number of integration substeps

    Args:
        discretization_history: List of V matrices from each SCP iteration
        n_x: Number of states
        n_u: Number of controls
        position_slice: Slice for extracting position from state vector
        scene_scale: Divide positions by this factor for visualization

    Returns:
        List of propagation trajectories per iteration.
        Each iteration contains a list of (n_substeps, 3) arrays, one per segment.
    """
    if not discretization_history:
        return []

    i4 = n_x + n_x * n_x + 2 * n_x * n_u
    propagations = []

    for V in discretization_history:
        # V shape: (flattened_size, n_timesteps)
        n_timesteps = V.shape[1]
        n_segments = V.shape[0] // i4  # N-1 segments

        iteration_segments = []
        for seg_idx in range(n_segments):
            # Extract this segment's data across all timesteps
            seg_start = seg_idx * i4
            seg_end = seg_start + i4

            # For each timestep, extract the position from the state
            segment_positions = []
            for t_idx in range(n_timesteps):
                # Get full state at this segment and timestep
                state = V[seg_start:seg_end, t_idx][:n_x]
                # Extract position components
                pos = state[position_slice] / scene_scale
                segment_positions.append(pos)

            iteration_segments.append(np.array(segment_positions, dtype=np.float32))

        propagations.append(iteration_segments)

    return propagations


def add_scp_propagation_lines(
    server: viser.ViserServer,
    propagations: list[list[np.ndarray]],
    line_width: float = 2.0,
    cmap_name: str = "viridis",
) -> tuple[list, UpdateCallback]:
    """Add animated nonlinear propagation lines that update per SCP iteration.

    Shows the actual integrated trajectory between optimization nodes,
    revealing defects (gaps) in early iterations that close as SCP converges.
    All iterations up to the current one are shown with viridis coloring,
    similar to ghost iterations for nodes.

    Args:
        server: ViserServer instance
        propagations: List of propagation trajectories per iteration from
            extract_propagation_positions(). Each iteration contains a list
            of (n_substeps, 3) position arrays, one per segment.
        line_width: Width of propagation lines
        cmap_name: Matplotlib colormap name (default: "viridis")

    Returns:
        Tuple of (list of line handles, update_callback)
    """
    if not propagations:
        return [], lambda _: None

    n_iterations = len(propagations)
    n_segments = len(propagations[0])
    cmap = plt.get_cmap(cmap_name)

    # Pre-compute colors for each iteration
    iteration_colors = []
    for i in range(n_iterations):
        t = i / max(n_iterations - 1, 1)
        rgb = cmap(t)[:3]
        iteration_colors.append(np.array([int(c * 255) for c in rgb], dtype=np.uint8))

    # Create line handles for each (iteration, segment) pair
    # Structure: handles[iter_idx][seg_idx]
    all_handles = []

    for iter_idx in range(n_iterations):
        iter_handles = []
        color = iteration_colors[iter_idx]

        for seg_idx in range(n_segments):
            seg_pos = propagations[iter_idx][seg_idx]  # Shape (n_substeps, 3)

            if len(seg_pos) < 2:
                iter_handles.append(None)
                continue

            # Create line segments connecting consecutive substeps
            segments = np.array(
                [[seg_pos[i], seg_pos[i + 1]] for i in range(len(seg_pos) - 1)],
                dtype=np.float32,
            )

            handle = server.scene.add_line_segments(
                f"/scp/propagation/iter_{iter_idx}/segment_{seg_idx}",
                points=segments,
                colors=color,
                line_width=line_width,
                visible=(iter_idx == 0),  # Only first iteration visible initially
            )
            iter_handles.append(handle)

        all_handles.append(iter_handles)

    def update(iter_idx: int) -> None:
        idx = min(iter_idx, n_iterations - 1)

        # Show all iterations up to and including current, hide the rest
        for i in range(n_iterations):
            should_show = i <= idx
            for handle in all_handles[i]:
                if handle is not None:
                    handle.visible = should_show

    return all_handles, update


def add_scp_animation_controls(
    server: viser.ViserServer,
    n_iterations: int,
    update_callbacks: list[UpdateCallback],
    autoplay: bool = False,
    frame_duration_ms: int = 500,
    folder_name: str = "SCP Animation",
) -> None:
    """Add GUI controls for stepping through SCP iterations.

    Creates play/pause button, step buttons, iteration slider, and speed control.

    Args:
        server: ViserServer instance
        n_iterations: Total number of SCP iterations
        update_callbacks: List of update functions to call each iteration
        autoplay: Whether to start playing automatically
        frame_duration_ms: Default milliseconds per iteration frame
        folder_name: Name for the GUI folder
    """
    # Filter out None callbacks
    callbacks = [cb for cb in update_callbacks if cb is not None]

    def update_all(iter_idx: int) -> None:
        """Update all visualization components."""
        for callback in callbacks:
            callback(iter_idx)

    # --- GUI Controls ---
    with server.gui.add_folder(folder_name):
        play_button = server.gui.add_button("Play")
        with server.gui.add_folder("Step Controls", expand_by_default=False):
            prev_button = server.gui.add_button("◀ Previous")
            next_button = server.gui.add_button("Next ▶")
        iter_slider = server.gui.add_slider(
            "Iteration",
            min=0,
            max=n_iterations - 1,
            step=1,
            initial_value=0,
        )
        speed_slider = server.gui.add_slider(
            "Speed (ms/iter)",
            min=50,
            max=2000,
            step=50,
            initial_value=frame_duration_ms,
        )
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=True)

    # Animation state
    state = {"playing": autoplay, "iteration": 0, "needs_update": True}

    @play_button.on_click
    def _(_) -> None:
        state["playing"] = not state["playing"]
        state["needs_update"] = True  # Trigger immediate update on play
        play_button.name = "Pause" if state["playing"] else "Play"

    @prev_button.on_click
    def _(_) -> None:
        if state["iteration"] > 0:
            state["iteration"] -= 1
            iter_slider.value = state["iteration"]
            update_all(state["iteration"])

    @next_button.on_click
    def _(_) -> None:
        if state["iteration"] < n_iterations - 1:
            state["iteration"] += 1
            iter_slider.value = state["iteration"]
            update_all(state["iteration"])

    @iter_slider.on_update
    def _(_) -> None:
        if not state["playing"]:
            state["iteration"] = int(iter_slider.value)
            update_all(state["iteration"])

    def animation_loop() -> None:
        """Background thread for SCP iteration playback."""
        last_update = time.time()
        while True:
            time.sleep(0.016)  # ~60 fps check rate

            # Handle immediate update requests (e.g., on play button click)
            if state["needs_update"]:
                state["needs_update"] = False
                last_update = time.time()
                update_all(state["iteration"])
                continue

            if state["playing"]:
                current_time = time.time()
                elapsed_ms = (current_time - last_update) * 1000

                if elapsed_ms >= speed_slider.value:
                    last_update = current_time
                    state["iteration"] += 1

                    if state["iteration"] >= n_iterations:
                        if loop_checkbox.value:
                            state["iteration"] = 0
                        else:
                            state["iteration"] = n_iterations - 1
                            state["playing"] = False
                            play_button.name = "Play"

                    iter_slider.value = state["iteration"]
                    update_all(state["iteration"])

    # Start animation thread
    thread = threading.Thread(target=animation_loop, daemon=True)
    thread.start()

    # Initial update to ensure first frame is fully rendered
    update_all(0)


# =============================================================================
# Animation Controller
# =============================================================================


def add_animation_controls(
    server: viser.ViserServer,
    traj_time: np.ndarray,
    update_callbacks: list[UpdateCallback],
    loop: bool = True,
    folder_name: str = "Animation",
) -> None:
    """Add animation GUI controls and start the animation loop.

    Creates play/pause button, reset button, time slider, speed slider, and loop checkbox.
    Runs animation in a background daemon thread.

    Args:
        server: ViserServer instance
        traj_time: Time array of shape (N,) with timestamps for each frame
        update_callbacks: List of update functions to call each frame
        loop: Whether to loop animation by default
        folder_name: Name for the GUI folder
    """
    traj_time = traj_time.flatten()
    n_frames = len(traj_time)
    t_start, t_end = float(traj_time[0]), float(traj_time[-1])
    duration = t_end - t_start

    # Filter out None callbacks
    callbacks = [cb for cb in update_callbacks if cb is not None]

    def time_to_frame(t: float) -> int:
        """Convert simulation time to frame index."""
        return int(np.clip(np.searchsorted(traj_time, t, side="right") - 1, 0, n_frames - 1))

    def update_all(sim_t: float) -> None:
        """Update all visualization components."""
        idx = time_to_frame(sim_t)
        for callback in callbacks:
            callback(idx)

    # --- GUI Controls ---
    with server.gui.add_folder(folder_name):
        play_button = server.gui.add_button("Play")
        reset_button = server.gui.add_button("Reset")
        time_slider = server.gui.add_slider(
            "Time (s)",
            min=t_start,
            max=t_end,
            step=duration / 100,
            initial_value=t_start,
        )
        speed_slider = server.gui.add_slider(
            "Speed",
            min=0.1,
            max=5.0,
            step=0.1,
            initial_value=1.0,
        )
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=loop)

    # Animation state
    state = {"playing": False, "sim_time": t_start}

    @play_button.on_click
    def _(_) -> None:
        state["playing"] = not state["playing"]
        play_button.name = "Pause" if state["playing"] else "Play"

    @reset_button.on_click
    def _(_) -> None:
        state["sim_time"] = t_start
        time_slider.value = t_start
        update_all(t_start)

    @time_slider.on_update
    def _(_) -> None:
        if not state["playing"]:
            state["sim_time"] = float(time_slider.value)
            update_all(state["sim_time"])

    def animation_loop() -> None:
        """Background thread for realtime animation playback."""
        last_time = time.time()
        while True:
            time.sleep(0.016)  # ~60 fps
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            if state["playing"]:
                # Advance simulation time (speed=1.0 is realtime)
                state["sim_time"] += dt * speed_slider.value

                if state["sim_time"] >= t_end:
                    if loop_checkbox.value:
                        state["sim_time"] = t_start
                    else:
                        state["sim_time"] = t_end
                        state["playing"] = False
                        play_button.name = "Play"

                time_slider.value = state["sim_time"]
                update_all(state["sim_time"])

    # Start animation thread
    thread = threading.Thread(target=animation_loop, daemon=True)
    thread.start()
