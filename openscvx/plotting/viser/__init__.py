"""Viser-based 3D trajectory visualization.

This module provides composable primitives for visualizing trajectory optimization
results using viser, a web-based 3D visualization library.

Example usage:
    from openscvx.plotting import viser

    # Create server with basic scene setup
    server = viser.create_server(positions)

    # Add static elements
    viser.add_gates(server, gate_vertices)
    viser.add_ellipsoid_obstacles(server, centers, radii)
    viser.add_ghost_trajectory(server, positions, colors)

    # Add animated elements (returns handle and update callback)
    _, update_trail = viser.add_animated_trail(server, positions, colors)
    _, update_marker = viser.add_position_marker(server, positions)
    _, update_thrust = viser.add_thrust_vector(server, positions, thrust)

    # Add animation controls and start playback
    viser.add_animation_controls(
        server, time_array,
        [update_trail, update_marker, update_thrust]
    )
    server.sleep_forever()

For higher-level convenience functions, see examples/plotting_viser.py.
"""

# Server setup
from .server import compute_grid_size, compute_velocity_colors, create_server

# Static primitives
from .primitives import (
    add_ellipsoid_obstacles,
    add_gates,
    add_ghost_trajectory,
    add_glideslope_cone,
)

# Animated components
from .animated import (
    UpdateCallback,
    add_animated_trail,
    add_attitude_frame,
    add_position_marker,
    add_target_marker,
    add_target_markers,
    add_thrust_vector,
    add_viewcone,
)

# Animation controls
from .controls import add_animation_controls

# SCP iteration visualization
from .scp import (
    add_scp_animation_controls,
    add_scp_ghost_iterations,
    add_scp_iteration_attitudes,
    add_scp_iteration_nodes,
    add_scp_propagation_lines,
    extract_propagation_positions,
)

__all__ = [
    # Server
    "create_server",
    "compute_velocity_colors",
    "compute_grid_size",
    # Static primitives
    "add_gates",
    "add_ellipsoid_obstacles",
    "add_glideslope_cone",
    "add_ghost_trajectory",
    # Animated components
    "UpdateCallback",
    "add_animated_trail",
    "add_position_marker",
    "add_target_marker",
    "add_target_markers",
    "add_thrust_vector",
    "add_attitude_frame",
    "add_viewcone",
    # Animation controls
    "add_animation_controls",
    # SCP visualization
    "add_scp_iteration_nodes",
    "add_scp_iteration_attitudes",
    "add_scp_ghost_iterations",
    "extract_propagation_positions",
    "add_scp_propagation_lines",
    "add_scp_animation_controls",
]
