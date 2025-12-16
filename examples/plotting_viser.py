"""Viser-based trajectory visualization templates.

This module provides convenience functions for common visualization patterns.
These are templates meant to be copied and customized for specific problems.

For the composable primitives, see openscvx.plotting.traj.
"""

import numpy as np
import viser

from openscvx.algorithms import OptimizationResults
from openscvx.plotting.traj import (
    add_animated_trail,
    add_animation_controls,
    add_attitude_frame,
    add_ellipsoid_obstacles,
    add_gates,
    add_ghost_trajectory,
    add_position_marker,
    add_thrust_vector,
    add_viewcone,
    compute_velocity_colors,
    create_server,
)


def create_plotting_server(results: OptimizationResults) -> viser.ViserServer:
    """Create a basic (non-animated) plotting server.

    Args:
        results: Optimization result dictionary containing trajectory data

    Returns:
        ViserServer instance
    """
    pos = results.trajectory["position"]
    return create_server(pos)


def add_velocity_trace(server: viser.ViserServer, results: OptimizationResults) -> None:
    """Add a static velocity-colored trajectory trace.

    Args:
        server: ViserServer instance
        results: Optimization result dictionary
    """
    pos = results.trajectory["position"]
    vel = results.trajectory["velocity"]
    colors = compute_velocity_colors(vel)
    server.scene.add_point_cloud("/traj", points=pos, colors=colors)


def create_animated_plotting_server(
    results: OptimizationResults,
    show_ghost_trajectory: bool = True,
    loop_animation: bool = True,
    thrust_key: str = "force",
    thrust_scale: float = 0.3,
    attitude_key: str = "attitude",
    attitude_axes_length: float = 2.0,
    show_viewcone: bool = True,
    viewcone_fov: float = 60.0,
    viewcone_scale: float = 5.0,
) -> viser.ViserServer:
    """Create an animated trajectory visualization server.

    This is a convenience function that composes the modular components.
    For custom visualizations, use the individual add_* functions directly.

    Features:
    - Play/pause button for animation
    - Time slider to scrub through trajectory (realtime playback)
    - Speed control slider
    - Velocity-colored trail that grows as animation progresses
    - Current position marker
    - Thrust vector visualization (if thrust data available)
    - Body frame attitude visualization (if attitude data available, for 6DOF)
    - Viewcone/camera frustum (if R_sb (body-to-sensor) in results and show_viewcone=True)
    - Optional ghost trajectory showing full path
    - Static obstacles/gates if present in results
    - Ellipsoidal obstacles (if obstacles_centers, obstacles_radii, obstacles_axes in results)

    Args:
        results: Optimization result dictionary containing trajectory data
        show_ghost_trajectory: If True, show faint full trajectory
        loop_animation: If True, loop animation when it reaches the end
        thrust_key: Key for thrust/force data in trajectory dict (default: "force")
        thrust_scale: Scale factor for thrust vector visualization
        attitude_key: Key for attitude quaternion data (default: "attitude")
        attitude_axes_length: Length of body frame axes
        show_viewcone: If True and R_sb is in results, show camera viewcone
        viewcone_fov: Field of view for viewcone in degrees
        viewcone_scale: Size/depth of viewcone frustum

    Returns:
        ViserServer instance (animation runs in background thread)
    """
    # Extract data
    pos = results.trajectory["position"]
    vel = results.trajectory["velocity"]
    thrust = results.trajectory.get(thrust_key)
    attitude = results.trajectory.get(attitude_key)
    traj_time = results.trajectory["time"]

    # Viewcone parameters (body-to-sensor rotation)
    # Note: In many problems, R_sb is named as such but is actually body-to-sensor
    R_sb = results.get("R_sb")

    # Precompute colors
    colors = compute_velocity_colors(vel)

    # Create server
    server = create_server(pos)

    # Add static elements
    if "vertices" in results:
        add_gates(server, results["vertices"])

    # Add ellipsoidal obstacles if present
    if "obstacles_centers" in results:
        add_ellipsoid_obstacles(
            server,
            centers=results["obstacles_centers"],
            radii=results.get("obstacles_radii", [np.ones(3)] * len(results["obstacles_centers"])),
            axes=results.get("obstacles_axes"),
        )

    if show_ghost_trajectory:
        add_ghost_trajectory(server, pos, colors)

    # Add animated elements (collect update callbacks)
    update_callbacks = []

    _, update_trail = add_animated_trail(server, pos, colors)
    update_callbacks.append(update_trail)

    # Use position marker for point-mass, attitude frame for 6DOF
    if attitude is not None:
        _, update_attitude = add_attitude_frame(
            server, pos, attitude, axes_length=attitude_axes_length
        )
        update_callbacks.append(update_attitude)
    else:
        _, update_marker = add_position_marker(server, pos)
        update_callbacks.append(update_marker)

    _, update_thrust = add_thrust_vector(server, pos, thrust, attitude=attitude, scale=thrust_scale)
    update_callbacks.append(update_thrust)  # Will be filtered out if None

    # Add viewcone if R_sb is available and enabled
    if show_viewcone and R_sb is not None and attitude is not None:
        _, update_viewcone = add_viewcone(
            server,
            pos,
            attitude,
            fov=viewcone_fov,
            scale=viewcone_scale,
            R_sb=R_sb,
        )
        update_callbacks.append(update_viewcone)

    # Add animation controls
    add_animation_controls(server, traj_time, update_callbacks, loop=loop_animation)

    return server
