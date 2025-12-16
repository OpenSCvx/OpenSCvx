"""Viser-based trajectory visualization templates.

This module provides convenience functions for common visualization patterns.
These are templates meant to be copied and customized for specific problems.

For the composable primitives, see openscvx.plotting.animation.
"""

import numpy as np
import viser

from openscvx.algorithms import OptimizationResults
from openscvx.plotting.animation import (
    add_animated_trail,
    add_animation_controls,
    add_attitude_frame,
    add_ellipsoid_obstacles,
    add_gates,
    add_ghost_trajectory,
    add_glideslope_cone,
    add_position_marker,
    add_scp_animation_controls,
    add_scp_ghost_iterations,
    add_scp_iteration_attitudes,
    add_scp_iteration_nodes,
    add_target_markers,
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
    viewcone_fov: float | None = None,
    viewcone_scale: float = 5.0,
    show_targets: bool = True,
    target_radius: float = 1.0,
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
    - Viewcone/camera frustum (if R_sb in results and show_viewcone=True)
    - Target markers for viewplanning (if init_poses in results and show_targets=True)
    - Optional ghost trajectory showing full path
    - Static obstacles/gates if present in results
    - Ellipsoidal obstacles (if obstacles_centers, obstacles_radii, obstacles_axes in results)

    Args:
        results: Optimization result dictionary containing trajectory data.
            Expected keys in results (beyond trajectory data):
            - vertices: Gate/obstacle vertices (optional)
            - R_sb: Body-to-sensor rotation matrix for viewcone (optional)
            - alpha_x: Sensor cone half-angle parameter for FOV calculation (optional)
            - init_poses: List of viewplanning target positions (optional)
            - obstacles_centers, obstacles_radii, obstacles_axes: Ellipsoid obstacles (optional)
        show_ghost_trajectory: If True, show faint full trajectory
        loop_animation: If True, loop animation when it reaches the end
        thrust_key: Key for thrust/force data in trajectory dict (default: "force")
        thrust_scale: Scale factor for thrust vector visualization
        attitude_key: Key for attitude quaternion data (default: "attitude")
        attitude_axes_length: Length of body frame axes
        show_viewcone: If True and R_sb is in results, show camera viewcone
        viewcone_fov: Field of view for viewcone in degrees. If None, computed from
            alpha_x in results (fov = 180/alpha_x degrees), or defaults to 60.0.
        viewcone_scale: Size/depth of viewcone frustum
        show_targets: If True and init_poses in results, show target markers
        target_radius: Radius of target marker spheres

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
    alpha_x = results.get("alpha_x")

    # Compute FOV from alpha_x if not explicitly provided
    # alpha_x defines the cone half-angle as pi/alpha_x radians
    if viewcone_fov is None:
        if alpha_x is not None:
            viewcone_fov = np.degrees(np.pi / alpha_x)
        else:
            viewcone_fov = 60.0  # Default

    # Viewplanning target positions
    init_poses = results.get("init_poses")

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

    # Add target markers for viewplanning problems
    if show_targets and init_poses is not None:
        target_results = add_target_markers(server, init_poses, radius=target_radius)
        for _, update in target_results:
            if update is not None:
                update_callbacks.append(update)

    # Add animation controls
    add_animation_controls(server, traj_time, update_callbacks, loop=loop_animation)

    return server


def create_scp_animated_plotting_server(
    results: OptimizationResults,
    position_slice: slice | None = None,
    attitude_slice: slice | None = None,
    show_ghost_iterations: bool = True,
    show_attitudes: bool = True,
    attitude_stride: int = 3,
    attitude_axes_length: float = 1.5,
    node_point_size: float = 0.3,
    show_lines: bool = True,
    frame_duration_ms: int = 500,
    scene_scale: float = 1.0,
) -> viser.ViserServer:
    """Create an animated visualization of SCP iteration convergence.

    This shows how the optimization nodes evolve across SCP iterations,
    allowing you to visualize the convergence process.

    Features:
    - Play/pause button for iteration animation
    - Previous/Next buttons to step through iterations
    - Iteration slider to scrub through convergence history
    - Speed control for playback
    - Node positions colored by iteration (red -> green as it converges)
    - Optional ghost trails showing previous iterations
    - Optional attitude frames at each node (for 6DOF problems)
    - Static obstacles/gates if present in results

    Args:
        results: Optimization results containing SCP iteration history (results.X).
        position_slice: Slice for extracting position from state vector.
            If None, auto-detected from results._states looking for "position".
        attitude_slice: Slice for extracting attitude quaternion from state vector.
            If None, auto-detected from results._states looking for "attitude".
        show_ghost_iterations: If True, show all previous iterations with viridis coloring
        show_attitudes: If True and attitude data available, show body frames
        attitude_stride: Show attitude frame every N nodes (reduces clutter)
        attitude_axes_length: Length of attitude coordinate frame axes
        node_point_size: Size of node markers
        show_lines: If True, connect nodes with line segments
        frame_duration_ms: Default milliseconds per iteration frame
        scene_scale: Divide all positions by this factor. Use >1 for large-scale
            trajectories (e.g., 100.0 for km-scale problems).

    Returns:
        ViserServer instance (animation runs in background thread)
    """
    # Get iteration history
    X_history = results.X  # List of state arrays per iteration
    n_iterations = len(X_history)

    if n_iterations == 0:
        raise ValueError("No SCP iteration history available in results.X")

    # Auto-detect slices from state metadata if not provided
    if position_slice is None or attitude_slice is None:
        states = getattr(results, "_states", [])
        for state in states:
            if position_slice is None and state.name.lower() == "position":
                position_slice = state._slice
            if attitude_slice is None and state.name.lower() == "attitude":
                attitude_slice = state._slice

    # Default position slice if still not found (assume first 3 states)
    if position_slice is None:
        position_slice = slice(0, 3)

    # Extract position history and apply scene scale
    positions = [X[:, position_slice] / scene_scale for X in X_history]

    # Extract attitude history if available
    attitudes = None
    if attitude_slice is not None:
        attitudes = [X[:, attitude_slice] for X in X_history]

    # Create server using final iteration's positions for grid sizing
    server = create_server(positions[-1])

    # Add static elements (gates, obstacles) if present
    if "vertices" in results:
        add_gates(server, results["vertices"])

    if "obstacles_centers" in results:
        add_ellipsoid_obstacles(
            server,
            centers=results["obstacles_centers"],
            radii=results.get("obstacles_radii", [np.ones(3)] * len(results["obstacles_centers"])),
            axes=results.get("obstacles_axes"),
        )

    # Collect update callbacks
    update_callbacks = []

    # Add ghost iterations (previous iterations with viridis coloring)
    if show_ghost_iterations:
        _, update_ghosts = add_scp_ghost_iterations(server, positions)
        update_callbacks.append(update_ghosts)

    # Add main iteration nodes
    _, _, update_nodes = add_scp_iteration_nodes(
        server,
        positions,
        point_size=node_point_size,
        show_lines=show_lines,
    )
    update_callbacks.append(update_nodes)

    # Add attitude frames if available and enabled
    if show_attitudes and attitudes is not None:
        _, update_attitudes = add_scp_iteration_attitudes(
            server,
            positions,
            attitudes,
            axes_length=attitude_axes_length,
            stride=attitude_stride,
        )
        update_callbacks.append(update_attitudes)

    # Add SCP animation controls
    add_scp_animation_controls(
        server,
        n_iterations,
        update_callbacks,
        frame_duration_ms=frame_duration_ms,
    )

    return server


def create_pdg_animated_plotting_server(
    results: OptimizationResults,
    show_ghost_trajectory: bool = True,
    loop_animation: bool = True,
    thrust_key: str = "thrust",
    thrust_scale: float = 0.0001,
    thrust_vector_scale: float = 1.0,
    show_glideslope: bool = True,
    glideslope_angle_deg: float | None = None,
    glideslope_height: float | None = None,
    marker_radius: float = 0.3,
    trail_point_size: float = 0.15,
    ghost_point_size: float = 0.05,
    scene_scale: float = 100.0,
) -> viser.ViserServer:
    """Create an animated visualization for Powered Descent Guidance problems.

    This is specialized for rocket landing trajectories with:
    - 3D position and velocity
    - Thrust vector visualization
    - Glideslope constraint cone

    All positions are divided by scene_scale to bring large-scale trajectories
    (e.g., 2000m) into a range that viser handles well (~20m).

    Args:
        results: Optimization result dictionary containing trajectory data.
            Expected keys:
            - trajectory["position"]: 3D position (N, 3)
            - trajectory["velocity"]: 3D velocity (N, 3)
            - trajectory[thrust_key]: Thrust vector (N, 3)
            - glideslope_angle_deg: Glideslope angle in degrees (optional, for cone)
        show_ghost_trajectory: If True, show faint full trajectory
        loop_animation: If True, loop animation when it reaches the end
        thrust_key: Key for thrust data in trajectory dict
        thrust_scale: Converts thrust magnitude (Newtons) to scene units.
            E.g., 0.0001 means 10000N becomes 1 scene unit.
        thrust_vector_scale: Additional multiplier for thrust vector length.
        show_glideslope: If True, show glideslope constraint cone
        glideslope_angle_deg: Glideslope angle in degrees. If None, uses value from
            results["glideslope_angle_deg"] or defaults to 86.0.
        glideslope_height: Height of glideslope cone visualization (in original units).
            If None, uses 10% of the initial altitude.
        marker_radius: Radius of position marker (in scaled scene units).
        trail_point_size: Size of trail points.
        ghost_point_size: Size of ghost trajectory points.
        scene_scale: Divide all positions by this factor. Default 100.0 brings
            km-scale trajectories into a ~10-20m range for viser.

    Returns:
        ViserServer instance (animation runs in background thread)
    """
    # Extract and scale position data
    pos = results.trajectory["position"] / scene_scale
    vel = results.trajectory["velocity"]
    thrust = results.trajectory.get(thrust_key)
    traj_time = results.trajectory["time"]

    # Combined thrust scale factor
    combined_thrust_scale = thrust_scale * thrust_vector_scale

    # Get glideslope parameters
    if glideslope_angle_deg is None:
        glideslope_angle_deg = results.get("glideslope_angle_deg", 86.0)

    if glideslope_height is None:
        # Default to 20% of initial altitude - just show near landing point
        glideslope_height = float(results.trajectory["position"][0, 2]) * 0.1
    glideslope_height_scaled = glideslope_height / scene_scale

    # Precompute colors
    colors = compute_velocity_colors(vel)

    # Create server
    server = create_server(pos)

    # Add static elements
    if show_glideslope:
        add_glideslope_cone(
            server,
            apex=(0, 0, 0),
            height=glideslope_height_scaled,
            glideslope_angle_deg=glideslope_angle_deg,
        )

    if show_ghost_trajectory:
        add_ghost_trajectory(server, pos, colors, point_size=ghost_point_size)

    # Add animated elements
    update_callbacks = []

    _, update_trail = add_animated_trail(server, pos, colors, point_size=trail_point_size)
    update_callbacks.append(update_trail)

    _, update_marker = add_position_marker(server, pos, radius=marker_radius)
    update_callbacks.append(update_marker)

    # Thrust vector (no attitude for 3DoF, thrust is in world frame)
    _, update_thrust = add_thrust_vector(
        server, pos, thrust, attitude=None, scale=combined_thrust_scale
    )
    update_callbacks.append(update_thrust)

    # Add animation controls
    add_animation_controls(server, traj_time, update_callbacks, loop=loop_animation)

    return server
