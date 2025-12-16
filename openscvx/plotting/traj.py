"""Viser-based trajectory visualization.

This module provides modular components for visualizing trajectory optimization results.
Components can be composed together for custom visualizations, or use the convenience
function `create_animated_plotting_server` for a complete out-of-the-box experience.

Example (modular usage):
    server = create_server(pos)
    add_gates(server, vertices)
    add_ghost_trajectory(server, pos, colors)

    _, update_trail = add_animated_trail(server, pos, colors)
    _, update_marker = add_position_marker(server, pos)
    _, update_thrust = add_thrust_vector(server, pos, thrust)

    add_animation_controls(server, traj_time, [update_trail, update_marker, update_thrust])
    server.sleep_forever()

Example (convenience function):
    server = create_animated_plotting_server(results)
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


def add_thrust_vector(
    server: viser.ViserServer,
    pos: np.ndarray,
    thrust: np.ndarray | None,
    scale: float = 0.3,
    color: tuple[int, int, int] = (255, 100, 100),
    line_width: float = 4.0,
) -> tuple[viser.LineSegmentsHandle | None, UpdateCallback | None]:
    """Add an animated thrust/force vector visualization.

    Args:
        server: ViserServer instance
        pos: Position array of shape (N, 3)
        thrust: Thrust/force array of shape (N, 3), or None to skip
        scale: Scale factor for thrust vector length
        color: RGB color tuple
        line_width: Line width

    Returns:
        Tuple of (handle, update_callback), or (None, None) if thrust is None
    """
    if thrust is None:
        return None, None

    thrust_end = pos[0] + thrust[0] * scale
    handle = server.scene.add_line_segments(
        "/thrust_vector",
        points=np.array([[pos[0], thrust_end]]),  # Shape (1, 2, 3)
        colors=color,
        line_width=line_width,
    )

    def update(frame_idx: int) -> None:
        thrust_end = pos[frame_idx] + thrust[frame_idx] * scale
        handle.points = np.array([[pos[frame_idx], thrust_end]])

    return handle, update


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


# =============================================================================
# Convenience Functions
# =============================================================================


def create_plotting_server(results: dict) -> viser.ViserServer:
    """Create a basic (non-animated) plotting server.

    Args:
        results: Optimization result dictionary containing trajectory data

    Returns:
        ViserServer instance
    """
    pos = results.trajectory["position"]
    return create_server(pos)


def add_velocity_trace(server: viser.ViserServer, results: dict) -> None:
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
    results: dict,
    show_ghost_trajectory: bool = True,
    loop_animation: bool = True,
    thrust_key: str = "force",
    thrust_scale: float = 0.3,
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
    - Optional ghost trajectory showing full path
    - Static obstacles/gates if present in results

    Args:
        results: Optimization result dictionary containing trajectory data
        show_ghost_trajectory: If True, show faint full trajectory
        loop_animation: If True, loop animation when it reaches the end
        thrust_key: Key for thrust/force data in trajectory dict (default: "force")
        thrust_scale: Scale factor for thrust vector visualization

    Returns:
        ViserServer instance (animation runs in background thread)
    """
    # Extract data
    pos = results.trajectory["position"]
    vel = results.trajectory["velocity"]
    thrust = results.trajectory.get(thrust_key)
    traj_time = results.trajectory["time"]

    # Precompute colors
    colors = compute_velocity_colors(vel)

    # Create server
    server = create_server(pos)

    # Add static elements
    if "vertices" in results:
        add_gates(server, results["vertices"])

    if show_ghost_trajectory:
        add_ghost_trajectory(server, pos, colors)

    # Add animated elements (collect update callbacks)
    update_callbacks = []

    _, update_trail = add_animated_trail(server, pos, colors)
    update_callbacks.append(update_trail)

    _, update_marker = add_position_marker(server, pos)
    update_callbacks.append(update_marker)

    _, update_thrust = add_thrust_vector(server, pos, thrust, scale=thrust_scale)
    update_callbacks.append(update_thrust)  # Will be filtered out if None

    # Add animation controls
    add_animation_controls(server, traj_time, update_callbacks, loop=loop_animation)

    return server
