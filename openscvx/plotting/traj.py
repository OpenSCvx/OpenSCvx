import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import viser


def _compute_velocity_colors(vel: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
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


def _compute_grid_size(pos: np.ndarray, padding: float = 1.2) -> float:
    """Compute grid size based on trajectory extent."""
    max_x = np.abs(pos[:, 0]).max()
    max_y = np.abs(pos[:, 1]).max()
    return max(max_x, max_y) * 2 * padding


def _add_static_elements(server: viser.ViserServer, results: dict) -> None:
    """Add static scene elements like obstacles and gates."""
    # Add gate/obstacle vertices if present
    if "vertices" in results:
        for i, verts in enumerate(results["vertices"]):
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
                colors=(255, 165, 0),  # orange
                line_width=3.0,
            )


def create_plotting_server(results: dict) -> viser.ViserServer:
    """Create a basic (non-animated) plotting server.

    Args:
        results: Optimization result dictionary containing trajectory data

    Returns:
        ViserServer instance
    """
    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)

    pos = results.trajectory["position"]
    grid_size = _compute_grid_size(pos)

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


def add_velocity_trace(server: viser.ViserServer, results: dict) -> None:
    """Add a static velocity-colored trajectory trace.

    Args:
        server: ViserServer instance
        results: Optimization result dictionary
    """
    pos = results.trajectory["position"]
    vel = results.trajectory["velocity"]
    colors = _compute_velocity_colors(vel)
    server.scene.add_point_cloud("/traj", points=pos, colors=colors)


def create_animated_plotting_server(
    results: dict,
    show_ghost_trajectory: bool = True,
    loop_animation: bool = True,
    thrust_key: str = "force",
) -> viser.ViserServer:
    """Create an animated trajectory visualization server.

    Features:
    - Play/pause button for animation
    - Frame slider to scrub through trajectory
    - Speed control slider
    - Velocity-colored trail that grows as animation progresses
    - Current position marker
    - Thrust vector visualization
    - Optional ghost trajectory showing full path
    - Static obstacles/gates if present in results

    Args:
        results: Optimization result dictionary containing trajectory data
        show_ghost_trajectory: If True, show faint full trajectory
        loop_animation: If True, loop animation when it reaches the end
        thrust_key: Key for thrust/force data in trajectory dict (default: "force")

    Returns:
        ViserServer instance (animation runs in background thread)
    """
    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)

    pos = results.trajectory["position"]
    vel = results.trajectory["velocity"]
    thrust = results.trajectory.get(thrust_key)
    traj_time = results.trajectory["time"].flatten()  # Time array
    n_frames = pos.shape[0]
    t_start, t_end = traj_time[0], traj_time[-1]
    duration = t_end - t_start

    # Setup scene
    grid_size = _compute_grid_size(pos)
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

    # Add static elements (gates, obstacles)
    _add_static_elements(server, results)

    # Precompute velocity colors
    colors = _compute_velocity_colors(vel)

    # Ghost trajectory (faint full path)
    if show_ghost_trajectory:
        ghost_colors = (colors * 0.3).astype(np.uint8)  # Dim the colors
        server.scene.add_point_cloud(
            "/ghost_traj",
            points=pos,
            colors=ghost_colors,
            point_size=0.05,
        )

    # --- GUI Controls ---
    with server.gui.add_folder("Animation"):
        play_button = server.gui.add_button("Play")
        reset_button = server.gui.add_button("Reset")
        time_slider = server.gui.add_slider(
            "Time (s)",
            min=float(t_start),
            max=float(t_end),
            step=float(duration / 100),
            initial_value=float(t_start),
        )
        speed_slider = server.gui.add_slider(
            "Speed",
            min=0.1,
            max=5.0,
            step=0.1,
            initial_value=1.0,
        )
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=loop_animation)

    # --- Dynamic Scene Handles ---
    # Trail showing trajectory up to current frame
    trail_handle = server.scene.add_point_cloud(
        "/trail",
        points=pos[:1],
        colors=colors[:1],
        point_size=0.15,
    )

    # Current position marker
    marker_handle = server.scene.add_icosphere(
        "/current_pos",
        radius=0.5,
        color=(100, 200, 255),  # Cyan/light blue
        position=pos[0],
    )

    # Thrust vector at current position
    thrust_scale = 0.3  # Scale factor for thrust visualization
    thrust_line_handle = None
    if thrust is not None:
        thrust_end = pos[0] + thrust[0] * thrust_scale
        thrust_line_handle = server.scene.add_line_segments(
            "/thrust_vector",
            points=np.array([[pos[0], thrust_end]]),  # Shape (1, 2, 3)
            colors=(255, 100, 100),  # Red for thrust
            line_width=4.0,
        )

    # Animation state (track simulation time, not frame)
    state = {"playing": False, "sim_time": float(t_start)}

    def time_to_frame(t: float) -> int:
        """Convert simulation time to frame index."""
        return int(np.clip(np.searchsorted(traj_time, t, side="right") - 1, 0, n_frames - 1))

    def update_scene(sim_t: float) -> None:
        """Update all dynamic scene elements for given simulation time."""
        idx = time_to_frame(sim_t)

        # Update trail (show all points up to current time)
        trail_handle.points = pos[: idx + 1]
        trail_handle.colors = colors[: idx + 1]

        # Update marker position
        marker_handle.position = pos[idx]

        # Update thrust vector
        if thrust_line_handle is not None:
            thrust_end = pos[idx] + thrust[idx] * thrust_scale
            thrust_line_handle.points = np.array([[pos[idx], thrust_end]])  # Shape (1, 2, 3)

    @play_button.on_click
    def _(_) -> None:
        state["playing"] = not state["playing"]
        play_button.name = "Pause" if state["playing"] else "Play"

    @reset_button.on_click
    def _(_) -> None:
        state["sim_time"] = float(t_start)
        time_slider.value = float(t_start)
        update_scene(t_start)

    @time_slider.on_update
    def _(_) -> None:
        if not state["playing"]:
            state["sim_time"] = float(time_slider.value)
            update_scene(state["sim_time"])

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
                        state["sim_time"] = float(t_start)
                    else:
                        state["sim_time"] = float(t_end)
                        state["playing"] = False
                        play_button.name = "Play"

                time_slider.value = state["sim_time"]
                update_scene(state["sim_time"])

    # Start animation thread
    thread = threading.Thread(target=animation_loop, daemon=True)
    thread.start()

    return server
