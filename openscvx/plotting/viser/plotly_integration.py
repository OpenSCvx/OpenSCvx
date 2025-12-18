"""Plotly integration for viser - animated 2D plots synchronized with 3D visualization.

This module provides utilities for embedding plotly figures in viser's GUI with
animated markers that synchronize with the 3D animation timeline.
"""

import numpy as np
import plotly.graph_objects as go
import viser

from openscvx.algorithms import OptimizationResults


def add_animated_plotly_marker(
    server: viser.ViserServer,
    fig: go.Figure,
    time_array: np.ndarray,
    marker_x_data: np.ndarray,
    marker_y_data: np.ndarray,
    use_trajectory_indexing: bool = True,
    marker_name: str = "Current",
    marker_color: str = "red",
    marker_size: int = 12,
    folder_name: str | None = None,
    aspect: float = 1.5,
) -> tuple:
    """Add a plotly figure to viser GUI with an animated time marker.

    This function takes any plotly figure and adds an animated marker that
    synchronizes with viser's 3D animation timeline. The marker shows the
    current position on the plot as the animation plays.

    Args:
        server: ViserServer instance
        fig: Plotly figure to display
        time_array: Time values corresponding to animation frames (N,).
            This should match the time array passed to add_animation_controls().
        marker_x_data: X-axis values for marker position (N,)
        marker_y_data: Y-axis values for marker position (N,)
        use_trajectory_indexing: If True, frame_idx maps directly to data indices.
            If False, searches for nearest time value (use for node-only data).
        marker_name: Legend name for the marker trace
        marker_color: Color of the animated marker
        marker_size: Size of the animated marker in points
        folder_name: Optional GUI folder name to organize plots
        aspect: Aspect ratio for plot display (width/height)

    Returns:
        Tuple of (plot_handle, update_callback)

    Example::

        from openscvx.plotting import plot_vector_norm, viser

        # Create any plotly figure
        fig = plot_vector_norm(results, "thrust")
        thrust_norms = np.linalg.norm(results.trajectory["thrust"], axis=1)

        # Add to viser with animated marker
        _, update_plot = viser.add_animated_plotly_marker(
            server, fig,
            time_array=results.trajectory["time"].flatten(),
            marker_x_data=results.trajectory["time"].flatten(),
            marker_y_data=thrust_norms,
        )

        # Add to animation callbacks
        update_callbacks.append(update_plot)
    """
    # Add marker trace to figure
    marker_trace = go.Scatter(
        x=[marker_x_data[0]],
        y=[marker_y_data[0]],
        mode="markers",
        marker={"color": marker_color, "size": marker_size, "symbol": "circle"},
        name=marker_name,
    )
    fig.add_trace(marker_trace)
    marker_trace_idx = len(fig.data) - 1

    # Add to viser GUI
    if folder_name:
        with server.gui.add_folder(folder_name):
            plot_handle = server.gui.add_plotly(figure=fig, aspect=aspect)
    else:
        plot_handle = server.gui.add_plotly(figure=fig, aspect=aspect)

    # Create update callback
    def update(frame_idx: int) -> None:
        """Update marker position based on current frame."""
        if use_trajectory_indexing:
            # Direct indexing: frame_idx corresponds to data index
            idx = min(frame_idx, len(marker_y_data) - 1)
        else:
            # Search for nearest time (for node-only data)
            current_time = time_array[frame_idx]
            idx = min(np.searchsorted(marker_x_data, current_time), len(marker_y_data) - 1)

        # Update marker position
        fig.data[marker_trace_idx].x = [marker_x_data[idx]]
        fig.data[marker_trace_idx].y = [marker_y_data[idx]]

        # Trigger viser update
        plot_handle.figure = fig

    return plot_handle, update


def add_animated_vector_norm_plot(
    server: viser.ViserServer,
    results: OptimizationResults,
    var_name: str,
    bounds: tuple[float, float] | None = None,
    show: str = "both",
    title: str | None = None,
    folder_name: str | None = None,
    aspect: float = 1.5,
    marker_color: str = "red",
    marker_size: int = 12,
) -> tuple:
    """Add animated norm plot for a state or control variable.

    Convenience wrapper around add_animated_plotly_marker() that uses
    the existing plot_vector_norm() function to create the base plot.

    Args:
        server: ViserServer instance
        results: Optimization results containing variable data
        var_name: Name of the state or control variable to plot
        bounds: Optional (min, max) bounds to display on plot
        show: What to plot - "both", "nodes", or "trajectory"
        title: Optional custom title for the plot (defaults to "‖{var_name}‖₂")
        folder_name: Optional GUI folder name to organize plots
        aspect: Aspect ratio for plot display (width/height)
        marker_color: Color of the animated marker
        marker_size: Size of the animated marker in points

    Returns:
        Tuple of (plot_handle, update_callback), or (None, None) if variable not found

    Example::

        from openscvx.plotting import viser

        # Add animated thrust norm plot
        _, update_thrust = viser.add_animated_vector_norm_plot(
            server, results, "thrust",
            title="Thrust Magnitude",
            bounds=(0.0, max_thrust),
            folder_name="Control Plots"
        )
        if update_thrust is not None:
            update_callbacks.append(update_thrust)
    """
    from openscvx.plotting import plot_vector_norm

    # Check if variable exists in results
    has_in_trajectory = bool(results.trajectory) and var_name in results.trajectory
    has_in_nodes = var_name in results.nodes

    if not (has_in_trajectory or has_in_nodes):
        import warnings

        warnings.warn(f"Variable '{var_name}' not found in results, skipping plot")
        return None, None

    # Create figure using existing plotting function
    fig = plot_vector_norm(results, var_name, bounds=bounds, show=show)

    # Update title if custom title provided
    if title is not None:
        fig.update_layout(title_text=title)

    # Determine data source and compute norms
    if has_in_trajectory:
        time_data = results.trajectory["time"].flatten()
        var_data = results.trajectory[var_name]
        use_trajectory_indexing = True
    else:
        time_data = results.nodes["time"].flatten()
        var_data = results.nodes[var_name]
        use_trajectory_indexing = False

    # Compute norms
    norm_data = np.linalg.norm(var_data, axis=1) if var_data.ndim > 1 else np.abs(var_data)

    # Add animated marker
    return add_animated_plotly_marker(
        server,
        fig,
        time_array=time_data,
        marker_x_data=time_data,
        marker_y_data=norm_data,
        use_trajectory_indexing=use_trajectory_indexing,
        marker_name="Current",
        marker_color=marker_color,
        marker_size=marker_size,
        folder_name=folder_name,
        aspect=aspect,
    )
