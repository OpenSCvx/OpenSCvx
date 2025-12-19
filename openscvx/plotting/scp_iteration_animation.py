import plotly.graph_objects as go
from plotly.subplots import make_subplots

from openscvx.algorithms import OptimizationResults

from .plotting import _get_var


def plot_scp_iteration_animation(
    result: OptimizationResults,
    state_names: list[str] | None = None,
    control_names: list[str] | None = None,
):
    """Create an animated plot showing SCP iteration convergence.

    Shows the evolution of states and controls across SCP iterations, including
    multi-shot propagation trajectories (if available) and optimization nodes.

    Args:
        result: Optimization results containing iteration history
        state_names: Optional list of state names to include. If None, plots all states.
        control_names: Optional list of control names to include. If None, plots all controls.

    Returns:
        Plotly figure with animation frames for each SCP iteration

    Example:
        >>> results = problem.solve()
        >>> plot_scp_iteration_animation(results, ["position", "velocity"]).show()
    """
    import numpy as np

    if not result.X:
        raise ValueError("No iteration history available in result.X")

    # Derive dimensions from result data
    n_x = result.X[0].shape[1]  # n_states from first iteration's X
    n_u = result.U[0].shape[1]  # n_controls from first iteration's U

    # Find time slice by looking for "time" state
    time_slice = None
    for state in result._states:
        if state.name.lower() == "time":
            time_slice = state._slice
            break

    # Get iteration history
    V_history = result.discretization_history if result.discretization_history else []
    U_history = result.U

    # Extract multi-shot propagation trajectories from V_history
    X_prop_history = []  # Multi-shot propagated trajectories
    if V_history:
        i4 = n_x + n_x * n_x + 2 * n_x * n_u

        for V in V_history:
            # V shape: (flattened_size, n_timesteps) where flattened_size = (N-1) * i4
            # Extract positions for each time step in the multi-shoot
            pos_traj = []
            for i_multi in range(V.shape[1]):
                # Reshape each time column to (N-1, i4) and extract position (first n_x columns)
                pos_traj.append(V[:, i_multi].reshape(-1, i4)[:, :n_x])
            X_prop_history.append(np.array(pos_traj))  # Shape: (n_timesteps, N-1, n_x)
    else:
        # Fallback to X history if V_history not available
        X_prop_history = None

    n_iterations = len(result.X)

    if n_iterations == 0:
        raise ValueError("No iteration history available")

    # Limit iterations to those with available propagation history
    if X_prop_history:
        n_iterations = min(n_iterations, len(X_prop_history))

    # Get states and controls, filter CTCS
    # For propagated states, use x_prop metadata
    states = result._states if hasattr(result, "_states") and result._states else []
    controls = result._controls if hasattr(result, "_controls") and result._controls else []

    # Filter out augmented states
    filtered_states = [s for s in states if "ctcs_aug" not in s.name.lower()]
    states = filtered_states
    controls = controls if controls else []

    # Optional filtering by provided names
    state_filter = set(state_names) if state_names else None
    control_filter = set(control_names) if control_names else None

    # If only one group is specified, drop the other entirely
    if state_filter and control_filter is None:
        controls = []
    if control_filter and state_filter is None:
        states = []

    if state_filter:
        states = [s for s in states if s.name in state_filter]
    if control_filter:
        controls = [c for c in controls if c.name in control_filter]

    # Expand multi-dimensional states/controls
    def expand_variables(variables):
        expanded = []
        for var in variables:
            var_slice = var._slice
            if isinstance(var_slice, slice):
                start = var_slice.start if var_slice.start is not None else 0
                stop = var_slice.stop if var_slice.stop is not None else start + 1
                n_comp = stop - start
            else:
                start = var_slice
                n_comp = 1

            if n_comp > 1:
                for i in range(n_comp):

                    class Component:
                        def __init__(self, idx, parent, comp_idx):
                            self.name = f"{parent}_{comp_idx}"
                            self._slice = slice(idx, idx + 1)

                    expanded.append(Component(start + i, var.name, i))
            else:

                class Single:
                    def __init__(self, name, idx):
                        self.name = name
                        self._slice = slice(idx, idx + 1)

                expanded.append(Single(var.name, start))
        return expanded

    expanded_states = expand_variables(states)
    expanded_controls = expand_variables(controls)

    # Grid dimensions
    n_states = len(expanded_states)
    n_controls = len(expanded_controls)
    n_state_cols = min(7, n_states) if n_states > 0 else 1
    n_control_cols = min(3, n_controls) if n_controls > 0 else 1
    n_state_rows = (n_states + n_state_cols - 1) // n_state_cols if n_states > 0 else 0
    n_control_rows = (n_controls + n_control_cols - 1) // n_control_cols if n_controls > 0 else 0

    total_rows = n_state_rows + n_control_rows
    # Use n_state_cols for state rows, n_control_cols for control rows - don't pad to max
    actual_cols = n_state_cols if n_state_rows > 0 else n_control_cols

    # Create figure with proper column counts per section
    subplot_titles = [s.name for s in expanded_states] + [c.name for c in expanded_controls]
    # For mixed grids, we need to handle states and controls separately
    if n_states > 0 and n_controls > 0:
        # Create a grid that can accommodate both sections
        fig = make_subplots(
            rows=total_rows,
            cols=max(n_state_cols, n_control_cols),
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
            specs=[
                [{"secondary_y": False}] * max(n_state_cols, n_control_cols)
                for _ in range(total_rows)
            ],
        )
    else:
        fig = make_subplots(
            rows=total_rows,
            cols=actual_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
        )

    # Add 500 blank traces for animation placeholder
    for _ in range(2000):
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False))

    # Prepare bounds data for each subplot (use bounds from variable metadata)
    state_bounds_data = {}
    for state_idx, exp_state in enumerate(expanded_states):
        idx = exp_state._slice.start
        # Find the parent state to get bounds
        parent_name = exp_state.name.rsplit("_", 1)[0] if "_" in exp_state.name else exp_state.name
        parent_state = _get_var(result, parent_name, result._states)
        comp_idx = idx - (
            parent_state._slice.start
            if isinstance(parent_state._slice, slice)
            else parent_state._slice
        )
        x_min = parent_state.min[comp_idx] if parent_state.min is not None else -np.inf
        x_max = parent_state.max[comp_idx] if parent_state.max is not None else np.inf
        state_bounds_data[state_idx] = (x_min, x_max)

    control_bounds_data = {}
    for control_idx, exp_control in enumerate(expanded_controls):
        idx = exp_control._slice.start
        # Find the parent control to get bounds
        parent_name = (
            exp_control.name.rsplit("_", 1)[0] if "_" in exp_control.name else exp_control.name
        )
        parent_control = _get_var(result, parent_name, result._controls)
        comp_idx = idx - (
            parent_control._slice.start
            if isinstance(parent_control._slice, slice)
            else parent_control._slice
        )
        u_min = parent_control.min[comp_idx] if parent_control.min is not None else -np.inf
        u_max = parent_control.max[comp_idx] if parent_control.max is not None else np.inf
        control_bounds_data[control_idx] = (u_min, u_max)

    # Create animation frames
    frames = []
    for iter_idx in range(n_iterations):
        X_nodes = result.X[iter_idx]  # Optimization nodes
        U_iter = U_history[iter_idx]

        # Time for nodes (N points)
        t_nodes = (
            X_nodes[:, time_slice].flatten()
            if time_slice is not None
            else np.linspace(0, result.t_final, X_nodes.shape[0])
        )

        frame_data = []

        # States: multi-shot trajectories + nodes
        for state_idx, state in enumerate(expanded_states):
            idx = state._slice.start
            row = (state_idx // n_state_cols) + 1
            col = (state_idx % n_state_cols) + 1

            # Plot multi-shot trajectories (one line per time interval between nodes)
            if X_prop_history and iter_idx < len(X_prop_history):
                pos_traj = X_prop_history[iter_idx]  # Shape: (n_timesteps, N-1, n_x)

                # Loop through each segment (N-1 segments between N nodes)
                for j in range(pos_traj.shape[1]):
                    # Extract time and state values for this segment across all timesteps
                    segment_states = pos_traj[:, j, idx]  # Shape: (n_timesteps,)
                    segment_times = pos_traj[:, j, time_slice].flatten()

                    frame_data.append(
                        go.Scatter(
                            x=segment_times,
                            y=segment_states,
                            mode="lines",
                            line={"color": "blue", "width": 2},
                            showlegend=False,
                            xaxis=f"x{1 if (row == 1 and col == 1) else state_idx + 1}",
                            yaxis=f"y{1 if (row == 1 and col == 1) else state_idx + 1}",
                        )
                    )

            # Optimization nodes (markers only)
            frame_data.append(
                go.Scatter(
                    x=t_nodes,
                    y=X_nodes[:, idx],
                    mode="markers",
                    marker={"color": "cyan", "size": 6, "symbol": "circle"},
                    showlegend=False,
                    xaxis=f"x{1 if (row == 1 and col == 1) else state_idx + 1}",
                    yaxis=f"y{1 if (row == 1 and col == 1) else state_idx + 1}",
                )
            )

        # Controls: plot on separate subplots
        for control_idx, control in enumerate(expanded_controls):
            idx = control._slice.start
            row = n_state_rows + (control_idx // n_control_cols) + 1
            col = (control_idx % n_control_cols) + 1

            frame_data.append(
                go.Scatter(
                    x=t_nodes,
                    y=U_iter[:, idx],
                    mode="markers",
                    marker={"color": "orange", "size": 6, "symbol": "circle"},
                    showlegend=False,
                    xaxis=f"x{1 if (row == 1 and col == 1) else n_states + control_idx + 1}",
                    yaxis=f"y{1 if (row == 1 and col == 1) else n_states + control_idx + 1}",
                )
            )

        # Time range for bounds spans (use t_nodes for the full time range)
        t_min = t_nodes.min() if len(t_nodes) > 0 else 0
        t_max = t_nodes.max() if len(t_nodes) > 0 else 1

        # Add bounds to each frame
        # State bounds
        for state_idx, (x_min, x_max) in state_bounds_data.items():
            axis_num = state_idx + 1
            if not np.isinf(x_min):
                frame_data.append(
                    go.Scatter(
                        x=[t_min, t_max],
                        y=[x_min, x_min],
                        mode="lines",
                        line={"color": "red", "width": 1, "dash": "dot"},
                        showlegend=False,
                        xaxis=f"x{axis_num}",
                        yaxis=f"y{axis_num}",
                    )
                )
            if not np.isinf(x_max):
                frame_data.append(
                    go.Scatter(
                        x=[t_min, t_max],
                        y=[x_max, x_max],
                        mode="lines",
                        line={"color": "red", "width": 1, "dash": "dot"},
                        showlegend=False,
                        xaxis=f"x{axis_num}",
                        yaxis=f"y{axis_num}",
                    )
                )

        # Control bounds
        for control_idx, (u_min, u_max) in control_bounds_data.items():
            axis_num = n_states + control_idx + 1
            if not np.isinf(u_min):
                frame_data.append(
                    go.Scatter(
                        x=[t_min, t_max],
                        y=[u_min, u_min],
                        mode="lines",
                        line={"color": "red", "width": 1, "dash": "dot"},
                        showlegend=False,
                        xaxis=f"x{axis_num}",
                        yaxis=f"y{axis_num}",
                    )
                )
            if not np.isinf(u_max):
                frame_data.append(
                    go.Scatter(
                        x=[t_min, t_max],
                        y=[u_max, u_max],
                        mode="lines",
                        line={"color": "red", "width": 1, "dash": "dot"},
                        showlegend=False,
                        xaxis=f"x{axis_num}",
                        yaxis=f"y{axis_num}",
                    )
                )

        frames.append(go.Frame(data=frame_data, name=f"Iteration {iter_idx}"))

    # Animation controls (60 FPS = ~16.67ms per frame)
    fig.frames = frames
    fig.update_layout(
        title_text=f"SCP Iteration History ({n_iterations} iterations)",
        template="plotly_dark",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.1,
                "y": -0.15,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 17, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "y": -0.1,
                "xanchor": "left",
                "x": 0.4,
                "currentvalue": {"prefix": "Iteration: ", "visible": True, "xanchor": "right"},
                "pad": {"b": 10, "t": 50},
                "len": 0.5,
                "steps": [
                    {
                        "args": [
                            [f"Iteration {i}"],
                            {
                                "frame": {"duration": 17, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": str(i),
                        "method": "animate",
                    }
                    for i in range(n_iterations)
                ],
            }
        ],
    )

    # Add legend entries for the traces
    # Add dummy traces for legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Multishot Trajectory",
            line={"color": "blue", "width": 2},
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="State Nodes",
            marker={"color": "cyan", "size": 6, "symbol": "circle"},
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Control Nodes",
            marker={"color": "orange", "size": 6, "symbol": "circle"},
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Bounds",
            line={"color": "red", "width": 1, "dash": "dot"},
            showlegend=True,
        )
    )

    # Update legend configuration
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(0, 0, 0, 0.5)",
            bordercolor="white",
            borderwidth=1,
        )
    )

    for i in range(1, total_rows + 1):
        fig.update_xaxes(title_text="Time (s)", row=i, col=1)

    return fig
