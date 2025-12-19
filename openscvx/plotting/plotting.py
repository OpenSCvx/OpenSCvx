import plotly.graph_objects as go
from plotly.subplots import make_subplots

from openscvx.algorithms import OptimizationResults


def _get_var(result: OptimizationResults, var_name: str, var_list: list):
    """Get a variable object by name from the metadata list."""
    for var in var_list:
        if var.name == var_name:
            return var
    raise ValueError(f"Variable '{var_name}' not found")


def _get_var_dim(result: OptimizationResults, var_name: str, var_list: list) -> int:
    """Get dimensionality of a variable from the metadata."""
    var = _get_var(result, var_name, var_list)
    s = var._slice
    if isinstance(s, slice):
        return (s.stop or 1) - (s.start or 0)
    return 1


def _add_component_traces(
    fig: go.Figure,
    result: OptimizationResults,
    var_name: str,
    component_idx: int,
    row: int,
    col: int,
    show_legend: bool,
    min_val: float | None = None,
    max_val: float | None = None,
):
    """Add traces for a single component of a variable to a subplot.

    Args:
        fig: Plotly figure to add traces to
        result: Optimization results
        var_name: Name of the variable
        component_idx: Index of the component to plot
        row: Subplot row
        col: Subplot column
        show_legend: Whether to show legend entries
        min_val: Optional minimum bound to show as horizontal line
        max_val: Optional maximum bound to show as horizontal line
    """
    import numpy as np

    t_nodes = result.nodes["time"].flatten()
    has_trajectory = bool(result.trajectory) and var_name in result.trajectory
    t_full = result.trajectory["time"].flatten() if has_trajectory else None

    # Plot propagated trajectory if available
    if has_trajectory:
        data = result.trajectory[var_name]
        y = data if data.ndim == 1 else data[:, component_idx]
        fig.add_trace(
            go.Scatter(
                x=t_full,
                y=y,
                mode="lines",
                name="Trajectory",
                showlegend=show_legend,
                legendgroup="trajectory",
                line={"color": "green", "width": 2},
            ),
            row=row,
            col=col,
        )

    # Plot optimization nodes
    if var_name in result.nodes:
        data = result.nodes[var_name]
        y = data if data.ndim == 1 else data[:, component_idx]
        fig.add_trace(
            go.Scatter(
                x=t_nodes,
                y=y,
                mode="markers",
                name="Nodes",
                showlegend=show_legend,
                legendgroup="nodes",
                marker={"color": "cyan", "size": 6, "symbol": "circle"},
            ),
            row=row,
            col=col,
        )

    # Add horizontal bound lines if provided
    # Only add if finite (skip -inf/inf bounds)
    if min_val is not None and np.isfinite(min_val):
        fig.add_hline(
            y=min_val,
            line={"color": "red", "width": 1.5, "dash": "dash"},
            row=row,
            col=col,
        )
    if max_val is not None and np.isfinite(max_val):
        fig.add_hline(
            y=max_val,
            line={"color": "red", "width": 1.5, "dash": "dash"},
            row=row,
            col=col,
        )


# =============================================================================
# State Plotting
# =============================================================================


def plot_state_component(
    result: OptimizationResults,
    state_name: str,
    component: int = 0,
) -> go.Figure:
    """Plot a single component of a state variable vs time.

    This is the low-level function for plotting one scalar value over time.
    For plotting all components of a state, use plot_states().

    Args:
        result: Optimization results containing state trajectories
        state_name: Name of the state variable
        component: Component index (0-indexed). For scalar states, use 0.

    Returns:
        Plotly figure with single plot

    Example:
        >>> plot_state_component(result, "position", 2)  # Plot z-component
    """
    available = {s.name for s in result._states}
    if state_name not in available:
        raise ValueError(f"State '{state_name}' not found. Available: {sorted(available)}")

    dim = _get_var_dim(result, state_name, result._states)
    if component < 0 or component >= dim:
        raise ValueError(f"Component {component} out of range for '{state_name}' (dim={dim})")

    t_nodes = result.nodes["time"].flatten()
    has_trajectory = bool(result.trajectory) and state_name in result.trajectory
    t_full = result.trajectory["time"].flatten() if has_trajectory else None

    label = f"{state_name}_{component}" if dim > 1 else state_name

    fig = go.Figure()
    fig.update_layout(title_text=label, template="plotly_dark")

    if has_trajectory:
        data = result.trajectory[state_name]
        y = data if data.ndim == 1 else data[:, component]
        fig.add_trace(
            go.Scatter(
                x=t_full,
                y=y,
                mode="lines",
                name="Trajectory",
                line={"color": "green", "width": 2},
            )
        )

    if state_name in result.nodes:
        data = result.nodes[state_name]
        y = data if data.ndim == 1 else data[:, component]
        fig.add_trace(
            go.Scatter(
                x=t_nodes,
                y=y,
                mode="markers",
                name="Nodes",
                marker={"color": "cyan", "size": 6},
            )
        )

    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text=label)
    return fig


def plot_states(
    result: OptimizationResults,
    state_names: list[str] | None = None,
    include_private: bool = False,
    cols: int = 4,
) -> go.Figure:
    """Plot state variables in a subplot grid.

    Each component of each state gets its own subplot with individual y-axis
    scaling. This is the primary function for visualizing state trajectories.

    Args:
        result: Optimization results containing state trajectories
        state_names: List of state names to plot. If None, plots all states.
        include_private: Whether to include private states (names starting with '_')
        cols: Maximum number of columns in subplot grid

    Returns:
        Plotly figure with subplot grid

    Examples:
        >>> plot_states(result, ["position"])  # 3 subplots for x, y, z
        >>> plot_states(result, ["position", "velocity"])  # 6 subplots
        >>> plot_states(result)  # All states
    """

    states = result._states
    if not include_private:
        states = [s for s in states if not s.name.startswith("_")]

    if state_names is not None:
        available = {s.name for s in states}
        missing = set(state_names) - available
        if missing:
            raise ValueError(f"States not found in result: {missing}")
        # Preserve order from state_names
        state_order = {name: i for i, name in enumerate(state_names)}
        states = sorted(
            [s for s in states if s.name in state_names],
            key=lambda s: state_order[s.name],
        )

    # Build list of (display_name, var_name, component_idx)
    components = []
    for s in states:
        dim = _get_var_dim(result, s.name, result._states)
        if dim == 1:
            components.append((s.name, s.name, 0))
        else:
            for i in range(dim):
                components.append((f"{s.name}_{i}", s.name, i))

    if not components:
        raise ValueError("No state components to plot")

    n_cols = min(cols, len(components))
    n_rows = (len(components) + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[c[0] for c in components])
    fig.update_layout(title_text="State Trajectories", template="plotly_dark")

    for idx, (_, var_name, comp_idx) in enumerate(components):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1

        # Get bounds for this component
        var = _get_var(result, var_name, result._states)
        min_val = var.min[comp_idx] if var.min is not None else None
        max_val = var.max[comp_idx] if var.max is not None else None

        _add_component_traces(
            fig,
            result,
            var_name,
            comp_idx,
            row,
            col,
            show_legend=(idx == 0),
            min_val=min_val,
            max_val=max_val,
        )

    # Add x-axis labels to bottom row
    for col_idx in range(1, n_cols + 1):
        fig.update_xaxes(title_text="Time (s)", row=n_rows, col=col_idx)

    return fig


# =============================================================================
# Control Plotting
# =============================================================================


def plot_control_component(
    result: OptimizationResults,
    control_name: str,
    component: int = 0,
) -> go.Figure:
    """Plot a single component of a control variable vs time.

    This is the low-level function for plotting one scalar control over time.
    For plotting all components of a control, use plot_controls().

    Args:
        result: Optimization results containing control trajectories
        control_name: Name of the control variable
        component: Component index (0-indexed). For scalar controls, use 0.

    Returns:
        Plotly figure with single plot

    Example:
        >>> plot_control_component(result, "thrust", 0)  # Plot thrust_x
    """
    available = {c.name for c in result._controls}
    if control_name not in available:
        raise ValueError(f"Control '{control_name}' not found. Available: {sorted(available)}")

    dim = _get_var_dim(result, control_name, result._controls)
    if component < 0 or component >= dim:
        raise ValueError(f"Component {component} out of range for '{control_name}' (dim={dim})")

    t_nodes = result.nodes["time"].flatten()
    has_trajectory = bool(result.trajectory) and control_name in result.trajectory
    t_full = result.trajectory["time"].flatten() if has_trajectory else None

    label = f"{control_name}_{component}" if dim > 1 else control_name

    fig = go.Figure()
    fig.update_layout(title_text=label, template="plotly_dark")

    if has_trajectory:
        data = result.trajectory[control_name]
        y = data if data.ndim == 1 else data[:, component]
        fig.add_trace(
            go.Scatter(
                x=t_full,
                y=y,
                mode="lines",
                name="Trajectory",
                line={"color": "green", "width": 2},
            )
        )

    if control_name in result.nodes:
        data = result.nodes[control_name]
        y = data if data.ndim == 1 else data[:, component]
        fig.add_trace(
            go.Scatter(
                x=t_nodes,
                y=y,
                mode="markers",
                name="Nodes",
                marker={"color": "cyan", "size": 6},
            )
        )

    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text=label)
    return fig


def plot_controls(
    result: OptimizationResults,
    control_names: list[str] | None = None,
    include_private: bool = False,
    cols: int = 3,
) -> go.Figure:
    """Plot control variables in a subplot grid.

    Each component of each control gets its own subplot with individual y-axis
    scaling. This is the primary function for visualizing control trajectories.

    Args:
        result: Optimization results containing control trajectories
        control_names: List of control names to plot. If None, plots all controls.
        include_private: Whether to include private controls (names starting with '_')
        cols: Maximum number of columns in subplot grid

    Returns:
        Plotly figure with subplot grid

    Examples:
        >>> plot_controls(result, ["thrust"])  # 3 subplots for x, y, z
        >>> plot_controls(result)  # All controls
    """

    controls = result._controls
    if not include_private:
        controls = [c for c in controls if not c.name.startswith("_")]

    if control_names is not None:
        available = {c.name for c in controls}
        missing = set(control_names) - available
        if missing:
            raise ValueError(f"Controls not found in result: {missing}")
        # Preserve order from control_names
        control_order = {name: i for i, name in enumerate(control_names)}
        controls = sorted(
            [c for c in controls if c.name in control_names],
            key=lambda c: control_order[c.name],
        )

    # Build list of (display_name, var_name, component_idx)
    components = []
    for c in controls:
        dim = _get_var_dim(result, c.name, result._controls)
        if dim == 1:
            components.append((c.name, c.name, 0))
        else:
            for i in range(dim):
                components.append((f"{c.name}_{i}", c.name, i))

    if not components:
        raise ValueError("No control components to plot")

    n_cols = min(cols, len(components))
    n_rows = (len(components) + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[c[0] for c in components])
    fig.update_layout(title_text="Control Trajectories", template="plotly_dark")

    for idx, (_, var_name, comp_idx) in enumerate(components):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1

        # Get bounds for this component
        var = _get_var(result, var_name, result._controls)
        min_val = var.min[comp_idx] if var.min is not None else None
        max_val = var.max[comp_idx] if var.max is not None else None

        _add_component_traces(
            fig,
            result,
            var_name,
            comp_idx,
            row,
            col,
            show_legend=(idx == 0),
            min_val=min_val,
            max_val=max_val,
        )

    # Add x-axis labels to bottom row
    for col_idx in range(1, n_cols + 1):
        fig.update_xaxes(title_text="Time (s)", row=n_rows, col=col_idx)

    return fig


def plot_trust_region_heatmap(result: OptimizationResults):
    """Plot heatmap of the final trust-region deltas (TR_history[-1])."""
    if not result.TR_history:
        raise ValueError("Result has no TR_history to plot")

    tr_mat = result.TR_history[-1]

    # Build variable names list
    var_names = []
    for var_list in [result._states, result._controls]:
        for var in var_list:
            dim = _get_var_dim(result, var.name, var_list)
            if dim == 1:
                var_names.append(var.name)
            else:
                var_names.extend(f"{var.name}_{i}" for i in range(dim))

    # TR matrix is (n_states+n_controls, n_nodes): rows = variables, cols = nodes
    if tr_mat.shape[0] == len(var_names):
        z = tr_mat
    elif tr_mat.shape[1] == len(var_names):
        z = tr_mat.T
    else:
        raise ValueError("TR matrix dimensions do not align with state/control components")

    x_len = z.shape[1]
    t_nodes = result.nodes["time"].flatten()
    x_labels = t_nodes if len(t_nodes) == x_len else list(range(x_len))

    fig = go.Figure(data=go.Heatmap(z=z, x=x_labels, y=var_names, colorscale="Viridis"))
    fig.update_layout(
        title="Trust Region Delta Magnitudes (last iteration)", template="plotly_dark"
    )
    fig.update_xaxes(title_text="Node / Time", side="bottom")
    fig.update_yaxes(title_text="State / Control component", side="left")
    return fig


def plot_projections_2d(
    result: OptimizationResults,
    var_name: str = "position",
    velocity_var_name: str | None = None,
    cmap: str = "viridis",
):
    """Plot XY, XZ, YZ projections of a 3D variable.

    Useful for visualizing 3D trajectories in 2D plane views.

    Args:
        result: Optimization results containing trajectories
        var_name: Name of the 3D variable to plot (default: "position")
        velocity_var_name: Optional name of velocity variable for coloring by speed.
            If provided, trajectory points are colored by velocity magnitude.
        cmap: Matplotlib colormap name for velocity coloring (default: "viridis")

    Returns:
        Plotly figure with three subplots (XY, XZ, YZ planes)
    """
    import numpy as np

    has_trajectory = bool(result.trajectory) and var_name in result.trajectory
    has_nodes = var_name in result.nodes

    if not has_trajectory and not has_nodes:
        available_traj = set(result.trajectory.keys()) if result.trajectory else set()
        available_nodes = set(result.nodes.keys())
        raise ValueError(
            f"Variable '{var_name}' not found. "
            f"Available in trajectory: {sorted(available_traj)}, nodes: {sorted(available_nodes)}"
        )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("XY Plane", "XZ Plane", "YZ Plane"),
        specs=[[{}, {}], [{}, None]],
    )

    # Subplot positions: (x_idx, y_idx, row, col)
    subplots = [(0, 1, 1, 1), (0, 2, 1, 2), (1, 2, 2, 1)]

    # Compute velocity norms if velocity variable is provided
    traj_vel_norm = None
    node_vel_norm = None
    if velocity_var_name is not None:
        if has_trajectory and velocity_var_name in result.trajectory:
            traj_vel_norm = np.linalg.norm(result.trajectory[velocity_var_name], axis=1)
        if has_nodes and velocity_var_name in result.nodes:
            node_vel_norm = np.linalg.norm(result.nodes[velocity_var_name], axis=1)

    # Colorbar config (only shown once)
    colorbar_cfg = {"title": "‖velocity‖", "x": 1.02, "y": 0.5, "len": 0.9}

    # Plot trajectory if available
    if has_trajectory:
        data = result.trajectory[var_name]
        for i, (xi, yi, row, col) in enumerate(subplots):
            if traj_vel_norm is not None:
                marker = {
                    "size": 4,
                    "color": traj_vel_norm,
                    "colorscale": cmap,
                    "showscale": (i == 0),
                    "colorbar": colorbar_cfg if i == 0 else None,
                }
                fig.add_trace(
                    go.Scatter(
                        x=data[:, xi],
                        y=data[:, yi],
                        mode="markers",
                        marker=marker,
                        name="Trajectory",
                        legendgroup="trajectory",
                        showlegend=(i == 0),
                    ),
                    row=row,
                    col=col,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=data[:, xi],
                        y=data[:, yi],
                        mode="lines",
                        line={"color": "green", "width": 2},
                        name="Trajectory",
                        legendgroup="trajectory",
                        showlegend=(i == 0),
                    ),
                    row=row,
                    col=col,
                )

    # Plot nodes if available
    if has_nodes:
        data = result.nodes[var_name]
        # Only show colorbar on nodes if trajectory doesn't have one
        show_node_colorbar = (traj_vel_norm is None) and (node_vel_norm is not None)
        for i, (xi, yi, row, col) in enumerate(subplots):
            if node_vel_norm is not None:
                marker = {
                    "size": 8,
                    "color": node_vel_norm,
                    "colorscale": cmap,
                    "showscale": show_node_colorbar and (i == 0),
                    "colorbar": colorbar_cfg if (show_node_colorbar and i == 0) else None,
                    "line": {"color": "white", "width": 1},
                }
            else:
                marker = {"color": "cyan", "size": 6}
            fig.add_trace(
                go.Scatter(
                    x=data[:, xi],
                    y=data[:, yi],
                    mode="markers",
                    marker=marker,
                    name="Nodes",
                    legendgroup="nodes",
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

    # Set axis titles
    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=1)
    fig.update_xaxes(title_text="X", row=1, col=2)
    fig.update_yaxes(title_text="Z", row=1, col=2)
    fig.update_xaxes(title_text="Y", row=2, col=1)
    fig.update_yaxes(title_text="Z", row=2, col=1)

    # Set equal aspect ratio for each subplot
    layout_opts = {
        "title": f"{var_name} - XY, XZ, YZ Projections",
        "template": "plotly_dark",
        "xaxis": {"scaleanchor": "y"},
        "xaxis2": {"scaleanchor": "y2"},
        "xaxis3": {"scaleanchor": "y3"},
    }
    # Move legend to bottom-right when using colorbar to avoid overlap
    if velocity_var_name is not None:
        layout_opts["legend"] = {
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.15,
            "xanchor": "center",
            "x": 0.5,
        }
    fig.update_layout(**layout_opts)

    return fig


def plot_vector_norm(
    result: OptimizationResults,
    var_name: str,
    bounds: tuple[float, float] | None = None,
):
    """Plot the 2-norm of a vector variable over time.

    Useful for visualizing thrust magnitude, velocity magnitude, etc.

    Args:
        result: Optimization results containing trajectories
        var_name: Name of the vector variable (state or control)
        bounds: Optional (min, max) bounds to show as horizontal dashed lines

    Returns:
        Plotly figure
    """
    import numpy as np

    has_trajectory = bool(result.trajectory) and var_name in result.trajectory
    has_nodes = var_name in result.nodes

    if not has_trajectory and not has_nodes:
        available_traj = set(result.trajectory.keys()) if result.trajectory else set()
        available_nodes = set(result.nodes.keys())
        raise ValueError(
            f"Variable '{var_name}' not found. "
            f"Available in trajectory: {sorted(available_traj)}, nodes: {sorted(available_nodes)}"
        )

    fig = go.Figure()

    # Plot trajectory norm if available
    if has_trajectory:
        t_full = result.trajectory["time"].flatten()
        data = result.trajectory[var_name]
        norm = np.linalg.norm(data, axis=1)
        fig.add_trace(
            go.Scatter(
                x=t_full,
                y=norm,
                mode="lines",
                line={"color": "green", "width": 2},
                name="Trajectory",
                legendgroup="trajectory",
            )
        )

    # Plot node norms if available
    if has_nodes:
        t_nodes = result.nodes["time"].flatten()
        data = result.nodes[var_name]
        norm = np.linalg.norm(data, axis=1)
        fig.add_trace(
            go.Scatter(
                x=t_nodes,
                y=norm,
                mode="markers",
                marker={"color": "cyan", "size": 6},
                name="Nodes",
                legendgroup="nodes",
            )
        )

    # Add bounds if provided
    if bounds is not None:
        min_bound, max_bound = bounds
        fig.add_hline(
            y=min_bound,
            line={"color": "red", "width": 2, "dash": "dash"},
            annotation_text="Min",
            annotation_position="right",
        )
        fig.add_hline(
            y=max_bound,
            line={"color": "red", "width": 2, "dash": "dash"},
            annotation_text="Max",
            annotation_position="right",
        )

    fig.update_layout(
        title=f"‖{var_name}‖₂",
        xaxis_title="Time (s)",
        yaxis_title="Norm",
        template="plotly_dark",
    )

    return fig


def plot_virtual_control_heatmap(result: OptimizationResults):
    """Plot heatmap of the final virtual control magnitudes (VC_history[-1])."""
    if not result.VC_history:
        raise ValueError("Result has no VC_history to plot")

    vc_mat = result.VC_history[-1]

    # Build state names list
    state_names = []
    for var in result._states:
        dim = _get_var_dim(result, var.name, result._states)
        if dim == 1:
            state_names.append(var.name)
        else:
            state_names.extend(f"{var.name}_{i}" for i in range(dim))

    # Align so rows = states, cols = nodes
    if vc_mat.shape[1] == len(state_names):
        z = vc_mat.T
    elif vc_mat.shape[0] == len(state_names):
        z = vc_mat
    else:
        raise ValueError("VC matrix shape does not align with state components")

    x_len = z.shape[1]
    t_nodes = result.nodes["time"].flatten()

    # Virtual control uses N-1 intervals
    if len(t_nodes) == x_len + 1:
        x_labels = t_nodes[:-1]
    elif len(t_nodes) == x_len:
        x_labels = t_nodes
    else:
        x_labels = list(range(x_len))

    fig = go.Figure(data=go.Heatmap(z=z, x=x_labels, y=state_names, colorscale="Magma"))
    fig.update_layout(title="Virtual Control Magnitudes (last iteration)", template="plotly_dark")
    fig.update_xaxes(title_text="Node Interval (N-1)")
    fig.update_yaxes(title_text="State component")
    return fig

def plot_scp_iteration_animation(
    result: OptimizationResults = None,
    params: Config = None,
    problem=None,
    state_names=None,
    control_names=None,
):
    """Create an animated plot showing SCP iteration convergence.

    Args:
        result: Optimization results containing iteration history (optional if problem provided)
        params: Configuration with state/control bounds and metadata (optional if problem provided)
        problem: Problem instance to extract result and params from (optional)
        state_names: Optional list of state names to include in the animation
        control_names: Optional list of control names to include in the animation

    Returns:
        Plotly figure with animation frames for each SCP iteration
    """
    # If problem provided, extract result and params from it
    if problem is not None:
        if result is None:
            # Check if post_process() was called and use propagated result
            if hasattr(problem._solution, "_propagated_result"):
                result = problem._solution._propagated_result
            else:
                from openscvx.algorithms import format_result

                result = format_result(problem, problem._solution, True)
        if params is None:
            params = problem.settings

    if result is None or params is None:
        raise ValueError("Must provide either (result, params) or problem")

    # Get iteration history
    V_history = (
        result.discretization_history
        if hasattr(result, "discretization_history") and result.discretization_history
        else []
    )
    U_history = result.U

    # Extract multi-shot propagation trajectories from V_history
    X_prop_history = []  # Multi-shot propagated trajectories
    if V_history:
        n_x = params.sim.n_states
        n_u = params.sim.n_controls
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

    time_slice = params.sim.time_slice

    # Prepare bounds data for each subplot
    state_bounds_data = {}
    for state_idx, state in enumerate(expanded_states):
        idx = state._slice.start
        x_min = params.sim.x.min[idx] if params.sim.x.min is not None else -np.inf
        x_max = params.sim.x.max[idx] if params.sim.x.max is not None else np.inf
        state_bounds_data[state_idx] = (x_min, x_max)

    control_bounds_data = {}
    for control_idx, control in enumerate(expanded_controls):
        idx = control._slice.start
        u_min = params.sim.u.min[idx] if params.sim.u.min is not None else -np.inf
        u_max = params.sim.u.max[idx] if params.sim.u.max is not None else np.inf
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
            else np.linspace(0, params.sim.total_time, X_nodes.shape[0])
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