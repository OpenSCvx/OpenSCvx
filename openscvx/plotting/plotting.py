import plotly.graph_objects as go
from plotly.subplots import make_subplots

from openscvx.algorithms import OptimizationResults


def _get_var_dim(result: OptimizationResults, var_name: str, var_list: list) -> int:
    """Get dimensionality of a variable from the metadata."""
    for var in var_list:
        if var.name == var_name:
            s = var._slice
            if isinstance(s, slice):
                return (s.stop or 1) - (s.start or 0)
            return 1
    raise ValueError(f"Variable '{var_name}' not found")


def _plot_timeseries(
    fig: go.Figure,
    result: OptimizationResults,
    var_name: str,
    component_idx: int,
    row: int,
    col: int,
    show_legend: bool,
):
    """Add traces for a single component of a variable to the figure."""
    t_nodes = result.nodes["time"].flatten()
    has_trajectory = bool(result.trajectory)
    t_full = result.trajectory["time"].flatten() if has_trajectory else None

    # Plot propagated trajectory if available
    if has_trajectory and var_name in result.trajectory:
        data = result.trajectory[var_name]
        y = data if data.ndim == 1 else data[:, component_idx]
        fig.add_trace(
            go.Scatter(
                x=t_full,
                y=y,
                mode="lines",
                name="Propagated",
                showlegend=show_legend,
                legendgroup="propagated",
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


def plot_state(result: OptimizationResults, state_name: str):
    """Plot a single state variable (all components on one figure).

    Args:
        result: Optimization results containing state trajectories
        state_name: Name of the state to plot

    Returns:
        Plotly figure
    """
    available = {s.name for s in result._states}
    if state_name not in available:
        raise ValueError(f"State '{state_name}' not found. Available: {sorted(available)}")

    dim = _get_var_dim(result, state_name, result._states)
    t_nodes = result.nodes["time"].flatten()
    has_trajectory = bool(result.trajectory)
    t_full = result.trajectory["time"].flatten() if has_trajectory else None

    fig = go.Figure()
    fig.update_layout(title_text=state_name, template="plotly_dark")

    for i in range(dim):
        label = f"{state_name}_{i}" if dim > 1 else state_name

        if has_trajectory and state_name in result.trajectory:
            data = result.trajectory[state_name]
            y = data if data.ndim == 1 else data[:, i]
            fig.add_trace(go.Scatter(x=t_full, y=y, mode="lines", name=f"{label} (prop)"))

        if state_name in result.nodes:
            data = result.nodes[state_name]
            y = data if data.ndim == 1 else data[:, i]
            fig.add_trace(go.Scatter(x=t_nodes, y=y, mode="markers", name=f"{label} (nodes)"))

    fig.update_xaxes(title_text="Time (s)")
    return fig


def plot_states(
    result: OptimizationResults,
    state_names: list[str] | None = None,
    include_private: bool = False,
):
    """Plot multiple state variables in a subplot grid.

    Args:
        result: Optimization results containing state trajectories
        state_names: Optional list of state names to plot; defaults to all states
        include_private: Whether to include private states (names starting with '_')

    Returns:
        Plotly figure with state trajectory subplots
    """
    states = result._states
    if not include_private:
        states = [s for s in states if not s.name.startswith("_")]

    if state_names is not None:
        available = {s.name for s in states}
        missing = set(state_names) - available
        if missing:
            raise ValueError(f"States not found in result: {missing}")
        states = [s for s in states if s.name in state_names]

    # Build list of (display_name, var_name, component_idx)
    components = []
    for s in states:
        dim = _get_var_dim(result, s.name, result._states)
        if dim == 1:
            components.append((s.name, s.name, 0))
        else:
            for i in range(dim):
                components.append((f"{s.name}_{i}", s.name, i))

    n_cols = min(7, len(components))
    n_rows = (len(components) + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[c[0] for c in components])
    fig.update_layout(title_text="State Trajectories", template="plotly_dark")

    for idx, (_, var_name, comp_idx) in enumerate(components):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        _plot_timeseries(fig, result, var_name, comp_idx, row, col, show_legend=(idx == 0))

    for i in range(1, n_rows + 1):
        fig.update_xaxes(title_text="Time (s)", row=i, col=1)

    return fig


def plot_control(result: OptimizationResults, control_name: str):
    """Plot a single control variable (all components on one figure).

    Args:
        result: Optimization results containing control trajectories
        control_name: Name of the control to plot

    Returns:
        Plotly figure
    """
    available = {c.name for c in result._controls}
    if control_name not in available:
        raise ValueError(f"Control '{control_name}' not found. Available: {sorted(available)}")

    dim = _get_var_dim(result, control_name, result._controls)
    t_nodes = result.nodes["time"].flatten()
    has_trajectory = bool(result.trajectory)
    t_full = result.trajectory["time"].flatten() if has_trajectory else None

    fig = go.Figure()
    fig.update_layout(title_text=control_name, template="plotly_dark")

    for i in range(dim):
        label = f"{control_name}_{i}" if dim > 1 else control_name

        if has_trajectory and control_name in result.trajectory:
            data = result.trajectory[control_name]
            y = data if data.ndim == 1 else data[:, i]
            fig.add_trace(go.Scatter(x=t_full, y=y, mode="lines", name=f"{label} (prop)"))

        if control_name in result.nodes:
            data = result.nodes[control_name]
            y = data if data.ndim == 1 else data[:, i]
            fig.add_trace(go.Scatter(x=t_nodes, y=y, mode="markers", name=f"{label} (nodes)"))

    fig.update_xaxes(title_text="Time (s)")
    return fig


def plot_controls(
    result: OptimizationResults,
    control_names: list[str] | None = None,
    include_private: bool = False,
):
    """Plot multiple control variables in a subplot grid.

    Args:
        result: Optimization results containing control trajectories
        control_names: Optional list of control names to plot; defaults to all controls
        include_private: Whether to include private controls (names starting with '_')

    Returns:
        Plotly figure with control trajectory subplots
    """
    controls = result._controls
    if not include_private:
        controls = [c for c in controls if not c.name.startswith("_")]

    if control_names is not None:
        available = {c.name for c in controls}
        missing = set(control_names) - available
        if missing:
            raise ValueError(f"Controls not found in result: {missing}")
        controls = [c for c in controls if c.name in control_names]

    # Build list of (display_name, var_name, component_idx)
    components = []
    for c in controls:
        dim = _get_var_dim(result, c.name, result._controls)
        if dim == 1:
            components.append((c.name, c.name, 0))
        else:
            for i in range(dim):
                components.append((f"{c.name}_{i}", c.name, i))

    n_cols = min(3, len(components))
    n_rows = (len(components) + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[c[0] for c in components])
    fig.update_layout(title_text="Control Trajectories", template="plotly_dark")

    for idx, (_, var_name, comp_idx) in enumerate(components):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        _plot_timeseries(fig, result, var_name, comp_idx, row, col, show_legend=(idx == 0))

    for i in range(1, n_rows + 1):
        fig.update_xaxes(title_text="Time (s)", row=i, col=1)

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
