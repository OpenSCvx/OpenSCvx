import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from openscvx.algorithms import OptimizationResults
from openscvx.config import Config


def plot_state(
    result: OptimizationResults = None,
    params: Config = None,
    problem=None,
    state_names=None,
):
    """Plot state trajectories over time with bounds.

    Shows the optimized state trajectory (nodes and full propagation if available),
    initial guess, and constraint bounds for all state variables.

    Args:
        result: Optimization results containing state trajectories (optional if problem provided)
        params: Configuration with state bounds and metadata (optional if problem provided)
        problem: Problem instance to extract result and params from (optional)
        state_names: Optional list of state names to plot; defaults to all non-CTCS states

    Returns:
        Plotly figure with state trajectory subplots
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

    # Optional filtering of which states to plot
    state_filter = set(state_names) if state_names else None

    # Get time values at nodes from the nodes dictionary
    t_nodes = result.nodes["time"].flatten()

    # Check if full propagation trajectory is available
    has_full_trajectory = result.trajectory and len(result.trajectory) > 0

    # Get time for full trajectory
    if has_full_trajectory:
        t_full = result.trajectory["time"].flatten()

    # Get all states (both user-defined and augmented)
    states = result._states if hasattr(result, "_states") and result._states else []

    # Filter out CTCS augmentation states
    filtered_states = []
    for state in states:
        # Check if this is a CTCS augmentation state (names like _ctcs_aug_0, _ctcs_aug_1, etc.)
        if "ctcs_aug" not in state.name.lower():
            filtered_states.append(state)

    states = filtered_states

    if state_filter:
        states = [s for s in states if s.name in state_filter]

    # Expand states into individual components for multi-dimensional states
    expanded_states = []
    for state in states:
        state_slice = state._slice
        if isinstance(state_slice, slice):
            slice_start = state_slice.start if state_slice.start is not None else 0
            slice_stop = state_slice.stop if state_slice.stop is not None else slice_start + 1
            n_components = slice_stop - slice_start
        else:
            slice_start = state_slice
            n_components = 1

        # Create a separate entry for each component
        if n_components > 1:
            for i in range(n_components):

                class ComponentState:
                    def __init__(self, name: str, idx: int, parent_name: str, comp_idx: int):
                        self.name = f"{parent_name}_{comp_idx}"
                        self._slice = slice(idx, idx + 1)
                        self.parent_name = parent_name
                        self.component_index = comp_idx

                expanded_states.append(ComponentState(state.name, slice_start + i, state.name, i))
        else:
            # Single component, keep as is
            class SingleState:
                def __init__(self, name: str, idx: int):
                    self.name = name
                    self._slice = slice(idx, idx + 1)
                    self.parent_name = name
                    self.component_index = 0

            expanded_states.append(SingleState(state.name, slice_start))

    # Calculate grid dimensions based on expanded states
    n_states_total = len(expanded_states)
    n_cols = min(7, n_states_total)  # Max 7 columns
    n_rows = (n_states_total + n_cols - 1) // n_cols  # Ceiling division

    # Create subplot titles from expanded state names
    subplot_titles = [state.name for state in expanded_states]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
    )
    fig.update_layout(title_text="State Trajectories", template="plotly_dark")

    # Plot each expanded state component
    for idx, state in enumerate(expanded_states):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1

        # Get state slice (now always a single index)
        state_slice = state._slice
        slice_start = state_slice.start if state_slice.start is not None else 0
        state_idx = slice_start

        # Get bounds for this state
        if params.sim.x.min is not None and params.sim.x.max is not None:
            x_min = params.sim.x.min[state_idx]
            x_max = params.sim.x.max[state_idx]
        else:
            x_min = -np.inf
            x_max = np.inf

        # Show legend only on first subplot
        show_legend = idx == 0

        # Plot full nonlinear propagation if available
        if has_full_trajectory and state.parent_name in result.trajectory and t_full is not None:
            state_data = result.trajectory[state.parent_name]
            # Handle both 1D and 2D trajectory data
            if state_data.ndim == 1:
                y_data = state_data
            else:
                # Extract the specific component for multi-dimensional states
                y_data = state_data[:, state.component_index]

            fig.add_trace(
                go.Scatter(
                    x=t_full,
                    y=y_data,
                    mode="lines",
                    name="Propagated",
                    showlegend=show_legend,
                    legendgroup="propagated",
                    line={"color": "green", "width": 2},
                ),
                row=row,
                col=col,
            )

        # Plot nodes from optimization - use nodes dictionary if available
        if result.nodes and state.parent_name in result.nodes:
            node_data = result.nodes[state.parent_name]
            # Handle both 1D and 2D node data
            if node_data.ndim == 1:
                y_nodes = node_data
            else:
                # Extract the specific component for multi-dimensional states
                y_nodes = node_data[:, state.component_index]

        fig.add_trace(
            go.Scatter(
                x=t_nodes,
                y=y_nodes,
                mode="markers",
                name="Nodes",
                showlegend=show_legend,
                legendgroup="nodes",
                marker={"color": "cyan", "size": 6, "symbol": "circle"},
            ),
            row=row,
            col=col,
        )

        # Add constraint bounds (hlines don't support legend, so we add invisible scatter traces)
        if not np.isinf(x_min):
            fig.add_hline(
                y=x_min,
                line={"color": "red", "width": 1, "dash": "dot"},
                row=row,
                col=col,
            )
        if not np.isinf(x_max):
            fig.add_hline(
                y=x_max,
                line={"color": "red", "width": 1, "dash": "dot"},
                row=row,
                col=col,
            )

        # Add bounds to legend (only once)
        if show_legend and (not np.isinf(x_min) or not np.isinf(x_max)):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name="Bounds",
                    showlegend=True,
                    legendgroup="bounds",
                    line={"color": "red", "width": 1, "dash": "dot"},
                ),
                row=row,
                col=col,
            )

    # Update axis labels
    for i in range(1, n_rows + 1):
        fig.update_xaxes(title_text="Time (s)", row=i, col=1)

    return fig


def _expanded_variable_names(states, controls):
    names = []

    def expand(items):
        expanded = []
        for item in items:
            var_slice = item._slice
            if isinstance(var_slice, slice):
                start = var_slice.start if var_slice.start is not None else 0
                stop = var_slice.stop if var_slice.stop is not None else start + 1
                n_comp = stop - start
            else:
                start = var_slice
                n_comp = 1

            if n_comp > 1:
                for i in range(n_comp):
                    expanded.append((f"{item.name}_{i}", start + i))
            else:
                expanded.append((item.name, start))
        return expanded

    names.extend(expand(states))
    names.extend(expand(controls))
    return [n for n, _ in names]


def plot_trust_region_heatmap(result: OptimizationResults, problem=None):
    """Plot heatmap of the final trust-region deltas (TR_history[-1])."""

    if result is None:
        if problem is None:
            raise ValueError("Provide a result or a problem with a cached solution")
        if not hasattr(problem, "_solution") or problem._solution is None:
            raise ValueError("Problem has no cached solution; run solve() first")
        from openscvx.algorithms import format_result

        result = format_result(problem, problem._solution, True)

    if not getattr(result, "TR_history", None):
        raise ValueError("Result has no TR_history to plot")

    tr_mat = result.TR_history[-1]
    var_names = _expanded_variable_names(
        getattr(result, "_states", []) or [], getattr(result, "_controls", []) or []
    )

    # TR matrix is (n_states+n_controls, n_nodes): rows = variables, cols = nodes
    if tr_mat.shape[0] == len(var_names):
        z = tr_mat
    elif tr_mat.shape[1] == len(var_names):
        z = tr_mat.T
    else:
        raise ValueError("TR matrix dimensions do not align with state/control components")

    x_len = z.shape[1]

    # Node labels
    if result.nodes and "time" in result.nodes and len(result.nodes["time"]) == x_len:
        x_labels = result.nodes["time"].flatten()
    else:
        x_labels = list(range(x_len))

    fig = go.Figure(data=go.Heatmap(z=z, x=x_labels, y=var_names, colorscale="Viridis"))
    fig.update_layout(
        title="Trust Region Delta Magnitudes (last iteration)", template="plotly_dark"
    )
    fig.update_xaxes(title_text="Node / Time", side="bottom")
    fig.update_yaxes(title_text="State / Control component", side="left")
    return fig


def plot_virtual_control_heatmap(result: OptimizationResults, problem=None):
    """Plot heatmap of the final virtual control magnitudes (VC_history[-1])."""

    if result is None:
        if problem is None:
            raise ValueError("Provide a result or a problem with a cached solution")
        if not hasattr(problem, "_solution") or problem._solution is None:
            raise ValueError("Problem has no cached solution; run solve() first")
        from openscvx.algorithms import format_result

        result = format_result(problem, problem._solution, True)

    if not getattr(result, "VC_history", None):
        raise ValueError("Result has no VC_history to plot")

    vc_mat = result.VC_history[-1]
    # Virtual control only applies to states, not controls
    state_names = _expanded_variable_names(getattr(result, "_states", []) or [], [])

    # Align so rows = states, cols = nodes
    if vc_mat.shape[1] == len(state_names):
        z = vc_mat.T  # (states, nodes)
    elif vc_mat.shape[0] == len(state_names):
        z = vc_mat
    else:
        raise ValueError("VC matrix shape does not align with state components")

    x_len = z.shape[1]

    # Node labels - virtual control uses N-1 nodes (between nodes)
    if result.nodes and "time" in result.nodes:
        t_all = result.nodes["time"].flatten()
        if len(t_all) == x_len + 1:
            # Use midpoints between nodes or just first N-1 time values
            x_labels = t_all[:-1]  # First N-1 nodes
        elif len(t_all) == x_len:
            x_labels = t_all
        else:
            x_labels = list(range(x_len))
    else:
        x_labels = list(range(x_len))

    fig = go.Figure(data=go.Heatmap(z=z, x=x_labels, y=state_names, colorscale="Magma"))
    fig.update_layout(title="Virtual Control Magnitudes (last iteration)", template="plotly_dark")
    fig.update_xaxes(title_text="Node Interval (N-1)")
    fig.update_yaxes(title_text="State component")
    return fig


def plot_control(
    result: OptimizationResults = None,
    params: Config = None,
    problem=None,
    control_names=None,
):
    """Plot control trajectories over time with bounds.

    Shows the optimized control trajectory (nodes and full propagation if available),
    initial guess, and constraint bounds for all control variables.

    Args:
        result: Optimization results containing control trajectories (optional if problem provided)
        params: Configuration with control bounds and metadata (optional if problem provided)
        problem: Problem instance to extract result and params from (optional)
        control_names: Optional list of control names to plot; defaults to all controls

    Returns:
        Plotly figure with control trajectory subplots
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

    # Get time values at nodes from the nodes dictionary
    t_nodes = result.nodes["time"].flatten()

    # Check if full propagation trajectory is available
    has_full_trajectory = result.trajectory and len(result.trajectory) > 0

    # Get time for full trajectory
    if has_full_trajectory:
        t_full = result.trajectory["time"].flatten()

    # Get all controls (both user-defined and augmented)
    controls = result._controls if hasattr(result, "_controls") and result._controls else []

    # Optional filtering of which controls to plot
    control_filter = set(control_names) if control_names else None
    if control_filter:
        controls = [c for c in controls if c.name in control_filter]

    # Expand controls into individual components for multi-dimensional controls
    expanded_controls = []
    for control in controls:
        control_slice = control._slice
        if isinstance(control_slice, slice):
            slice_start = control_slice.start if control_slice.start is not None else 0
            slice_stop = control_slice.stop if control_slice.stop is not None else slice_start + 1
            n_components = slice_stop - slice_start
        else:
            slice_start = control_slice
            n_components = 1

        # Create a separate entry for each component
        if n_components > 1:
            for i in range(n_components):

                class ComponentControl:
                    def __init__(self, idx: int, parent_name: str, comp_idx: int):
                        self.name = f"{parent_name}_{comp_idx}"
                        self._slice = slice(idx, idx + 1)
                        self.parent_name = parent_name
                        self.component_index = comp_idx

                expanded_controls.append(ComponentControl(slice_start + i, control.name, i))
        else:
            # Single component, keep as is
            class SingleControl:
                def __init__(self, name: str, idx: int):
                    self.name = name
                    self._slice = slice(idx, idx + 1)
                    self.parent_name = name
                    self.component_index = 0

            expanded_controls.append(SingleControl(control.name, slice_start))

    # Calculate grid dimensions based on expanded controls
    n_controls_total = len(expanded_controls)
    n_cols = min(3, n_controls_total)  # Max 3 columns
    n_rows = (n_controls_total + n_cols - 1) // n_cols  # Ceiling division

    # Create subplot titles from expanded control names
    subplot_titles = [control.name for control in expanded_controls]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
    )
    fig.update_layout(title_text="Control Trajectories", template="plotly_dark")

    # Plot each expanded control component
    for idx, control in enumerate(expanded_controls):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1

        # Get control slice (now always a single index)
        control_slice = control._slice
        slice_start = control_slice.start if control_slice.start is not None else 0
        control_idx = slice_start

        # Get bounds for this control
        u_min = params.sim.u.min[control_idx]
        u_max = params.sim.u.max[control_idx]

        # Show legend only on first subplot
        show_legend = idx == 0

        # Plot full propagated control trajectory if available
        if has_full_trajectory and control.parent_name in result.trajectory and t_full is not None:
            control_data = result.trajectory[control.parent_name]
            # Handle both 1D and 2D trajectory data
            if control_data.ndim == 1:
                y_data = control_data
            else:
                # Extract the specific component for multi-dimensional controls
                y_data = control_data[:, control.component_index]

            fig.add_trace(
                go.Scatter(
                    x=t_full,
                    y=y_data,
                    mode="lines",
                    name="Propagated",
                    showlegend=show_legend,
                    legendgroup="propagated",
                    line={"color": "green", "width": 2},
                ),
                row=row,
                col=col,
            )

        # Plot nodes from optimization - use nodes dictionary if available
        node_data = result.nodes[control.parent_name]
        # Handle both 1D and 2D node data
        if node_data.ndim == 1:
            y_nodes = node_data
        else:
            # Extract the specific component for multi-dimensional controls
            y_nodes = node_data[:, control.component_index]

        fig.add_trace(
            go.Scatter(
                x=t_nodes,
                y=y_nodes,
                mode="markers",
                name="Nodes",
                showlegend=show_legend,
                legendgroup="nodes",
                marker={"color": "cyan", "size": 6, "symbol": "circle"},
            ),
            row=row,
            col=col,
        )

        # Add constraint bounds (hlines don't support legend, so we add invisible scatter traces)
        if not np.isinf(u_min):
            fig.add_hline(
                y=u_min,
                line={"color": "red", "width": 1, "dash": "dot"},
                row=row,
                col=col,
            )
        if not np.isinf(u_max):
            fig.add_hline(
                y=u_max,
                line={"color": "red", "width": 1, "dash": "dot"},
                row=row,
                col=col,
            )

        # Add bounds to legend (only once)
        if show_legend and (not np.isinf(u_min) or not np.isinf(u_max)):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name="Bounds",
                    showlegend=True,
                    legendgroup="bounds",
                    line={"color": "red", "width": 1, "dash": "dot"},
                ),
                row=row,
                col=col,
            )

    # Update axis labels
    for i in range(1, n_rows + 1):
        fig.update_xaxes(title_text="Time (s)", row=i, col=1)

    return fig
