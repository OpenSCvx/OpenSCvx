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


class ProblemPlotMixin:
    """Mixin class that adds plotting methods to Problem.

    This can be inherited by the Problem class to add plotting functionality
    without coupling the plotting module to the problem module.

    Note: This mixin expects the class using it to have a `settings` attribute
    of type Config.
    """

    settings: Config  # Type hint for the settings attribute from Problem class

    def plot_state(self, result: OptimizationResults = None, state_names=None):
        """Plot state trajectories with bounds.

        Shows the optimized state trajectory (nodes and full propagation if available),
        initial guess, and constraint bounds for all state variables.

        Args:
            result: OptimizationResults object from solve() or post_process()
                (optional, uses internal solution if not provided)

        Returns:
            Plotly figure with state trajectory subplots

        Example:
            results = problem.solve()
            fig = problem.plot_state()  # Uses internal solution
            fig.show()

            # Or explicitly pass a result
            results = problem.post_process()
            fig = problem.plot_state(results)
            fig.show()
        """
        return plot_state(
            result=result, params=self.settings, problem=self, state_names=state_names
        )

    def plot_control(self, result: OptimizationResults = None, control_names=None):
        """Plot control trajectories with bounds.

        Shows the optimized control trajectory (nodes and full propagation if available),
        initial guess, and constraint bounds for all control variables.

        Args:
            result: OptimizationResults object from solve() or post_process()
                (optional, uses internal solution if not provided)

        Returns:
            Plotly figure with control trajectory subplots

        Example:
            results = problem.solve()
            fig = problem.plot_control()  # Uses internal solution
            fig.show()

            # Or explicitly pass a result
            results = problem.post_process()
            fig = problem.plot_control(results)
            fig.show()
        """
        return plot_control(
            result=result, params=self.settings, problem=self, control_names=control_names
        )

    def plot_trust_region_heatmap(self, result: OptimizationResults = None):
        """Plot heatmap of trust-region deltas (last iteration)."""
        if result is None:
            if hasattr(self, "_solution") and self._solution is not None:
                from openscvx.algorithms import format_result

                result = format_result(self, self._solution, True)
            else:
                raise ValueError("Provide a result or solve the problem first")
        return plot_trust_region_heatmap(result=result, problem=self)

    def plot_virtual_control_heatmap(self, result: OptimizationResults = None):
        """Plot heatmap of virtual control magnitudes (last iteration)."""
        if result is None:
            if hasattr(self, "_solution") and self._solution is not None:
                from openscvx.algorithms import format_result

                result = format_result(self, self._solution, True)
            else:
                raise ValueError("Provide a result or solve the problem first")
        return plot_virtual_control_heatmap(result=result, problem=self)

    def plot_scp_animation(
        self,
        result: OptimizationResults = None,
        state_names=None,
        control_names=None,
    ):
        """Create an animated plot showing SCP iteration convergence.

        This function creates a Plotly animation that shows how the state and control
        trajectories evolve through each SCP iteration using the discretization history.

        Args:
            result: OptimizationResults object from solve() containing iteration history
                (optional, uses internal solution if not provided)

        Returns:
            Plotly figure with animation frames for each SCP iteration

        Example:
            results = problem.solve()
            fig = problem.plot_scp_animation()  # Uses internal solution
            fig.show()

            # Or explicitly pass a result
            fig = problem.plot_scp_animation(results)
            fig.show()
        """
        return plot_scp_iteration_animation(
            result=result,
            params=self.settings,
            problem=self,
            state_names=state_names,
            control_names=control_names,
        )


def register_plotting_methods(problem_class):
    """Register plotting methods on a Problem class.

    This function can be called from problem.py to add plotting methods
    without creating a circular import dependency.

    Args:
        problem_class: The Problem class to add plotting methods to

    Example:
        from openscvx.plotting.plotting import register_plotting_methods
        register_plotting_methods(Problem)
    """
    problem_class.plot_state = ProblemPlotMixin.plot_state
    problem_class.plot_control = ProblemPlotMixin.plot_control
    problem_class.plot_scp_animation = ProblemPlotMixin.plot_scp_animation
