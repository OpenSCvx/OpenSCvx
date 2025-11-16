import random

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional pyqtgraph imports
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from PyQt5 import QtWidgets

    PYQTPHOT_AVAILABLE = True
except ImportError:
    PYQTPHOT_AVAILABLE = False
    pg = None
    gl = None
    QtWidgets = None

from openscvx.config import Config
from openscvx.results import OptimizationResults
from openscvx.utils import get_kp_pose


def qdcm(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion to a direction cosine matrix (DCM).

    Args:
        q: Quaternion array [w, x, y, z] where w is the scalar part

    Returns:
        3x3 rotation matrix (direction cosine matrix)
    """
    q_norm = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
    w, x, y, z = q / q_norm
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


# Helper functions for plotting polytope connections
def create_connection_line_3d(positions, i, node_a, node_b, color="red", width=3):
    """Create a 3D line connecting two nodes."""
    return go.Scatter3d(
        x=[positions[node_a][i][0], positions[node_b][i][0]],
        y=[positions[node_a][i][1], positions[node_b][i][1]],
        z=[positions[node_a][i][2], positions[node_b][i][2]],
        mode="lines",
        line={"color": color, "width": width},
        showlegend=False,
    )


def create_connection_line_2d_projected(positions, i, node_a, node_b, color="red", width=3):
    """Create a 2D line connecting two nodes with z-normalization (perspective projection)."""
    return go.Scatter(
        x=[
            positions[node_a][i][0] / positions[node_a][i][2],
            positions[node_b][i][0] / positions[node_b][i][2],
        ],
        y=[
            positions[node_a][i][1] / positions[node_a][i][2],
            positions[node_b][i][1] / positions[node_b][i][2],
        ],
        mode="lines",
        line={"color": color, "width": width},
        showlegend=False,
    )


def create_closed_polygon_3d(vertices, color="blue", width=10):
    """Create a 3D closed polygon from a list of vertices."""
    # Close the polygon by appending the first vertex
    closed_vertices = vertices + [vertices[0]]
    x_coords = [v[0] for v in closed_vertices]
    y_coords = [v[1] for v in closed_vertices]
    z_coords = [v[2] for v in closed_vertices]

    return go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode="lines",
        showlegend=False,
        line={"color": color, "width": width},
    )


def create_sphere_surface(center, radius=1.0, rotation_matrix=None, n_points=20):
    """Create a sphere or ellipsoid surface mesh.

    Args:
        center: 3D center point (array-like)
        radius: scalar or 3-element array for ellipsoid radii
        rotation_matrix: optional 3x3 rotation matrix
        n_points: mesh resolution

    Returns:
        points: (n_points**2, 3) array of mesh vertices
        n_points: resolution (useful for reshaping)
    """
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Handle scalar or vector radius
    if np.isscalar(radius):
        radius = np.array([radius, radius, radius])
    else:
        radius = np.asarray(radius)

    x = radius[0] * x
    y = radius[1] * y
    z = radius[2] * z

    points = np.array([x.flatten(), y.flatten(), z.flatten()])

    if rotation_matrix is not None:
        points = rotation_matrix @ points

    points = points.T + np.asarray(center)
    return points, n_points


def generate_subject_colors(result_or_count, min_rgb=0, max_rgb=255):
    """Generate random RGB colors for subjects/keypoints.

    Args:
        result_or_count: either a result dictionary (checks for 'init_poses') or an integer count
        min_rgb: minimum RGB value (0-255)
        max_rgb: maximum RGB value (0-255)

    Returns:
        List of RGB color strings
    """
    if isinstance(result_or_count, int):
        n_subjects = result_or_count
    else:
        n_subjects = len(result_or_count["init_poses"]) if "init_poses" in result_or_count else 1
    return [
        f"rgb({random.randint(min_rgb, max_rgb)}, {random.randint(min_rgb, max_rgb)}, "
        f"{random.randint(min_rgb, max_rgb)})"
        for _ in range(n_subjects)
    ]


def compute_cone_projection(values, A, norm_type, fixed_axis_value=0, axis_index=0):
    """Compute 1D projection of conic constraint.

    Args:
        values: array of values to evaluate along one axis
        A: conic constraint matrix (2x2)
        norm_type: "inf" or numeric norm order
        fixed_axis_value: value for the fixed axis (default 0)
        axis_index: 0 for x-projection, 1 for y-projection

    Returns:
        Array of z-values defining the cone boundary
    """
    z = []
    for val in values:
        vector = [0, 0]
        vector[axis_index] = val
        vector[1 - axis_index] = fixed_axis_value

        if norm_type == "inf":
            z.append(np.linalg.norm(A @ np.array(vector), axis=0, ord=np.inf))
        else:
            z.append(np.linalg.norm(A @ np.array(vector), axis=0, ord=norm_type))
    return np.array(z)


def frame_args(duration):
    """Create frame arguments for plotly animations.

    Args:
        duration: duration in milliseconds

    Returns:
        Dictionary of frame arguments
    """
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }


# Modular component functions for building plots
def add_animation_controls(
    fig, slider_x=0.15, slider_y=0.32, play_speed=50, frame_speed=500, button_x=None, button_y=None
):
    """Add animation slider and play/pause controls to a plotly figure.

    Args:
        fig: plotly figure with frames already set
        slider_x: x position of slider (0-1)
        slider_y: y position of slider (0-1)
        play_speed: play button frame duration in ms
        frame_speed: slider frame duration in ms
        button_x: optional x position of buttons (defaults to slider_x)
        button_y: optional y position of buttons (defaults to slider_y)

    Returns:
        fig: modified figure
    """
    # Default button position to slider position if not specified
    if button_x is None:
        button_x = slider_x
    if button_y is None:
        button_y = slider_y

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.8,
            "x": slider_x,
            "y": slider_y,
            "steps": [
                {
                    "args": [[f.name], frame_args(frame_speed)],
                    "label": f.name,
                    "method": "animate",
                }
                for f in fig.frames
            ],
        }
    ]

    updatemenus = [
        {
            "buttons": [
                {
                    "args": [None, frame_args(play_speed)],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [[None], frame_args(0)],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "type": "buttons",
            "x": button_x,
            "y": button_y,
        }
    ]

    fig.update_layout(updatemenus=updatemenus, sliders=sliders)
    return fig


def add_obstacles(fig, centers, axes, radii, opacity=0.5, color=None, n_points=20):
    """Add ellipsoidal obstacle surfaces to a plotly figure.

    Args:
        fig: plotly figure
        centers: list of 3D center points
        axes: list of 3x3 rotation matrices
        radii: list of 3-element radius arrays
        opacity: surface opacity (0-1)
        color: optional colorscale name
        n_points: mesh resolution

    Returns:
        fig: modified figure
    """
    for center, axis, radius in zip(centers, axes, radii):
        # Use inverse radius for scaling (matches original behavior)
        inv_radius = 1.0 / np.asarray(radius)
        points, n = create_sphere_surface(center, inv_radius, axis, n_points)

        trace_kwargs = {
            "x": points[:, 0].reshape(n, n),
            "y": points[:, 1].reshape(n, n),
            "z": points[:, 2].reshape(n, n),
            "opacity": opacity,
            "showscale": False,
        }
        if color:
            trace_kwargs["colorscale"] = color

        fig.add_trace(go.Surface(**trace_kwargs))
    return fig


def add_range_spheres(fig, center, min_range, max_range, n_points=20):
    """Add min and max range sphere surfaces around a center point.

    Args:
        fig: plotly figure
        center: 3D center point
        min_range: minimum range radius
        max_range: maximum range radius
        n_points: mesh resolution

    Returns:
        fig: modified figure
    """
    points_min, n = create_sphere_surface(center, min_range, n_points=n_points)
    points_max, _ = create_sphere_surface(center, max_range, n_points=n_points)

    fig.add_trace(
        go.Surface(
            x=points_min[:, 0].reshape(n, n),
            y=points_min[:, 1].reshape(n, n),
            z=points_min[:, 2].reshape(n, n),
            opacity=0.2,
            colorscale="reds",
            name="Minimum Range",
            showlegend=True,
            showscale=False,
        )
    )
    fig.add_trace(
        go.Surface(
            x=points_max[:, 0].reshape(n, n),
            y=points_max[:, 1].reshape(n, n),
            z=points_max[:, 2].reshape(n, n),
            opacity=0.2,
            colorscale="blues",
            name="Maximum Range",
            showlegend=True,
            showscale=False,
        )
    )
    return fig


def add_ground_plane(fig, size=200, z_level=0, opacity=0.3):
    """Add a ground plane surface to a plotly figure.

    Args:
        fig: plotly figure
        size: half-width of square plane
        z_level: z-coordinate of plane
        opacity: surface opacity (0-1)

    Returns:
        fig: modified figure
    """
    fig.add_trace(
        go.Surface(
            x=[-size, size, size, -size],
            y=[-size, -size, size, size],
            z=[[z_level, z_level], [z_level, z_level], [z_level, z_level], [z_level, z_level]],
            opacity=opacity,
            showscale=False,
            colorscale="Greys",
            showlegend=True,
            name="Ground Plane",
        )
    )
    return fig


def add_velocity_trajectory(
    fig,
    positions,
    velocities,
    end_index=None,
    marker_size=5,
    colorscale="Viridis",
    name="Trajectory",
):
    """Add a trajectory trace colored by velocity magnitude.

    Args:
        fig: plotly figure
        positions: Nx3 array of positions
        velocities: Nx3 array of velocities
        end_index: optional end index (None for full trajectory)
        marker_size: marker size
        colorscale: plotly colorscale name
        name: trace name

    Returns:
        fig: modified figure
    """
    if end_index is None:
        end_index = len(positions) - 1

    pos_slice = positions[: end_index + 1]
    vel_slice = velocities[: end_index + 1]

    fig.add_trace(
        go.Scatter3d(
            x=pos_slice[:, 0],
            y=pos_slice[:, 1],
            z=pos_slice[:, 2],
            mode="markers",
            marker={
                "size": marker_size,
                "color": np.linalg.norm(vel_slice, axis=1),
                "colorscale": colorscale,
                "colorbar": {
                    "title": "Velocity Norm (m/s)",
                    "x": 0.02,
                    "y": 0.55,
                    "len": 0.75,
                },
            },
            name=name,
        )
    )
    return fig


def add_cone_surface(fig, A, norm_type, x_range, y_range, opacity=0.25, n_points=20):
    """Add a second-order cone surface to a plotly figure.

    Args:
        fig: plotly figure
        A: 2x2 conic constraint matrix
        norm_type: "inf" or numeric norm order
        x_range: (min, max) tuple for x values
        y_range: (min, max) tuple for y values
        opacity: surface opacity (0-1)
        n_points: mesh resolution

    Returns:
        fig: modified figure
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)

    z = []
    for x_val in x:
        for y_val in y:
            if norm_type == "inf":
                z.append(np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord=np.inf))
            else:
                z.append(np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord=norm_type))
    z = np.array(z)

    fig.add_trace(
        go.Surface(x=X, y=Y, z=z.reshape(n_points, n_points), opacity=opacity, showscale=False)
    )
    return fig


def add_cone_projections(
    fig, A, norm_type, x_vals, y_vals, x_range, y_range, color="grey", width=3
):
    """Add x-z and y-z plane projections of a cone to a plotly figure.

    Args:
        fig: plotly figure
        A: 2x2 conic constraint matrix
        norm_type: "inf" or numeric norm order
        x_vals: x-coordinate(s) for y-z projection plane
        y_vals: y-coordinate(s) for x-z projection plane
        x_range: (min, max) tuple for x values
        y_range: (min, max) tuple for y values
        color: line color
        width: line width

    Returns:
        fig: modified figure
    """
    x = np.linspace(x_range[0], x_range[1], 20)
    y = np.linspace(y_range[0], y_range[1], 20)

    # X-Z plane projection (y fixed)
    z_x = compute_cone_projection(x, A, norm_type, fixed_axis_value=0, axis_index=0)
    fig.add_trace(
        go.Scatter3d(
            y=x,
            x=y_vals,
            z=z_x,
            mode="lines",
            showlegend=False,
            line={"color": color, "width": width},
        )
    )

    # Y-Z plane projection (x fixed)
    z_y = compute_cone_projection(y, A, norm_type, fixed_axis_value=0, axis_index=1)
    fig.add_trace(
        go.Scatter3d(
            y=x_vals,
            x=y,
            z=z_y,
            mode="lines",
            showlegend=False,
            line={"color": color, "width": width},
        )
    )

    return fig


# Helper functions that return traces (for use in animation frames)
def create_velocity_trajectory_trace(
    positions, velocities, end_index, marker_size=5, colorscale="Viridis", name="Trajectory"
):
    """Create a trajectory trace colored by velocity magnitude.

    Args:
        positions: Nx3 array of positions
        velocities: Nx3 array of velocities
        end_index: end index for slicing
        marker_size: marker size
        colorscale: plotly colorscale name
        name: trace name

    Returns:
        go.Scatter3d trace
    """
    pos_slice = positions[: end_index + 1]
    vel_slice = velocities[: end_index + 1]

    return go.Scatter3d(
        x=pos_slice[:, 0],
        y=pos_slice[:, 1],
        z=pos_slice[:, 2],
        mode="markers",
        marker={
            "size": marker_size,
            "color": np.linalg.norm(vel_slice, axis=1),
            "colorscale": colorscale,
            "colorbar": {
                "title": "Velocity Norm (m/s)",
                "x": 0.02,
                "y": 0.55,
                "len": 0.75,
            },
        },
        name=name,
    )


def create_range_sphere_traces(center, min_range, max_range, n_points=20):
    """Create min and max range sphere surface traces.

    Args:
        center: 3D center point
        min_range: minimum range radius
        max_range: maximum range radius
        n_points: mesh resolution

    Returns:
        List of two go.Surface traces
    """
    points_min, n = create_sphere_surface(center, min_range, n_points=n_points)
    points_max, _ = create_sphere_surface(center, max_range, n_points=n_points)

    return [
        go.Surface(
            x=points_min[:, 0].reshape(n, n),
            y=points_min[:, 1].reshape(n, n),
            z=points_min[:, 2].reshape(n, n),
            opacity=0.2,
            colorscale="reds",
            name="Minimum Range",
            showlegend=True,
            showscale=False,
        ),
        go.Surface(
            x=points_max[:, 0].reshape(n, n),
            y=points_max[:, 1].reshape(n, n),
            z=points_max[:, 2].reshape(n, n),
            opacity=0.2,
            colorscale="blues",
            name="Maximum Range",
            showlegend=True,
            showscale=False,
        ),
    ]


def plot_dubins_car(results: OptimizationResults, params: Config):
    # Plot the trajectory of the Dubins car in 3d as an animaiton
    fig = go.Figure()

    position = results.trajectory["position"]
    x = position[:, 0]
    y = position[:, 1]

    obs_center = results.plotting_data["obs_center"]
    obs_radius = results.plotting_data["obs_radius"]

    # Create a 2D scatter plot
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", line={"color": "blue", "width": 2}, name="Trajectory")
    )

    # Plot the circular obstacle
    fig.add_trace(
        go.Scatter(
            x=obs_center[0] + obs_radius * np.cos(np.linspace(0, 2 * np.pi, 100)),
            y=obs_center[1] + obs_radius * np.sin(np.linspace(0, 2 * np.pi, 100)),
            mode="lines",
            line={"color": "red", "width": 2},
            name="Obstacle",
        )
    )

    fig.update_layout(title="Dubins Car Trajectory", title_x=0.5, template="plotly_dark")

    # Set axis to be equal
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig


def plot_dubins_car_disjoint(results: OptimizationResults, params: Config):
    # Plot the trajectory of the Dubins car, but show wp1 and wp2 as circles with centers and radii
    fig = go.Figure()

    position = results.trajectory["position"]
    x = position[:, 0]
    y = position[:, 1]
    # Use the forward velocity from the control input
    velocity = results.trajectory.get("speed")
    if velocity is not None:
        # Flatten to 1D array for Plotly color mapping
        velocity = np.asarray(velocity).flatten()
    else:
        velocity = np.zeros_like(x)

    # Plot the trajectory colored by velocity
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line={"color": "rgba(0,0,0,0)"},  # Hide default line
            marker={
                "color": velocity,
                "colorscale": "Viridis",
                "size": 6,
                "colorbar": {"title": "Velocity"},
                "showscale": True,
            },
            name="Trajectory (velocity)",
        )
    )

    # Plot waypoints wp1 and wp2 as circles and their centers
    # Handle 0, 1, or 2 waypoints
    # Handle wp1 (optional)
    if "wp1_center" in results and "wp1_radius" in results:
        wp1_center = results.get("wp1_center")
        wp1_radius = results.get("wp1_radius")

        # Extract values if they are Parameter objects or other non-array types
        if hasattr(wp1_center, "value"):
            wp1_center = np.asarray(wp1_center.value)
        else:
            wp1_center = np.asarray(wp1_center)

        if hasattr(wp1_radius, "value"):
            wp1_radius = np.asarray(wp1_radius.value)
        else:
            wp1_radius = np.asarray(wp1_radius)

        # Ensure they are scalars/arrays
        wp1_center = np.asarray(wp1_center).flatten()
        wp1_radius = float(np.asarray(wp1_radius).item())

        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = wp1_center[0] + wp1_radius * np.cos(theta)
        circle_y = wp1_center[1] + wp1_radius * np.sin(theta)
        fig.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode="lines",
                line={"color": "green", "width": 2, "dash": "dash"},
                name="Waypoint 1 Area",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[wp1_center[0]],
                y=[wp1_center[1]],
                mode="markers",
                marker={"color": "green", "size": 12, "symbol": "x"},
                name="Waypoint 1 Center",
            )
        )

    # Handle wp2 (optional)
    if "wp2_center" in results and "wp2_radius" in results:
        wp2_center = results.get("wp2_center")
        wp2_radius = results.get("wp2_radius")

        # Extract values if they are Parameter objects or other non-array types
        if hasattr(wp2_center, "value"):
            wp2_center = np.asarray(wp2_center.value)
        else:
            wp2_center = np.asarray(wp2_center)

        if hasattr(wp2_radius, "value"):
            wp2_radius = np.asarray(wp2_radius.value)
        else:
            wp2_radius = np.asarray(wp2_radius)

        # Ensure they are scalars/arrays
        wp2_center = np.asarray(wp2_center).flatten()
        wp2_radius = float(np.asarray(wp2_radius).item())

        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = wp2_center[0] + wp2_radius * np.cos(theta)
        circle_y = wp2_center[1] + wp2_radius * np.sin(theta)
        fig.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode="lines",
                line={"color": "orange", "width": 2, "dash": "dash"},
                name="Waypoint 2 Area",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[wp2_center[0]],
                y=[wp2_center[1]],
                mode="markers",
                marker={"color": "orange", "size": 12, "symbol": "x"},
                name="Waypoint 2 Center",
            )
        )

    fig.update_layout(
        title="Dubins Car Trajectory with Waypoints", title_x=0.5, template="plotly_dark"
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig


def full_subject_traj_time(results: OptimizationResults, params: Config):
    x_full = results.x_full
    x_nodes = results.x.guess
    t_nodes = x_nodes[:, params.sim.time_slice]
    t_full = results.t_full
    subs_traj = []
    subs_traj_node = []
    subs_traj_sen = []
    subs_traj_sen_node = []

    # if hasattr(params.dyn, 'get_kp_pose'):
    if "moving_subject" in results and "init_poses" in results:
        init_poses = results.plotting_data["init_poses"]
        subs_traj.append(get_kp_pose(t_full, init_poses))
        subs_traj_node.append(get_kp_pose(t_nodes, init_poses))
        subs_traj_node[0] = subs_traj_node[0].squeeze()
    elif "init_poses" in results:
        init_poses = results.plotting_data["init_poses"]
        for pose in init_poses:
            # repeat the pose for all time steps
            pose_full = np.repeat(pose[:, np.newaxis], x_full.shape[0], axis=1).T
            subs_traj.append(pose_full)

            pose_node = np.repeat(pose[:, np.newaxis], x_nodes.shape[0], axis=1).T
            subs_traj_node.append(pose_node)
    else:
        raise ValueError("No valid method to get keypoint poses.")

    if "R_sb" in results:
        R_sb = results.plotting_data["R_sb"]
        for sub_traj in subs_traj:
            sub_traj_sen = []
            for i in range(x_full.shape[0]):
                sub_pose = sub_traj[i]
                sub_traj_sen.append(R_sb @ qdcm(x_full[i, 6:10]).T @ (sub_pose - x_full[i, 0:3]))
            subs_traj_sen.append(np.array(sub_traj_sen).squeeze())

        for sub_traj_node in subs_traj_node:
            sub_traj_sen_node = []
            for i in range(x_nodes.shape[0]):
                sub_pose = sub_traj_node[i]
                sub_traj_sen_node.append(
                    R_sb @ qdcm(x_nodes[i, 6:10]).T @ (sub_pose - x_nodes[i, 0:3]).T
                )
            subs_traj_sen_node.append(np.array(sub_traj_sen_node).squeeze())
        return subs_traj, subs_traj_sen, subs_traj_node, subs_traj_sen_node
    else:
        raise ValueError("`R_sb` not found in results. Cannot compute sensor frame.")


def plot_camera_view(result: OptimizationResults, params: Config) -> None:
    title = r"$\text{Camera View}$"
    _, sub_positions_sen, _, sub_positions_sen_node = full_subject_traj_time(result, params)
    fig = go.Figure()

    # Create a cone plot
    A = np.diag(
        [
            1 / np.tan(np.pi / result.plotting_data["alpha_y"]),
            1 / np.tan(np.pi / result.plotting_data["alpha_x"]),
        ]
    )  # Conic Matrix

    # Meshgrid
    if "moving_subject" in result:
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        z = np.linspace(-10, 10, 100)
    else:
        x = np.linspace(-80, 80, 100)
        y = np.linspace(-80, 80, 100)
        z = np.linspace(-80, 80, 100)

    X, Y = np.meshgrid(x, y)

    # Define the condition for the second order cone
    z = []
    for x_val in x:
        for y_val in y:
            if result.plotting_data["norm_type"] == "inf":
                z.append(np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord=np.inf))
            else:
                z.append(
                    np.linalg.norm(
                        A @ np.array([x_val, y_val]), axis=0, ord=result.plotting_data["norm_type"]
                    )
                )
    z = np.array(z)

    # Extract the points from the meshgrid
    X = X.flatten()
    Y = Y.flatten()
    Z = z.flatten()

    # Normalize the coordinates by the Z value
    X = X / Z
    Y = Y / Z

    # Order the points so they are connected in radial order about the origin
    order = np.argsort(np.arctan2(Y, X))
    X = X[order]
    Y = Y[order]

    # Repeat the first point to close the cone
    X = np.append(X, X[0])
    Y = np.append(Y, Y[0])

    # Plot the points on a red scatter plot
    fig.add_trace(
        go.Scatter(
            x=X, y=Y, mode="lines", line={"color": "red", "width": 5}, name=r"$\text{Camera Frame}$"
        )
    )

    sub_idx = 0
    for sub_traj in sub_positions_sen:
        color = (
            f"rgb({random.randint(10, 255)}, {random.randint(10, 255)}, {random.randint(10, 255)})"
        )
        sub_traj = np.array(sub_traj)
        sub_traj[:, 0] = sub_traj[:, 0] / sub_traj[:, 2]
        sub_traj[:, 1] = sub_traj[:, 1] / sub_traj[:, 2]
        fig.add_trace(
            go.Scatter(
                x=sub_traj[:, 0],
                y=sub_traj[:, 1],
                mode="lines",
                line={"color": color, "width": 3},
                name=r"$\text{Subject }" + str(sub_idx) + "$",
            )
        )

        sub_traj_nodal = np.array(sub_positions_sen_node[sub_idx])
        sub_traj_nodal[:, 0] = sub_traj_nodal[:, 0] / sub_traj_nodal[:, 2]
        sub_traj_nodal[:, 1] = sub_traj_nodal[:, 1] / sub_traj_nodal[:, 2]
        fig.add_trace(
            go.Scatter(
                x=sub_traj_nodal[:, 0],
                y=sub_traj_nodal[:, 1],
                mode="markers",
                marker={"color": color, "size": 20},
                name=r"$\text{Subject }" + str(sub_idx) + r"\text{ Node}$",
            )
        )
        sub_idx += 1

    # Center the title for the plot
    fig.update_layout(title=title, title_x=0.5)
    fig.update_layout(template="simple_white")

    # Increase title size
    fig.update_layout(title_font_size=20)

    # Increase legend size
    fig.update_layout(legend_font_size=15)

    # fig.update_yaxes(scaleanchor="x", scaleratio=1,)
    fig.update_layout(height=600)

    # Set x axis and y axis limits
    fig.update_xaxes(range=[-1.0, 1.0])
    fig.update_yaxes(range=[-1.0, 1.0])
    # Set aspect ratio to be equal
    fig.update_layout(autosize=False, width=800, height=800)

    # Save figure as svg
    fig.write_image("figures/camera_view.svg")

    return fig


def plot_camera_animation(result: dict, params: Config, path="") -> None:
    title = r"$\text{Camera Animation}$"
    _, subs_positions_sen, _, subs_positions_sen_node = full_subject_traj_time(result, params)
    fig = go.Figure()

    # Add blank plots for the subjects
    for _ in range(50):
        fig.add_trace(
            go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", line={"color": "blue", "width": 2})
        )

    # Create a cone plot
    if "alpha_x" in result and "alpha_y" in result:
        A = np.diag(
            [1 / np.tan(np.pi / result["alpha_y"]), 1 / np.tan(np.pi / result["alpha_x"])]
        )  # Conic Matrix
    else:
        raise ValueError("`alpha_x` and `alpha_y` not found in result dictionary.")

    # Meshgrid
    range_limit = 10 if "moving_subject" in result else 80
    x = np.linspace(-range_limit, range_limit, 50)
    y = np.linspace(-range_limit, range_limit, 50)
    X, Y = np.meshgrid(x, y)

    # Define the condition for the second order cone
    if "norm_type" in result:
        z = np.array(
            [
                np.linalg.norm(
                    A @ np.array([x_val, y_val]),
                    axis=0,
                    ord=(np.inf if result["norm_type"] == "inf" else result["norm_type"]),
                )
                for x_val in x
                for y_val in y
            ]
        )
    else:
        raise ValueError("`norm_type` not found in result dictionary.")

    # Extract the points from the meshgrid
    X, Y, Z = X.flatten(), Y.flatten(), z.flatten()

    # Normalize the coordinates by the Z value
    X, Y = X / Z, Y / Z

    # Order the points so they are connected in radial order about the origin
    order = np.argsort(np.arctan2(Y, X))
    X, Y = X[order], Y[order]

    # Repeat the first point to close the cone
    X, Y = np.append(X, X[0]), np.append(Y, Y[0])

    # Plot the points on a red scatter plot
    fig.add_trace(
        go.Scatter(
            x=X,
            y=Y,
            mode="lines",
            line={"color": "red", "width": 5},
            name=r"$\text{Camera Frame}$",
            showlegend=False,
        )
    )

    # Choose a random color for each subject
    colors = generate_subject_colors(len(subs_positions_sen), min_rgb=10, max_rgb=255)

    frames = []
    # Animate the subjects along their trajectories
    for i in range(0, len(subs_positions_sen[0]), 2):
        frame_data = []
        for sub_idx, sub_traj in enumerate(subs_positions_sen):
            color = colors[sub_idx]
            sub_traj = np.array(sub_traj)
            sub_traj_nodal = np.array(subs_positions_sen_node[sub_idx])
            sub_traj[:, 0] /= sub_traj[:, 2]
            sub_traj[:, 1] /= sub_traj[:, 2]
            frame_data.append(
                go.Scatter(
                    x=sub_traj[: i + 1, 0],
                    y=sub_traj[: i + 1, 1],
                    mode="lines",
                    line={"color": color, "width": 3},
                    showlegend=False,
                )
            )

            # Add in node when loop has reached point where node is present
            scaled_index = int((i // (sub_traj.shape[0] / sub_traj_nodal.shape[0])) + 1)
            sub_node_plot = sub_traj_nodal[:scaled_index]
            sub_node_plot[:, 0] /= sub_node_plot[:, 2]
            sub_node_plot[:, 1] /= sub_node_plot[:, 2]
            frame_data.append(
                go.Scatter(
                    x=sub_node_plot[:, 0],
                    y=sub_node_plot[:, 1],
                    mode="markers",
                    marker={"color": color, "size": 10},
                    showlegend=False,
                )
            )

        frames.append(go.Frame(name=str(i), data=frame_data))

    fig.frames = frames

    # Add animation controls using modular component
    add_animation_controls(fig, slider_x=0.15, slider_y=0.15, play_speed=50, frame_speed=500)

    # Center the title for the plot
    fig.update_layout(title=title, title_x=0.5)
    fig.update_layout(template="plotly_dark")
    # Remove grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Remove center line
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)

    # Increase title size
    fig.update_layout(title_font_size=20)

    # Increase legend size
    fig.update_layout(legend_font_size=15)

    # Remove the axis numbers
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    # Remove ticks enrtirely
    fig.update_xaxes(ticks="outside", tickwidth=0, tickcolor="black")
    fig.update_yaxes(ticks="outside", tickwidth=0, tickcolor="black")

    # Set x axis and y axis limits
    fig.update_xaxes(range=[-1.1, 1.1])
    fig.update_yaxes(range=[-1.1, 1.1])

    # Move Title down
    fig.update_layout(title_y=0.9)

    # Set aspect ratio to be equal
    # fig.update_layout(autosize=False, width=650, height=650)
    # Remove marigns
    fig.update_layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})

    return fig


def plot_camera_polytope_animation(result: dict, params: Config, path="") -> None:
    sub_positions_sen, _, sub_positions_sen_node = full_subject_traj_time(
        result["x_full"], params, False
    )
    fig = go.Figure()

    # Add blank plots for the subjects
    for _ in range(500):
        fig.add_trace(
            go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", line={"color": "blue", "width": 2})
        )

    # Create a cone plot
    A = np.diag(
        [1 / np.tan(np.pi / params.vp.alpha_y), 1 / np.tan(np.pi / params.vp.alpha_x)]
    )  # Conic Matrix

    # Meshgrid
    range_limit = 10 if params.vp.tracking else 80
    x = np.linspace(-range_limit, range_limit, 50)
    y = np.linspace(-range_limit, range_limit, 50)
    X, Y = np.meshgrid(x, y)

    # Define the condition for the second order cone
    z = np.array(
        [
            np.linalg.norm(
                A @ np.array([x_val, y_val]),
                axis=0,
                ord=(np.inf if params.vp.norm == "inf" else params.vp.norm),
            )
            for x_val in x
            for y_val in y
        ]
    )

    # Extract the points from the meshgrid
    X, Y, Z = X.flatten(), Y.flatten(), z.flatten()

    # Normalize the coordinates by the Z value
    X, Y = X / Z, Y / Z

    # Order the points so they are connected in radial order about the origin
    order = np.argsort(np.arctan2(Y, X))
    X, Y = X[order], Y[order]

    # Repeat the first point to close the cone
    X, Y = np.append(X, X[0]), np.append(Y, Y[0])

    # Plot the points on a red scatter plot
    fig.add_trace(
        go.Scatter(
            x=X,
            y=Y,
            mode="lines",
            line={"color": "red", "width": 5},
            name=r"$\text{Camera Frame}$",
            showlegend=False,
        )
    )

    # Choose a random color for each subject
    [
        f"rgb({random.randint(10, 255)}, {random.randint(10, 255)}, {random.randint(10, 255)})"
        for _ in sub_positions_sen
    ]

    frames = []
    # Animate the subjects along their trajectories
    for i in range(0, len(sub_positions_sen[0]), 2):
        frame_data = []
        for sub_idx, sub_traj in enumerate(sub_positions_sen):
            sub_traj = np.array(sub_traj)
            sub_traj_nodal = np.array(sub_positions_sen_node[sub_idx])
            sub_traj[:, 0] /= sub_traj[:, 2]
            sub_traj[:, 1] /= sub_traj[:, 2]
            frame_data.append(
                go.Scatter(
                    x=sub_traj[: i + 1, 0],
                    y=sub_traj[: i + 1, 1],
                    mode="lines",
                    line={"color": "darkblue", "width": 3},
                    showlegend=False,
                )
            )

            # Add in node when loop has reached point where node is present
            scaled_index = int((i // (sub_traj.shape[0] / sub_traj_nodal.shape[0])) + 1)
            sub_node_plot = sub_traj_nodal[:scaled_index]
            sub_node_plot[:, 0] /= sub_node_plot[:, 2]
            sub_node_plot[:, 1] /= sub_node_plot[:, 2]
            frame_data.append(
                go.Scatter(
                    x=sub_node_plot[:, 0],
                    y=sub_node_plot[:, 1],
                    mode="markers",
                    marker={"color": "darkblue", "size": 10},
                    showlegend=False,
                )
            )

        # Polytope connection topology: node -> [connected nodes]
        polytope_connections = {
            0: [16, 8, 12],
            1: [17, 9, 12],
            2: [16, 13, 10],
            3: [17, 11, 13],
            4: [18, 14, 8],
            5: [19, 9, 14],
            6: [18, 15, 10],
            7: [19, 11, 15],
            8: [0, 4, 10],
            9: [1, 5, 11],
            10: [8, 2, 6],
            11: [3, 7, 9],
            12: [0, 1, 14],
            13: [2, 3, 15],
            14: [4, 5, 12],
            15: [13, 6, 7],
            16: [0, 2, 17],
            17: [1, 3, 16],
            18: [4, 6, 19],
            19: [5, 7, 18],
        }
        # Connect polytope vertices using helper function and topology dictionary
        frame_data.extend(
            [
                create_connection_line_2d_projected(sub_positions_sen, i, node_a, node_b)
                for node_a, connections in polytope_connections.items()
                for node_b in connections
            ]
        )
        frames.append(go.Frame(name=str(i), data=frame_data))

    fig.frames = frames

    # Add animation controls using modular component
    add_animation_controls(fig, slider_x=0.15, slider_y=0.15, play_speed=50, frame_speed=500)

    # Center the title for the plot
    # fig.update_layout(title=title, title_x=0.5)
    fig.update_layout(template="plotly_dark")
    # Remove grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Remove center line
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)

    # Increase title size
    fig.update_layout(title_font_size=20)

    # Increase legend size
    fig.update_layout(legend_font_size=15)

    # Remove the axis numbers
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    # Remove ticks enrtirely
    fig.update_xaxes(ticks="outside", tickwidth=0, tickcolor="black")
    fig.update_yaxes(ticks="outside", tickwidth=0, tickcolor="black")

    # Set x axis and y axis limits
    fig.update_xaxes(range=[-1.1, 1.1])
    fig.update_yaxes(range=[-1.1, 1.1])

    # Move Title down
    fig.update_layout(title_y=0.9)

    # Set aspect ratio to be equal
    # fig.update_layout(autosize=False, width=650, height=650)
    # Remove marigns
    fig.update_layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})

    return fig


def plot_conic_view_animation(result: dict, params: Config, path="") -> None:
    sub_positions_sen, _, sub_positions_sen_node = full_subject_traj_time(
        result["x_full"], params, False
    )
    fig = go.Figure()
    for i in range(100):
        fig.add_trace(
            go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", line={"color": "blue", "width": 2})
        )

    # Create a cone plot
    if "alpha_x" in result and "alpha_y" in result:
        A = np.diag(
            [1 / np.tan(np.pi / result["alpha_y"]), 1 / np.tan(np.pi / result["alpha_x"])]
        )  # Conic Matrix
    else:
        raise ValueError("`alpha_x` and `alpha_y` not found in result dictionary.")

    # Meshgrid
    if "moving_subject" in result:
        x = np.linspace(-6, 6, 20)
        y = np.linspace(-6, 6, 20)
    else:
        x = np.linspace(-80, 80, 20)
        y = np.linspace(-80, 80, 20)

    X, Y = np.meshgrid(x, y)

    if "norm_type" in result:
        # Add cone surface using helper function
        x_range = (x[0], x[-1])
        y_range = (y[0], y[-1])
        add_cone_surface(fig, A, result["norm_type"], x_range, y_range, opacity=0.25, n_points=20)
        frames = []

        if "moving_subject" in result:
            x_vals = 12 * np.ones_like(np.array(sub_positions_sen[0])[:, 0])
            y_vals = 12 * np.ones_like(np.array(sub_positions_sen[0])[:, 0])
        else:
            x_vals = 110 * np.ones_like(np.array(sub_positions_sen[0])[:, 0])
            y_vals = 110 * np.ones_like(np.array(sub_positions_sen[0])[:, 0])

        # Add cone projections using helper function
        x_range = (x[0], x[-1])
        y_range = (y[0], y[-1])
        add_cone_projections(fig, A, result["norm_type"], x_vals, y_vals, x_range, y_range)
    else:
        raise ValueError("`norm_type` not found in result dictionary.")

    # Choose a random color for each subject
    colors = generate_subject_colors(len(sub_positions_sen), min_rgb=10, max_rgb=255)

    sub_node_plot = []
    for i in range(0, len(sub_positions_sen[0]), 4):
        frame = go.Frame(name=str(i))
        data = []
        sub_idx = 0

        for sub_traj in sub_positions_sen:
            sub_traj = np.array(sub_traj)
            sub_traj_nodal = np.array(sub_positions_sen_node[sub_idx])

            if "moving_subject" in result:
                x_vals = 12 * np.ones_like(sub_traj[: i + 1, 0])
                y_vals = 12 * np.ones_like(sub_traj[: i + 1, 0])
            else:
                x_vals = 110 * np.ones_like(sub_traj[: i + 1, 0])
                y_vals = 110 * np.ones_like(sub_traj[: i + 1, 0])

            data.append(
                go.Scatter3d(
                    x=sub_traj[: i + 1, 0],
                    y=y_vals,
                    z=sub_traj[: i + 1, 2],
                    mode="lines",
                    showlegend=False,
                    line={"color": "grey", "width": 4},
                )
            )
            data.append(
                go.Scatter3d(
                    x=x_vals,
                    y=sub_traj[: i + 1, 1],
                    z=sub_traj[: i + 1, 2],
                    mode="lines",
                    showlegend=False,
                    line={"color": "grey", "width": 4},
                )
            )

            # Add subject position to data
            sub_traj = np.array(sub_traj)
            data.append(
                go.Scatter3d(
                    x=sub_traj[: i + 1, 0],
                    y=sub_traj[: i + 1, 1],
                    z=sub_traj[: i + 1, 2],
                    mode="lines",
                    line={"color": colors[sub_idx], "width": 3},
                    showlegend=False,
                )
            )

            # Add in node when loop has reached point where node is present
            scaled_index = int((i // (sub_traj.shape[0] / sub_traj_nodal.shape[0])) + 1)
            sub_node_plot = sub_traj_nodal[:scaled_index]

            data.append(
                go.Scatter3d(
                    x=sub_node_plot[:, 0],
                    y=sub_node_plot[:, 1],
                    z=sub_node_plot[:, 2],
                    mode="markers",
                    marker={"color": colors[sub_idx], "size": 5},
                    showlegend=False,
                )
            )

            sub_idx += 1

        frame.data = data
        frames.append(frame)

    fig.frames = frames

    # Add animation controls using modular component
    add_animation_controls(fig, slider_x=0.15, slider_y=0.32, play_speed=50, frame_speed=500)

    # Set camera position
    fig.update_layout(
        scene_camera={
            "up": {"x": 0, "y": 0, "z": 10},
            "center": {"x": -2, "y": 0, "z": -3},
            "eye": {"x": -28, "y": -22, "z": 15},
        }
    )

    # Set axis labels
    fig.update_layout(
        scene={"xaxis_title": "x (m)", "yaxis_title": "y (m)", "zaxis_title": "z (m)"}
    )

    fig.update_layout(template="plotly_dark")

    # Make only the grid lines thicker in the template
    fig.update_layout(
        scene={
            "xaxis": {"showgrid": True, "gridwidth": 5},
            "yaxis": {"showgrid": True, "gridwidth": 5},
            "zaxis": {"showgrid": True, "gridwidth": 5},
        }
    )

    fig.update_layout(scene={"aspectmode": "manual", "aspectratio": {"x": 20, "y": 20, "z": 20}})
    # fig.update_layout(autosize=False, width=600, height=600)

    # Remove marigns
    fig.update_layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})

    return fig


def plot_conic_view_polytope_animation(result: dict, params: Config, path="") -> None:
    sub_positions_sen, _, sub_positions_sen_node = full_subject_traj_time(
        result["x_full"], params, False
    )
    fig = go.Figure()
    for i in range(500):
        fig.add_trace(
            go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", line={"color": "blue", "width": 2})
        )

    # Create a cone plot
    A = np.diag(
        [1 / np.tan(np.pi / params.vp.alpha_y), 1 / np.tan(np.pi / params.vp.alpha_x)]
    )  # Conic Matrix

    # Meshgrid
    if params.vp.tracking:
        x = np.linspace(-6, 6, 20)
        y = np.linspace(-6, 6, 20)
    else:
        x = np.linspace(-80, 80, 20)
        y = np.linspace(-80, 80, 20)

    X, Y = np.meshgrid(x, y)

    # Add cone surface using helper function
    x_range = (x[0], x[-1])
    y_range = (y[0], y[-1])
    add_cone_surface(fig, A, params.vp.norm, x_range, y_range, opacity=0.25, n_points=20)
    frames = []

    if params.vp.tracking:
        x_vals = 12 * np.ones_like(np.array(sub_positions_sen[0])[:, 0])
        y_vals = 12 * np.ones_like(np.array(sub_positions_sen[0])[:, 0])
    else:
        x_vals = 110 * np.ones_like(np.array(sub_positions_sen[0])[:, 0])
        y_vals = 110 * np.ones_like(np.array(sub_positions_sen[0])[:, 0])

    # Add cone projections using helper function
    x_range = (x[0], x[-1])
    y_range = (y[0], y[-1])
    add_cone_projections(fig, A, params.vp.norm, x_vals, y_vals, x_range, y_range)

    for i in range(0, len(sub_positions_sen[0]), 4):
        frame = go.Frame(name=str(i))
        data = []
        sub_idx = 0

        for sub_traj in sub_positions_sen:
            sub_traj = np.array(sub_traj)
            sub_traj_nodal = np.array(sub_positions_sen_node[sub_idx])

            if params.vp.tracking:
                x_vals = 12 * np.ones_like(sub_traj[: i + 1, 0])
                y_vals = 12 * np.ones_like(sub_traj[: i + 1, 0])
            else:
                x_vals = 110 * np.ones_like(sub_traj[: i + 1, 0])
                y_vals = 110 * np.ones_like(sub_traj[: i + 1, 0])

            data.append(
                go.Scatter3d(
                    x=sub_traj[: i + 1, 0],
                    y=y_vals,
                    z=sub_traj[: i + 1, 2],
                    mode="lines",
                    showlegend=False,
                    line={"color": "grey", "width": 4},
                )
            )
            data.append(
                go.Scatter3d(
                    x=x_vals,
                    y=sub_traj[: i + 1, 1],
                    z=sub_traj[: i + 1, 2],
                    mode="lines",
                    showlegend=False,
                    line={"color": "grey", "width": 4},
                )
            )

            # Add subject position to data
            sub_traj = np.array(sub_traj)
            data.append(
                go.Scatter3d(
                    x=sub_traj[: i + 1, 0],
                    y=sub_traj[: i + 1, 1],
                    z=sub_traj[: i + 1, 2],
                    mode="lines",
                    line={"color": "darkblue", "width": 3},
                    showlegend=False,
                )
            )

            # Add in node when loop has reached point where node is present
            scaled_index = int((i // (sub_traj.shape[0] / sub_traj_nodal.shape[0])) + 1)
            sub_traj_nodal[:scaled_index]

            sub_idx += 1

        # Polytope connection topology: node -> [connected nodes]
        polytope_connections = {
            0: [16, 8, 12],
            1: [17, 9, 12],
            2: [16, 13, 10],
            3: [17, 11, 13],
            4: [18, 14, 8],
            5: [19, 9, 14],
            6: [18, 15, 10],
            7: [19, 11, 15],
            8: [0, 4, 12],
            9: [1, 5, 11],
            10: [8, 2, 6],
            11: [3, 7, 9],
            12: [0, 1, 14],
            13: [2, 3, 15],
            14: [4, 5, 12],
            15: [13, 6, 7],
            16: [0, 2, 17],
            17: [1, 3, 16],
            18: [4, 6, 19],
            19: [5, 7, 18],
        }

        # Connect polytope vertices using helper function and topology dictionary
        data.extend(
            [
                create_connection_line_3d(sub_positions_sen, i, node_a, node_b)
                for node_a, connections in polytope_connections.items()
                for node_b in connections
            ]
        )

        frame.data = data
        frames.append(frame)

    fig.frames = frames

    # Add animation controls using modular component
    add_animation_controls(fig, slider_x=0.15, slider_y=0.32, play_speed=50, frame_speed=500)

    # Set camera position
    fig.update_layout(
        scene_camera={
            "up": {"x": 0, "y": 0, "z": 10},
            "center": {"x": -2, "y": 0, "z": -3},
            "eye": {"x": -28, "y": -22, "z": 15},
        }
    )

    # Set axis labels
    fig.update_layout(
        scene={"xaxis_title": "x (m)", "yaxis_title": "y (m)", "zaxis_title": "z (m)"}
    )

    fig.update_layout(template="plotly_dark")

    # Make only the grid lines thicker in the template
    fig.update_layout(
        scene={
            "xaxis": {"showgrid": True, "gridwidth": 5},
            "yaxis": {"showgrid": True, "gridwidth": 5},
            "zaxis": {"showgrid": True, "gridwidth": 5},
        }
    )

    fig.update_layout(scene={"aspectmode": "manual", "aspectratio": {"x": 20, "y": 20, "z": 20}})
    # fig.update_layout(autosize=False, width=600, height=600)

    # Remove marigns
    fig.update_layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})

    return fig


def plot_animation(
    result: dict,
    params: Config,
    path="",
) -> None:
    result["t_final"]
    # Make title say quadrotor simulation and insert the variable tof into the title
    # title = 'Quadrotor Simulation: Time of Flight = ' + str(tof) + 's'
    drone_positions = result.trajectory["position"]
    drone_velocities = result.trajectory["velocity"]
    drone_attitudes = result.trajectory.get("attitude", None)
    if "moving_subject" in result or "init_poses" in result:
        subs_positions, _, _, _ = full_subject_traj_time(result, params)

    step = 2
    indices = np.array(
        [*list(range(drone_positions.shape[0] - 1)[::step]), drone_positions.shape[0] - 1]
    )

    fig = go.Figure(
        go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", line={"color": "gray", "width": 2})
    )
    for i in range(100):
        fig.add_trace(
            go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", line={"color": "red", "width": 2})
        )

    frames = []
    i = 0
    # Generate a color for each keypoint
    if "init_poses" in result or "moving_subject" in result:
        color_kp = generate_subject_colors(result)

    # Draw drone attitudes as axes
    for i in range(0, len(indices) - 1, step):
        att = drone_attitudes[indices[i]]
        frame = go.Frame(name=str(i))

        subs_pose = []

        if "moving_subject" in result or "init_poses" in result:
            for sub_positions in subs_positions:
                subs_pose.append(sub_positions[indices[i]])

        # Convert quaternion to rotation matrix
        rotation_matrix = qdcm(att)

        # Extract axes from rotation matrix
        axes = 20 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        rotated_axes = np.dot(rotation_matrix, axes).T

        # Meshgrid
        if "moving_subject" in result:
            x = np.linspace(-5, 5, 20)
            y = np.linspace(-5, 5, 20)
            z = np.linspace(-5, 5, 20)
        elif "covariance" in result:
            x = np.linspace(-2000, 2000, 20)
            y = np.linspace(-2000, 2000, 20)
            z = np.linspace(-2000, 2000, 20)
        else:
            x = np.linspace(-30, 30, 20)
            y = np.linspace(-30, 30, 20)
            z = np.linspace(-30, 30, 20)

        X, Y = np.meshgrid(x, y)

        data = []

        # Define the condition for the second order cone
        if "init_poses" in result or "moving_subject" in result:
            if "alpha_x" in result and "alpha_y" in result:
                A = np.diag(
                    [1 / np.tan(np.pi / result["alpha_y"]), 1 / np.tan(np.pi / result["alpha_x"])]
                )  # Conic Matrix
            else:
                raise ValueError("`alpha_x` and `alpha_y` not found in result dictionary.")
            if "norm_type" in result:
                z = []
                for x_val in x:
                    for y_val in y:
                        if result["norm_type"] == "inf":
                            z.append(
                                np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord=np.inf)
                            )
                        else:
                            z.append(
                                np.linalg.norm(
                                    A @ np.array([x_val, y_val]), axis=0, ord=result["norm_type"]
                                )
                            )
                Z = np.array(z).reshape(20, 20)
            else:
                raise ValueError("`norm_type` not found in result dictionary.")

            # Transform X,Y, and Z from the Sensor frame to the Body frame using R_sb
            if "R_sb" in result:
                R_sb = result["R_sb"]
            else:
                raise ValueError("`R_sb` not found in result dictionary.")
            X, Y, Z = R_sb.T @ np.array([X.flatten(), Y.flatten(), Z.flatten()])
            # Transform X,Y, and Z from the Body frame to the Inertial frame
            R_bi = qdcm(drone_attitudes[indices[i]])
            X, Y, Z = R_bi @ np.array([X, Y, Z])
            # Shift the meshgrid to the drone position
            X += drone_positions[indices[i], 0]
            Y += drone_positions[indices[i], 1]
            Z += drone_positions[indices[i], 2]

            # Make X, Y, Z back into a meshgrid
            X = X.reshape(20, 20)
            Y = Y.reshape(20, 20)
            Z = Z.reshape(20, 20)

            data.append(
                go.Surface(
                    x=X, y=Y, z=Z, opacity=0.5, showscale=False, showlegend=True, name="Viewcone"
                )
            )

        colors = ["#FF0000", "#00FF00", "#0000FF"]
        labels = ["X", "Y", "Z"]

        for k in range(3):
            if k < 3:
                axis = rotated_axes[k]
            color = colors[k]
            labels[k]

            data.append(
                go.Scatter3d(
                    x=[drone_positions[indices[i], 0], drone_positions[indices[i], 0] + axis[0]],
                    y=[drone_positions[indices[i], 1], drone_positions[indices[i], 1] + axis[1]],
                    z=[drone_positions[indices[i], 2], drone_positions[indices[i], 2] + axis[2]],
                    mode="lines+text",
                    line={"color": color, "width": 4},
                    showlegend=False,
                )
            )
        # Add subject position to data
        j = 0
        for sub_pose in subs_pose:
            # Use color iter to change the color of the subject in rgb
            data.append(
                go.Scatter3d(
                    x=[sub_pose[0]],
                    y=[sub_pose[1]],
                    z=[sub_pose[2]],
                    mode="markers",
                    marker={"size": 10, "color": color_kp[j]},
                    showlegend=False,
                    name="Subject",
                )
            )
            # if params.vp.n_subs != 1:
            j += 1

        # Add velocity-colored trajectory using helper function
        data.append(
            create_velocity_trajectory_trace(
                drone_positions, drone_velocities, indices[i], name="Nonlinear Propagation"
            )
        )

        # Make the subject draw a line as it moves
        if "moving_subject" in result:
            if result["moving_subject"]:
                for sub_positions in subs_positions:
                    data.append(
                        go.Scatter3d(
                            x=sub_positions[: indices[i] + 1, 0],
                            y=sub_positions[: indices[i] + 1, 1],
                            z=sub_positions[: indices[i] + 1, 2],
                            mode="lines",
                            line={"color": "red", "width": 10},
                            name="Subject Position",
                        )
                    )

                    sub_position = sub_positions[indices[i]]

                    # Add range spheres using helper function
                    if "min_range" in result and "max_range" in result:
                        data.extend(
                            create_range_sphere_traces(
                                sub_position, result["min_range"], result["max_range"]
                            )
                        )
                    else:
                        raise ValueError(
                            "`min_range` and `max_range` not found in result dictionary."
                        )

        frame.data = data
        frames.append(frame)

    fig.frames = frames

    # Add obstacles using modular component
    if "obstacles_centers" in result:
        add_obstacles(
            fig,
            result["obstacles_centers"],
            result["obstacles_axes"],
            result["obstacles_radii"],
            opacity=0.5,
        )

    # Add gate vertices
    if "vertices" in result:
        for vertices in result["vertices"]:
            fig.add_trace(create_closed_polygon_3d(vertices))

    # Add ground plane using modular component
    add_ground_plane(fig, size=200, z_level=0, opacity=0.3)

    # Add animation controls using modular component
    add_animation_controls(
        fig,
        slider_x=0.15,
        slider_y=0.32,
        play_speed=50,
        frame_speed=500,
        button_x=0.22,
        button_y=0.37,
    )

    fig.update_layout(template="plotly_dark")  # , title=title)

    fig.update_layout(scene={"aspectmode": "manual", "aspectratio": {"x": 10, "y": 10, "z": 10}})

    # Check if covariance exists
    if "covariance" in result:
        fig.update_layout(
            scene={
                "xaxis": {"range": [0, 4000]},
                "yaxis": {"range": [0, 4000]},
                "zaxis": {"range": [-1000, 3000]},
            }
        )
    else:
        fig.update_layout(
            scene={
                "xaxis": {"range": [-200, 200]},
                "yaxis": {"range": [-200, 200]},
                "zaxis": {"range": [-200, 200]},
            }
        )

    # Overlay the title onto the plot
    fig.update_layout(title_y=0.95, title_x=0.5)

    # Show the legend overlayed on the plot
    fig.update_layout(legend={"yanchor": "top", "y": 0.9, "xanchor": "left", "x": 0.75})

    # fig.update_layout(height=450, width = 800)

    # Remove the black border around the fig
    fig.update_layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})

    # Rmeove the background from the legend
    fig.update_layout(legend={"bgcolor": "rgba(0,0,0,0)"})

    fig.update_xaxes(dtick=1.0, showline=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, showline=False, dtick=1.0)

    return fig


def plot_brachistochrone_position(result: OptimizationResults, params=None):
    # Plot the position of the brachistochrone problem
    fig = go.Figure()

    position = result.trajectory["position"]
    x = position[:, 0]
    y = position[:, 1]

    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", line={"color": "blue", "width": 2}, name="Position")
    )
    fig.add_trace(
        go.Scatter(
            x=[x[0]], y=[y[0]], mode="markers", marker={"color": "green", "size": 10}, name="Start"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x[-1]], y=[y[-1]], mode="markers", marker={"color": "red", "size": 10}, name="End"
        )
    )

    fig.update_layout(title="Brachistochrone Position", title_x=0.5, template="plotly_dark")
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig


def plot_brachistochrone_velocity(results: OptimizationResults, params=None):
    # Plot the velocity of the brachistochrone problem
    fig = go.Figure()

    tof = results.t_final
    t_full = results.t_full

    v = results.trajectory["velocity"].squeeze()  # scalar velocity

    fig.add_trace(
        go.Scatter(x=t_full, y=v, mode="lines", line={"color": "blue", "width": 2}, name="Velocity")
    )

    fig.update_layout(
        title=f"Brachistochrone Velocity: {tof} seconds", title_x=0.5, template="plotly_dark"
    )
    return fig


def plot_scp_animation(result: dict, params=None, path=""):
    tof = result["t_final"]
    title = f"SCP Simulation: {tof} seconds"
    drone_positions = result.trajectory["position"]
    drone_attitudes = result.trajectory.get("attitude", None)
    result.trajectory.get("force", None)
    scp_traj_interp(result["x_history"], params)
    scp_ctcs_trajs = result["x_history"]
    scp_multi_shoot = result["discretization_history"]
    # obstacles = result_ctcs["obstacles"]
    # gates = result_ctcs["gates"]
    if "moving_subject" in result or "init_poses" in result:
        subs_positions, _, _, _ = full_subject_traj_time(result, params)
    fig = go.Figure(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="lines+markers",
            line={"color": "gray", "width": 2},
            name="SCP Iterations",
        )
    )
    for j in range(200):
        fig.add_trace(
            go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", line={"color": "gray", "width": 2})
        )

    # fig.update_layout(height=1000)

    fig.add_trace(
        go.Scatter3d(
            x=drone_positions[:, 0],
            y=drone_positions[:, 1],
            z=drone_positions[:, 2],
            mode="lines",
            line={"color": "green", "width": 5},
            name="Nonlinear Propagation",
        )
    )

    fig.update_layout(template="plotly_dark", title=title)

    fig.update_layout(scene={"aspectmode": "manual", "aspectratio": {"x": 10, "y": 10, "z": 10}})

    # Extract the number of states and controls from the parameters
    n_x = params.sim.n_states
    n_u = params.sim.n_controls

    # Define indices for slicing the augmented state vector
    i1 = n_x
    i2 = i1 + n_x * n_x
    i3 = i2 + n_x * n_u
    i4 = i3 + n_x * n_u

    # Plot the attitudes of the SCP Trajs
    frames = []
    traj_iter = 0

    for scp_traj in scp_ctcs_trajs:
        drone_positions = scp_traj[:, 0:3]
        drone_attitudes = scp_traj[:, 6:10]
        frame = go.Frame(name=str(traj_iter))
        data = []
        # Plot the multiple shooting trajectories
        pos_traj = []
        if traj_iter < len(scp_multi_shoot):
            for i_multi in range(scp_multi_shoot[traj_iter].shape[1]):
                pos_traj.append(scp_multi_shoot[traj_iter][:, i_multi].reshape(-1, i4)[:, 0:3])
            pos_traj = np.array(pos_traj)

            for j in range(pos_traj.shape[1]):
                if j == 0:
                    data.append(
                        go.Scatter3d(
                            x=pos_traj[:, j, 0],
                            y=pos_traj[:, j, 1],
                            z=pos_traj[:, j, 2],
                            mode="lines",
                            legendgroup="Multishot Trajectory",
                            name="Multishot Trajectory " + str(traj_iter),
                            showlegend=True,
                            line={"color": "blue", "width": 5},
                        )
                    )
                else:
                    data.append(
                        go.Scatter3d(
                            x=pos_traj[:, j, 0],
                            y=pos_traj[:, j, 1],
                            z=pos_traj[:, j, 2],
                            mode="lines",
                            legendgroup="Multishot Trajectory",
                            showlegend=False,
                            line={"color": "blue", "width": 5},
                        )
                    )

        for i in range(drone_attitudes.shape[0]):
            att = drone_attitudes[i]

            # Convert quaternion to rotation matrix
            rotation_matrix = qdcm(att)

            # Extract axes from rotation matrix
            axes = 2 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            rotated_axes = np.dot(rotation_matrix, axes.T).T

            colors = ["#FF0000", "#00FF00", "#0000FF"]

            for k in range(3):
                axis = rotated_axes[k]
                color = colors[k]

                data.append(
                    go.Scatter3d(
                        x=[scp_traj[i, 0], scp_traj[i, 0] + axis[0]],
                        y=[scp_traj[i, 1], scp_traj[i, 1] + axis[1]],
                        z=[scp_traj[i, 2], scp_traj[i, 2] + axis[2]],
                        mode="lines+text",
                        line={"color": color, "width": 4},
                        showlegend=False,
                    )
                )
        traj_iter += 1
        frame.data = data
        frames.append(frame)
    fig.frames = frames

    i = 1
    # Add obstacles using helper function (extract .value from centers if needed)
    if "obstacles_centers" in result:
        centers = [c.value if hasattr(c, "value") else c for c in result["obstacles_centers"]]
        add_obstacles(
            fig,
            centers,
            result["obstacles_axes"],
            result["obstacles_radii"],
            opacity=0.5,
        )

    if "vertices" in result:
        for vertices in result["vertices"]:
            # Plot a line through the vertices of the gate
            fig.add_trace(create_closed_polygon_3d(vertices))

    # Add the subject positions
    if "n_subs" in result and result["n_subs"] != 0:
        if "moving_subject" in result:
            if result["moving_subject"]:
                for sub_positions in subs_positions:
                    fig.add_trace(
                        go.Scatter3d(
                            x=sub_positions[:, 0],
                            y=sub_positions[:, 1],
                            z=sub_positions[:, 2],
                            mode="lines",
                            line={"color": "red", "width": 5},
                            showlegend=False,
                        )
                    )
        else:
            # Plot the subject positions as points
            for sub_positions in subs_positions:
                fig.add_trace(
                    go.Scatter3d(
                        x=sub_positions[:, 0],
                        y=sub_positions[:, 1],
                        z=sub_positions[:, 2],
                        mode="markers",
                        marker={"size": 10, "color": "red"},
                        showlegend=False,
                    )
                )

    # Add ground plane using helper function
    add_ground_plane(fig, size=2000, z_level=0, opacity=0.3)
    fig.update_layout(scene={"aspectmode": "manual", "aspectratio": {"x": 10, "y": 10, "z": 10}})

    # Add animation controls using helper function
    add_animation_controls(fig, slider_x=0.15, slider_y=0.32, play_speed=50, frame_speed=0)

    fig.update_layout(scene={"aspectmode": "manual", "aspectratio": {"x": 10, "y": 10, "z": 10}})

    # Overlay the title onto the plot
    fig.update_layout(title_y=0.95, title_x=0.5)

    # Show the legend overlayed on the plot
    fig.update_layout(legend={"yanchor": "top", "y": 0.9, "xanchor": "left", "x": 0.75})

    # fig.update_layout(height=450, width = 800)

    # Remove the black border around the fig
    fig.update_layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})

    # Rmeove the background from the legend
    fig.update_layout(legend={"bgcolor": "rgba(0,0,0,0)"})

    fig.update_xaxes(dtick=1.0, showline=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, showline=False, dtick=1.0)

    # Rotate the camera view to the left
    if "moving_subject" not in result:
        fig.update_layout(
            scene_camera={
                "up": {"x": 0, "y": 0, "z": 90},
                "center": {"x": 1, "y": 0.3, "z": 1},
                "eye": {"x": -1, "y": 2, "z": 1},
            }
        )

    return fig


def plot_xy_xz_yz(result: dict, params: Config):
    position = result.trajectory["position"]
    result["t_full"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("XY Plane", "XZ Plane", "YZ Plane"),
        specs=[[{}, {}], [{}, None]],
    )

    # Add trajectory traces
    fig.add_trace(
        go.Scatter(
            x=position[:, 0],
            y=position[:, 1],
            mode="lines",
            line={"color": "blue", "width": 2},
            name="Trajectory XY Plane",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=position[:, 0],
            y=position[:, 2],
            mode="lines",
            line={"color": "blue", "width": 2},
            name="Trajectory XZ Plane",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=position[:, 1],
            y=position[:, 2],
            mode="lines",
            line={"color": "blue", "width": 2},
            name="Trajectory YZ Plane",
        ),
        row=2,
        col=1,
    )

    # Set axis titles
    fig.update_xaxes(title_text="X (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y (m)", row=1, col=1)
    fig.update_xaxes(title_text="X (m)", row=1, col=2)
    fig.update_yaxes(title_text="Z (m)", row=1, col=2)
    fig.update_xaxes(title_text="Y (m)", row=2, col=1)
    fig.update_yaxes(title_text="Z (m)", row=2, col=1)

    # Set equal aspect ratio for each subplot
    fig.update_layout(
        title="Trajectory in XY, XZ, and YZ Planes",
        template="plotly_dark",
        xaxis={"scaleanchor": "y"},  # row=1, col=1
        xaxis2={"scaleanchor": "y2"},  # row=1, col=2
        xaxis3={"scaleanchor": "y3"},  # row=2, col=1
    )

    return fig


def plot_control_norm(results: OptimizationResults, params: Config):
    # Plot the control norm over time
    fig = go.Figure()

    u_full = results.trajectory["thrust"]
    t_full = results.t_full

    # Compute the norm of the control vector
    u_norm = np.linalg.norm(u_full, axis=1)

    rho_min = results.plotting_data["rho_min"]
    rho_max = results.plotting_data["rho_max"]

    fig.add_trace(
        go.Scatter(
            x=t_full,
            y=u_norm,
            mode="lines",
            line={"color": "blue", "width": 2},
            name="Control Norm",
        )
    )
    fig.add_hline(y=rho_min, line={"color": "red", "width": 2, "dash": "dash"}, name="Min Thrust")
    fig.add_hline(y=rho_max, line={"color": "red", "width": 2, "dash": "dash"}, name="Max Thrust")

    title = f"Control Norm: {results.t_final} seconds"
    fig.update_layout(
        title=title, xaxis_title="Time (s)", yaxis_title="Control Norm", template="plotly_dark"
    )
    return fig


def scp_traj_interp(scp_trajs, params: Config):
    scp_prop_trajs = []
    for traj in scp_trajs:
        states = []
        for k in range(params.scp.n):
            traj_temp = np.repeat(
                np.expand_dims(traj[k], axis=1), params.prp.inter_sample - 1, axis=1
            )
            for i in range(1, params.prp.inter_sample - 1):
                states.append(traj_temp[:, i])
        scp_prop_trajs.append(np.array(states))
    return scp_prop_trajs


def plot_animation_double_integrator(
    result: dict,
    params: Config,
    path="",
) -> None:
    result["t_final"]
    # Make title say quadrotor simulation and insert the variable tof into the title
    # title = 'Quadrotor Simulation: Time of Flight = ' + str(tof) + 's'
    drone_positions = result.trajectory["position"]
    drone_velocities = result.trajectory["velocity"]
    if "moving_subject" in result or "init_poses" in result:
        subs_positions, _, _, _ = full_subject_traj_time(result, params)

    step = 2
    indices = np.array(
        [*list(range(drone_positions.shape[0] - 1)[::step]), drone_positions.shape[0] - 1]
    )

    fig = go.Figure(
        go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", line={"color": "gray", "width": 2})
    )
    for i in range(100):
        fig.add_trace(
            go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", line={"color": "red", "width": 2})
        )

    frames = []
    i = 0
    # Generate a color for each keypoint
    if "init_poses" in result or "moving_subject" in result:
        color_kp = generate_subject_colors(result)

    # Draw drone attitudes as axes
    for i in range(0, len(indices) - 1, step):
        frame = go.Frame(name=str(i))

        subs_pose = []

        if "moving_subject" in result or "init_poses" in result:
            for sub_positions in subs_positions:
                subs_pose.append(sub_positions[indices[i]])

        # Meshgrid
        if "moving_subject" in result:
            x = np.linspace(-5, 5, 20)
            y = np.linspace(-5, 5, 20)
        elif "covariance" in result:
            x = np.linspace(-2000, 2000, 20)
            y = np.linspace(-2000, 2000, 20)
        else:
            x = np.linspace(-30, 30, 20)
            y = np.linspace(-30, 30, 20)

        X, Y = np.meshgrid(x, y)

        data = []

        colors = ["#FF0000", "#00FF00", "#0000FF"]
        labels = ["X", "Y", "Z"]

        for k in range(3):
            color = colors[k]
            labels[k]

            data.append(
                go.Scatter3d(
                    x=[drone_positions[indices[i], 0], drone_positions[indices[i], 0]],
                    y=[drone_positions[indices[i], 1], drone_positions[indices[i], 1]],
                    z=[drone_positions[indices[i], 2], drone_positions[indices[i], 2]],
                    mode="lines+text",
                    line={"color": color, "width": 4},
                    showlegend=False,
                )
            )
        # Add subject position to data
        j = 0
        for sub_pose in subs_pose:
            # Use color iter to change the color of the subject in rgb
            data.append(
                go.Scatter3d(
                    x=[sub_pose[0]],
                    y=[sub_pose[1]],
                    z=[sub_pose[2]],
                    mode="markers",
                    marker={"size": 10, "color": color_kp[j]},
                    showlegend=False,
                    name="Subject",
                )
            )
            # if params.vp.n_subs != 1:
            j += 1

        # Add velocity-colored trajectory using helper function
        data.append(
            create_velocity_trajectory_trace(
                drone_positions, drone_velocities, indices[i], name="Nonlinear Propagation"
            )
        )

        # Make the subject draw a line as it moves
        if "moving_subject" in result:
            if result["moving_subject"]:
                for sub_positions in subs_positions:
                    data.append(
                        go.Scatter3d(
                            x=sub_positions[: indices[i] + 1, 0],
                            y=sub_positions[: indices[i] + 1, 1],
                            z=sub_positions[: indices[i] + 1, 2],
                            mode="lines",
                            line={"color": "red", "width": 10},
                            name="Subject Position",
                        )
                    )

                    sub_position = sub_positions[indices[i]]

                    # Add range spheres using helper function
                    if "min_range" in result and "max_range" in result:
                        data.extend(
                            create_range_sphere_traces(
                                sub_position, result["min_range"], result["max_range"]
                            )
                        )
                    else:
                        raise ValueError(
                            "`min_range` and `max_range` not found in result dictionary."
                        )

        frame.data = data
        frames.append(frame)

    fig.frames = frames

    # Add obstacles using modular component
    if "obstacles_centers" in result:
        add_obstacles(
            fig,
            result["obstacles_centers"],
            result["obstacles_axes"],
            result["obstacles_radii"],
            opacity=0.5,
        )

    # Add gate vertices
    if "vertices" in result:
        for vertices in result["vertices"]:
            fig.add_trace(create_closed_polygon_3d(vertices))

    # Add ground plane using modular component
    add_ground_plane(fig, size=200, z_level=0, opacity=0.3)

    # Add animation controls using modular component
    add_animation_controls(
        fig,
        slider_x=0.15,
        slider_y=0.32,
        play_speed=50,
        frame_speed=500,
        button_x=0.22,
        button_y=0.37,
    )

    fig.update_layout(template="plotly_dark")  # , title=title)

    fig.update_layout(scene={"aspectmode": "manual", "aspectratio": {"x": 10, "y": 10, "z": 10}})

    # Check if covariance exists
    if "covariance" in result:
        fig.update_layout(
            scene={
                "xaxis": {"range": [0, 4000]},
                "yaxis": {"range": [0, 4000]},
                "zaxis": {"range": [-1000, 3000]},
            }
        )
    else:
        fig.update_layout(
            scene={
                "xaxis": {"range": [-200, 200]},
                "yaxis": {"range": [-200, 200]},
                "zaxis": {"range": [-200, 200]},
            }
        )

    # Overlay the title onto the plot
    fig.update_layout(title_y=0.95, title_x=0.5)

    # Show the legend overlayed on the plot
    fig.update_layout(legend={"yanchor": "top", "y": 0.9, "xanchor": "left", "x": 0.75})

    # fig.update_layout(height=450, width = 800)

    # Remove the black border around the fig
    fig.update_layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})

    # Rmeove the background from the legend
    fig.update_layout(legend={"bgcolor": "rgba(0,0,0,0)"})

    fig.update_xaxes(dtick=1.0, showline=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, showline=False, dtick=1.0)

    return fig


def plot_animation_3DoF_rocket(
    result: dict,
    params: Config,
    path="",
) -> None:
    result["t_final"]
    # Make title say quadrotor simulation and insert the variable tof into the title
    # title = 'Quadrotor Simulation: Time of Flight = ' + str(tof) + 's'
    drone_positions = result.trajectory["position"]
    drone_velocities = result.trajectory["velocity"]
    drone_forces = 0.01 * result.trajectory["force"]

    step = 2
    indices = np.array(
        [*list(range(drone_positions.shape[0] - 1)[::step]), drone_positions.shape[0] - 1]
    )

    fig = go.Figure(
        go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", line={"color": "gray", "width": 2})
    )
    for i in range(100):
        fig.add_trace(
            go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", line={"color": "red", "width": 2})
        )

    frames = []
    i = 0

    # Draw drone attitudes as axes
    for i in range(0, len(indices) - 1, step):
        frame = go.Frame(name=str(i))

        # Extract axes from rotation matrix
        20 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        data = []

        colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFFFF"]
        labels = ["X", "Y", "Z", "Force"]

        for k in range(4):
            color = colors[k]
            labels[k]

            if labels[k] != "Force":
                data.append(
                    go.Scatter3d(
                        x=[drone_positions[indices[i], 0], drone_positions[indices[i], 0]],
                        y=[drone_positions[indices[i], 1], drone_positions[indices[i], 1]],
                        z=[drone_positions[indices[i], 2], drone_positions[indices[i], 2]],
                        mode="lines+text",
                        line={"color": color, "width": 4},
                        showlegend=False,
                    )
                )
            else:
                data.append(
                    go.Scatter3d(
                        x=[
                            drone_positions[indices[i], 0],
                            drone_positions[indices[i], 0] - drone_forces[indices[i], 0],
                        ],
                        y=[
                            drone_positions[indices[i], 1],
                            drone_positions[indices[i], 1] - drone_forces[indices[i], 1],
                        ],
                        z=[
                            drone_positions[indices[i], 2],
                            drone_positions[indices[i], 2] - drone_forces[indices[i], 2],
                        ],
                        mode="lines+text",
                        line={"color": color, "width": 10},
                        showlegend=False,
                    )
                )

        # Add velocity-colored trajectory using helper function
        data.append(
            create_velocity_trajectory_trace(
                drone_positions, drone_velocities, indices[i], name="Nonlinear Propagation"
            )
        )

        frame.data = data
        frames.append(frame)

    fig.frames = frames

    # Add ground plane using helper function
    add_ground_plane(fig, size=200, z_level=0, opacity=0.3)

    # Add animation controls using helper function
    add_animation_controls(
        fig,
        slider_x=0.15,
        slider_y=0.32,
        play_speed=50,
        frame_speed=500,
        button_x=0.22,
        button_y=0.37,
    )

    fig.update_layout(template="plotly_dark")  # , title=title)

    fig.update_layout(scene={"aspectmode": "manual", "aspectratio": {"x": 10, "y": 10, "z": 10}})

    # Check if covariance exists
    fig.update_layout(
        scene={
            "xaxis": {"range": [-3000, 3000]},
            "yaxis": {"range": [-3000, 3000]},
            "zaxis": {"range": [-200, 2000]},
        }
    )

    # Overlay the title onto the plot
    fig.update_layout(title_y=0.95, title_x=0.5)

    # Show the legend overlayed on the plot
    fig.update_layout(legend={"yanchor": "top", "y": 0.9, "xanchor": "left", "x": 0.75})

    # fig.update_layout(height=450, width = 800)

    # Remove the black border around the fig
    fig.update_layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})

    # Rmeove the background from the legend
    fig.update_layout(legend={"bgcolor": "rgba(0,0,0,0)"})

    fig.update_xaxes(dtick=1.0, showline=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, showline=False, dtick=1.0)

    return fig


def plot_animation_pyqtgraph(result, params, step=2):
    if not PYQTPHOT_AVAILABLE:
        raise ImportError(
            "pyqtgraph is required for this function but not installed. "
            "Install it with: pip install openscvx[gui] or pip install pyqtgraph PyQt5"
        )

    import sys

    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
    from scipy.spatial.transform import Rotation as R

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # Main window and layout
    main_widget = QWidget()
    main_layout = QVBoxLayout(main_widget)
    w = gl.GLViewWidget()
    w.setWindowTitle("Quadrotor Simulation (pyqtgraph)")
    w.setGeometry(0, 110, 1280, 720)

    # Extract data
    drone_positions = result.trajectory["position"]
    drone_velocities = result.trajectory["velocity"]
    drone_attitudes = result.trajectory.get("attitude", None)
    velocity_norm = np.linalg.norm(drone_velocities, axis=1)
    n_points = drone_positions.shape[0]
    indices = np.array([*list(range(n_points - 1)[::step]), n_points - 1])

    # Auto-calculate plotting bounds
    pos_min = drone_positions.min(axis=0)
    pos_max = drone_positions.max(axis=0)
    pos_range = pos_max - pos_min

    # Add padding to bounds (20% of range)
    padding = pos_range * 0.2
    bounds_min = pos_min - padding
    bounds_max = pos_max + padding

    # Ensure minimum bounds for small trajectories
    min_bounds_size = 10.0
    for i in range(3):
        if bounds_max[i] - bounds_min[i] < min_bounds_size:
            center = (bounds_max[i] + bounds_min[i]) / 2
            bounds_min[i] = center - min_bounds_size / 2
            bounds_max[i] = center + min_bounds_size / 2

    # Auto-calculate camera distance based on bounds
    max_range = max(bounds_max - bounds_min)
    camera_distance = max_range * 1.5  # 1.5x the maximum range

    # Auto-calculate vehicle axes length based on trajectory size
    axes_length = max(pos_range) * 0.1  # 10% of trajectory range
    axes_length = max(axes_length, 2.0)  # Minimum 2 units
    axes_length = min(axes_length, 20.0)  # Maximum 20 units

    w.setCameraPosition(distance=camera_distance, elevation=20, azimuth=45)
    main_layout.addWidget(w)

    # Controls
    controls_layout = QHBoxLayout()
    play_btn = QPushButton("Play")
    pause_btn = QPushButton("Pause")
    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(0)
    slider.setSingleStep(1)
    controls_layout.addWidget(play_btn)
    controls_layout.addWidget(pause_btn)
    controls_layout.addWidget(QLabel("Time:"))
    controls_layout.addWidget(slider)
    main_layout.addLayout(controls_layout)
    main_widget.show()

    slider.setMaximum(len(indices) - 1)

    cmap = pg.colormap.get("viridis")
    vmin, vmax = velocity_norm.min(), velocity_norm.max()
    normed_vel = (velocity_norm - vmin) / (vmax - vmin + 1e-8)
    colors = cmap.map(normed_vel, mode="float")

    # Thicker trajectory line
    traj_line = gl.GLLinePlotItem(
        pos=np.zeros((0, 3)), color=np.ones((0, 4)), width=5, antialias=True
    )
    w.addItem(traj_line)
    drone_dot = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1, 1, 1, 1), size=10)
    w.addItem(drone_dot)

    axis_items = [gl.GLLinePlotItem(width=3) for _ in range(3)]
    for item in axis_items:
        w.addItem(item)
    axis_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]

    # Add a nice ground grid - auto-scale based on bounds
    grid_size = max(bounds_max[:2] - bounds_min[:2]) * 0.3  # 30% of XY range (reduced from 60%)
    grid_spacing = grid_size / 8  # 8 grid lines (reduced from 10 for tighter spacing)
    grid_lines = []
    for x in np.arange(bounds_min[0] - grid_size / 2, bounds_max[0] + grid_size / 2, grid_spacing):
        pts = np.array(
            [
                [x, bounds_min[1] - grid_size / 2, bounds_min[2]],
                [x, bounds_max[1] + grid_size / 2, bounds_min[2]],
            ]
        )
        line = gl.GLLinePlotItem(pos=pts, color=(0.7, 0.7, 0.7, 0.5), width=1, antialias=True)
        w.addItem(line)
        grid_lines.append(line)
    for y in np.arange(bounds_min[1] - grid_size / 2, bounds_max[1] + grid_size / 2, grid_spacing):
        pts = np.array(
            [
                [bounds_min[0] - grid_size / 2, y, bounds_min[2]],
                [bounds_max[0] + grid_size / 2, y, bounds_min[2]],
            ]
        )
        line = gl.GLLinePlotItem(pos=pts, color=(0.7, 0.7, 0.7, 0.5), width=1, antialias=True)
        w.addItem(line)
        grid_lines.append(line)

    obstacle_items = []
    if "obstacles_centers" in result and "obstacles_radii" in result and "obstacles_axes" in result:
        for center, axes, radius in zip(
            result["obstacles_centers"], result["obstacles_axes"], result["obstacles_radii"]
        ):
            # Create a sphere mesh and transform it to an ellipsoid
            sphere_mesh = gl.MeshData.sphere(rows=20, cols=40)
            verts = sphere_mesh.vertexes()
            # Scale by 1/radius to match the original plot_animation function
            verts = verts * np.array([1 / radius[0], 1 / radius[1], 1 / radius[2]])
            verts = (axes @ verts.T).T  # rotate
            center_val = center.value if hasattr(center, "value") else center
            verts = verts + center_val  # translate
            sphere_mesh.setVertexes(verts)
            obstacle = gl.GLMeshItem(
                meshdata=sphere_mesh,
                smooth=True,
                color=(1, 0, 0, 0.6),
                shader="shaded",
                drawEdges=False,
                glOptions="translucent",
            )
            w.addItem(obstacle)
            obstacle_items.append(obstacle)

    gate_items = []
    if "vertices" in result:
        for vertices in result["vertices"]:
            gate = gl.GLLinePlotItem(
                pos=np.array([*vertices, vertices[0]]), color=(0, 0, 1, 1), width=5, antialias=True
            )
            w.addItem(gate)
            gate_items.append(gate)

    subject_lines = []
    subject_dots = []
    if "init_poses" in result or "moving_subject" in result:
        if "init_poses" in result:
            subs_positions, _, _, _ = full_subject_traj_time(result, params)
        else:
            subs_positions, _, _, _ = full_subject_traj_time(result, params)
        for _sub_traj in subs_positions:
            line = gl.GLLinePlotItem(pos=np.zeros((0, 3)), color=(1, 0, 0, 1), width=3)
            dot = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1, 0, 0, 1), size=10)
            w.addItem(line)
            w.addItem(dot)
            subject_lines.append(line)
            subject_dots.append(dot)

    range_spheres = []
    if "min_range" in result and "max_range" in result and "init_poses" in result:
        n = 20
        u = np.linspace(0, 2 * np.pi, n)
        v = np.linspace(0, np.pi, n)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        for _sub_traj in subs_positions:
            min_sphere = gl.GLMeshItem(
                meshdata=gl.MeshData.sphere(rows=10, cols=20, radius=result["min_range"]),
                color=(1, 0, 0, 0.4),
                smooth=True,
                shader="shaded",
                drawEdges=False,
                glOptions="translucent",
            )
            max_sphere = gl.GLMeshItem(
                meshdata=gl.MeshData.sphere(rows=10, cols=20, radius=result["max_range"]),
                color=(0, 0, 1, 0.4),
                smooth=True,
                shader="shaded",
                drawEdges=False,
                glOptions="translucent",
            )
            w.addItem(min_sphere)
            w.addItem(max_sphere)
            range_spheres.append((min_sphere, max_sphere))

    # Viewcone as a transparent surface
    viewcone_mesh = None
    cone_meshdata = None
    if (
        drone_attitudes is not None
        and "alpha_x" in result
        and "alpha_y" in result
        and "R_sb" in result
    ):
        n_cone = 40
        theta = np.linspace(0, 2 * np.pi, n_cone)
        alpha_x = result["alpha_x"]
        alpha_y = result["alpha_y"]
        A = np.diag([1 / np.tan(np.pi / alpha_x), 1 / np.tan(np.pi / alpha_y)])
        # Make a circle in the sensor frame, scale Z for a larger cone
        cone_length = max(pos_range) * 0.3  # 30% of trajectory range
        cone_length = max(cone_length, 10.0)  # Minimum 10 units
        cone_length = min(cone_length, 50.0)  # Maximum 50 units
        circle = np.stack([np.cos(theta), np.sin(theta)])

        # Use norm_type from results to plot the cone correctly
        if "norm_type" in result:
            if result["norm_type"] == np.inf or result["norm_type"] == "inf":
                z = np.linalg.norm(A @ circle, axis=0, ord=np.inf)
            else:
                z = np.linalg.norm(A @ circle, axis=0, ord=result["norm_type"])
        else:
            # Default to 2-norm if norm_type not specified
            z = np.linalg.norm(A @ circle, axis=0, ord=2)

        X = circle[0] / z
        Y = circle[1] / z
        Z = np.ones_like(X)
        base_points = np.stack([X, Y, Z], axis=1) * cone_length
        apex = np.array([[0, 0, 0]])
        vertices = np.vstack([apex, base_points])
        faces = []
        for i in range(1, n_cone):
            faces.append([0, i, i + 1])
        faces.append([0, n_cone, 1])
        faces = np.array(faces)
        cone_meshdata = gl.MeshData(vertexes=vertices, faces=faces)
        # Draw cone last for correct depth compositing
        viewcone_mesh = gl.GLMeshItem(
            meshdata=cone_meshdata,
            smooth=True,
            color=(1, 1, 0, 0.5),
            shader="shaded",
            drawEdges=False,
            glOptions="additive",
        )
        # Do not add yet, will add after all other objects

    ptr = [0]
    playing = [False]

    def update():
        i = ptr[0]
        if i >= len(indices):
            timer.stop()
            playing[0] = False
            play_btn.setEnabled(True)
            pause_btn.setEnabled(False)
            return
        slider.blockSignals(True)
        slider.setValue(i)
        slider.blockSignals(False)
        idx = indices[: i + 1]
        pos = drone_positions[idx]
        col = colors[idx]
        traj_line.setData(pos=pos, color=col)
        drone_dot.setData(pos=pos[-1:], color=(1, 1, 1, 1))
        if drone_attitudes is not None:
            att = drone_attitudes[indices[i]]
            r = R.from_quat([att[1], att[2], att[3], att[0]])
            rotmat = r.as_matrix()
            axes = axes_length * np.eye(3)
            axes_rot = rotmat @ axes
            for k in range(3):
                axis_pts = np.stack(
                    [drone_positions[indices[i]], drone_positions[indices[i]] + axes_rot[:, k]]
                )
                axis_items[k].setData(pos=axis_pts, color=axis_colors[k])
        if "init_poses" in result or "moving_subject" in result:
            for j, sub_traj in enumerate(subs_positions):
                subject_lines[j].setData(pos=sub_traj[: indices[i] + 1], color=(1, 0, 0, 1))
                subject_dots[j].setData(
                    pos=sub_traj[indices[i] : indices[i] + 1], color=(1, 0, 0, 1)
                )
                if j < len(range_spheres):
                    min_sphere, max_sphere = range_spheres[j]
                    min_sphere.resetTransform()
                    max_sphere.resetTransform()
                    min_sphere.translate(*sub_traj[indices[i]])
                    max_sphere.translate(*sub_traj[indices[i]])
        # Update viewcone mesh
        if viewcone_mesh is not None and drone_attitudes is not None:
            att = drone_attitudes[indices[i]]
            r = R.from_quat([att[1], att[2], att[3], att[0]])
            rotmat = r.as_matrix()
            R_sb = result["R_sb"]
            # Transform cone vertices
            verts = cone_meshdata.vertexes()
            verts_tf = (rotmat @ R_sb.T @ verts.T).T + drone_positions[indices[i]]
            viewcone_mesh.setMeshData(vertexes=verts_tf, faces=cone_meshdata.faces())
            # Remove and re-add cone to ensure it is drawn last
            if viewcone_mesh in w.items:
                w.removeItem(viewcone_mesh)
            w.addItem(viewcone_mesh)
        ptr[0] += 1

    def set_frame(i):
        ptr[0] = i
        update()

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.setInterval(30)

    def play():
        if not playing[0]:
            playing[0] = True
            play_btn.setEnabled(False)
            pause_btn.setEnabled(True)
            timer.start()

    def pause():
        playing[0] = False
        play_btn.setEnabled(True)
        pause_btn.setEnabled(False)
        timer.stop()

    play_btn.clicked.connect(play)
    pause_btn.clicked.connect(pause)
    slider.valueChanged.connect(set_frame)
    pause_btn.setEnabled(False)

    main_widget.setWindowTitle("Quadrotor Simulation (pyqtgraph)")
    main_widget.resize(1280, 800)

    update()  # Draw initial frame

    if not hasattr(QtWidgets.QApplication.instance(), "exec_"):
        app.exec()
    else:
        QtWidgets.QApplication.instance().exec_()


def plot_animation_vispy(result, params, step=2):
    """
    VisPy-based 3D animation of quadrotor simulation.
    Provides the same functionality as plot_animation_pyqtgraph but uses VisPy backend.
    """
    import sys

    import numpy as np
    import vispy
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
    from scipy.spatial.transform import Rotation as R
    from vispy import scene
    from vispy.geometry import create_sphere
    from vispy.scene import visuals

    # Initialize Qt application
    app = QApplication.instance() or QApplication(sys.argv)

    # Main window and layout
    main_widget = QWidget()
    main_layout = QVBoxLayout(main_widget)

    # Create canvas and embed it properly
    canvas = scene.SceneCanvas(keys="interactive", size=(1280, 720))
    view = canvas.central_widget.add_view()
    view.camera = "turntable"
    view.camera.fov = 45
    view.camera.distance = 150
    view.camera.elevation = 20
    view.camera.azimuth = 45

    # Embed VisPy canvas in Qt widget
    canvas_widget = canvas.native
    main_layout.addWidget(canvas_widget)

    # Controls
    controls_layout = QHBoxLayout()
    play_btn = QPushButton("Play")
    pause_btn = QPushButton("Pause")
    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(0)
    slider.setSingleStep(1)
    controls_layout.addWidget(play_btn)
    controls_layout.addWidget(pause_btn)
    controls_layout.addWidget(QLabel("Time:"))
    controls_layout.addWidget(slider)
    main_layout.addLayout(controls_layout)
    main_widget.show()

    # Extract data and convert JAX arrays to NumPy arrays
    drone_positions = np.array(result.trajectory["position"])
    drone_velocities = np.array(result.trajectory["velocity"])
    drone_attitudes = (
        np.array(result.trajectory["attitude"]) if "attitude" in result.trajectory else None
    )
    velocity_norm = np.linalg.norm(drone_velocities, axis=1)
    n_points = drone_positions.shape[0]
    indices = np.array([*list(range(n_points - 1)[::step]), n_points - 1])
    slider.setMaximum(len(indices) - 1)

    # Color mapping for velocity
    vmin, vmax = velocity_norm.min(), velocity_norm.max()
    normed_vel = (velocity_norm - vmin) / (vmax - vmin + 1e-8)
    colors = vispy.color.get_colormap("viridis").map(normed_vel)

    # Create trajectory line - initialize with first point
    traj_line = visuals.Line(pos=drone_positions[:1], color=colors[:1], width=5, connect="strip")
    view.add(traj_line)

    # Create drone position marker
    drone_dot = visuals.Markers(pos=drone_positions[:1], face_color="white", size=10)
    view.add(drone_dot)

    # Create coordinate axes
    axis_lines = []
    axis_colors = ["red", "green", "blue"]
    for color in axis_colors:
        axis_line = visuals.Line(pos=np.zeros((2, 3)), color=color, width=3)
        view.add(axis_line)
        axis_lines.append(axis_line)

    # Create ground grid
    grid_size = 200
    grid_spacing = 20
    grid_lines = []
    for x in range(-grid_size, grid_size + 1, grid_spacing):
        pts = np.array([[x, -grid_size, 0], [x, grid_size, 0]])
        line = visuals.Line(pos=pts, color=(0.7, 0.7, 0.7, 0.5), width=1)
        view.add(line)
        grid_lines.append(line)
    for y in range(-grid_size, grid_size + 1, grid_spacing):
        pts = np.array([[-grid_size, y, 0], [grid_size, y, 0]])
        line = visuals.Line(pos=pts, color=(0.7, 0.7, 0.7, 0.5), width=1)
        view.add(line)
        grid_lines.append(line)

    # Create obstacles
    obstacle_meshes = []
    if "obstacles_centers" in result and "obstacles_radii" in result and "obstacles_axes" in result:
        for center, axes, radius in zip(
            result["obstacles_centers"], result["obstacles_axes"], result["obstacles_radii"]
        ):
            # Convert to NumPy arrays
            center = np.array(center)
            axes = np.array(axes)
            radius = np.array(radius)

            # Create sphere mesh and transform to ellipsoid
            sphere_mesh = create_sphere(radius=1.0, rows=20, cols=40)
            sphere_verts = sphere_mesh.get_vertices()
            sphere_faces = sphere_mesh.get_faces()
            sphere_verts = sphere_verts * radius  # scale to ellipsoid radii
            sphere_verts = (axes @ sphere_verts.T).T  # rotate
            sphere_verts = sphere_verts + center  # translate

            obstacle = visuals.Mesh(
                vertices=sphere_verts, faces=sphere_faces, color=(1, 0, 0, 0.3), shading="smooth"
            )
            view.add(obstacle)
            obstacle_meshes.append(obstacle)

    # Create gates
    gate_lines = []
    if "vertices" in result:
        for vertices in result["vertices"]:
            gate_pts = np.array([*vertices, vertices[0]])
            gate = visuals.Line(pos=gate_pts, color=(0, 0, 1, 1), width=5)
            view.add(gate)
            gate_lines.append(gate)

    # Create subject trajectories
    subject_lines = []
    subject_dots = []
    subs_positions = []
    if "init_poses" in result or "moving_subject" in result:
        if "init_poses" in result:
            subs_positions, _, _, _ = full_subject_traj_time(result, params)
        else:
            subs_positions, _, _, _ = full_subject_traj_time(result, params)

        # Convert subject positions to NumPy arrays
        subs_positions = [np.array(sub_pos) for sub_pos in subs_positions]

        for sub_traj in subs_positions:
            line = visuals.Line(pos=sub_traj[:1], color=(1, 0, 0, 1), width=3)
            dot = visuals.Markers(pos=sub_traj[:1], face_color=(1, 0, 0, 1), size=10)
            view.add(line)
            view.add(dot)
            subject_lines.append(line)
            subject_dots.append(dot)

    # Create range spheres
    range_spheres = []
    if "min_range" in result and "max_range" in result and "init_poses" in result:
        for sub_traj in subs_positions:
            # Min range sphere
            min_mesh = create_sphere(radius=result["min_range"], rows=10, cols=20)
            min_sphere = visuals.Mesh(
                vertices=min_mesh.get_vertices(),
                faces=min_mesh.get_faces(),
                color=(1, 0, 0, 0.2),
                shading="smooth",
            )
            view.add(min_sphere)

            # Max range sphere
            max_mesh = create_sphere(radius=result["max_range"], rows=10, cols=20)
            max_sphere = visuals.Mesh(
                vertices=max_mesh.get_vertices(),
                faces=max_mesh.get_faces(),
                color=(0, 0, 1, 0.2),
                shading="smooth",
            )
            view.add(max_sphere)

            range_spheres.append((min_sphere, max_sphere))

    # Create viewcone
    viewcone_mesh = None
    if (
        drone_attitudes is not None
        and "alpha_x" in result
        and "alpha_y" in result
        and "R_sb" in result
    ):
        n_cone = 40
        theta = np.linspace(0, 2 * np.pi, n_cone)
        alpha_x = result["alpha_x"]
        alpha_y = result["alpha_y"]
        A = np.diag([1 / np.tan(np.pi / alpha_y), 1 / np.tan(np.pi / alpha_x)])

        # Create cone geometry
        cone_length = 30.0
        circle = np.stack([np.cos(theta), np.sin(theta)])
        z = np.linalg.norm(A @ circle, axis=0)
        X = circle[0] / z
        Y = circle[1] / z
        Z = np.ones_like(X)
        base_points = np.stack([X, Y, Z], axis=1) * cone_length
        apex = np.array([[0, 0, 0]])
        vertices = np.vstack([apex, base_points])

        # Create faces for cone
        faces = []
        for i in range(1, n_cone):
            faces.append([0, i, i + 1])
        faces.append([0, n_cone, 1])
        faces = np.array(faces)

        viewcone_mesh = visuals.Mesh(
            vertices=vertices, faces=faces, color=(1, 0, 0, 0.3), shading="smooth"
        )
        view.add(viewcone_mesh)

    # Animation state
    ptr = [0]
    playing = [False]

    def update():
        i = ptr[0]
        if i >= len(indices):
            timer.stop()
            playing[0] = False
            play_btn.setEnabled(True)
            pause_btn.setEnabled(False)
            return

        slider.blockSignals(True)
        slider.setValue(i)
        slider.blockSignals(False)

        idx = indices[: i + 1]
        pos = drone_positions[idx]
        col = colors[idx]

        # Update trajectory
        traj_line.set_data(pos=pos, color=col)
        drone_dot.set_data(pos=pos[-1:], face_color="white")

        # Update coordinate axes
        if drone_attitudes is not None:
            att = drone_attitudes[indices[i]]
            r = R.from_quat([att[1], att[2], att[3], att[0]])
            rotmat = r.as_matrix()
            axes = 20 * np.eye(3)
            axes_rot = rotmat @ axes

            for k in range(3):
                axis_pts = np.stack(
                    [drone_positions[indices[i]], drone_positions[indices[i]] + axes_rot[:, k]]
                )
                axis_lines[k].set_data(pos=axis_pts)

        # Update subject trajectories
        if len(subs_positions) > 0:
            for j, sub_traj in enumerate(subs_positions):
                subject_lines[j].set_data(pos=sub_traj[: indices[i] + 1])
                subject_dots[j].set_data(pos=sub_traj[indices[i] : indices[i] + 1])

                # Update range spheres
                if j < len(range_spheres):
                    min_sphere, max_sphere = range_spheres[j]
                    min_sphere.transform = vispy.visuals.transforms.STTransform(
                        translate=sub_traj[indices[i]]
                    )
                    max_sphere.transform = vispy.visuals.transforms.STTransform(
                        translate=sub_traj[indices[i]]
                    )

        # Update viewcone
        if viewcone_mesh is not None and drone_attitudes is not None:
            att = drone_attitudes[indices[i]]
            r = R.from_quat([att[1], att[2], att[3], att[0]])
            rotmat = r.as_matrix()
            R_sb = np.array(result["R_sb"])

            # Transform cone vertices
            cone_mesh = create_sphere(radius=1.0, rows=10, cols=20)
            verts = cone_mesh.get_vertices() * 30.0  # Scale to cone size
            verts = (rotmat @ R_sb.T @ verts.T).T + drone_positions[indices[i]]
            viewcone_mesh.set_data(vertices=verts, faces=cone_mesh.get_faces())

        ptr[0] += 1
        canvas.update()

    def set_frame(i):
        ptr[0] = i
        update()

    timer = QTimer()
    timer.timeout.connect(update)
    timer.setInterval(30)

    def play():
        if not playing[0]:
            playing[0] = True
            play_btn.setEnabled(False)
            pause_btn.setEnabled(True)
            timer.start()

    def pause():
        playing[0] = False
        play_btn.setEnabled(True)
        pause_btn.setEnabled(False)
        timer.stop()

    play_btn.clicked.connect(play)
    pause_btn.clicked.connect(pause)
    slider.valueChanged.connect(set_frame)
    pause_btn.setEnabled(False)

    main_widget.setWindowTitle("Quadrotor Simulation (VisPy)")
    main_widget.resize(1280, 800)

    # Draw initial frame
    update()

    # Start Qt event loop
    if not hasattr(QApplication.instance(), "exec_"):
        app.exec()
    else:
        QApplication.instance().exec_()


def plot_scp_animation_pyqtgraph(result, params, step=2):
    if not PYQTPHOT_AVAILABLE:
        raise ImportError(
            "pyqtgraph is required for this function but not installed. "
            "Install it with: pip install openscvx[gui] or pip install pyqtgraph PyQt5"
        )

    import sys

    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
    from scipy.spatial.transform import Rotation as R

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # Main window and layout
    main_widget = QWidget()
    main_layout = QVBoxLayout(main_widget)
    w = gl.GLViewWidget()
    w.setWindowTitle("SCP Animation (pyqtgraph)")
    w.setGeometry(0, 110, 1280, 720)

    # Extract data
    tof = result["t_final"]
    drone_positions = result.trajectory["position"]
    result.trajectory["velocity"]
    drone_attitudes = result.trajectory.get("attitude", None)
    scp_traj_interp(result["x_history"], params)
    scp_ctcs_trajs = result["x_history"]
    scp_multi_shoot = result["discretization_history"]

    if "moving_subject" in result or "init_poses" in result:
        subs_positions, _, _, _ = full_subject_traj_time(result, params)

    # Auto-calculate plotting bounds
    pos_min = drone_positions.min(axis=0)
    pos_max = drone_positions.max(axis=0)
    pos_range = pos_max - pos_min

    # Add padding to bounds (20% of range)
    padding = pos_range * 0.2
    bounds_min = pos_min - padding
    bounds_max = pos_max + padding

    # Ensure minimum bounds for small trajectories
    min_bounds_size = 10.0
    for i in range(3):
        if bounds_max[i] - bounds_min[i] < min_bounds_size:
            center = (bounds_max[i] + bounds_min[i]) / 2
            bounds_min[i] = center - min_bounds_size / 2
            bounds_max[i] = center + min_bounds_size / 2

    # Auto-calculate camera distance based on bounds
    max_range = max(bounds_max - bounds_min)
    camera_distance = max_range * 1.5  # 1.5x the maximum range

    w.setCameraPosition(distance=camera_distance, elevation=20, azimuth=45)
    main_layout.addWidget(w)

    # Controls
    controls_layout = QHBoxLayout()
    play_btn = QPushButton("Play")
    pause_btn = QPushButton("Pause")
    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(0)
    slider.setSingleStep(1)
    controls_layout.addWidget(play_btn)
    controls_layout.addWidget(pause_btn)
    controls_layout.addWidget(QLabel("SCP Iteration:"))
    controls_layout.addWidget(slider)
    main_layout.addLayout(controls_layout)
    main_widget.show()

    # Set slider range based on number of SCP iterations
    n_iterations = len(scp_ctcs_trajs)
    slider.setMaximum(n_iterations - 1)

    # Final trajectory (nonlinear propagation)
    final_traj_line = gl.GLLinePlotItem(
        pos=drone_positions, color=(0, 1, 0, 1), width=5, antialias=True
    )
    w.addItem(final_traj_line)

    # Add a nice ground grid - auto-scale based on bounds
    grid_size = max(bounds_max[:2] - bounds_min[:2]) * 0.3  # 30% of XY range
    grid_spacing = grid_size / 8  # 8 grid lines
    grid_lines = []
    for x in np.arange(bounds_min[0] - grid_size / 2, bounds_max[0] + grid_size / 2, grid_spacing):
        pts = np.array(
            [
                [x, bounds_min[1] - grid_size / 2, bounds_min[2]],
                [x, bounds_max[1] + grid_size / 2, bounds_min[2]],
            ]
        )
        line = gl.GLLinePlotItem(pos=pts, color=(0.7, 0.7, 0.7, 0.5), width=1, antialias=True)
        w.addItem(line)
        grid_lines.append(line)
    for y in np.arange(bounds_min[1] - grid_size / 2, bounds_max[1] + grid_size / 2, grid_spacing):
        pts = np.array(
            [
                [bounds_min[0] - grid_size / 2, y, bounds_min[2]],
                [bounds_max[0] + grid_size / 2, y, bounds_min[2]],
            ]
        )
        line = gl.GLLinePlotItem(pos=pts, color=(0.7, 0.7, 0.7, 0.5), width=1, antialias=True)
        w.addItem(line)
        grid_lines.append(line)

    # Create obstacles
    obstacle_items = []
    if "obstacles_centers" in result and "obstacles_radii" in result and "obstacles_axes" in result:
        for center, axes, radius in zip(
            result["obstacles_centers"], result["obstacles_axes"], result["obstacles_radii"]
        ):
            # Create a sphere mesh and transform it to an ellipsoid
            sphere_mesh = gl.MeshData.sphere(rows=20, cols=40)
            verts = sphere_mesh.vertexes()
            # Scale by 1/radius to match the original plot_animation function
            verts = verts * np.array([1 / radius[0], 1 / radius[1], 1 / radius[2]])
            verts = (axes @ verts.T).T  # rotate
            center_val = center.value if hasattr(center, "value") else center
            verts = verts + center_val  # translate
            sphere_mesh.setVertexes(verts)
            obstacle = gl.GLMeshItem(
                meshdata=sphere_mesh,
                smooth=True,
                color=(1, 0, 0, 0.6),
                shader="shaded",
                drawEdges=False,
                glOptions="translucent",
            )
            w.addItem(obstacle)
            obstacle_items.append(obstacle)

    # Create gates
    gate_items = []
    if "vertices" in result:
        for vertices in result["vertices"]:
            gate = gl.GLLinePlotItem(
                pos=np.array([*vertices, vertices[0]]), color=(0, 0, 1, 1), width=5, antialias=True
            )
            w.addItem(gate)
            gate_items.append(gate)

    # Create subject trajectories
    subject_lines = []
    subject_dots = []
    if "init_poses" in result or "moving_subject" in result:
        for sub_traj in subs_positions:
            line = gl.GLLinePlotItem(pos=sub_traj, color=(1, 0, 0, 1), width=3)
            dot = gl.GLScatterPlotItem(pos=sub_traj[-1:], color=(1, 0, 0, 1), size=10)
            w.addItem(line)
            w.addItem(dot)
            subject_lines.append(line)
            subject_dots.append(dot)

    # SCP iteration trajectories
    scp_traj_lines = []
    multishot_traj_lines = []
    scp_axes_items = []  # Store axes for each SCP iteration

    # Extract the number of states and controls from the parameters
    n_x = params.sim.n_states
    n_u = params.sim.n_controls

    # Define indices for slicing the augmented state vector
    i1 = n_x
    i2 = i1 + n_x * n_x
    i3 = i2 + n_x * n_u
    i4 = i3 + n_x * n_u

    # Auto-calculate vehicle axes length based on trajectory size
    axes_length = max(pos_range) * 0.1  # 10% of trajectory range
    axes_length = max(axes_length, 2.0)  # Minimum 2 units
    axes_length = min(axes_length, 20.0)  # Maximum 20 units

    # Create trajectory lines for each SCP iteration
    for traj_iter, scp_traj in enumerate(scp_ctcs_trajs):
        # SCP trajectory line - only show actual data points, not connecting lines
        scp_positions = scp_traj[:, 0:3]
        scp_line = gl.GLScatterPlotItem(pos=scp_positions, color=(0.5, 0.5, 0.5, 0.7), size=3)
        w.addItem(scp_line)
        scp_traj_lines.append(scp_line)

        # Create axes for this SCP iteration
        if drone_attitudes is not None:
            iteration_axes = []
            for i in range(scp_traj.shape[0]):
                att = scp_traj[i, 6:10]  # Extract attitude from SCP trajectory
                r = R.from_quat([att[1], att[2], att[3], att[0]])
                rotmat = r.as_matrix()
                axes = axes_length * np.eye(3)
                axes_rot = rotmat @ axes

                # Create axis lines for this position
                pos_axes = []
                axis_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]  # Red, Green, Blue
                for k in range(3):
                    axis_pts = np.stack([scp_positions[i], scp_positions[i] + axes_rot[:, k]])
                    axis_line = gl.GLLinePlotItem(pos=axis_pts, color=axis_colors[k], width=2)
                    w.addItem(axis_line)
                    pos_axes.append(axis_line)
                iteration_axes.append(pos_axes)
            scp_axes_items.append(iteration_axes)
        else:
            scp_axes_items.append([])

        # Multiple shooting trajectories - only show actual data points
        if traj_iter < len(scp_multi_shoot):
            pos_traj = []
            for i_multi in range(scp_multi_shoot[traj_iter].shape[1]):
                pos_traj.append(scp_multi_shoot[traj_iter][:, i_multi].reshape(-1, i4)[:, 0:3])
            pos_traj = np.array(pos_traj)

            iteration_multishot_lines = []
            for j in range(pos_traj.shape[1]):
                multishot_line = gl.GLScatterPlotItem(
                    pos=pos_traj[:, j], color=(0.3, 0.7, 1.0, 0.8), size=4
                )
                w.addItem(multishot_line)
                iteration_multishot_lines.append(multishot_line)
            multishot_traj_lines.append(iteration_multishot_lines)

    # Animation state
    ptr = [0]
    playing = [False]

    def update():
        i = ptr[0]
        if i >= n_iterations:
            timer.stop()
            playing[0] = False
            play_btn.setEnabled(True)
            pause_btn.setEnabled(False)
            return

        slider.blockSignals(True)
        slider.setValue(i)
        slider.blockSignals(False)

        # Show only the current SCP iteration and all previous ones
        for j, scp_line in enumerate(scp_traj_lines):
            if j <= i:
                scp_line.setVisible(True)
                scp_line.setData(color=(0.5, 0.5, 0.5, 0.7))
            else:
                scp_line.setVisible(False)

        # Show/hide axes for each iteration
        for j, iteration_axes in enumerate(scp_axes_items):
            if j == i:  # Only show axes for current iteration
                for pos_axes in iteration_axes:
                    for axis_line in pos_axes:
                        axis_line.setVisible(True)
            else:
                for pos_axes in iteration_axes:
                    for axis_line in pos_axes:
                        axis_line.setVisible(False)

        # Show multiple shooting trajectories for current iteration
        for j, iteration_lines in enumerate(multishot_traj_lines):
            if j == i:
                for line in iteration_lines:
                    line.setVisible(True)
                    line.setData(color=(0.3, 0.7, 1.0, 0.8))
            else:
                for line in iteration_lines:
                    line.setVisible(False)

        ptr[0] += 1

    def set_frame(i):
        ptr[0] = i
        update()

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.setInterval(1000)  # Slower for SCP iterations

    def play():
        if not playing[0]:
            playing[0] = True
            play_btn.setEnabled(False)
            pause_btn.setEnabled(True)
            timer.start()

    def pause():
        playing[0] = False
        play_btn.setEnabled(True)
        pause_btn.setEnabled(False)
        timer.stop()

    play_btn.clicked.connect(play)
    pause_btn.clicked.connect(pause)
    slider.valueChanged.connect(set_frame)
    pause_btn.setEnabled(False)

    main_widget.setWindowTitle(f"SCP Animation: {tof} seconds (pyqtgraph)")
    main_widget.resize(1280, 800)

    # Initialize display
    set_frame(0)

    if not hasattr(QtWidgets.QApplication.instance(), "exec_"):
        app.exec()
    else:
        QtWidgets.QApplication.instance().exec_()


def plot_camera_animation_pyqtgraph(result, params, step=2):
    if not PYQTPHOT_AVAILABLE:
        raise ImportError(
            "pyqtgraph is required for this function but not installed. "
            "Install it with: pip install openscvx[gui] or pip install pyqtgraph PyQt5"
        )

    """
    PyQtGraph version of plot_camera_animation: animates subject projections in camera frame (2D).
    """
    import sys

    import numpy as np
    import pyqtgraph as pg
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

    app = pg.QtWidgets.QApplication.instance() or pg.QtWidgets.QApplication(sys.argv)

    # Get subject projections in the camera frame
    _, subs_positions_sen, _, subs_positions_sen_node = full_subject_traj_time(result, params)
    n_frames = len(subs_positions_sen[0])
    indices = np.arange(0, n_frames, step)
    if indices[-1] != n_frames - 1:
        indices = np.append(indices, n_frames - 1)

    # Camera viewcone boundary (red curve)
    if "alpha_x" in result and "alpha_y" in result:
        A = np.diag([1 / np.tan(np.pi / result["alpha_x"]), 1 / np.tan(np.pi / result["alpha_y"])])
    else:
        raise ValueError("`alpha_x` and `alpha_y` not found in result dictionary.")
    n_cone = 400
    theta = np.linspace(0, 2 * np.pi, n_cone)
    circle = np.stack([np.cos(theta), np.sin(theta)])

    # Use the correct norm type from the result
    if "norm_type" in result:
        if result["norm_type"] == np.inf or result["norm_type"] == "inf":
            z = np.linalg.norm(A @ circle, axis=0, ord=np.inf)
        else:
            z = np.linalg.norm(A @ circle, axis=0, ord=result["norm_type"])
    else:
        # Default to 2-norm if norm_type not specified
        z = np.linalg.norm(A @ circle, axis=0)

    X = circle[0] / z
    Y = circle[1] / z
    # Repeat first point to close the curve
    X = np.append(X, X[0])
    Y = np.append(Y, Y[0])

    # Main window and layout
    main_widget = QWidget()
    main_layout = QVBoxLayout(main_widget)
    plot_widget = pg.PlotWidget()
    plot_widget.setAspectLocked(True)
    plot_widget.setRange(xRange=[-1.1, 1.1], yRange=[-1.1, 1.1])
    plot_widget.showGrid(x=False, y=False)  # Remove grid lines
    main_layout.addWidget(plot_widget)

    # Controls
    controls_layout = QHBoxLayout()
    play_btn = QPushButton("Play")
    pause_btn = QPushButton("Pause")
    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(len(indices) - 1)
    slider.setSingleStep(1)
    controls_layout.addWidget(play_btn)
    controls_layout.addWidget(pause_btn)
    controls_layout.addWidget(QLabel("Frame:"))
    controls_layout.addWidget(slider)
    main_layout.addLayout(controls_layout)
    main_widget.show()

    # Plot the camera viewcone boundary with thicker red line
    plot_widget.plot(X, Y, pen=pg.mkPen("r", width=5))

    # Prepare subject curves
    subject_curves = []
    subject_dots = []
    colors = [pg.intColor(i, hues=len(subs_positions_sen)) for i in range(len(subs_positions_sen))]
    for color in colors:
        curve = plot_widget.plot([], [], pen=pg.mkPen(color, width=2))
        dot = plot_widget.plot(
            [], [], pen=None, symbol="o", symbolBrush=color, symbolSize=6
        )  # Even smaller dots
        subject_curves.append(curve)
        subject_dots.append(dot)

    # Prepare nodal points - remove outline and make smaller
    subject_node_dots = []
    for color in colors:
        node_dot = plot_widget.plot(
            [], [], pen=None, symbol="o", symbolBrush=color, symbolSize=8
        )  # Smaller nodal points
        subject_node_dots.append(node_dot)

    ptr = [0]
    playing = [False]

    def update():
        i = ptr[0]
        if i >= len(indices):
            timer.stop()
            playing[0] = False
            play_btn.setEnabled(True)
            pause_btn.setEnabled(False)
            return
        slider.blockSignals(True)
        slider.setValue(i)
        slider.blockSignals(False)
        frame_idx = indices[i]
        for j, sub_traj in enumerate(subs_positions_sen):
            # Project trajectory up to current frame
            traj = np.array(sub_traj[: frame_idx + 1])
            x = traj[:, 0] / traj[:, 2]
            y = traj[:, 1] / traj[:, 2]
            subject_curves[j].setData(x, y)
            # Current dot
            subject_dots[j].setData([x[-1]], [y[-1]])
            # Nodal points (if available)
            if subs_positions_sen_node:
                node_traj = np.array(subs_positions_sen_node[j])
                node_idx = int((frame_idx // (len(sub_traj) / len(node_traj))) + 1)
                node_traj = node_traj[:node_idx]
                if len(node_traj) > 0:
                    node_x = node_traj[:, 0] / node_traj[:, 2]
                    node_y = node_traj[:, 1] / node_traj[:, 2]
                    subject_node_dots[j].setData(node_x, node_y)
                else:
                    subject_node_dots[j].setData([], [])
        ptr[0] += 1

    def set_frame(i):
        ptr[0] = i
        update()

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.setInterval(50)

    def play():
        if not playing[0]:
            playing[0] = True
            play_btn.setEnabled(False)
            pause_btn.setEnabled(True)
            timer.start()

    def pause():
        playing[0] = False
        play_btn.setEnabled(True)
        pause_btn.setEnabled(False)
        timer.stop()

    play_btn.clicked.connect(play)
    pause_btn.clicked.connect(pause)
    slider.valueChanged.connect(set_frame)
    pause_btn.setEnabled(False)

    main_widget.setWindowTitle("Camera Animation (pyqtgraph)")
    main_widget.resize(800, 800)

    update()  # Draw initial frame

    if not hasattr(pg.QtWidgets.QApplication.instance(), "exec_"):
        app.exec()
    else:
        pg.QtWidgets.QApplication.instance().exec_()
