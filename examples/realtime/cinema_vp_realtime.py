"""Interactive real-time visualization for cinematic viewpoint planning.

This module provides a PyQt5-based GUI for interactively solving and visualizing
the cinematic viewpoint planning trajectory optimization problem in real-time.
"""

import os
import sys
import threading
import time

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from examples.drone.cinema_vp_realtime_base import (
    kp_pose,
    plotting_dict,
    problem,
)

# Import PyQtGraph OpenGL modules
try:
    from pyqtgraph.opengl import (
        GLGridItem,
        GLLinePlotItem,
        GLMeshItem,
        GLScatterPlotItem,
        GLViewWidget,
        MeshData,
    )

    HAS_OPENGL = True
except ImportError:
    print("PyQtGraph OpenGL not available, falling back to 2D")
    HAS_OPENGL = False

# Import scipy Rotation for quaternion to rotation matrix conversion
# (matching plot_animation_pyqtgraph)
from scipy.spatial.transform import Rotation as R

running = {"stop": False}
reset_requested = {"reset": False}
latest_results = {"results": None}
new_result_event = threading.Event()


class CinemaVPPlotWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QVBoxLayout()
        self.setLayout(layout)
        if HAS_OPENGL:
            # Create 3D view
            self.view = GLViewWidget()
            self.view.setCameraPosition(distance=30)
            # Add grid
            grid = GLGridItem()
            self.view.addItem(grid)
            # Add trajectory scatter plot
            self.traj_scatter = GLScatterPlotItem(pos=np.zeros((1, 3)), color=(0, 0, 1, 1), size=5)
            self.view.addItem(self.traj_scatter)
            # Add keypoint scatter plot
            self.kp_scatter = GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1, 0, 0, 1), size=10)
            self.view.addItem(self.kp_scatter)
            # Add line-of-sight visualization (line from drone to keypoint)
            self.los_line = GLLinePlotItem(color=(1, 1, 0, 0.5), width=2)
            self.view.addItem(self.los_line)
            # Add drone axes visualization (x=red, y=green, z=blue)
            self.axis_lines = []
            axis_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]  # Red, Green, Blue
            for color in axis_colors:
                axis_line = GLLinePlotItem(color=color, width=3)
                self.view.addItem(axis_line)
                self.axis_lines.append(axis_line)
            # Add view cone mesh
            self.viewcone_mesh = None
            self.cone_meshdata = None  # Store cone mesh data for reuse
            # Create main layout with view and control panel
            main_layout = QHBoxLayout()
            # Create control panel
            self.create_control_panel()
            # Add widgets to main layout
            main_layout.addWidget(self.view, stretch=3)
            main_layout.addWidget(self.control_panel, stretch=1)
            layout.addLayout(main_layout)
        else:
            # Fallback to 2D
            label = QLabel("3D OpenGL not available")
            layout.addWidget(label)

    def create_control_panel(self):
        """Create the control panel with sliders for keypoint position"""
        self.control_panel = QWidget()
        control_layout = QVBoxLayout()
        self.control_panel.setLayout(control_layout)
        # Title
        title = QLabel("Cinema View Planning Control")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        control_layout.addWidget(title)
        # Optimization Metrics Display
        metrics_group = QGroupBox("Optimization Metrics")
        metrics_layout = QVBoxLayout()
        metrics_group.setLayout(metrics_layout)
        # Create labels for each metric
        self.iter_label = QLabel("Iteration: 0")
        self.j_tr_label = QLabel("J_tr: 0.00e+00")
        self.j_vb_label = QLabel("J_vb: 0.00e+00")
        self.j_vc_label = QLabel("J_vc: 0.00e+00")
        self.objective_label = QLabel("Objective: 0.00e+00")
        self.lam_cost_display_label = QLabel(f"λ_cost: {problem.settings.scp.lam_cost:.2E}")
        self.dis_time_label = QLabel("Dis Time: 0.0ms")
        self.solve_time_label = QLabel("Solve Time: 0.0ms")
        self.status_label = QLabel("Status: --")
        # Style the labels
        for label in [
            self.iter_label,
            self.j_tr_label,
            self.j_vb_label,
            self.j_vc_label,
            self.objective_label,
            self.lam_cost_display_label,
            self.dis_time_label,
            self.solve_time_label,
            self.status_label,
        ]:
            label.setStyleSheet("font-family: monospace; font-size: 11px; padding: 2px;")
            metrics_layout.addWidget(label)
        control_layout.addWidget(metrics_group)
        # Optimization Weights
        weights_group = QGroupBox("Optimization Weights")
        weights_layout = QVBoxLayout()
        weights_group.setLayout(weights_layout)
        # Lambda cost input - Input on left, label on right
        lam_cost_layout = QHBoxLayout()
        lam_cost_input = QLineEdit()
        lam_cost_input.setText(f"{problem.settings.scp.lam_cost:.2E}")
        lam_cost_input.setFixedWidth(80)
        lam_cost_input.returnPressed.connect(lambda: on_lam_cost_changed(lam_cost_input))
        lam_cost_label = QLabel("λ_cost:")
        lam_cost_label.setAlignment(Qt.AlignLeft)
        lam_cost_layout.addWidget(lam_cost_input)
        lam_cost_layout.addWidget(lam_cost_label)
        lam_cost_layout.addStretch()  # Push everything to the left
        weights_layout.addLayout(lam_cost_layout)
        # Lambda trust region input - Input on left, label on right
        lam_tr_layout = QHBoxLayout()
        lam_tr_input = QLineEdit()
        lam_tr_input.setText(f"{problem.settings.scp.w_tr:.2E}")
        lam_tr_input.setFixedWidth(80)
        lam_tr_input.returnPressed.connect(lambda: on_lam_tr_changed(lam_tr_input))
        lam_tr_label = QLabel("λ_tr:")
        lam_tr_label.setAlignment(Qt.AlignLeft)
        lam_tr_layout.addWidget(lam_tr_input)
        lam_tr_layout.addWidget(lam_tr_label)
        lam_tr_layout.addStretch()  # Push everything to the left
        weights_layout.addLayout(lam_tr_layout)
        control_layout.addWidget(weights_group)
        # Reset Button
        reset_group = QGroupBox("Problem Control")
        reset_layout = QVBoxLayout()
        reset_group.setLayout(reset_layout)
        reset_button = QPushButton("Reset Problem")
        reset_button.clicked.connect(self.on_reset_clicked)
        reset_layout.addWidget(reset_button)
        control_layout.addWidget(reset_group)
        # Keypoint Position Controls
        kp_group = QGroupBox("Keypoint Position (Line-of-Sight Target)")
        kp_layout = QVBoxLayout()
        kp_group.setLayout(kp_layout)
        # X, Y, Z sliders
        self.kp_sliders = []
        for j, coord in enumerate(["X", "Y", "Z"]):
            slider_layout = QHBoxLayout()
            label = QLabel(f"{coord}:")
            slider = QSlider(Qt.Horizontal)
            # Set range based on position bounds (scaled for slider)
            slider.setRange(-200, 200)
            # Set initial value based on current keypoint position
            initial_value = int(kp_pose.value[j] * 10)  # Scale by 10 for precision
            slider.setValue(initial_value)
            value_label = QLabel(f"{kp_pose.value[j]:.2f}")
            # Connect slider to update function
            slider.valueChanged.connect(
                lambda val, axis=j, label=value_label: self.on_slider_changed(val, axis, label)
            )
            slider_layout.addWidget(label)
            slider_layout.addWidget(slider)
            slider_layout.addWidget(value_label)
            kp_layout.addLayout(slider_layout)
            self.kp_sliders.append((slider, value_label))
        # Text inputs for precise control
        text_inputs_layout = QHBoxLayout()
        self.kp_x_input = QLineEdit()
        self.kp_x_input.setText(f"{kp_pose.value[0]:.2f}")
        self.kp_x_input.setFixedWidth(60)
        self.kp_x_input.returnPressed.connect(
            lambda: self.on_text_input_changed(0, self.kp_x_input)
        )
        self.kp_y_input = QLineEdit()
        self.kp_y_input.setText(f"{kp_pose.value[1]:.2f}")
        self.kp_y_input.setFixedWidth(60)
        self.kp_y_input.returnPressed.connect(
            lambda: self.on_text_input_changed(1, self.kp_y_input)
        )
        self.kp_z_input = QLineEdit()
        self.kp_z_input.setText(f"{kp_pose.value[2]:.2f}")
        self.kp_z_input.setFixedWidth(60)
        self.kp_z_input.returnPressed.connect(
            lambda: self.on_text_input_changed(2, self.kp_z_input)
        )
        text_inputs_layout.addWidget(QLabel("Precise:"))
        text_inputs_layout.addWidget(self.kp_x_input)
        text_inputs_layout.addWidget(self.kp_y_input)
        text_inputs_layout.addWidget(self.kp_z_input)
        kp_layout.addLayout(text_inputs_layout)
        control_layout.addWidget(kp_group)
        control_layout.addStretch()
        # Create labels dictionary for metrics update
        self.labels_dict = {
            "iter_label": self.iter_label,
            "j_tr_label": self.j_tr_label,
            "j_vb_label": self.j_vb_label,
            "j_vc_label": self.j_vc_label,
            "objective_label": self.objective_label,
            "lam_cost_display_label": self.lam_cost_display_label,
            "dis_time_label": self.dis_time_label,
            "solve_time_label": self.solve_time_label,
            "status_label": self.status_label,
        }

    def on_slider_changed(self, value, axis, label):
        """Handle slider value changes"""
        # Convert slider value (-200 to 200) to world coordinates (-20 to 20)
        world_value = value * 0.1
        # Update the parameter
        new_kp_pose = kp_pose.value.copy()
        new_kp_pose[axis] = world_value
        # Update both the Parameter object's value and problem.parameters
        kp_pose.value = new_kp_pose
        problem.parameters["kp_pose"] = new_kp_pose
        # Update label
        label.setText(f"{world_value:.2f}")
        # Update text input
        if axis == 0:
            self.kp_x_input.setText(f"{world_value:.2f}")
        elif axis == 1:
            self.kp_y_input.setText(f"{world_value:.2f}")
        elif axis == 2:
            self.kp_z_input.setText(f"{world_value:.2f}")

    def on_text_input_changed(self, axis, input_widget):
        """Handle text input changes"""
        try:
            value = float(input_widget.text())
            # Update the parameter
            new_kp_pose = kp_pose.value.copy()
            new_kp_pose[axis] = value
            # Update both the Parameter object's value and problem.parameters
            kp_pose.value = new_kp_pose
            problem.parameters["kp_pose"] = new_kp_pose
            # Update slider
            slider, label = self.kp_sliders[axis]
            slider_value = int(value * 10)  # Scale by 10 for slider
            slider.setValue(slider_value)
            label.setText(f"{value:.2f}")
        except ValueError:
            print(f"Invalid input for axis {axis}")

    def on_reset_clicked(self):
        """Handle reset button click"""
        reset_requested["reset"] = True
        print("Reset requested - problem will reset on next iteration")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


def on_lam_cost_changed(input_widget):
    """Handle lambda cost input changes"""
    new_value = input_widget.text()
    try:
        lam_cost_value = float(new_value)
        problem.settings.scp.lam_cost = lam_cost_value
        input_widget.setText(f"{lam_cost_value:.2E}")
    except ValueError:
        print("Invalid input. Please enter a valid number.")


def on_lam_tr_changed(input_widget):
    """Handle lambda trust region input changes"""
    new_value = input_widget.text()
    try:
        lam_tr_value = float(new_value)
        problem.settings.scp.w_tr = lam_tr_value
        input_widget.setText(f"{lam_tr_value:.2E}")
    except ValueError:
        print("Invalid input. Please enter a valid number.")


def update_optimization_metrics(results, labels_dict):
    """Update the optimization metrics display"""
    if results is None:
        return
    # Extract metrics from results
    iter_num = results.get("iter", 0)
    j_tr = results.get("J_tr", 0.0)
    j_vb = results.get("J_vb", 0.0)
    j_vc = results.get("J_vc", 0.0)
    cost = results.get("cost", 0.0)
    status = results.get("prob_stat", "--")
    # Get timing information
    dis_time = results.get("dis_time", 0.0)
    solve_time = results.get("solve_time", 0.0)
    # Update labels
    labels_dict["iter_label"].setText(f"Iteration: {iter_num}")
    labels_dict["j_tr_label"].setText(f"J_tr: {j_tr:.2E}")
    labels_dict["j_vb_label"].setText(f"J_vb: {j_vb:.2E}")
    labels_dict["j_vc_label"].setText(f"J_vc: {j_vc:.2E}")
    labels_dict["objective_label"].setText(f"Objective: {cost:.2E}")
    labels_dict["lam_cost_display_label"].setText(f"λ_cost: {problem.settings.scp.lam_cost:.2E}")
    labels_dict["dis_time_label"].setText(f"Dis Time: {dis_time:.1f}ms")
    labels_dict["solve_time_label"].setText(f"Solve Time: {solve_time:.1f}ms")
    labels_dict["status_label"].setText(f"Status: {status}")


def update_viewcone(plot_widget, drone_pos, att, plotting_dict, pos_range=None):
    """Create or update the view cone mesh visualization"""
    if not HAS_OPENGL:
        return

    # Get view cone parameters from plotting_dict
    alpha_x = plotting_dict.get("alpha_x", 6.0)
    alpha_y = plotting_dict.get("alpha_y", 8.0)
    R_sb = plotting_dict.get("R_sb", np.eye(3))
    norm_type = plotting_dict.get("norm_type", "inf")

    # Create cone mesh data if it doesn't exist
    if not hasattr(plot_widget, "cone_meshdata") or plot_widget.cone_meshdata is None:
        n_cone = 40
        theta = np.linspace(0, 2 * np.pi, n_cone)
        A = np.diag([1 / np.tan(np.pi / alpha_x), 1 / np.tan(np.pi / alpha_y)])

        # Calculate cone length based on trajectory range (similar to plot_animation_pyqtgraph)
        if pos_range is not None:
            cone_length = max(pos_range) * 0.3  # 30% of trajectory range
            cone_length = max(cone_length, 10.0)  # Minimum 10 units
            cone_length = min(cone_length, 50.0)  # Maximum 50 units
        else:
            cone_length = 15.0  # Default length

        # Create circle in sensor frame
        circle = np.stack([np.cos(theta), np.sin(theta)])

        # Calculate z values based on norm type
        if norm_type == "inf" or norm_type == np.inf:
            z = np.linalg.norm(A @ circle, axis=0, ord=np.inf)
        else:
            z = np.linalg.norm(A @ circle, axis=0, ord=norm_type)

        # Create cone base points
        X = circle[0] / z
        Y = circle[1] / z
        Z = np.ones_like(X)
        base_points = np.stack([X, Y, Z], axis=1) * cone_length

        # Create cone mesh (apex at origin, base at cone_length)
        apex = np.array([[0, 0, 0]])
        vertices = np.vstack([apex, base_points])

        # Create faces
        faces = []
        for i in range(1, n_cone):
            faces.append([0, i, i + 1])
        faces.append([0, n_cone, 1])
        faces = np.array(faces)

        # Store the mesh data for reuse
        plot_widget.cone_meshdata = MeshData(vertexes=vertices, faces=faces)

    # Transform cone from sensor frame to body frame to inertial frame
    # Use scipy Rotation for quaternion conversion (matching plot_animation_pyqtgraph)
    # Quaternion format: [qw, qx, qy, qz] -> scipy expects [qx, qy, qz, qw]
    r = R.from_quat([att[1], att[2], att[3], att[0]])
    rotmat = r.as_matrix()

    # Get original vertices and transform them
    verts = plot_widget.cone_meshdata.vertexes()
    verts_tf = (rotmat @ R_sb.T @ verts.T).T + drone_pos

    # Create or update mesh
    if plot_widget.viewcone_mesh is None:
        plot_widget.viewcone_mesh = GLMeshItem(
            meshdata=MeshData(vertexes=verts_tf, faces=plot_widget.cone_meshdata.faces()),
            smooth=True,
            color=(1, 1, 0, 0.5),  # Yellow, semi-transparent (matching plot_animation_pyqtgraph)
            shader="shaded",
            drawEdges=False,
            glOptions="additive",  # Use additive blending like plot_animation_pyqtgraph
        )
        plot_widget.view.addItem(plot_widget.viewcone_mesh)
    else:
        # Update existing mesh
        plot_widget.viewcone_mesh.setMeshData(
            vertexes=verts_tf, faces=plot_widget.cone_meshdata.faces()
        )
        # Remove and re-add cone to ensure it is drawn last (matching plot_animation_pyqtgraph)
        if plot_widget.viewcone_mesh in plot_widget.view.items:
            plot_widget.view.removeItem(plot_widget.viewcone_mesh)
        plot_widget.view.addItem(plot_widget.viewcone_mesh)


def optimization_loop():
    problem.initialize()
    try:
        while not running["stop"]:
            # Check if reset was requested
            if reset_requested["reset"]:
                problem.reset()
                reset_requested["reset"] = False
                print("Problem reset to initial conditions")

            # Perform a single SCP step (automatically warm-starts from previous iteration)
            step_result = problem.step()

            # Build results dict for visualization
            results = {
                "iter": step_result["scp_k"] - 1,  # Display iteration (0-indexed)
                "J_tr": step_result["scp_J_tr"],
                "J_vb": step_result["scp_J_vb"],
                "J_vc": step_result["scp_J_vc"],
                "converged": step_result["converged"],
                "V_multi_shoot": problem.state.V_history[-1] if problem.state.V_history else [],
                "x": problem.state.x,  # Current state trajectory
                "u": problem.state.u,  # Current control trajectory
            }

            # Get timing from the print queue (emitted data)
            try:
                if hasattr(problem, "print_queue") and not problem.print_queue.empty():
                    # Get the latest emitted data
                    emitted_data = problem.print_queue.get_nowait()
                    results["dis_time"] = emitted_data.get("dis_time", 0.0)
                    results["solve_time"] = emitted_data.get("subprop_time", 0.0)
                    results["prob_stat"] = emitted_data.get("prob_stat", "--")
                    results["cost"] = emitted_data.get("cost", 0.0)
                else:
                    results["dis_time"] = 0.0
                    results["solve_time"] = 0.0
                    results["prob_stat"] = "--"
                    results["cost"] = 0.0
            except Exception:
                results["dis_time"] = 0.0
                results["solve_time"] = 0.0
                results["prob_stat"] = "--"
                results["cost"] = 0.0

            results.update(plotting_dict)
            latest_results["results"] = results
            new_result_event.set()
    except KeyboardInterrupt:
        running["stop"] = True
        print("Stopped by user.")


def plot_thread_func():
    # Initialize PyQtGraph
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    print(f"Creating plot window... OpenGL available: {HAS_OPENGL}")
    # Create 3D plot window
    plot_widget = CinemaVPPlotWidget()
    plot_widget.setWindowTitle("Cinema View Planning Real-time Trajectory")
    plot_widget.resize(1200, 800)
    plot_widget.show()
    print("Plot window created and shown")
    # Force the window to be visible
    plot_widget.raise_()
    plot_widget.activateWindow()
    # Small delay to ensure window appears
    time.sleep(0.1)
    # Update timer
    timer = QTimer()

    def update_plot():
        if latest_results["results"] is not None:
            try:
                V_multi_shoot = np.array(latest_results["results"]["V_multi_shoot"])
                # Extract 3D position data (first 3 elements of state)
                n_x = problem.settings.sim.n_states
                n_u = problem.settings.sim.n_controls
                i1 = n_x
                i2 = i1 + n_x * n_x
                i3 = i2 + n_x * n_u
                i4 = i3 + n_x * n_u
                all_pos_segments = []
                for i_node in range(V_multi_shoot.shape[1]):
                    node_data = V_multi_shoot[:, i_node]
                    segments_for_node = node_data.reshape(-1, i4)
                    pos_segments = segments_for_node[:, :3]  # 3D positions
                    all_pos_segments.append(pos_segments)
                if all_pos_segments:
                    full_traj = np.vstack(all_pos_segments)
                    if HAS_OPENGL:
                        plot_widget.traj_scatter.setData(pos=full_traj)
                        # Update keypoint position
                        current_kp = problem.parameters["kp_pose"]
                        plot_widget.kp_scatter.setData(pos=current_kp.reshape(1, 3))
                        # Update line-of-sight line (from last position to keypoint)
                        if len(full_traj) > 0:
                            los_points = np.vstack([full_traj[-1], current_kp])
                            plot_widget.los_line.setData(pos=los_points)
                            # Extract attitude from last node and draw axes
                            # State order: position[0:3], velocity[3:6], attitude[6:10], ...
                            if "x" in latest_results["results"]:
                                x_traj = latest_results["results"]["x"]  # Now a numpy array
                                if len(x_traj) > 0 and x_traj.shape[1] >= 10:
                                    # Get attitude quaternion [qw, qx, qy, qz] at last node
                                    att = x_traj[-1, 6:10]
                                    # Convert quaternion to rotation matrix using scipy
                                    # (matching plot_animation_pyqtgraph)
                                    r = R.from_quat([att[1], att[2], att[3], att[0]])
                                    rotmat = r.as_matrix()
                                    # Draw axes (x=red, y=green, z=blue)
                                    axes_length = 2.0
                                    axes = axes_length * np.eye(3)
                                    axes_rot = rotmat @ axes
                                    current_pos = full_traj[-1]
                                    for k in range(3):
                                        axis_pts = np.vstack(
                                            [current_pos, current_pos + axes_rot[:, k]]
                                        )
                                        plot_widget.axis_lines[k].setData(pos=axis_pts)
                                    # Update view cone
                                    # Calculate position range for cone length
                                    if len(full_traj) > 1:
                                        pos_range = np.max(full_traj, axis=0) - np.min(
                                            full_traj, axis=0
                                        )
                                    else:
                                        pos_range = None
                                    update_viewcone(
                                        plot_widget,
                                        current_pos,
                                        att,
                                        plotting_dict,
                                        pos_range=pos_range,
                                    )
                    else:
                        # 2D fallback
                        pass
                # Update optimization metrics display
                update_optimization_metrics(latest_results["results"], plot_widget.labels_dict)
            except Exception as e:
                print(f"Plot update error: {e}")
                if "x" in latest_results["results"]:
                    x_traj = latest_results["results"]["x"]  # Now a numpy array
                    if HAS_OPENGL:
                        plot_widget.traj_scatter.setData(pos=x_traj[:, :3])
                        # Update keypoint position
                        current_kp = problem.parameters["kp_pose"]
                        plot_widget.kp_scatter.setData(pos=current_kp.reshape(1, 3))
                        # Update line-of-sight line
                        if len(x_traj) > 0:
                            los_points = np.vstack([x_traj[-1, :3], current_kp])
                            plot_widget.los_line.setData(pos=los_points)
                            # Draw axes at last position
                            if x_traj.shape[1] >= 10:
                                att = x_traj[-1, 6:10]
                                # Convert quaternion to rotation matrix using scipy
                                # (matching plot_animation_pyqtgraph)
                                r = R.from_quat([att[1], att[2], att[3], att[0]])
                                rotmat = r.as_matrix()
                                axes_length = 2.0
                                axes = axes_length * np.eye(3)
                                axes_rot = rotmat @ axes
                                current_pos = x_traj[-1, :3]
                                for k in range(3):
                                    axis_pts = np.vstack(
                                        [current_pos, current_pos + axes_rot[:, k]]
                                    )
                                    plot_widget.axis_lines[k].setData(pos=axis_pts)
                                # Update view cone
                                # Calculate position range for cone length
                                if len(x_traj) > 1:
                                    pos_range = np.max(x_traj[:, :3], axis=0) - np.min(
                                        x_traj[:, :3], axis=0
                                    )
                                else:
                                    pos_range = None
                                update_viewcone(
                                    plot_widget,
                                    current_pos,
                                    att,
                                    plotting_dict,
                                    pos_range=pos_range,
                                )

    timer.timeout.connect(update_plot)
    timer.start(50)  # Update every 50ms
    print("Starting Qt event loop...")
    # Start the Qt event loop
    app.exec_()


if __name__ == "__main__":
    # Start optimization thread
    opt_thread = threading.Thread(target=optimization_loop)
    opt_thread.daemon = True
    opt_thread.start()
    # Start plotting in main thread (this will block and run the Qt event loop)
    plot_thread_func()
