import os
import sys
import threading
import time

import numpy as np
import pyqtgraph.opengl as gl
from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph import Vector
from pyqtgraph.opengl import GLViewWidget

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from examples.drone.drone_racing import (
    gate_center_params,
    gen_vertices,
    initial_gate_centers,
    problem,
)

# Add the OpenSCvx path
# Import the drone racing problem and parameters
problem.initialize()


class OptimizationWorker(QObject):
    finished = pyqtSignal()
    results_ready = pyqtSignal(dict)
    metrics_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = False
        self.problem = problem
        self.gate_center_params = gate_center_params

    def update_gate_position(self, gate_idx, x, y, z):
        if 0 <= gate_idx < len(self.gate_center_params):
            # User input now represents the actual center position (no offset needed)
            new_center = np.array([x, y, z])
            # Update gate center parameter value
            self.gate_center_params[gate_idx].value = new_center
            # Sync the parameter to the problem's parameter dictionary
            param_name = self.gate_center_params[gate_idx].name
            self.problem.parameters[param_name] = new_center

    def run_optimization(self):
        self.running = True
        iteration = 0
        while self.running:
            try:
                start_time = time.time()
                results = self.problem.step()
                solve_time = time.time() - start_time
                # Add timing information to results
                results["iter"] = self.problem.scp_k - 1
                results["J_tr"] = self.problem.scp_J_tr
                results["J_vb"] = self.problem.scp_J_vb
                results["J_vc"] = self.problem.scp_J_vc
                results["solve_time"] = solve_time * 1000  # Convert to milliseconds
                # Get timing from the print queue (emitted data) if available
                try:
                    if (
                        hasattr(self.problem, "print_queue")
                        and not self.problem.print_queue.empty()
                    ):
                        # Get the latest emitted data
                        emitted_data = self.problem.print_queue.get_nowait()
                        results["dis_time"] = emitted_data.get("dis_time", 0.0)
                        results["prob_stat"] = emitted_data.get("prob_stat", "--")
                        results["cost"] = emitted_data.get("cost", 0.0)
                    else:
                        results["dis_time"] = 0.0
                        results["prob_stat"] = "--"
                        results["cost"] = results.get("cost", 0.0)
                except Exception:
                    results["dis_time"] = 0.0
                    results["prob_stat"] = "--"
                    results["cost"] = results.get("cost", 0.0)
                # Update vertices for visualization
                radii = np.array([2.5, 1e-4, 2.5])
                vertices = []
                for center_param in self.gate_center_params:
                    center = center_param.value
                    if center is not None:
                        vertices.append(gen_vertices(center, radii))
                    else:
                        vertices.append([])
                results.update(
                    {
                        "vertices": vertices,
                        "gate_center_params": self.gate_center_params,
                    }
                )
                self.results_ready.emit(results)
                # Emit metrics for compatibility
                metrics = {
                    "iteration": iteration,
                    "objective": results.get("cost", 0.0),
                    "solve_time": solve_time,
                    "status": results.get("prob_stat", "Unknown"),
                }
                self.metrics_updated.emit(metrics)
                iteration += 1
                time.sleep(0.1)
            except Exception as e:
                print(f"Optimization error: {e}")
                time.sleep(1.0)
        self.finished.emit()


class DraggableGLViewWidget(GLViewWidget):
    """Custom GLViewWidget that supports manual camera interaction"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manual_camera_interaction = False
        self.camera_interaction_timer = None

    def mousePressEvent(self, event):
        # User is interacting with camera - disable auto-adjustment temporarily
        self.manual_camera_interaction = True
        if self.camera_interaction_timer:
            self.camera_interaction_timer.stop()
        self.camera_interaction_timer = QTimer()
        self.camera_interaction_timer.timeout.connect(self.re_enable_auto_camera)
        self.camera_interaction_timer.start(2000)  # Re-enable after 2 seconds
        # Update status label
        if hasattr(self.parent(), "camera_status_label"):
            self.parent().camera_status_label.setText("Camera: Manual control (2s)")
            self.parent().camera_status_label.setStyleSheet("font-size: 10px; color: #ff6600;")
        super().mousePressEvent(event)

    def re_enable_auto_camera(self):
        """Re-enable auto camera adjustment after manual interaction"""
        self.manual_camera_interaction = False
        if self.camera_interaction_timer:
            self.camera_interaction_timer.stop()
        # Update status label
        if hasattr(self.parent(), "camera_status_label"):
            self.parent().camera_status_label.setText("Camera: Auto-adjust enabled")
            self.parent().camera_status_label.setStyleSheet("font-size: 10px; color: #666;")

    def mouseMoveEvent(self, event):
        # Only handle camera interaction, no gate dragging
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Disable auto-camera during zoom
        self.manual_camera_interaction = True
        if self.camera_interaction_timer:
            self.camera_interaction_timer.stop()
        self.camera_interaction_timer = QTimer()
        self.camera_interaction_timer.timeout.connect(self.re_enable_auto_camera)
        self.camera_interaction_timer.start(2000)
        # Update status label
        if hasattr(self.parent(), "camera_status_label"):
            self.parent().camera_status_label.setText("Camera: Manual control (2s)")
            self.parent().camera_status_label.setStyleSheet("font-size: 10px; color: #ff6600;")
        super().wheelEvent(event)


class DroneRacingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Racing Real-time Optimization")
        self.setGeometry(100, 100, 1400, 900)
        # Setup the worker
        self.worker = OptimizationWorker()
        self.worker_thread = threading.Thread(target=self.worker.run_optimization)
        self.worker.results_ready.connect(self.update_visualization)
        self.worker.metrics_updated.connect(self.update_metrics)
        # Setup GUI
        self.setup_gui()
        # Setup the problem and start optimization
        # self.worker.setup_problem()
        self.worker_thread.start()

    def setup_gui(self):
        """Set up the GUI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        # Title
        title = QLabel("Drone Racing Control")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_layout.addWidget(title)
        # Optimization Metrics Display
        metrics_group = QGroupBox("Optimization Metrics")
        metrics_layout = QVBoxLayout()
        metrics_group.setLayout(metrics_layout)
        # Create labels for each metric
        self.iter_label = QLabel("Iteration: 0")
        self.j_tr_label = QLabel("J_tr: 0.00E+00")
        self.j_vb_label = QLabel("J_vb: 0.00E+00")
        self.j_vc_label = QLabel("J_vc: 0.00E+00")
        self.objective_label = QLabel("Objective: 0.00E+00")
        self.lam_cost_display_label = QLabel(
            f"λ_cost: {self.worker.problem.settings.scp.lam_cost:.2E}"
        )
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
        left_layout.addWidget(metrics_group)
        # Optimization Weights
        weights_group = QGroupBox("Optimization Weights")
        weights_layout = QVBoxLayout()
        weights_group.setLayout(weights_layout)
        # Lambda cost input - Input on left, label on right
        lam_cost_layout = QHBoxLayout()
        self.lam_cost_input = QLineEdit()
        self.lam_cost_input.setText(f"{self.worker.problem.settings.scp.lam_cost:.2E}")
        self.lam_cost_input.setFixedWidth(80)
        self.lam_cost_input.returnPressed.connect(
            lambda: self.on_lam_cost_changed(self.lam_cost_input)
        )
        lam_cost_label = QLabel("λ_cost:")
        lam_cost_label.setAlignment(Qt.AlignLeft)
        lam_cost_layout.addWidget(self.lam_cost_input)
        lam_cost_layout.addWidget(lam_cost_label)
        lam_cost_layout.addStretch()  # Push everything to the left
        weights_layout.addLayout(lam_cost_layout)
        # Lambda trust region input - Input on left, label on right
        lam_tr_layout = QHBoxLayout()
        self.lam_tr_input = QLineEdit()
        self.lam_tr_input.setText(f"{self.worker.problem.settings.scp.w_tr:.2E}")
        self.lam_tr_input.setFixedWidth(80)
        self.lam_tr_input.returnPressed.connect(lambda: self.on_lam_tr_changed(self.lam_tr_input))
        lam_tr_label = QLabel("λ_tr:")
        lam_tr_label.setAlignment(Qt.AlignLeft)
        lam_tr_layout.addWidget(self.lam_tr_input)
        lam_tr_layout.addWidget(lam_tr_label)
        lam_tr_layout.addStretch()  # Push everything to the left
        weights_layout.addLayout(lam_tr_layout)
        left_layout.addWidget(weights_group)
        # Camera Controls
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QVBoxLayout()
        camera_group.setLayout(camera_layout)
        camera_reset_button = QPushButton("Reset Camera View")
        camera_reset_button.clicked.connect(self.reset_camera_view)
        camera_layout.addWidget(camera_reset_button)
        # Add auto-camera checkbox
        self.auto_camera_checkbox = QCheckBox("Auto-adjust camera to trajectory")
        self.auto_camera_checkbox.setChecked(True)
        camera_layout.addWidget(self.auto_camera_checkbox)
        # Add status indicator for manual camera interaction
        self.camera_status_label = QLabel("Camera: Auto-adjust enabled")
        self.camera_status_label.setStyleSheet("font-size: 10px; color: #666;")
        camera_layout.addWidget(self.camera_status_label)
        left_layout.addWidget(camera_group)
        # Gate Controls
        gate_group = QGroupBox("Gate Positions")
        gate_layout = QVBoxLayout()
        gate_group.setLayout(gate_layout)
        # Reset button for gates
        reset_gates_button = QPushButton("Reset Gate Positions")
        reset_gates_button.clicked.connect(self.reset_gate_positions)
        gate_layout.addWidget(reset_gates_button)
        # Scrollable area for gate controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(300)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        self.gate_controls = []
        for i in range(10):  # 10 gates
            gate_control_group = QWidget()
            gate_control_layout = QGridLayout(gate_control_group)
            gate_label = QLabel(f"Gate {i + 1}")
            gate_control_layout.addWidget(gate_label, 0, 0, 1, 3)
            # X, Y, Z text inputs
            x_input = QLineEdit()
            x_input.setFixedWidth(60)
            # Show actual center position (with offset) to user
            actual_center = self.worker.gate_center_params[i].value
            x_input.setText(str(actual_center[0]))
            x_input.editingFinished.connect(
                lambda gate_idx=i, field=x_input: self.update_gate_x_text(gate_idx, field)
            )
            y_input = QLineEdit()
            y_input.setFixedWidth(60)
            y_input.setText(str(actual_center[1]))
            y_input.editingFinished.connect(
                lambda gate_idx=i, field=y_input: self.update_gate_y_text(gate_idx, field)
            )
            z_input = QLineEdit()
            z_input.setFixedWidth(60)
            z_input.setText(str(actual_center[2]))
            z_input.editingFinished.connect(
                lambda gate_idx=i, field=z_input: self.update_gate_z_text(gate_idx, field)
            )
            gate_control_layout.addWidget(QLabel("X:"), 1, 0)
            gate_control_layout.addWidget(x_input, 1, 1)
            gate_control_layout.addWidget(QLabel("Y:"), 2, 0)
            gate_control_layout.addWidget(y_input, 2, 1)
            gate_control_layout.addWidget(QLabel("Z:"), 3, 0)
            gate_control_layout.addWidget(z_input, 3, 1)
            self.gate_controls.append((x_input, y_input, z_input))
            scroll_layout.addWidget(gate_control_group)
        scroll_layout.addStretch()
        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        gate_layout.addWidget(scroll_area)
        left_layout.addWidget(gate_group)
        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        layout.addWidget(left_panel, 1)
        # 3D visualization
        self.gl_widget = DraggableGLViewWidget()
        self.gl_widget.setCameraPosition(distance=150, elevation=30, azimuth=45)
        # Set the camera center separately
        self.gl_widget.opts["center"] = Vector(100, -50, 20)
        # Add grid
        grid = gl.GLGridItem()
        grid.setSize(200, 100)  # Match the fixed bounds: x (0,200), y (-100,0)
        grid.setSpacing(20, 20)
        self.gl_widget.addItem(grid)
        self.grid_item = grid  # Store reference for updates
        # Initialize visualization objects
        self.trajectory_plot = gl.GLScatterPlotItem(color=(1, 1, 0, 1), size=5)
        self.gl_widget.addItem(self.trajectory_plot)
        self.gate_plots = []
        for i in range(10):
            gate_plot = gl.GLLinePlotItem(color=(0, 1, 0, 1), width=3)
            self.gl_widget.addItem(gate_plot)
            self.gate_plots.append(gate_plot)
        layout.addWidget(self.gl_widget, 3)

    def update_gate_x_text(self, gate_idx, field):
        try:
            value = float(field.text())
            current_pos = self.worker.gate_center_params[gate_idx].value
            self.worker.update_gate_position(gate_idx, value, current_pos[1], current_pos[2])
        except Exception as e:
            print(f"Invalid X input for gate {gate_idx}: {e}")

    def update_gate_y_text(self, gate_idx, field):
        try:
            value = float(field.text())
            current_pos = self.worker.gate_center_params[gate_idx].value
            self.worker.update_gate_position(gate_idx, current_pos[0], value, current_pos[2])
        except Exception as e:
            print(f"Invalid Y input for gate {gate_idx}: {e}")

    def update_gate_z_text(self, gate_idx, field):
        try:
            value = float(field.text())
            current_pos = self.worker.gate_center_params[gate_idx].value
            self.worker.update_gate_position(gate_idx, current_pos[0], current_pos[1], value)
        except Exception as e:
            print(f"Invalid Z input for gate {gate_idx}: {e}")

    def update_visualization(self, results):
        """Update the 3D visualization with new results"""
        try:
            # Update optimization metrics display
            self.update_optimization_metrics(results)
            # Trajectory extraction
            if "V_multi_shoot" in results:
                try:
                    V_multi_shoot = np.array(results["V_multi_shoot"])
                    n_x = self.worker.problem.settings.sim.n_states
                    n_u = self.worker.problem.settings.sim.n_controls
                    i1 = n_x
                    i2 = i1 + n_x * n_x
                    i3 = i2 + n_x * n_u
                    i4 = i3 + n_x * n_u
                    all_pos_segments = []
                    for i_node in range(V_multi_shoot.shape[1]):
                        node_data = V_multi_shoot[:, i_node]
                        segments_for_node = node_data.reshape(-1, i4)
                        pos_segments = segments_for_node[:, :3]
                        all_pos_segments.append(pos_segments)
                    if all_pos_segments:
                        full_traj = np.vstack(all_pos_segments)
                        self.trajectory_plot.setData(pos=full_traj)
                        # Auto-adjust camera to fit trajectory bounds
                        if self.auto_camera_checkbox.isChecked():
                            self.adjust_camera_to_trajectory(full_traj)
                except Exception as e:
                    print(f"Trajectory extraction error: {e}")
            # Gate plotting (always runs)
            if "vertices" in results:
                for i, vertices in enumerate(results["vertices"]):
                    if i < len(self.gate_plots) and vertices is not None and len(vertices) >= 4:
                        try:
                            # Plot a line through the vertices of the gate
                            closed_vertices = np.vstack([vertices, vertices[0]])
                            self.gate_plots[i].setData(pos=closed_vertices)
                        except Exception as e:
                            print(f"Gate plotting error for gate {i}: {e}")
                    elif i < len(self.gate_plots):
                        self.gate_plots[i].setData(pos=np.zeros((0, 3)))
        except Exception as e:
            print(f"Error in update_visualization: {e}")

    def adjust_camera_to_trajectory(self, trajectory):
        """Automatically adjust camera to fit trajectory bounds with fixed x,y bounds"""
        try:
            # Skip auto-adjustment if user is manually interacting with camera
            if (
                hasattr(self.gl_widget, "manual_camera_interaction")
                and self.gl_widget.manual_camera_interaction
            ):
                return
            if trajectory is None or len(trajectory) == 0:
                return
            # Use fixed bounds for x and y, calculate z from trajectory
            min_x, max_x = 0, 200
            min_y, max_y = -100, 0
            # Calculate z bounds from trajectory
            min_z = np.min(trajectory[:, 2])
            max_z = np.max(trajectory[:, 2])
            # Add padding to z bounds
            padding_z = 10.0
            min_z -= padding_z
            max_z += padding_z
            # Calculate center and size
            center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2])
            size_x = max_x - min_x
            size_y = max_y - min_y
            size_z = max_z - min_z
            max_size = max(size_x, size_y, size_z)
            # Set camera distance based on the largest dimension
            distance = max_size * 1.2
            # Update camera position
            self.gl_widget.setCameraPosition(distance=distance, elevation=30, azimuth=45)
            # Set the camera center separately
            self.gl_widget.opts["center"] = Vector(center[0], center[1], center[2])
            # Update grid size to match the fixed bounds
            grid_size_x = size_x
            grid_size_y = size_y
            grid_spacing = 20  # Fixed spacing for better readability
            # Update the grid
            if hasattr(self, "grid_item"):
                self.grid_item.setSize(grid_size_x, grid_size_y)
                self.grid_item.setSpacing(grid_spacing, grid_spacing)
        except Exception as e:
            print(f"Error adjusting camera: {e}")

    def on_lam_cost_changed(self, input_widget):
        """Handle lambda cost input changes"""
        # Extract the new value from the input widget
        new_value = input_widget.text()
        try:
            # Convert the new value to a float
            lam_cost_value = float(new_value)
            self.worker.problem.settings.scp.lam_cost = lam_cost_value
            # Update the display with scientific notation
            input_widget.setText(f"{lam_cost_value:.2E}")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    def on_lam_tr_changed(self, input_widget):
        """Handle lambda trust region input changes"""
        # Extract the new value from the input widget
        new_value = input_widget.text()
        try:
            # Convert the new value to a float
            lam_tr_value = float(new_value)
            self.worker.problem.settings.scp.w_tr = lam_tr_value
            # Update the display with scientific notation
            input_widget.setText(f"{lam_tr_value:.2E}")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    def update_optimization_metrics(self, results):
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
        self.iter_label.setText(f"Iteration: {iter_num}")
        self.j_tr_label.setText(f"J_tr: {j_tr:.2E}")
        self.j_vb_label.setText(f"J_vb: {j_vb:.2E}")
        self.j_vc_label.setText(f"J_vc: {j_vc:.2E}")
        self.objective_label.setText(f"Objective: {cost:.2E}")
        self.lam_cost_display_label.setText(
            f"λ_cost: {self.worker.problem.settings.scp.lam_cost:.2E}"
        )
        self.dis_time_label.setText(f"Dis Time: {dis_time:.1f}ms")
        self.solve_time_label.setText(f"Solve Time: {solve_time:.1f}ms")
        self.status_label.setText(f"Status: {status}")

    def update_metrics(self, metrics):
        """Update the metrics display - kept for compatibility"""
        # This method is kept for compatibility but the main update is done
        # in update_optimization_metrics
        pass

    def reset_gate_positions(self):
        """Reset gate positions to their original values"""
        for i in range(10):
            original_center = initial_gate_centers[i]
            self.worker.update_gate_position(
                i, original_center[0], original_center[1], original_center[2]
            )
            # Update text inputs
            if i < len(self.gate_controls):
                x_input, y_input, z_input = self.gate_controls[i]
                x_input.setText(f"{original_center[0]:.2f}")
                y_input.setText(f"{original_center[1]:.2f}")
                z_input.setText(f"{original_center[2]:.2f}")

    def reset_camera_view(self):
        """Reset the camera view to default"""
        self.gl_widget.setCameraPosition(distance=150, elevation=30, azimuth=45)

    def closeEvent(self, event):
        """Clean up when closing the window"""
        self.worker.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DroneRacingGUI()
    window.show()
    sys.exit(app.exec_())
