import threading
import time
import numpy as np
import sys
import os
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QGraphicsEllipseItem, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QGroupBox, QPushButton, QButtonGroup, QLineEdit, QGridLayout
from PyQt5.QtCore import QTimer, Qt
from pyqtgraph.Qt import QtGui

# Import PyQtGraph OpenGL modules
try:
    from pyqtgraph.opengl import GLViewWidget, GLGridItem, GLScatterPlotItem, GLMeshItem, MeshData
    HAS_OPENGL = True
except ImportError:
    print("PyQtGraph OpenGL not available, falling back to 2D")
    HAS_OPENGL = False

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from drone.obstacle_avoidance_realtime_test import (
    x, u, obs_center_1, obs_center_2, obs_center_3, problem, plotting_dict
)
from openscvx.utils import generate_orthogonal_unit_vectors

running = {'stop': False}
latest_results = {'results': None}
new_result_event = threading.Event()

class Obstacle3DPlotWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Generate axes and radii as in obstacle_avoidance.py (only one call per obstacle)
        np.random.seed(0)
        self.ellipsoid_axes = []
        self.ellipsoid_radii = []
        for _ in range(3):
            ax = generate_orthogonal_unit_vectors()
            self.ellipsoid_axes.append(ax)
            rad = np.random.rand(3) + 0.1 * np.ones(3)
            self.ellipsoid_radii.append(rad)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        if HAS_OPENGL:
            # Create 3D view
            self.view = GLViewWidget()
            self.view.setCameraPosition(distance=15)
            
            # Add grid
            grid = GLGridItem()
            self.view.addItem(grid)
            
            # Add trajectory scatter plot
            self.traj_scatter = GLScatterPlotItem(pos=np.zeros((1, 3)), color=(0, 0, 1, 1), size=5)
            self.view.addItem(self.traj_scatter)
            
            # Create main layout with view and control panel
            main_layout = QHBoxLayout()
            
            # Create control panel
            self.create_control_panel()
            
            # Create obstacle ellipsoids
            self.obs_ellipsoids = []
            self.create_obstacle_ellipsoids()
            
            # Add widgets to main layout
            main_layout.addWidget(self.view, stretch=3)
            main_layout.addWidget(self.control_panel, stretch=1)
            
            layout.addLayout(main_layout)
        else:
            # Fallback to 2D
            label = QLabel("3D OpenGL not available")
            layout.addWidget(label)

    def create_control_panel(self):
        """Create the control panel with sliders for each obstacle"""
        self.control_panel = QWidget()
        control_layout = QVBoxLayout()
        self.control_panel.setLayout(control_layout)
        
        # Title
        title = QLabel("3D Obstacle Avoidance Control")
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
        for label in [self.iter_label, self.j_tr_label, self.j_vb_label, self.j_vc_label, self.objective_label, self.lam_cost_display_label, self.dis_time_label, self.solve_time_label, self.status_label]:
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
        lam_cost_input.returnPressed.connect(lambda: self.on_lam_cost_changed(lam_cost_input))
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
        lam_tr_input.returnPressed.connect(lambda: self.on_lam_tr_changed(lam_tr_input))
        lam_tr_label = QLabel("λ_tr:")
        lam_tr_label.setAlignment(Qt.AlignLeft)
        
        lam_tr_layout.addWidget(lam_tr_input)
        lam_tr_layout.addWidget(lam_tr_label)
        lam_tr_layout.addStretch()  # Push everything to the left
        weights_layout.addLayout(lam_tr_layout)
        
        control_layout.addWidget(weights_group)
        
        # Sliders for each obstacle
        for i in range(3):
            obs_group = QGroupBox(f"Obstacle {i+1} Position")
            obs_layout = QVBoxLayout()
            obs_group.setLayout(obs_layout)
            
            # X, Y, Z sliders
            sliders = []
            for j, coord in enumerate(['X', 'Y', 'Z']):
                slider_layout = QHBoxLayout()
                label = QLabel(f"{coord}:")
                slider = QSlider(Qt.Horizontal)
                slider.setRange(-100, 100)
                slider.setValue(0)
                value_label = QLabel("0.00")
                
                # Connect slider to update function
                slider.valueChanged.connect(lambda val, obs=i, axis=j, label=value_label: self.on_slider_changed(val, obs, axis, label))
                
                slider_layout.addWidget(label)
                slider_layout.addWidget(slider)
                slider_layout.addWidget(value_label)
                obs_layout.addLayout(slider_layout)
                sliders.append((slider, value_label))
            
            # Store sliders for this obstacle
            setattr(self, f'obs_{i}_sliders', sliders)
            control_layout.addWidget(obs_group)
        
        control_layout.addStretch()

    def create_obstacle_ellipsoids(self):
        if not HAS_OPENGL:
            return
        for i, (ax, rad) in enumerate(zip(self.ellipsoid_axes, self.ellipsoid_radii)):
            # Create main ellipsoid
            mesh = MeshData.sphere(rows=20, cols=20, radius=1.0)
            verts = mesh.vertexes()
            verts = verts * 1/(rad)  # scale to ellipsoid
            verts = verts @ ax.T  # rotate by axes
            mesh.setVertexes(verts)
            ellipsoid = GLMeshItem(
                meshdata=mesh,
                color=(0, 1, 0, 0.3),  # RGBA, green, transparent
                shader='shaded',
                smooth=True
            )
            ellipsoid.setGLOptions('translucent')  # Enable transparency
            # Set initial position using translate
            ellipsoid.translate(0, 0, 0)
            self.obs_ellipsoids.append(ellipsoid)
            self.view.addItem(ellipsoid)

    def on_slider_changed(self, value, obstacle_idx, axis, label):
        """Handle slider value changes"""
        # Convert slider value (-100 to 100) to world coordinates (-5 to 5)
        world_value = value * 0.05
        
        # Update the parameter
        centers = [obs_center_1.value, obs_center_2.value, obs_center_3.value]
        center = centers[obstacle_idx].copy()
        center[axis] = world_value
        
        if obstacle_idx == 0:
            obs_center_1.value = center
        elif obstacle_idx == 1:
            obs_center_2.value = center
        else:
            obs_center_3.value = center
        
        # Update visualization
        self.update_obstacle_position(obstacle_idx)
        
        # Update label
        label.setText(f"{world_value:.2f}")

    def update_obstacle_position(self, obstacle_idx):
        """Update obstacle position in 3D view"""
        if not HAS_OPENGL:
            return
            
        centers = [obs_center_1.value, obs_center_2.value, obs_center_3.value]
        center = centers[obstacle_idx]
        
        # Update ellipsoid position
        ellipsoid = self.obs_ellipsoids[obstacle_idx]
        ellipsoid.resetTransform()
        ellipsoid.translate(center[0], center[1], center[2])

    def update_slider_values(self, obstacle_idx):
        """Update slider values to match current obstacle position"""
        centers = [obs_center_1.value, obs_center_2.value, obs_center_3.value]
        center = centers[obstacle_idx]
        
        sliders = getattr(self, f'obs_{obstacle_idx}_sliders')
        for i, (slider, label) in enumerate(sliders):
            # Convert world coordinates to slider values
            slider_value = int(center[i] / 0.05)
            slider.setValue(slider_value)
            label.setText(f"{center[i]:.2f}")

    def on_lam_cost_changed(self, input_widget):
        """Handle lambda cost input changes"""
        # Extract the new value from the input widget
        new_value = input_widget.text()
        try:
            # Convert the new value to a float
            lam_cost_value = float(new_value)
            problem.settings.scp.lam_cost = lam_cost_value
            # Update the display
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
            problem.settings.scp.w_tr = lam_tr_value
            # Update the display
            input_widget.setText(f"{lam_tr_value:.2E}")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    def update_optimization_metrics(self, results):
        """Update the optimization metrics display"""
        if results is None:
            return
            
        # Extract metrics from results
        iter_num = results.get('iter', 0)
        j_tr = results.get('J_tr', 0.0)
        j_vb = results.get('J_vb', 0.0)
        j_vc = results.get('J_vc', 0.0)
        cost = results.get('cost', 0.0)
        status = results.get('prob_stat', '--')
        
        # Get timing information (these would need to be tracked separately)
        dis_time = results.get('dis_time', 0.0)
        solve_time = results.get('solve_time', 0.0)
        
        # Update labels
        self.iter_label.setText(f"Iteration: {iter_num}")
        self.j_tr_label.setText(f"J_tr: {j_tr:.2e}")
        self.j_vb_label.setText(f"J_vb: {j_vb:.2e}")
        self.j_vc_label.setText(f"J_vc: {j_vc:.2e}")
        self.objective_label.setText(f"Objective: {cost:.2e}")
        self.lam_cost_display_label.setText(f"λ_cost: {problem.settings.scp.lam_cost:.2E}")
        self.dis_time_label.setText(f"Dis Time: {dis_time:.1f}ms")
        self.solve_time_label.setText(f"Solve Time: {solve_time:.1f}ms")
        self.status_label.setText(f"Status: {status}")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

def optimization_loop():
    problem.initialize()
    try:
        while not running['stop']:
            # Warm start: set guess to last solution if available
            if latest_results['results'] is not None:
                problem.settings.sim.x.guess = latest_results['results']['x'].guess
                problem.settings.sim.u.guess = latest_results['results']['u'].guess
            
            # Perform a single SCP step
            results = problem.step()
            
            # Add timing information to results
            results['iter'] = problem.scp_k - 1
            results['J_tr'] = problem.scp_J_tr
            results['J_vb'] = problem.scp_J_vb
            results['J_vc'] = problem.scp_J_vc
            
            # Get timing from the print queue (emitted data)
            try:
                if hasattr(problem, 'print_queue') and not problem.print_queue.empty():
                    # Get the latest emitted data
                    emitted_data = problem.print_queue.get_nowait()
                    results['dis_time'] = emitted_data.get('dis_time', 0.0)
                    results['solve_time'] = emitted_data.get('subprop_time', 0.0)
                    results['prob_stat'] = emitted_data.get('prob_stat', '--')
                    results['cost'] = emitted_data.get('cost', 0.0)
                else:
                    results['dis_time'] = 0.0
                    results['solve_time'] = 0.0
                    results['prob_stat'] = '--'
                    results['cost'] = 0.0
            except:
                results['dis_time'] = 0.0
                results['solve_time'] = 0.0
                results['prob_stat'] = '--'
                results['cost'] = 0.0
            
            # Optionally skip post_process for speed
            # results = problem.post_process(results)
            results.update(plotting_dict)
            latest_results['results'] = results
            new_result_event.set()
    except KeyboardInterrupt:
        running['stop'] = True
        print("Stopped by user.")

def plot_thread_func():
    # Initialize PyQtGraph
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    print(f"Creating plot window... OpenGL available: {HAS_OPENGL}")
    
    # Create 3D plot window
    plot_widget = Obstacle3DPlotWidget()
    plot_widget.setWindowTitle('3D Obstacle Avoidance Real-time Trajectory')
    plot_widget.resize(800, 600)  # Set explicit size
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
        if latest_results['results'] is not None:
            try:
                V_multi_shoot = np.array(latest_results['results']['V_multi_shoot'])
                
                # Extract 3D position data (first 3 elements of state)
                n_x = problem.settings.sim.n_states
                n_u = problem.settings.sim.n_controls
                i1 = n_x
                i2 = i1 + n_x * n_x
                i3 = i2 + n_x * n_u
                i4 = i3 + n_x * n_u
                i5 = i4 + n_x
                
                all_pos_segments = []
                for i_node in range(V_multi_shoot.shape[1]):
                    node_data = V_multi_shoot[:, i_node]
                    segments_for_node = node_data.reshape(-1, i5)
                    pos_segments = segments_for_node[:, :3]  # 3D positions
                    all_pos_segments.append(pos_segments)
                
                if all_pos_segments:
                    full_traj = np.vstack(all_pos_segments)
                    
                    if HAS_OPENGL:
                        plot_widget.traj_scatter.setData(pos=full_traj)
                        
                        # Update obstacle positions (reset and translate for ellipsoids)
                        centers = [obs_center_1.value, obs_center_2.value, obs_center_3.value]
                        for i, ellipsoid in enumerate(plot_widget.obs_ellipsoids):
                            ellipsoid.resetTransform()
                            ellipsoid.translate(*centers[i])
                    else:
                        # 2D fallback - plot X vs Y
                        plot_widget.traj_curve.setData(full_traj[:, 0], full_traj[:, 1])
                        
                        # Update obstacle positions in 2D
                        plot_widget.obs_scatters[0].setData([obs_center_1.value[0]], [obs_center_1.value[1]])
                        plot_widget.obs_scatters[1].setData([obs_center_2.value[0]], [obs_center_2.value[1]])
                        plot_widget.obs_scatters[2].setData([obs_center_3.value[0]], [obs_center_3.value[1]])
                
                # Update optimization metrics display
                plot_widget.update_optimization_metrics(latest_results['results'])
                    
            except Exception as e:
                print(f"Plot update error: {e}")
                if 'x' in latest_results['results']:
                    x_traj = latest_results['results']['x'].guess
                    if HAS_OPENGL:
                        plot_widget.traj_scatter.setData(pos=x_traj[:, :3])
                    else:
                        plot_widget.traj_curve.setData(x_traj[:, 0], x_traj[:, 1])
    
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