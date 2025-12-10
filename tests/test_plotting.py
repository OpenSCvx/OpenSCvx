"""
Unit tests for plotting functions.

Tests the plotting functions:
- plot_state: Plot state trajectories over time with bounds
- plot_control: Plot control trajectories over time with bounds
- plot_scp_iteration_animation: Create animated plot showing SCP iteration convergence
"""

from unittest.mock import Mock

import numpy as np
import pytest

from openscvx.algorithms import OptimizationResults
from openscvx.config import Config
from openscvx.plotting.plotting import (
    plot_control,
    plot_scp_iteration_animation,
    plot_state,
)


class TestPlotStateFunction:
    """Test suite for plot_state function."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object with required attributes."""
        config = Mock(spec=Config)
        config.sim = Mock()
        config.sim.x = Mock()
        config.sim.x.min = np.array([-10.0, -10.0, -10.0])
        config.sim.x.max = np.array([10.0, 10.0, 10.0])
        config.sim.n_states = 3
        return config

    @pytest.fixture
    def mock_result_basic(self):
        """Create a basic mock OptimizationResults object."""
        result = Mock(spec=OptimizationResults)

        # Mock nodes dictionary
        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "state_x": np.random.randn(10, 3),
        }

        # Mock trajectory
        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "state_x": np.random.randn(100, 3),
        }

        # Mock states
        state1 = Mock()
        state1.name = "state_x"
        state1._slice = slice(0, 3)
        result._states = [state1]
        result._controls = []

        return result

    def test_plot_state_returns_figure(self, mock_result_basic, mock_config):
        """Test that plot_state returns a valid Plotly figure."""
        fig = plot_state(mock_result_basic, mock_config)

        assert fig is not None
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")
        assert fig.layout.title.text == "State Trajectories"

    def test_plot_state_with_multiple_states(self, mock_config):
        """Test plot_state with multiple state variables."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "position": np.random.randn(10, 2),
            "velocity": np.random.randn(10, 2),
        }

        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "position": np.random.randn(100, 2),
            "velocity": np.random.randn(100, 2),
        }

        pos_state = Mock()
        pos_state.name = "position"
        pos_state._slice = slice(0, 2)

        vel_state = Mock()
        vel_state.name = "velocity"
        vel_state._slice = slice(2, 4)

        result._states = [pos_state, vel_state]
        result._controls = []

        # Update config to match state dimensions
        mock_config.sim.x.min = np.array([-10.0] * 4)
        mock_config.sim.x.max = np.array([10.0] * 4)
        mock_config.sim.n_states = 4

        fig = plot_state(result, mock_config)

        # Should have subplots for each state component (4 total)
        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_state_with_empty_trajectory(self, mock_config):
        """Test plot_state when trajectory is empty."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "state_x": np.random.randn(10, 3),
        }

        result.trajectory = {}  # Empty trajectory

        state = Mock()
        state.name = "state_x"
        state._slice = slice(0, 3)
        result._states = [state]
        result._controls = []

        fig = plot_state(result, mock_config)

        assert fig is not None
        # Should still plot node markers even without full trajectory

    def test_plot_state_filters_ctcs_augmentation(self, mock_config):
        """Test that plot_state filters out CTCS augmentation states."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "state_x": np.random.randn(10, 2),
            "_ctcs_aug_0": np.random.randn(10, 1),
        }

        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "state_x": np.random.randn(100, 2),
            "_ctcs_aug_0": np.random.randn(100, 1),
        }

        state = Mock()
        state.name = "state_x"
        state._slice = slice(0, 2)

        aug_state = Mock()
        aug_state.name = "_ctcs_aug_0"
        aug_state._slice = slice(2, 3)

        result._states = [state, aug_state]
        result._controls = []

        fig = plot_state(result, mock_config)

        assert fig is not None
        # CTCS states should be filtered out, so we should only see state_x

    def test_plot_state_with_unbounded_states(self, mock_result_basic, mock_config):
        """Test plot_state with infinite bounds (unbounded states)."""
        mock_config.sim.x.min = np.array([-np.inf, -10.0, -10.0])
        mock_config.sim.x.max = np.array([np.inf, 10.0, 10.0])

        fig = plot_state(mock_result_basic, mock_config)

        assert fig is not None
        # Should handle infinite bounds gracefully


class TestPlotControlFunction:
    """Test suite for plot_control function."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object with required attributes."""
        config = Mock(spec=Config)
        config.sim = Mock()
        config.sim.u = Mock()
        config.sim.u.min = np.array([-5.0, -5.0])
        config.sim.u.max = np.array([5.0, 5.0])
        config.sim.n_controls = 2
        return config

    @pytest.fixture
    def mock_result_basic(self):
        """Create a basic mock OptimizationResults object."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "control_u": np.random.randn(10, 2),
        }

        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "control_u": np.random.randn(100, 2),
        }

        control = Mock()
        control.name = "control_u"
        control._slice = slice(0, 2)
        result._controls = [control]
        result._states = []

        return result

    def test_plot_control_returns_figure(self, mock_result_basic, mock_config):
        """Test that plot_control returns a valid Plotly figure."""
        fig = plot_control(mock_result_basic, mock_config)

        assert fig is not None
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")
        assert fig.layout.title.text == "Control Trajectories"

    def test_plot_control_with_multiple_controls(self, mock_config):
        """Test plot_control with multiple control variables."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "thrust": np.random.randn(10, 2),
            "torque": np.random.randn(10, 1),
        }

        result.trajectory = {
            "time": np.linspace(0, 1, 100).reshape(-1, 1),
            "thrust": np.random.randn(100, 2),
            "torque": np.random.randn(100, 1),
        }

        thrust_control = Mock()
        thrust_control.name = "thrust"
        thrust_control._slice = slice(0, 2)

        torque_control = Mock()
        torque_control.name = "torque"
        torque_control._slice = slice(2, 3)

        result._controls = [thrust_control, torque_control]
        result._states = []

        mock_config.sim.u.min = np.array([-10.0] * 3)
        mock_config.sim.u.max = np.array([10.0] * 3)

        fig = plot_control(result, mock_config)

        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_control_with_empty_trajectory(self, mock_config):
        """Test plot_control when trajectory is empty."""
        result = Mock(spec=OptimizationResults)

        result.nodes = {
            "time": np.linspace(0, 1, 10).reshape(-1, 1),
            "control_u": np.random.randn(10, 2),
        }

        result.trajectory = {}  # Empty trajectory

        control = Mock()
        control.name = "control_u"
        control._slice = slice(0, 2)
        result._controls = [control]
        result._states = []

        fig = plot_control(result, mock_config)

        assert fig is not None

    def test_plot_control_with_unbounded_controls(self, mock_result_basic, mock_config):
        """Test plot_control with infinite bounds (unbounded controls)."""
        mock_config.sim.u.min = np.array([-np.inf, -5.0])
        mock_config.sim.u.max = np.array([np.inf, 5.0])

        fig = plot_control(mock_result_basic, mock_config)

        assert fig is not None
        # Should handle infinite bounds gracefully

    def test_plot_control_legend_only_on_first_subplot(self, mock_result_basic, mock_config):
        """Test that legend items only appear on first subplot."""
        fig = plot_control(mock_result_basic, mock_config)

        # Count how many traces have showlegend=True
        legend_traces = [trace for trace in fig.data if trace.showlegend]

        # Should have some legend traces (propagated, nodes, bounds)
        assert len(legend_traces) > 0


class TestPlotSCPIterationAnimationFunction:
    """Test suite for plot_scp_iteration_animation function."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object with required attributes."""
        config = Mock(spec=Config)
        config.sim = Mock()
        config.sim.n_states = 3  # time + 2 state variables
        config.sim.n_controls = 1
        config.sim.x = Mock()
        # Bounds for all state indices (including time at index 0)
        config.sim.x.min = np.array([-np.inf, -10.0, -10.0])  # index 0=time, 1-2=states
        config.sim.x.max = np.array([np.inf, 10.0, 10.0])
        config.sim.u = Mock()
        config.sim.u.min = np.array([-5.0])
        config.sim.u.max = np.array([5.0])
        config.sim.total_time = 1.0
        config.sim.time_slice = slice(0, 1)
        config.sim.true_state_slice_prop = slice(0, 2)
        config.scp = Mock()
        config.scp.n = 5  # Number of SCP nodes
        return config

    @pytest.fixture
    def mock_result_with_animation(self, mock_config):
        """Create a mock OptimizationResults object with animation history."""
        result = Mock(spec=OptimizationResults)

        n_iterations = 3
        n_nodes = 5
        n_x = 3  # Must match params.sim.n_states
        n_u = 1  # Must match params.sim.n_controls
        n_timesteps = 10

        # Create X history (node values across iterations) - [time, state_x, state_y]
        result.X = [
            np.hstack(
                [
                    np.linspace(0, 1, n_nodes).reshape(-1, 1),  # time (slice 0:1)
                    np.random.randn(
                        n_nodes, n_x - 1
                    ),  # state values (slice 1:3) - only 2 actual states
                ]
            )
            for _ in range(n_iterations)
        ]

        # Create U history (control values across iterations)
        result.U = [np.random.randn(n_nodes, n_u) for _ in range(n_iterations)]

        # Create discretization history (V_history)
        N = n_nodes
        i4 = n_x + n_x * n_x + 2 * n_x * n_u
        result.discretization_history = [
            np.random.randn((N - 1) * i4, n_timesteps) for _ in range(n_iterations)
        ]

        # Mock states (slice starts at 1 to skip time column)
        state = Mock()
        state.name = "state_x"
        state._slice = slice(1, 3)  # indices 1-2 for the two state components
        result._states = [state]

        # Mock controls
        control = Mock()
        control.name = "control_u"
        control._slice = slice(0, 1)
        result._controls = [control]

        return result

    def test_plot_scp_animation_returns_figure(self, mock_result_with_animation, mock_config):
        """Test that plot_scp_iteration_animation returns a valid Plotly figure."""
        fig = plot_scp_iteration_animation(mock_result_with_animation, mock_config)

        assert fig is not None
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")
        assert hasattr(fig, "frames")

    def test_plot_scp_animation_has_frames(self, mock_result_with_animation, mock_config):
        """Test that animation has correct number of frames."""
        fig = plot_scp_iteration_animation(mock_result_with_animation, mock_config)

        n_iterations = len(mock_result_with_animation.X)
        assert len(fig.frames) == n_iterations

    def test_plot_scp_animation_has_animation_controls(
        self, mock_result_with_animation, mock_config
    ):
        """Test that animation has play/pause controls."""
        fig = plot_scp_iteration_animation(mock_result_with_animation, mock_config)

        # Check for updatemenus (play/pause buttons)
        assert "updatemenus" in fig.layout
        assert len(fig.layout.updatemenus) > 0

        buttons = fig.layout.updatemenus[0].buttons
        button_labels = [btn.label for btn in buttons]
        assert "Play" in button_labels
        assert "Pause" in button_labels

    def test_plot_scp_animation_has_slider(self, mock_result_with_animation, mock_config):
        """Test that animation has a slider control."""
        fig = plot_scp_iteration_animation(mock_result_with_animation, mock_config)

        # Check for sliders
        assert "sliders" in fig.layout
        assert len(fig.layout.sliders) > 0

        slider = fig.layout.sliders[0]
        n_iterations = len(mock_result_with_animation.X)
        assert len(slider.steps) == n_iterations

    def test_plot_scp_animation_no_iterations_raises_error(self, mock_config):
        """Test that function raises error when no iteration history available."""
        result = Mock(spec=OptimizationResults)
        result.X = []  # Empty iteration history
        result.U = []  # Empty U history
        result.discretization_history = None
        result._states = []
        result._controls = []

        with pytest.raises(ValueError, match="No iteration history available"):
            plot_scp_iteration_animation(result, mock_config)

    def test_plot_scp_animation_with_single_iteration(self, mock_config):
        """Test animation with only one iteration."""
        result = Mock(spec=OptimizationResults)

        n_nodes = 5
        n_x = 3  # Must match mock_config.sim.n_states
        n_u = 1

        # Single iteration
        result.X = [
            np.hstack(
                [np.linspace(0, 1, n_nodes).reshape(-1, 1), np.random.randn(n_nodes, n_x - 1)]
            )
        ]
        result.U = [np.random.randn(n_nodes, n_u)]

        N = n_nodes
        i4 = n_x + n_x * n_x + 2 * n_x * n_u
        result.discretization_history = [np.random.randn((N - 1) * i4, 10)]

        state = Mock()
        state.name = "state_x"
        state._slice = slice(1, 3)
        result._states = [state]

        control = Mock()
        control.name = "control_u"
        control._slice = slice(0, 1)
        result._controls = [control]

        fig = plot_scp_iteration_animation(result, mock_config)

        assert fig is not None
        assert len(fig.frames) == 1

    def test_plot_scp_animation_limits_iterations_to_v_history(self, mock_config):
        """Test that iterations are limited to V_history length."""
        result = Mock(spec=OptimizationResults)

        n_iterations = 5
        n_v_history = 3  # Fewer V_history entries than X
        n_nodes = 5
        n_x = 3  # Must match mock_config.sim.n_states
        n_u = 1
        N = n_nodes
        i4 = n_x + n_x * n_x + 2 * n_x * n_u

        # More X than V_history
        result.X = [
            np.hstack(
                [np.linspace(0, 1, n_nodes).reshape(-1, 1), np.random.randn(n_nodes, n_x - 1)]
            )
            for _ in range(n_iterations)
        ]
        result.U = [np.random.randn(n_nodes, n_u) for _ in range(n_iterations)]
        result.discretization_history = [
            np.random.randn((N - 1) * i4, 10) for _ in range(n_v_history)
        ]

        state = Mock()
        state.name = "state_x"
        state._slice = slice(1, 3)
        result._states = [state]

        control = Mock()
        control.name = "control_u"
        control._slice = slice(0, 1)
        result._controls = [control]

        fig = plot_scp_iteration_animation(result, mock_config)

        # Should be limited to n_v_history
        assert len(fig.frames) == n_v_history

    def test_plot_scp_animation_includes_bounds(self, mock_result_with_animation, mock_config):
        """Test that animation includes bounds visualization."""
        fig = plot_scp_iteration_animation(mock_result_with_animation, mock_config)

        # Check that frames exist and contain data
        assert len(fig.frames) > 0
        first_frame = fig.frames[0]

        # Frame should have traces (trajectories, nodes, bounds)
        assert len(first_frame.data) > 0

    def test_plot_scp_animation_has_subplots(self, mock_result_with_animation, mock_config):
        """Test that animation creates proper subplots."""
        fig = plot_scp_iteration_animation(mock_result_with_animation, mock_config)

        # Should have subplots for states and controls
        assert len(fig.data) > 0  # Initial placeholder traces


class TestPlottingIntegration:
    """Integration tests combining multiple plotting functions."""

    @pytest.fixture
    def complete_mock_result(self):
        """Create a complete mock result with states, controls, and animation history."""
        result = Mock(spec=OptimizationResults)

        n_nodes = 5
        n_iterations = 3

        # States
        result.nodes = {
            "time": np.linspace(0, 1, n_nodes).reshape(-1, 1),
            "position": np.random.randn(n_nodes, 2),
        }

        result.trajectory = {
            "time": np.linspace(0, 1, 50).reshape(-1, 1),
            "position": np.random.randn(50, 2),
        }

        # Controls
        result.nodes["control_u"] = np.random.randn(n_nodes, 1)
        result.trajectory["control_u"] = np.random.randn(50, 1)

        # Animation history
        result.X = [
            np.hstack([np.linspace(0, 1, n_nodes).reshape(-1, 1), np.random.randn(n_nodes, 2)])
            for _ in range(n_iterations)
        ]
        result.U = [np.random.randn(n_nodes, 1) for _ in range(n_iterations)]

        n_x = 2
        n_u = 1
        N = n_nodes
        i4 = n_x + n_x * n_x + 2 * n_x * n_u
        result.discretization_history = [
            np.random.randn((N - 1) * i4, 10) for _ in range(n_iterations)
        ]

        # Variables
        pos_state = Mock()
        pos_state.name = "position"
        pos_state._slice = slice(0, 2)
        result._states = [pos_state]

        control = Mock()
        control.name = "control_u"
        control._slice = slice(0, 1)
        result._controls = [control]

        return result

    @pytest.fixture
    def complete_mock_config(self):
        """Create a complete mock config."""
        config = Mock(spec=Config)
        config.sim = Mock()
        config.sim.x = Mock()
        config.sim.x.min = np.array([-10.0, -10.0])
        config.sim.x.max = np.array([10.0, 10.0])
        config.sim.u = Mock()
        config.sim.u.min = np.array([-5.0])
        config.sim.u.max = np.array([5.0])
        config.sim.n_states = 2
        config.sim.n_controls = 1
        config.sim.total_time = 1.0
        config.sim.time_slice = slice(0, 1)
        config.sim.true_state_slice_prop = slice(0, 2)
        config.sim.n_x = 2
        config.sim.n_u = 1
        config.scp = Mock()
        config.scp.n = 5
        return config

    def test_all_plotting_functions_work_together(self, complete_mock_result, complete_mock_config):
        """Test that all three plotting functions can be called on same result."""
        # Should not raise any exceptions
        fig_state = plot_state(complete_mock_result, complete_mock_config)
        fig_control = plot_control(complete_mock_result, complete_mock_config)
        fig_animation = plot_scp_iteration_animation(complete_mock_result, complete_mock_config)

        assert fig_state is not None
        assert fig_control is not None
        assert fig_animation is not None

    def test_plotting_functions_produce_different_figures(
        self, complete_mock_result, complete_mock_config
    ):
        """Test that different plotting functions produce visually different outputs."""
        fig_state = plot_state(complete_mock_result, complete_mock_config)
        fig_control = plot_control(complete_mock_result, complete_mock_config)
        fig_animation = plot_scp_iteration_animation(complete_mock_result, complete_mock_config)

        # State plot should have state title
        assert "State" in fig_state.layout.title.text

        # Control plot should have control title
        assert "Control" in fig_control.layout.title.text

        # Animation should have SCP in title
        assert (
            "SCP" in fig_animation.layout.title.text
            or "Iteration" in fig_animation.layout.title.text
        )
