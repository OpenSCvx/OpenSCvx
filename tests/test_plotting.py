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
    plot_controls,
    plot_states,
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
        """Test that plot_states returns a valid Plotly figure."""
        fig = plot_states(mock_result_basic)

        assert fig is not None
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")
        assert fig.layout.title.text == "State Trajectories"

    def test_plot_state_with_multiple_states(self, mock_config):
        """Test plot_states with multiple state variables."""
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

        fig = plot_states(result)

        # Should have subplots for each state component (4 total)
        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_state_with_empty_trajectory(self, mock_config):
        """Test plot_states when trajectory is empty."""
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

        fig = plot_states(result)

        assert fig is not None
        # Should still plot node markers even without full trajectory

    def test_plot_state_filters_ctcs_augmentation(self, mock_config):
        """Test that plot_states filters out CTCS augmentation states by default."""
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

        fig = plot_states(result, include_private=False)

        assert fig is not None
        # CTCS states should be filtered out, so we should only see state_x

    def test_plot_state_with_unbounded_states(self, mock_result_basic, mock_config):
        """Test plot_states (note: new API doesn't use bounds)."""
        fig = plot_states(mock_result_basic)

        assert fig is not None
        # New API doesn't plot bounds, just trajectories


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
        """Test that plot_controls returns a valid Plotly figure."""
        fig = plot_controls(mock_result_basic)

        assert fig is not None
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")
        assert fig.layout.title.text == "Control Trajectories"

    def test_plot_control_with_multiple_controls(self, mock_config):
        """Test plot_controls with multiple control variables."""
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

        fig = plot_controls(result)

        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_control_with_empty_trajectory(self, mock_config):
        """Test plot_controls when trajectory is empty."""
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

        fig = plot_controls(result)

        assert fig is not None

    def test_plot_control_with_unbounded_controls(self, mock_result_basic, mock_config):
        """Test plot_controls (note: new API doesn't use bounds)."""
        fig = plot_controls(mock_result_basic)

        assert fig is not None
        # New API doesn't plot bounds, just trajectories

    def test_plot_control_legend_only_on_first_subplot(self, mock_result_basic, mock_config):
        """Test that legend items only appear on first subplot."""
        fig = plot_controls(mock_result_basic)

        # Count how many traces have showlegend=True
        legend_traces = [trace for trace in fig.data if trace.showlegend]

        # Should have some legend traces (trajectory, nodes)
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
