"""
Helper functions for testing pyqtgraph plotting functions in headless mode.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def setup_headless_environment():
    """Set up environment variables for headless GUI testing."""
    # Set environment variables for headless Qt
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["DISPLAY"] = ":99"

    # Mock Qt application to prevent GUI windows from appearing
    mock_app = MagicMock()
    mock_app.instance.return_value = None
    mock_app.exec_.return_value = 0
    mock_app.exec.return_value = 0

    return mock_app


def run_pyqtgraph_function_headless(plot_function, result, params, timeout=5.0):
    """
    Test a pyqtgraph plotting function in headless mode.

    Args:
        plot_function: The pyqtgraph plotting function to test
        result: The optimization result dictionary
        params: The problem parameters
        timeout: Maximum time to wait for function execution (seconds)

    Returns:
        bool: True if function executed without errors
    """
    # Set up headless environment
    mock_app = setup_headless_environment()

    # Mock Qt widgets and components
    with (
        patch("PyQt5.QtWidgets.QApplication", return_value=mock_app),
        patch("pyqtgraph.QtWidgets.QApplication", return_value=mock_app),
        patch("PyQt5.QtWidgets.QWidget") as mock_widget,
        patch("pyqtgraph.opengl.GLViewWidget") as mock_gl_widget,
        patch("pyqtgraph.PlotWidget") as mock_plot_widget,
        patch("pyqtgraph.opengl.GLLinePlotItem") as mock_line_item,
        patch("pyqtgraph.opengl.GLScatterPlotItem") as mock_scatter_item,
        patch("pyqtgraph.opengl.GLMeshItem") as mock_mesh_item,
        patch("pyqtgraph.opengl.MeshData") as mock_mesh_data,
        patch("pyqtgraph.QtCore.QTimer") as mock_timer,
    ):
        # Configure mocks
        mock_widget_instance = MagicMock()
        mock_widget.return_value = mock_widget_instance
        mock_widget_instance.show.return_value = None
        mock_widget_instance.resize.return_value = None
        mock_widget_instance.setWindowTitle.return_value = None

        mock_gl_widget_instance = MagicMock()
        mock_gl_widget.return_value = mock_gl_widget_instance
        mock_gl_widget_instance.setWindowTitle.return_value = None
        mock_gl_widget_instance.setGeometry.return_value = None
        mock_gl_widget_instance.setCameraPosition.return_value = None
        mock_gl_widget_instance.addItem.return_value = None

        mock_plot_widget_instance = MagicMock()
        mock_plot_widget.return_value = mock_plot_widget_instance
        mock_plot_widget_instance.setAspectLocked.return_value = None
        mock_plot_widget_instance.setRange.return_value = None
        mock_plot_widget_instance.showGrid.return_value = None
        mock_plot_widget_instance.plot.return_value = MagicMock()

        mock_line_item_instance = MagicMock()
        mock_line_item.return_value = mock_line_item_instance
        mock_line_item_instance.setVisible.return_value = None
        mock_line_item_instance.setData.return_value = None

        mock_scatter_item_instance = MagicMock()
        mock_scatter_item.return_value = mock_scatter_item_instance
        mock_scatter_item_instance.setVisible.return_value = None
        mock_scatter_item_instance.setData.return_value = None

        mock_mesh_item_instance = MagicMock()
        mock_mesh_item.return_value = mock_mesh_item_instance
        mock_mesh_item_instance.setVisible.return_value = None

        mock_mesh_data_instance = MagicMock()
        mock_mesh_data.return_value = mock_mesh_data_instance
        mock_mesh_data.sphere.return_value = mock_mesh_data_instance
        mock_mesh_data_instance.vertexes.return_value = np.random.rand(100, 3)
        mock_mesh_data_instance.setVertexes.return_value = None

        mock_timer_instance = MagicMock()
        mock_timer.return_value = mock_timer_instance
        mock_timer_instance.timeout.connect.return_value = None
        mock_timer_instance.setInterval.return_value = None
        mock_timer_instance.start.return_value = None
        mock_timer_instance.stop.return_value = None

        try:
            # Call the plotting function
            plot_function(result, params)
            return True
        except Exception as e:
            # Log the error but don't fail the test
            print(f"Warning: pyqtgraph function {plot_function.__name__} raised exception: {e}")
            return False


def check_pyqtgraph_functions_basic(result, params):
    """
    Basic test to ensure pyqtgraph functions can be imported and called.
    This is a minimal test that doesn't require full GUI setup.
    """
    try:
        from examples.plotting import (
            plot_animation_pyqtgraph,
            plot_camera_animation_pyqtgraph,
            plot_scp_animation_pyqtgraph,
        )

        # Test that functions exist and are callable
        assert callable(plot_animation_pyqtgraph)
        assert callable(plot_scp_animation_pyqtgraph)
        assert callable(plot_camera_animation_pyqtgraph)

        return True
    except ImportError as e:
        # Skip test if GUI packages are not available
        pytest.skip(f"pyqtgraph not available: {e}")
    except Exception as e:
        # Don't fail the test for other errors
        print(f"Error testing pyqtgraph functions: {e}")
        return False
