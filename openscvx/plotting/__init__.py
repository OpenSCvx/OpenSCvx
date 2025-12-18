"""Trajectory visualization and plotting utilities.

This module provides reusable building blocks for visualizing trajectory
optimization results. It is intentionally minimal - we provide common utilities
that can be composed together, not a complete solution that tries to do
everything for you.

**2D Plots** (plotly-based):
    Direct imports for time series, projections, and heatmaps::

        from openscvx.plotting import plot_projections_2d, plot_vector_norm
        plot_vector_norm(results, "thrust", bounds=(rho_min, rho_max)).show()

**3D Visualization** (viser-based):
    The ``viser`` submodule provides composable primitives for building
    interactive 3D visualizations. See ``openscvx.plotting.viser`` for details::

        from openscvx.plotting import viser
        server = viser.create_server(positions)
        viser.add_gates(server, gate_vertices)
        server.sleep_forever()

For problem-specific visualization examples (drones, rockets, etc.), see
``examples/plotting_viser.py``.
"""

from . import viser
from .plotting import (
    plot_control,
    plot_projections_2d,
    plot_state,
    plot_trust_region_heatmap,
    plot_vector_norm,
    plot_virtual_control_heatmap,
)

__all__ = [
    # 2D plotting functions (plotly)
    "plot_state",
    "plot_control",
    "plot_projections_2d",
    "plot_vector_norm",
    "plot_trust_region_heatmap",
    "plot_virtual_control_heatmap",
    # 3D visualization submodule (viser)
    "viser",
]
