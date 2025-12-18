"""Trajectory visualization and plotting utilities.

This module provides visualization utilities for trajectory optimization results:

- **2D plots** (plotly-based): Time series, projections, heatmaps
  - Direct imports: `from openscvx.plotting import plot_projections_2d, plot_vector_norm`

- **3D visualization** (viser-based): Interactive trajectory animation
  - Submodule import: `from openscvx.plotting import viser`
  - See `openscvx.plotting.viser` for composable 3D primitives

Example:
    # 2D plots
    from openscvx.plotting import plot_projections_2d, plot_vector_norm
    plot_projections_2d(results, velocity_var_name="velocity").show()

    # 3D visualization
    from openscvx.plotting import viser
    server = viser.create_server(positions)
    viser.add_gates(server, gate_vertices)
    server.sleep_forever()
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
