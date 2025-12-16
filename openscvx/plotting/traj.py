import matplotlib.pyplot as plt
import numpy as np
import viser


def create_plotting_server(results: dict):
    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)

    # Calculate grid size based on trajectory extent
    pos = results.trajectory["position"]
    max_x = np.abs(pos[:, 0]).max()
    max_y = np.abs(pos[:, 1]).max()

    # Use the larger dimension and add 20% padding
    grid_size = max(max_x, max_y) * 2 * 1.2

    server.scene.add_grid(
        "/grid",
        width=grid_size,
        height=grid_size,
        position=np.array([0.0, 0.0, 0.0]),
    )
    server.scene.add_frame(
        "/origin",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
    )

    return server


def add_velocity_trace(server: viser._viser.ViserServer, results: dict):
    pos = results.trajectory["position"]
    vel = results.trajectory["velocity"]
    # Normalize your velocity magnitudes to [0, 1]
    vel_norms = np.linalg.norm(vel, axis=1)
    vel_normalized = (vel_norms - vel_norms.min()) / (vel_norms.max() - vel_norms.min())

    # Get the colormap
    cmap = plt.get_cmap("viridis")

    # Map each normalized velocity to an RGB color
    colors = [cmap(v) for v in vel_normalized]  # Returns RGBA tuples in [0, 1]

    rgb = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, a in colors]
    server.scene.add_point_cloud("/traj", points=pos, colors=rgb)
