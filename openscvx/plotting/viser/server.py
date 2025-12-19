"""Viser server setup utilities."""

import matplotlib.pyplot as plt
import numpy as np
import viser
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage


def compute_velocity_colors(vel: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """Compute RGB colors based on velocity magnitude.

    Args:
        vel: Velocity array of shape (N, 3)
        cmap_name: Matplotlib colormap name

    Returns:
        Array of RGB colors with shape (N, 3), values in [0, 255]
    """
    vel_norms = np.linalg.norm(vel, axis=1)
    vel_range = vel_norms.max() - vel_norms.min()
    if vel_range < 1e-8:
        vel_normalized = np.zeros_like(vel_norms)
    else:
        vel_normalized = (vel_norms - vel_norms.min()) / vel_range

    cmap = plt.get_cmap(cmap_name)
    colors = np.array([[int(c * 255) for c in cmap(v)[:3]] for v in vel_normalized])
    return colors


def compute_grid_size(pos: np.ndarray, padding: float = 1.2) -> float:
    """Compute grid size based on trajectory extent.

    Args:
        pos: Position array of shape (N, 3)
        padding: Padding factor (1.2 = 20% padding)

    Returns:
        Grid size (width and height)
    """
    max_x = np.abs(pos[:, 0]).max()
    max_y = np.abs(pos[:, 1]).max()
    return max(max_x, max_y) * 2 * padding


def create_server(
    pos: np.ndarray,
    dark_mode: bool = True,
) -> viser.ViserServer:
    """Create a viser server with basic scene setup.

    Args:
        pos: Position array for computing grid size
        dark_mode: Whether to use dark theme

    Returns:
        ViserServer instance with grid and origin frame
    """
    server = viser.ViserServer()

    # Configure theme with OpenSCvx branding
    # TitlebarButton and TitlebarConfig are TypedDict classes (create as plain dicts)
    buttons = (
        TitlebarButton(
            text="Getting Started",
            icon="Description",
            href="https://haynec.github.io/OpenSCvx/latest/getting-started/",
        ),
        TitlebarButton(
            text="Docs",
            icon="Description",
            href="https://haynec.github.io/OpenSCvx/",
        ),
        TitlebarButton(
            text="GitHub",
            icon="GitHub",
            href="https://github.com/haynec/OpenSCvx",
        ),
    )

    # Add OpenSCvx logo to titlebar (loaded from GitHub)
    logo_url = (
        "https://raw.githubusercontent.com/haynec/OpenSCvx/main/figures/openscvx_logo_square.png"
    )
    image = TitlebarImage(
        image_url_light=logo_url,
        image_url_dark=logo_url,  # Use same logo for both themes
        image_alt="OpenSCvx",
        href="https://github.com/haynec/OpenSCvx",
    )

    titlebar_config = TitlebarConfig(buttons=buttons, image=image)

    server.gui.configure_theme(
        titlebar_content=titlebar_config,
        dark_mode=dark_mode,
    )

    grid_size = compute_grid_size(pos)
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
