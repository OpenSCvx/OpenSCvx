"""
Analytical solution for the brachistochrone problem.

The brachistochrone curve is the path of fastest descent under gravity between
two points. The solution is a cycloid parametrized by:
    x(φ) = R(φ - sin(φ))
    y(φ) = -R(1 - cos(φ))

where R is determined by the boundary conditions and φ is the parameter.
"""

import numpy as np
from scipy.optimize import fsolve


def compute_brachistochrone_parameters(x0, y0, x1, y1, g=9.81):
    """
    Compute the cycloid parameters R and φ_final for the brachistochrone curve.

    The cycloid is parametrized as:
        x(φ) = x0 + R(φ - sin(φ))
        y(φ) = y0 - R(1 - cos(φ))

    Args:
        x0: Initial x position
        y0: Initial y position
        x1: Final x position
        y1: Final y position
        g: Gravitational acceleration (default: 9.81 m/s²)

    Returns:
        tuple: (R, φ_final, T) where:
            - R is the cycloid radius parameter
            - φ_final is the final parameter value
            - T is the optimal time
    """
    # Constraints from boundary conditions
    dx = x1 - x0
    dy = y1 - y0  # Note: dy is negative if descending

    def equations(params):
        """System of equations to solve for R and φ_final."""
        R, phi = params
        eq1 = R * (phi - np.sin(phi)) - dx
        eq2 = -R * (1 - np.cos(phi)) - dy
        return [eq1, eq2]

    # Initial guess: φ ~ π, R ~ dx/π
    phi_guess = np.pi
    R_guess = dx / np.pi

    # Solve for R and φ_final
    solution = fsolve(equations, [R_guess, phi_guess])
    R, phi_final = solution

    # Compute optimal time using the parametric time formula
    # For a cycloid starting from rest: t = sqrt(R/g) * φ
    T = np.sqrt(R / g) * phi_final

    return R, phi_final, T


def evaluate_brachistochrone_trajectory(x0, y0, R, phi_final, g=9.81, num_points=100):
    """
    Evaluate the brachistochrone trajectory at discrete points.

    Args:
        x0: Initial x position
        y0: Initial y position
        R: Cycloid radius parameter
        phi_final: Final parameter value
        g: Gravitational acceleration (default: 9.81 m/s²)
        num_points: Number of points to evaluate

    Returns:
        dict: Dictionary containing:
            - 'phi': Parameter values
            - 'x': x positions
            - 'y': y positions
            - 'v': velocities
            - 't': times
    """
    # Parameter values from 0 to φ_final
    phi = np.linspace(0, phi_final, num_points)

    # Position along cycloid
    x = x0 + R * (phi - np.sin(phi))
    y = y0 - R * (1 - np.cos(phi))

    # Velocity magnitude: v = sqrt(2*g*R*(1 - cos(φ)))
    v = np.sqrt(2 * g * R * (1 - np.cos(phi)))

    # Time: t = sqrt(R/g) * φ
    t = np.sqrt(R / g) * phi

    return {
        "phi": phi,
        "x": x,
        "y": y,
        "v": v,
        "t": t,
    }


def compute_cycloid_arc_length(R, phi_final):
    """
    Compute the arc length of a cycloid curve.

    For a cycloid parametrized as:
        x(φ) = R(φ - sin(φ))
        y(φ) = -R(1 - cos(φ))

    The arc length from φ=0 to φ=φ_final is:
        s = ∫[0 to φ_final] sqrt((dx/dφ)² + (dy/dφ)²) dφ
        s = ∫[0 to φ_final] sqrt(R²(1 - cos(φ))² + R²sin²(φ)) dφ
        s = ∫[0 to φ_final] R*sqrt(2(1 - cos(φ))) dφ
        s = ∫[0 to φ_final] 2R*sin(φ/2) dφ
        s = 4R[1 - cos(φ_final/2)]

    Args:
        R: Cycloid radius parameter
        phi_final: Final parameter value

    Returns:
        float: Arc length of the cycloid
    """
    return 4 * R * (1 - np.cos(phi_final / 2))


def compare_trajectory_to_analytical(
    t,
    position,
    velocity,
    x0,
    y0,
    x1,
    y1,
    g=9.81,
):
    """
    Compare a numerical trajectory to the analytical brachistochrone solution.

    Args:
        t: Time array, shape (N,)
        position: Position array, shape (N, 2) with columns [x, y]
        velocity: Velocity array, shape (N,) or (N, 1)
        x0: Initial x position
        y0: Initial y position
        x1: Final x position
        y1: Final y position
        g: Gravitational acceleration (default: 9.81 m/s²)

    Returns:
        dict: Comparison metrics including:
            - 'analytical_time': Optimal time from analytical solution
            - 'numerical_time': Final time from numerical solution
            - 'time_error_pct': Percentage error in final time
            - 'position_rmse': RMS error in position
            - 'position_max_error': Maximum position error
            - 'velocity_rmse': RMS error in velocity
            - 'analytical_trajectory': Analytical trajectory data
            - 'arc_length': Analytical arc length of cycloid
    """
    # Extract x, y from position array
    x_num = position[:, 0]
    y_num = position[:, 1]

    # Flatten velocity if needed
    v_num = velocity.flatten() if velocity.ndim > 1 else velocity

    # Compute analytical solution
    R, phi_final, T_analytical = compute_brachistochrone_parameters(x0, y0, x1, y1, g)
    analytical = evaluate_brachistochrone_trajectory(x0, y0, R, phi_final, g, num_points=len(t))

    # Interpolate analytical solution to match numerical time points
    x_analytical_interp = np.interp(t, analytical["t"], analytical["x"])
    y_analytical_interp = np.interp(t, analytical["t"], analytical["y"])
    v_analytical_interp = np.interp(t, analytical["t"], analytical["v"])

    # Compute errors
    position_errors = np.sqrt(
        (x_num - x_analytical_interp) ** 2 + (y_num - y_analytical_interp) ** 2
    )
    position_rmse = np.sqrt(np.mean(position_errors**2))
    position_max_error = np.max(position_errors)

    T_numerical = t[-1]
    time_error_pct = 100 * abs(T_numerical - T_analytical) / T_analytical

    # Velocity error
    velocity_errors = np.abs(v_num - v_analytical_interp)
    velocity_rmse = np.sqrt(np.mean(velocity_errors**2))

    # Compute analytical arc length
    arc_length = compute_cycloid_arc_length(R, phi_final)

    return {
        "analytical_time": T_analytical,
        "numerical_time": T_numerical,
        "time_error_pct": time_error_pct,
        "position_rmse": position_rmse,
        "position_max_error": position_max_error,
        "velocity_rmse": velocity_rmse,
        "analytical_trajectory": analytical,
        "R": R,
        "phi_final": phi_final,
        "arc_length": arc_length,
    }
