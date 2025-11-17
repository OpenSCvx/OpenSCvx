"""
Unit test for brachistochrone problem.

The brachistochrone problem asks for the curve of fastest descent under gravity
between two points. This has a known analytical solution: a cycloid curve.

For the problem setup in examples/abstract/brachistochrone.py:
- Start: (0, 10)
- End: (10, 5)
- g = 9.81 m/s²

The analytical solution gives an optimal time of approximately 1.808 seconds.
"""
import jax
import pytest

from tests.brachistochrone_analytical import compare_trajectory_to_analytical


def test_brachistochrone_example():
    """
    Test the brachistochrone example from examples/abstract/brachistochrone.py.

    This validates against the known analytical solution and checks timing.
    """
    from examples.abstract.brachistochrone import problem

    # Disable printing for cleaner test output
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False

    # Run optimization
    problem.initialize()
    result = problem.solve()
    result = problem.post_process(result)

    # Check convergence
    assert result["converged"], "Brachistochrone failed to converge"

    # Compare numerical solution to analytical brachistochrone solution
    # Extract boundary conditions from problem definition
    x0, y0 = 0.0, 10.0
    x1, y1 = 10.0, 5.0
    g = 9.81

    # Compare to analytical solution
    comparison = compare_trajectory_to_analytical(
        result.t_full,
        result.trajectory['position'],
        result.trajectory['velocity'],
        x0, y0, x1, y1, g
    )

    # Print comparison metrics (visible in verbose mode)
    print("\nBrachistochrone Validation Metrics:")
    print(f"  Analytical time:     {comparison['analytical_time']:.4f} s")
    print(f"  Numerical time:      {comparison['numerical_time']:.4f} s")
    print(f"  Time error:          {comparison['time_error_pct']:.2f}%")
    print(f"  Position RMSE:       {comparison['position_rmse']:.4f}")
    print(f"  Max position error:  {comparison['position_max_error']:.4f}")
    if comparison['velocity_rmse'] is not None:
        print(f"  Velocity RMSE:       {comparison['velocity_rmse']:.4f} m/s")
    print(f"  Cycloid parameters:  R={comparison['R']:.4f}, φ_final={comparison['phi_final']:.4f}")

    # Check time accuracy: numerical should be within 1% of analytical
    time_error_pct = comparison['time_error_pct']
    assert time_error_pct < 1.0, \
        f"Time error {time_error_pct:.2f}% exceeds 1% threshold " \
        f"(analytical: {comparison['analytical_time']:.4f}s, " \
        f"numerical: {comparison['numerical_time']:.4f}s)"

    # Check that numerical time is close to but not significantly better than analytical
    # (since analytical is theoretically optimal)
    assert comparison['numerical_time'] >= comparison['analytical_time'] * 0.95, \
        f"Numerical time {comparison['numerical_time']:.4f}s is suspiciously " \
        f"better than analytical {comparison['analytical_time']:.4f}s"

    # Check trajectory shape: position RMSE should be small
    # Current performance: ~0.01, so enforce < 0.05 with margin
    position_rmse = comparison['position_rmse']
    assert position_rmse < 0.05, \
        f"Position RMSE {position_rmse:.4f} exceeds threshold of 0.05"

    # Check maximum position error
    max_pos_error = comparison['position_max_error']
    assert max_pos_error < 0.1, \
        f"Maximum position error {max_pos_error:.4f} exceeds threshold of 0.1"

    # Check velocity accuracy
    # Current performance: ~0.01 m/s, so enforce < 0.05 m/s with margin
    velocity_rmse = comparison['velocity_rmse']
    assert velocity_rmse < 0.05, \
        f"Velocity RMSE {velocity_rmse:.4f} exceeds threshold of 0.05 m/s"

    # Check that we didn't take too many iterations
    if "discretization_history" in result:
        num_iters = len(result["discretization_history"])
        assert num_iters < 15, f"Took {num_iters} SCP iterations (expected < 15)"

    # Check timing - these are generous limits for a simple problem like brachistochrone
    assert problem.timing_init < 10.0, \
        f"Initialization took {problem.timing_init:.2f}s (expected < 10s)"
    assert problem.timing_solve < 1.0, \
        f"Solve took {problem.timing_solve:.2f}s (expected < 1s)"
    assert problem.timing_post < 5.0, \
        f"Post-processing took {problem.timing_post:.2f}s (expected < 5s)"

    # Clean up JAX caches
    jax.clear_caches()


def test_brachistochrone_inline():
    """
    Test brachistochrone with an inline problem definition.

    This allows testing different configurations and settings without
    modifying the example file.
    """
    import jax.numpy as jnp
    import numpy as np
    import openscvx as ox
    from openscvx import TrajOptProblem

    # Problem parameters
    n = 2
    total_time = 2.0
    g = 9.81

    # Boundary conditions
    x0, y0 = 0.0, 10.0
    x1, y1 = 10.0, 5.0

    # Define state components
    position = ox.State("position", shape=(2,))  # 2D position [x, y]
    position.max = np.array([10.0, 10.0])
    position.min = np.array([0.0, 0.0])
    position.initial = np.array([x0, y0])
    position.final = [x1, y1]
    position.guess = np.linspace(position.initial, position.final, n)

    velocity = ox.State("velocity", shape=(1,))  # Scalar speed
    velocity.max = np.array([10.0])
    velocity.min = np.array([0.0])
    velocity.initial = np.array([0.0])
    velocity.final = [("free", 10.0)]
    velocity.guess = np.linspace(0.0, 10.0, n).reshape(-1, 1)

    # Define control
    theta = ox.Control("theta", shape=(1,))  # Angle from vertical
    theta.max = np.array([100.5 * jnp.pi / 180])
    theta.min = np.array([0.0])
    theta.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

    # Define list of all states (needed for TrajOptProblem and constraints)
    states = [position, velocity]
    controls = [theta]

    # Define dynamics as dictionary mapping state names to their derivatives
    dynamics = {
        "position": ox.Concat(
            velocity[0] * ox.Sin(theta[0]),  # x_dot
            -velocity[0] * ox.Cos(theta[0]),  # y_dot
        ),
        "velocity": g * ox.Cos(theta[0]),
    }

    # Generate box constraints for all states
    constraint_exprs = []
    for state in states:
        constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

    time = ox.Time(
        initial=0.0,
        final=("minimize", total_time),
        min=0.0,
        max=total_time,
    )

    problem = TrajOptProblem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraint_exprs,
        N=n,
        licq_max=1e-8,
    )

    problem.settings.prp.dt = 0.01
    problem.settings.cvx.solver_args = {"abstol": 1e-6, "reltol": 1e-9}
    problem.settings.scp.w_tr = 1e1  # Weight on the Trust Region
    problem.settings.scp.lam_cost = 1e0  # Weight on the Minimal Time Objective
    problem.settings.scp.lam_vc = 1e1  # Weight on the Virtual Control Objective
    problem.settings.scp.uniform_time_grid = True
    problem.settings.sim.save_compiled = False

    # Disable printing for cleaner test output
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False

    # Run optimization
    problem.initialize()
    result = problem.solve()
    result = problem.post_process(result)

    # Check convergence
    assert result["converged"], "Brachistochrone (inline) failed to converge"

    # Compare to analytical solution
    comparison = compare_trajectory_to_analytical(
        result.t_full,
        result.trajectory['position'],
        result.trajectory['velocity'],
        x0, y0, x1, y1, g
    )

    # Print comparison metrics (visible in verbose mode)
    print("\nBrachistochrone Inline Validation Metrics:")
    print(f"  Analytical time:     {comparison['analytical_time']:.4f} s")
    print(f"  Numerical time:      {comparison['numerical_time']:.4f} s")
    print(f"  Time error:          {comparison['time_error_pct']:.2f}%")
    print(f"  Position RMSE:       {comparison['position_rmse']:.4f}")
    print(f"  Max position error:  {comparison['position_max_error']:.4f}")
    if comparison['velocity_rmse'] is not None:
        print(f"  Velocity RMSE:       {comparison['velocity_rmse']:.4f} m/s")

    # Check time accuracy: numerical should be within 1% of analytical
    time_error_pct = comparison['time_error_pct']
    assert time_error_pct < 1.0, \
        f"Time error {time_error_pct:.2f}% exceeds 1% threshold"

    # Check that numerical time is close to but not significantly better than analytical
    assert comparison['numerical_time'] >= comparison['analytical_time'] * 0.95, \
        f"Numerical time {comparison['numerical_time']:.4f}s is suspiciously " \
        f"better than analytical {comparison['analytical_time']:.4f}s"

    # Check trajectory shape: position RMSE should be small
    position_rmse = comparison['position_rmse']
    assert position_rmse < 0.05, \
        f"Position RMSE {position_rmse:.4f} exceeds threshold of 0.05"

    # Check maximum position error
    max_pos_error = comparison['position_max_error']
    assert max_pos_error < 0.1, \
        f"Maximum position error {max_pos_error:.4f} exceeds threshold of 0.1"

    # Check velocity accuracy
    velocity_rmse = comparison['velocity_rmse']
    assert velocity_rmse < 0.05, \
        f"Velocity RMSE {velocity_rmse:.4f} exceeds threshold of 0.05 m/s"

    # Check that we didn't take too many iterations
    if "discretization_history" in result:
        num_iters = len(result["discretization_history"])
        assert num_iters < 15, f"Took {num_iters} SCP iterations (expected < 15)"

    # Check timing - these are generous limits for a simple problem like brachistochrone
    assert problem.timing_init < 10.0, \
        f"Initialization took {problem.timing_init:.2f}s (expected < 10s)"
    assert problem.timing_solve < 1.0, \
        f"Solve took {problem.timing_solve:.2f}s (expected < 1s)"
    assert problem.timing_post < 5.0, \
        f"Post-processing took {problem.timing_post:.2f}s (expected < 5s)"

    # Clean up JAX caches
    jax.clear_caches()
