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

from tests.brachistochrone_analytical import compare_trajectory_to_analytical


def _print_comparison_metrics(comparison, test_name="Brachistochrone"):
    """Print comparison metrics for brachistochrone validation."""
    print(f"\n{test_name} Validation Metrics:")
    print(f"  Analytical time:     {comparison['analytical_time']:.4f} s")
    print(f"  Numerical time:      {comparison['numerical_time']:.4f} s")
    print(f"  Time error:          {comparison['time_error_pct']:.2f}%")
    print(f"  Position RMSE:       {comparison['position_rmse']:.4f}")
    print(f"  Max position error:  {comparison['position_max_error']:.4f}")
    if comparison["velocity_rmse"] is not None:
        print(f"  Velocity RMSE:       {comparison['velocity_rmse']:.4f} m/s")
    if "R" in comparison and "phi_final" in comparison:
        print(
            f"  Cycloid parameters:  R={comparison['R']:.4f}, φ_final={comparison['phi_final']:.4f}"
        )
    if "arc_length" in comparison:
        print(f"  Arc length:          {comparison['arc_length']:.4f} m")


def _assert_brachistochrone_accuracy(comparison, problem, result):
    """Common assertions for brachistochrone test validation."""
    # Check time accuracy: numerical should be within 1% of analytical
    time_error_pct = comparison["time_error_pct"]
    assert time_error_pct < 1.0, (
        f"Time error {time_error_pct:.2f}% exceeds 1% threshold "
        f"(analytical: {comparison['analytical_time']:.4f}s, "
        f"numerical: {comparison['numerical_time']:.4f}s)"
    )

    # Check that numerical time is close to but not significantly better than analytical
    # (since analytical is theoretically optimal)
    assert comparison["numerical_time"] >= comparison["analytical_time"] * 0.95, (
        f"Numerical time {comparison['numerical_time']:.4f}s is suspiciously "
        f"better than analytical {comparison['analytical_time']:.4f}s"
    )

    # Check trajectory shape: position RMSE should be small
    # Current performance: ~0.01, so enforce < 0.05 with margin
    position_rmse = comparison["position_rmse"]
    assert position_rmse < 0.05, f"Position RMSE {position_rmse:.4f} exceeds threshold of 0.05"

    # Check maximum position error
    max_pos_error = comparison["position_max_error"]
    assert max_pos_error < 0.1, (
        f"Maximum position error {max_pos_error:.4f} exceeds threshold of 0.1"
    )

    # Check velocity accuracy
    # Current performance: ~0.01 m/s, so enforce < 0.05 m/s with margin
    velocity_rmse = comparison["velocity_rmse"]
    assert velocity_rmse < 0.05, f"Velocity RMSE {velocity_rmse:.4f} exceeds threshold of 0.05 m/s"

    # Check that we didn't take too many iterations
    if "discretization_history" in result:
        num_iters = len(result["discretization_history"])
        assert num_iters < 15, f"Took {num_iters} SCP iterations (expected < 15)"

    # Check timing - these are generous limits for a simple problem like brachistochrone
    assert problem.timing_init < 10.0, (
        f"Initialization took {problem.timing_init:.2f}s (expected < 10s)"
    )
    assert problem.timing_solve < 1.0, f"Solve took {problem.timing_solve:.2f}s (expected < 1s)"
    assert problem.timing_post < 5.0, (
        f"Post-processing took {problem.timing_post:.2f}s (expected < 5s)"
    )


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
        result.trajectory["position"],
        result.trajectory["velocity"],
        x0,
        y0,
        x1,
        y1,
        g,
    )

    _print_comparison_metrics(comparison, "Brachistochrone")
    _assert_brachistochrone_accuracy(comparison, problem, result)

    # Clean up JAX caches
    jax.clear_caches()


def test_brachistochrone_monolithic():
    """
    Test brachistochrone with monolithic state representation.

    This tests the alternative problem formulation style where all state
    components (x, y, v) are packed into a single state vector instead of
    being defined as separate named states.
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

    # Define monolithic state vector (x, y, v)
    x = ox.State("x", shape=(3,))
    x.max = np.array([10.0, 10.0, 10.0])  # Upper bounds
    x.min = np.array([0.0, 0.0, 0.0])  # Lower bounds
    x.initial = np.array([x0, y0, 0.0])  # [x0, y0, v0]
    x.final = [x1, y1, ("free", 10.0)]  # [x1, y1, v_free]
    x.guess = np.linspace(x.initial, x.final, n)

    # Define control
    u = ox.Control("u", shape=(1,))  # Angle from vertical
    u.max = np.array([100.5 * jnp.pi / 180])
    u.min = np.array([0.0])
    u.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

    # Define dynamics for monolithic state
    x_dot = x[2] * ox.Sin(u[0])  # dx/dt = v * sin(theta)
    y_dot = -x[2] * ox.Cos(u[0])  # dy/dt = -v * cos(theta)
    v_dot = g * ox.Cos(u[0])  # dv/dt = g * cos(theta)
    dyn_expr = ox.Concat(x_dot, y_dot, v_dot)

    # Box constraints
    constraint_exprs = [
        ox.ctcs(x <= x.max),
        ox.ctcs(x.min <= x),
    ]

    time = ox.Time(
        initial=0.0,
        final=("minimize", total_time),
        min=0.0,
        max=total_time,
    )

    problem = TrajOptProblem(
        dynamics={"x": dyn_expr},
        states=[x],
        controls=[u],
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
    assert result["converged"], "Problem failed to converge"

    # Extract position and velocity from monolithic state
    # x.trajectory is shape (N, 3) with columns [x, y, v]
    position = result.trajectory["x"][:, :2]  # First 2 columns: [x, y]
    velocity = result.trajectory["x"][:, 2:3]  # Third column: [v]

    # Compare to analytical solution
    comparison = compare_trajectory_to_analytical(
        result.t_full, position, velocity, x0, y0, x1, y1, g
    )

    _print_comparison_metrics(comparison, "Brachistochrone Monolithic")
    _assert_brachistochrone_accuracy(comparison, problem, result)

    # Clean up JAX caches
    jax.clear_caches()


def test_brachistochrone_nodal():
    """
    Test brachistochrone with nodal constraints.
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
        constraint_exprs.extend([state <= state.max, state.min <= state])

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
    assert result["converged"], "Problem failed to converge"

    # Compare to analytical solution
    comparison = compare_trajectory_to_analytical(
        result.t_full,
        result.trajectory["position"],
        result.trajectory["velocity"],
        x0,
        y0,
        x1,
        y1,
        g,
    )

    _print_comparison_metrics(comparison, "Brachistochrone Nodal")
    _assert_brachistochrone_accuracy(comparison, problem, result)

    # Clean up JAX caches
    jax.clear_caches()


def test_brachistochrone_parameters():
    """
    Test brachistochrone with Parameter objects.

    This tests the use of ox.Parameter for problem constants (like gravity).
    Parameters allow symbolic representation of constants in the problem formulation,
    and can be modified between solves without re-initialization.
    """
    import jax.numpy as jnp
    import numpy as np

    import openscvx as ox
    from openscvx import TrajOptProblem

    # Problem parameters
    n = 2
    total_time = 2.0

    # Boundary conditions
    x0, y0 = 0.0, 10.0
    x1, y1 = 10.0, 5.0

    # Create parameter for gravity (instead of hardcoding)
    g_param = ox.Parameter("g", value=9.81)

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
        "velocity": g_param * ox.Cos(theta[0]),  # Use parameter instead of hardcoded value
    }

    # Generate box constraints for all states
    constraint_exprs = []
    for state in states:
        constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

    time = ox.Time(
        initial=0.0,
        final=("minimize", total_time),
        min=0.0,
        max=10.0,
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
    problem.settings.scp.w_tr = 1e1
    problem.settings.scp.lam_cost = 1e0
    problem.settings.scp.lam_vc = 1e1
    problem.settings.scp.uniform_time_grid = True
    problem.settings.sim.save_compiled = False

    # Apply scaling override for time state
    time_slice = problem.settings.sim.time_slice
    if time_slice is not None:
        # Use time bounds for scaling override
        problem.settings.sim.scaling_x_overrides = [
            (total_time, 0.0, time_slice),
        ]

    # Save original weight values for second problem setup
    original_w_tr = problem.settings.scp.w_tr
    original_lam_cost = problem.settings.scp.lam_cost
    original_lam_vc = problem.settings.scp.lam_vc

    # Disable printing for cleaner test output
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False

    # Run optimization with initial gravity parameter
    problem.initialize()
    result = problem.solve()
    result = problem.post_process(result)

    # Check convergence
    assert result["converged"], "Problem failed to converge (first run)"

    # Extract position and velocity
    position_traj = result.trajectory["position"]
    velocity_traj = result.trajectory["velocity"]

    # Compare to analytical solution (using g_param.value for comparison)
    comparison = compare_trajectory_to_analytical(
        result.t_full, position_traj, velocity_traj, x0, y0, x1, y1, g_param.value
    )

    _print_comparison_metrics(comparison, "Brachistochrone Parameters (g=9.81)")
    print(f"  Using g parameter:   {g_param.value} m/s^2")
    _assert_brachistochrone_accuracy(comparison, problem, result)

    # Second run with different gravity parameter (e.g., Moon gravity)
    # Similar to how dubins_car.py modifies obstacle parameters
    g_moon = 1.62  # Moon's gravity in m/s^2
    problem.parameters["g"] = g_moon

    # Update time bounds for Moon gravity (weaker gravity means longer optimal time)
    # Expected time scales as 1/sqrt(g), so Moon time ≈ 2.46x Earth time
    total_time_moon = 10.0  # Generous upper bound for Moon gravity
    time_slice = problem.settings.sim.x.time_slice
    problem.settings.sim.x.max[time_slice] = total_time_moon
    problem.settings.sim.x.final[time_slice] = total_time_moon

    # Reset guesses for second run
    position.guess = np.linspace(position.initial, position.final, n)
    velocity.guess = np.linspace(0.0, 10.0, n).reshape(-1, 1)
    theta.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

    # Restore original weight values for second problem setup
    problem.settings.scp.w_tr = original_w_tr
    problem.settings.scp.lam_cost = original_lam_cost
    problem.settings.scp.lam_vc = original_lam_vc

    # Solve again without re-initialization (parameters are updated)
    result2 = problem.solve()
    result2 = problem.post_process(result2)

    # Check convergence
    assert result2["converged"], "Problem failed to converge (second run with Moon gravity)"

    # Extract position and velocity
    position_traj2 = result2.trajectory["position"]
    velocity_traj2 = result2.trajectory["velocity"]

    # Compare to analytical solution with Moon gravity
    comparison2 = compare_trajectory_to_analytical(
        result2.t_full, position_traj2, velocity_traj2, x0, y0, x1, y1, g_moon
    )

    _print_comparison_metrics(comparison2, "Brachistochrone Parameters (g=1.62, Moon)")
    print(f"  Using g parameter:   {g_moon} m/s^2")
    _assert_brachistochrone_accuracy(comparison2, problem, result2)

    # Verify that the optimal time is longer with Moon gravity (weaker gravity)
    # Time should be proportional to 1/sqrt(g), so Moon time should be ~2.45x longer
    time_ratio = comparison2["analytical_time"] / comparison["analytical_time"]
    expected_ratio = np.sqrt(g_param.value / g_moon)  # sqrt(9.81/1.62) ≈ 2.46
    assert abs(time_ratio - expected_ratio) < 0.1, (
        f"Time ratio {time_ratio:.2f} doesn't match expected {expected_ratio:.2f} "
        f"for gravity scaling"
    )
    print(f"  Time ratio (Moon/Earth): {time_ratio:.2f} (expected: {expected_ratio:.2f})")

    # Clean up JAX caches
    jax.clear_caches()


def test_brachistochrone_propagation():
    """
    Test brachistochrone with propagation dynamics to track distance travelled.

    This test demonstrates using dynamics_prop and states_prop to add an
    extra state (distance) that is only propagated forward and not included
    in the optimization problem. The distance state integrates velocity to
    track the total arc length travelled along the brachistochrone curve.
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

    # Define state components (optimization states)
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

    # Optimization states and controls
    states = [position, velocity]
    controls = [theta]

    # Define propagation-only state for tracking total distance traveled
    distance = ox.State("distance", shape=(1,))
    distance.initial = np.array([0.0])
    distance.min = np.array([0.0])
    distance.max = np.array([100.0])
    distance.guess = np.zeros((n, 1))

    # Extra propagation states: only the NEW states, not optimization states
    states_prop_extra = [distance]

    # Define dynamics for optimization states
    dynamics = {
        "position": ox.Concat(
            velocity[0] * ox.Sin(theta[0]),  # x_dot
            -velocity[0] * ox.Cos(theta[0]),  # y_dot
        ),
        "velocity": g * ox.Cos(theta[0]),
    }

    # Define EXTRA propagation dynamics (only for new states)
    # distance_dot = velocity (total arc length traveled)
    dynamics_prop_extra = {
        "distance": velocity[0],
    }

    # Generate box constraints for optimization states
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
        dynamics_prop=dynamics_prop_extra,  # Only extra states
        states_prop=states_prop_extra,  # Only extra states
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
    assert result["converged"], "Problem failed to converge"

    # Compare to analytical solution
    comparison = compare_trajectory_to_analytical(
        result.t_full,
        result.trajectory["position"],
        result.trajectory["velocity"],
        x0,
        y0,
        x1,
        y1,
        g,
    )

    _print_comparison_metrics(comparison, "Brachistochrone Propagation")
    _assert_brachistochrone_accuracy(comparison, problem, result)

    # Verify that the distance state was propagated
    assert "distance" in result.trajectory, "Distance state not found in trajectory"
    distance_traj = result.trajectory["distance"]
    assert distance_traj.shape[0] == len(result.t_full), "Distance trajectory length mismatch"

    # Check that distance is monotonically increasing (since velocity >= 0)
    distance_values = distance_traj.flatten()
    assert np.all(np.diff(distance_values) >= -1e-6), "Distance should be monotonically increasing"

    # Check that initial distance is 0
    assert abs(distance_values[0]) < 1e-6, f"Initial distance should be 0, got {distance_values[0]}"

    # Check that final distance is positive and reasonable
    final_distance = distance_values[-1]
    assert final_distance > 0, "Final distance should be positive"

    # Compare numerical distance to analytical arc length
    analytical_arc_length = comparison["arc_length"]
    distance_error_pct = 100 * abs(final_distance - analytical_arc_length) / analytical_arc_length

    # Distance should match analytical arc length within 2%
    assert distance_error_pct < 2.0, (
        f"Distance error {distance_error_pct:.2f}% exceeds 2% threshold "
        f"(analytical: {analytical_arc_length:.4f}m, "
        f"numerical: {final_distance:.4f}m)"
    )

    # Sanity checks: arc length should be longer than straight line
    straight_line_distance = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    assert analytical_arc_length > straight_line_distance * 0.99, (
        f"Analytical arc length {analytical_arc_length:.4f}m should be greater than "
        f"straight-line distance {straight_line_distance:.4f}m"
    )

    print(f"\n  Distance travelled:    {final_distance:.4f} m")
    print(f"  Analytical arc length: {analytical_arc_length:.4f} m")
    print(f"  Distance error:        {distance_error_pct:.2f}%")
    print(f"  Straight-line dist:    {straight_line_distance:.4f} m")
    print(f"  Ratio (arc/line):      {analytical_arc_length / straight_line_distance:.4f}")

    # Clean up JAX caches
    jax.clear_caches()
