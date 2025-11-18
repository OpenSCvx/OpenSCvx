"""Example demonstrating inter-node constraints using NodeReference.

This example shows how to define constraints across different trajectory nodes,
which is useful for:
- Rate limiting (velocity/acceleration limits)
- Smoothness constraints
- Multi-step dependencies
- Consistency between time steps

The `.node(k)` method enables referencing state/control values at specific
trajectory nodes, allowing you to express temporal relationships.
"""

import numpy as np

import openscvx as ox


def example_position_rate_limiting():
    """Example: Limit the rate of change in position between consecutive nodes."""
    print("=" * 70)
    print("Example 1: Position Rate Limiting")
    print("=" * 70)

    # Define state variables
    position = ox.State("pos", shape=(3,))
    velocity = ox.State("vel", shape=(3,))

    # Trajectory parameters
    N = 50  # Number of nodes
    dt = 0.1  # Time step
    max_position_change = 0.2  # Maximum position change per time step

    # Constraint: position change between consecutive nodes must be limited
    # This would typically be written as:
    #   constraint = (position.node(k) - position.node(k-1) <= max_position_change).at(range(1, N))
    #
    # However, since we're just demonstrating the syntax, we'll show a specific instance:

    # Define node reference for current and previous positions
    # In practice, the node index 'k' would be symbolic, but here we show concrete examples
    pos_k = position.node(10)
    pos_k_prev = position.node(9)

    # Create rate limit constraint
    rate_constraint = pos_k - pos_k_prev <= max_position_change

    print(f"Position at node k: {pos_k}")
    print(f"Position at node k-1: {pos_k_prev}")
    print(f"Rate constraint: {rate_constraint}")
    print(f"Shape of constraint: {rate_constraint.check_shape()}")
    print()


def example_velocity_consistency():
    """Example: Ensure velocity is consistent with position change."""
    print("=" * 70)
    print("Example 2: Velocity Consistency")
    print("=" * 70)

    position = ox.State("pos", shape=(3,))
    velocity = ox.State("vel", shape=(3,))
    dt = 0.1

    # Node references
    pos_k = position.node(10)
    pos_k_prev = position.node(9)
    vel_k = velocity.node(10)

    # Constraint: velocity should match finite difference of position
    # vel[k] ≈ (pos[k] - pos[k-1]) / dt
    consistency_constraint = vel_k == (pos_k - pos_k_prev) / dt

    print(f"Velocity consistency constraint: {consistency_constraint}")
    print(f"This ensures vel[k] = (pos[k] - pos[k-1]) / {dt}")
    print()


def example_control_rate_limiting():
    """Example: Limit control input rate of change."""
    print("=" * 70)
    print("Example 3: Control Rate Limiting")
    print("=" * 70)

    thrust = ox.Control("thrust", shape=(3,))
    max_thrust_rate = 1.0  # Maximum change in thrust per time step

    # Control at different nodes
    thrust_k = thrust.node(5)
    thrust_k_prev = thrust.node(4)

    # Rate limit on control
    control_rate_constraint = thrust_k - thrust_k_prev <= max_thrust_rate

    print(f"Thrust at node k: {thrust_k}")
    print(f"Thrust at node k-1: {thrust_k_prev}")
    print(f"Control rate constraint: {control_rate_constraint}")
    print()


def example_spatial_indexing_with_nodes():
    """Example: Combine spatial indexing with node references."""
    print("=" * 70)
    print("Example 4: Spatial Indexing + Node References")
    print("=" * 70)

    position = ox.State("pos", shape=(3,))

    # Apply different rate limits to different components
    # For example, limit only the z-component (altitude) rate
    z_k = position[2].node(10)
    z_k_prev = position[2].node(9)

    max_z_rate = 0.05  # Stricter limit on altitude change
    z_rate_constraint = z_k - z_k_prev <= max_z_rate

    print(f"Z-component at node k: {z_k}")
    print(f"Z-component at node k-1: {z_k_prev}")
    print(f"Z-rate constraint: {z_rate_constraint}")
    print(f"Shape (scalar after indexing): {z_k.check_shape()}")
    print()


def example_multi_step_dependency():
    """Example: Multi-step dependencies like second-order differences."""
    print("=" * 70)
    print("Example 5: Multi-Step Dependencies (Acceleration)")
    print("=" * 70)

    state = ox.State("x", shape=(1,))
    dt = 0.1

    # Three consecutive time steps
    x_next = state.node(11)
    x_curr = state.node(10)
    x_prev = state.node(9)

    # Second-order finite difference approximates acceleration
    # accel ≈ (x[k+1] - 2*x[k] + x[k-1]) / dt^2
    acceleration = (x_next - 2 * x_curr + x_prev) / (dt**2)

    # Limit acceleration
    max_accel = 5.0
    accel_constraint = acceleration <= max_accel

    print(f"Acceleration (second-order difference): {acceleration}")
    print(f"Acceleration constraint: {accel_constraint}")
    print()


def example_periodic_boundary():
    """Example: Periodic boundary conditions (state at end equals state at start)."""
    print("=" * 70)
    print("Example 6: Periodic Boundary Conditions")
    print("=" * 70)

    state = ox.State("x", shape=(2,))
    N = 100

    # Initial and final states
    x_start = state.node(0)
    x_end = state.node(N - 1)

    # Periodicity constraint
    periodicity = x_start == x_end

    print(f"State at start: {x_start}")
    print(f"State at end: {x_end}")
    print(f"Periodicity constraint: {periodicity}")
    print()


def example_using_with_at():
    """Example: Combining node references with .at() for selective enforcement."""
    print("=" * 70)
    print("Example 7: Using NodeReference with .at()")
    print("=" * 70)

    velocity = ox.State("vel", shape=(3,))
    N = 50
    max_accel = 0.5

    # In a real problem, you would use this pattern:
    # This constraint would be applied at nodes 1 through N-1
    # At each node k in that range, it enforces: vel[k] - vel[k-1] <= max_accel

    # For demonstration, we show what the constraint looks like
    vel_k = velocity.node(10)  # Placeholder for symbolic 'k'
    vel_k_prev = velocity.node(9)  # Placeholder for symbolic 'k-1'

    # Create the constraint
    rate_constraint = vel_k - vel_k_prev <= max_accel

    # Apply it at specific nodes using .at()
    # In practice: (vel.node(k) - vel.node(k-1) <= max_accel).at(range(1, N))
    nodal_constraint = rate_constraint.at(list(range(1, N)))

    print(f"Rate constraint: {rate_constraint}")
    print(f"Applied at nodes: {list(range(1, 10))}... (showing first 10)")
    print(f"Type: {type(nodal_constraint).__name__}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("=" * 70)
    print("INTER-NODE CONSTRAINTS WITH NodeReference")
    print("=" * 70)
    print()

    example_position_rate_limiting()
    example_velocity_consistency()
    example_control_rate_limiting()
    example_spatial_indexing_with_nodes()
    example_multi_step_dependency()
    example_periodic_boundary()
    example_using_with_at()

    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    print("The .node(k) method enables:")
    print("  1. Rate limiting: limit change between consecutive time steps")
    print("  2. Consistency: enforce relationships across time")
    print("  3. Multi-step: express dependencies spanning multiple nodes")
    print("  4. Flexibility: works with spatial indexing and .at() constraints")
    print()
    print("Key syntax:")
    print("  state.node(k)        - Reference state at node k")
    print("  state[i].node(k)     - Reference component i at node k")
    print("  constraint.at(nodes) - Apply constraint at specific nodes")
    print("=" * 70)


if __name__ == "__main__":
    main()
