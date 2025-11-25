# Vectorization and Vmapping Across Decision Nodes

This page explains how OpenSCvx processes individual state and control definitions, combines them into unified vectors, and vectorizes computations across decision nodes using JAX's `vmap`. Understanding this pipeline is crucial for performance optimization and debugging.

## Overview

OpenSCvx transforms user-defined dynamics and constraints through several stages:

1. **User Definition**: Define individual states, controls, and their dynamics symbolically
2. **Symbolic Preprocessing**: Augment with time state, CTCS augmented states, and time dilation
3. **Unification**: Combine individual variables into monolithic "unified" vectors
4. **JAX Lowering**: Convert symbolic expressions to executable JAX functions
5. **Vectorization**: Apply `vmap` to batch computations across all decision nodes

This pipeline enables efficient computation by evaluating dynamics and constraints for all nodes simultaneously rather than sequentially.

## Stage 1: User Definition

Users define problems with individual states and controls, each with their own shape:

```python
import openscvx as ox
import numpy as np

# Define individual state components
position = ox.State("position", shape=(2,))  # 2D position [x, y]
position.initial = np.array([0.0, 10.0])
position.final = [10.0, 5.0]

velocity = ox.State("velocity", shape=(1,))  # Scalar speed
velocity.initial = np.array([0.0])

# Define control
theta = ox.Control("theta", shape=(1,))  # Angle from vertical

# Define dynamics as dictionary mapping state names to derivatives
dynamics = {
    "position": ox.Concat(
        velocity[0] * ox.Sin(theta[0]),  # x_dot
        -velocity[0] * ox.Cos(theta[0]),  # y_dot
    ),
    "velocity": 9.81 * ox.Cos(theta[0]),  # v_dot
}
```

**Key Points:**
- Each state/control is defined independently with its own shape
- Dynamics are specified per-state using symbolic expressions
- No explicit mention of decision nodes or batching

## Stage 2: Symbolic Preprocessing

When `TrajOptProblem` is constructed, preprocessing augments the problem:

```python
problem = TrajOptProblem(
    dynamics=dynamics,
    states=[position, velocity],
    controls=[theta],
    N=10,  # Number of decision nodes
    time=ox.Time(initial=0.0, final=2.0),
)
```

**Augmentation Process** (in `openscvx/symbolic/builder.py:preprocess_symbolic_problem`):

1. **Time State Added**: If not provided, a `"time"` state is auto-created
2. **CTCS Augmented States**: For each CTCS constraint, an augmented state is added (e.g., `"_ctcs_aug_0"`)
3. **Time Dilation Control**: A `"_time_dilation"` control is added for time-optimal problems

**After Augmentation:**
```python
# states_aug = [position, velocity, time, _ctcs_aug_0, ...]
# controls_aug = [theta, _time_dilation]
```

The augmented dynamics now includes derivatives for all states, including the augmented ones.

## Stage 3: Unification

Individual states and controls are combined into single monolithic vectors (in `openscvx/symbolic/lower.py:lower_symbolic_expressions`):

```python
x_unified: UnifiedState = unify_states(states_aug)
u_unified: UnifiedControl = unify_controls(controls_aug)
```

**Unification Process** (in `openscvx/symbolic/unified.py`):

1. **Sorting**: User-defined variables first, then augmented (starting with `_`)
2. **Concatenation**: All properties concatenated (min, max, guess, initial, final)
3. **Slice Assignment**: Each original State/Control gets a slice for indexing

**Example Unified Shapes:**

In general, for a problem with `N` decision nodes:

```python
# Individual states before unification:
# position: (2,) at each node
# velocity: (1,) at each node
# time: (1,) at each node
# _ctcs_aug_0: (K,) at each node  # K depends on CTCS constraints

# After unification:
x_unified.shape = (n_x,)  # Sum of all individual state dimensions
u_unified.shape = (n_u,)  # Sum of all control dimensions

# Guess trajectories:
x_unified.guess.shape = (N, n_x)    # States at all N nodes
u_unified.guess.shape = (N, n_u)    # Controls at all N nodes
```

**Concrete Example:** For the brachistochrone problem with `N=10` nodes, `position` (2D), `velocity` (1D), `time` (1D), and `theta` control (1D), `_time_dilation` control (1D):

```python
# After unification (no CTCS constraints for simplicity):
x_unified.shape = (4,)        # position(2) + velocity(1) + time(1)
u_unified.shape = (2,)        # theta(1) + _time_dilation(1)

# Guess trajectories:
x_unified.guess.shape = (10, 4)    # States at 10 nodes
u_unified.guess.shape = (10, 2)    # Controls at 10 nodes
```

**Accessing Individual Components:**

After unification, each state retains its slice:

```python
# position has ._slice = slice(0, 2)
# velocity has ._slice = slice(2, 3)
# time has ._slice = slice(3, 4)

# During lowering, extract values like:
x_unified = jnp.array([...])  # (n_x,) unified state
position_value = x_unified[position._slice]  # (2,) extract position
```

## Stage 4: JAX Lowering

Symbolic expressions are converted to executable JAX functions (in `openscvx/symbolic/lower.py`):

```python
# Convert symbolic dynamics to JAX function
dyn_fn = lower_to_jax(dynamics_aug)

# Create Dynamics object with Jacobians
dynamics_augmented = Dynamics(
    f=dyn_fn,                      # State derivative function
    A=jacfwd(dyn_fn, argnums=0),   # Jacobian df/dx
    B=jacfwd(dyn_fn, argnums=1),   # Jacobian df/du
)
```

**Function Signature (Before Vmap):**

The lowered JAX functions have a standardized signature:

```python
def f(x: Array, u: Array, node: int, params: dict) -> Array:
    """Compute state derivative at a single decision node.

    Args:
        x: State vector at this node, shape (n_x,)
        u: Control vector at this node, shape (n_u,)
        node: Node index (0 to N-1), used for time-varying constraints
        params: Dictionary of problem parameters

    Returns:
        State derivative dx/dt, shape (n_x,)
    """
    ...
```

Similarly for Jacobians:

```python
A(x, u, node, params) -> Array[n_x, n_x]  # df/dx
B(x, u, node, params) -> Array[n_x, n_u]  # df/du
```

**Constraints** follow the same pattern:

```python
# Non-convex nodal constraints (in openscvx/symbolic/lower.py:315-326)
constraint_fn(x, u, node, params) -> scalar  # Before vmap
```

## Stage 5: Vectorization with Vmap

Finally, dynamics and constraints are vectorized to operate on all decision nodes simultaneously (in `openscvx/trajoptproblem.py:356-358`):

```python
# Vectorize dynamics functions across decision nodes
self.dynamics_augmented.f = jax.vmap(
    self.dynamics_augmented.f,
    in_axes=(0, 0, 0, None)
)
self.dynamics_augmented.A = jax.vmap(
    self.dynamics_augmented.A,
    in_axes=(0, 0, 0, None)
)
self.dynamics_augmented.B = jax.vmap(
    self.dynamics_augmented.B,
    in_axes=(0, 0, 0, None)
)
```

**Vmap Configuration: `in_axes=(0, 0, 0, None)`**

This means:
- **Axis 0 of x**: Batch over states at different nodes
- **Axis 0 of u**: Batch over controls at different nodes
- **Axis 0 of node**: Batch over node indices
- **None for params**: Shared parameters (not batched)

**Function Signature (After Vmap):**

```python
def f_vmapped(x_batch: Array, u_batch: Array, nodes: Array, params: dict) -> Array:
    """Compute state derivatives at all decision nodes simultaneously.

    Args:
        x_batch: States at all nodes, shape (N-1, n_x)
        u_batch: Controls at all nodes, shape (N-1, n_u)
        nodes: Node indices, shape (N-1,) - typically jnp.arange(0, N-1)
        params: Dictionary of problem parameters (shared across all nodes)

    Returns:
        State derivatives at all nodes, shape (N-1, n_x)
    """
    ...
```

Similarly for Jacobians:

```python
A_vmapped(x_batch, u_batch, nodes, params) -> Array[N-1, n_x, n_x]
B_vmapped(x_batch, u_batch, nodes, params) -> Array[N-1, n_x, n_u]
```

**Why N-1 instead of N?**

Trajectory discretization operates on **intervals** between consecutive decision nodes:
- **N decision nodes**: Including initial and final states (e.g., nodes 0, 1, 2, ..., 9 for N=10)
- **N-1 intervals**: Between consecutive nodes (e.g., intervals [0→1], [1→2], ..., [8→9] for N=10)
- **Dynamics evaluation**: At the start of each interval, giving N-1 evaluations

This is why vmapped functions process batches of size `(N-1, ...)` rather than `(N, ...)`.

## Usage in Discretization

The vmapped functions are called during discretization (in `openscvx/discretization.py:77-85`):

```python
# Setup batch inputs
x = V[:, :n_x]                    # Shape: (N-1, n_x)
u = u[: x.shape[0], :-1]          # Shape: (N-1, n_u-1) - exclude time dilation
nodes = jnp.arange(0, N-1)        # Shape: (N-1,)

# Call vmapped dynamics - evaluates all intervals in parallel
f = state_dot(x, u, nodes, params)    # Shape: (N-1, n_x)
dfdx = A(x, u, nodes, params)         # Shape: (N-1, n_x, n_x)
dfdu = B(x, u, nodes, params)         # Shape: (N-1, n_x, n_u-1)
```

**Example with N=10:** This single call evaluates dynamics at all 9 intervals simultaneously, leveraging JAX's efficient vectorization on GPU/TPU.

## Constraints Vectorization

Non-convex nodal constraints are also vectorized (in `openscvx/symbolic/lower.py:320-323`):

```python
# Vectorize constraint functions
constraint = LoweredNodalConstraint(
    func=jax.vmap(fn, in_axes=(0, 0, None, None)),
    grad_g_x=jax.vmap(jacfwd(fn, argnums=0), in_axes=(0, 0, None, None)),
    grad_g_u=jax.vmap(jacfwd(fn, argnums=1), in_axes=(0, 0, None, None)),
    nodes=constraint.nodes,  # List of node indices where constraint applies
)
```

**Note the difference:** Constraints use `in_axes=(0, 0, None, None)` because they're only evaluated at specific nodes, not across a sequence.

When evaluated:
```python
# x_batch, u_batch only include states/controls at constraint.nodes
g = constraint.func(x_batch, u_batch, node_idx, params)  # Shape: (len(nodes),)
```

## Shape Summary Table

Here's a complete reference for shapes at each stage, shown with symbolic dimensions (`N`, `n_x`, `n_u`) and a concrete example:

| **Stage** | **Variable** | **Symbolic Shape** | **Concrete Example (N=10, n_x=4, n_u=2)** |
|-----------|--------------|-------------------|-------------------------------------------|
| **User Definition** | `position` | `(2,)` | `(2,)` - Single 2D position vector |
| | `velocity` | `(1,)` | `(1,)` - Single scalar velocity |
| | `theta` | `(1,)` | `(1,)` - Single scalar control |
| | | | |
| **After Augmentation** | `states_aug` | List of States | [position, velocity, time] (3 states) |
| | `controls_aug` | List of Controls | [theta, _time_dilation] (2 controls) |
| | | | |
| **After Unification** | `x_unified.shape` | `(n_x,)` | `(4,)` - position(2) + velocity(1) + time(1) |
| | `u_unified.shape` | `(n_u,)` | `(2,)` - theta(1) + _time_dilation(1) |
| | `x_unified.guess` | `(N, n_x)` | `(10, 4)` - States at 10 nodes |
| | `u_unified.guess` | `(N, n_u)` | `(10, 2)` - Controls at 10 nodes |
| | `position._slice` | `slice(0, 2)` | `slice(0, 2)` - Extract position |
| | `velocity._slice` | `slice(2, 3)` | `slice(2, 3)` - Extract velocity |
| | `time._slice` | `slice(3, 4)` | `slice(3, 4)` - Extract time |
| | | | |
| **JAX Functions (Pre-Vmap)** | `f(x, u, node, params)` | Input: `(n_x,), (n_u,), scalar, dict` | Input: `(4,), (2,), scalar, dict` |
| | | Output: `(n_x,)` | Output: `(4,)` - Single state derivative |
| | `A(x, u, node, params)` | Output: `(n_x, n_x)` | Output: `(4, 4)` - Jacobian df/dx |
| | `B(x, u, node, params)` | Output: `(n_x, n_u)` | Output: `(4, 2)` - Jacobian df/du |
| | | | |
| **JAX Functions (Post-Vmap)** | `f(x, u, nodes, params)` | Input: `(N-1, n_x), (N-1, n_u), (N-1,), dict` | Input: `(9, 4), (9, 2), (9,), dict` |
| | | Output: `(N-1, n_x)` | Output: `(9, 4)` - Derivatives at 9 intervals |
| | `A(x, u, nodes, params)` | Output: `(N-1, n_x, n_x)` | Output: `(9, 4, 4)` - Jacobians at 9 intervals |
| | `B(x, u, nodes, params)` | Output: `(N-1, n_x, n_u)` | Output: `(9, 4, 2)` - Jacobians at 9 intervals |

## Performance Implications

**Why This Architecture?**

1. **GPU/TPU Acceleration**: Vmapping enables SIMD parallelism across nodes
2. **JIT Compilation**: JAX compiles the vmapped function once, not per-node
3. **Automatic Differentiation**: Jacobians computed automatically via `jacfwd`
4. **Reduced Python Overhead**: Single JAX call instead of Python loop

**Performance Tips:**

- Use simple symbolic expressions to maximize JIT compilation effectiveness
- Keep the number of states/controls reasonable (hundreds, not thousands)
- Parameters should be problem constants, not optimization variables
- Reuse compiled solvers when possible (they can be cached based on problem structure)

## Implementation Files Reference

| **File** | **Function/Class** | **Purpose** |
|----------|-------------------|-------------|
| `examples/abstract/brachistochrone.py` | — | Example problem definition |
| `openscvx/trajoptproblem.py` | `TrajOptProblem.__init__` | Orchestrates preprocessing pipeline |
| `openscvx/symbolic/builder.py` | `preprocess_symbolic_problem` | Augments states/controls/dynamics |
| `openscvx/symbolic/lower.py` | `lower_symbolic_expressions` | Unification and JAX lowering |
| `openscvx/symbolic/unified.py` | `unify_states`, `unify_controls` | Combines individual variables |
| `openscvx/trajoptproblem.py:356-358` | `initialize` | Applies vmap to dynamics |
| `openscvx/discretization.py` | `dVdt`, `calculate_discretization` | Uses vmapped dynamics |

## Advanced: Accessing Unified Vectors

During problem setup, you can access the unified objects:

```python
problem = TrajOptProblem(...)
problem.initialize()

# Access unified state/control objects
x_unified = problem.x_unified
u_unified = problem.u_unified

print(f"Total state dimension: {x_unified.shape[0]}")
print(f"Total control dimension: {u_unified.shape[0]}")

# Access individual state slices
for state in problem.states:
    print(f"{state.name}: slice {state._slice}")
```

## Common Pitfalls

1. **Confusing nodes vs intervals**: Discretization operates on N-1 intervals between N nodes, so vmapped dynamics have batch size (N-1, ...)
2. **Forgetting augmented dimensions**: `n_x` and `n_u` include auto-added states/controls (time, CTCS augmented states, time dilation)
3. **Parameter mutability**: The `params` dict is shared across all evaluations - don't modify it during dynamics evaluation
4. **Node index usage**: The `node` parameter enables time-varying behavior (e.g., time-dependent constraints), not for indexing into trajectory arrays

## See Also

- [Basic Problem Setup](basic_problem_setup.md) - How to define problems
- [API: State](api_state.md) - State class documentation
- [API: Control](api_control.md) - Control class documentation
- [API: TrajOptProblem](api_trajoptproblem.md) - Main problem class
