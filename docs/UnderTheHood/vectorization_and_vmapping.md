# Vectorization and Vmapping Across Decision Nodes

This page explains how OpenSCvx internally processes symbolic problem definitions into vectorized JAX computations. After symbolic preprocessing and augmentation (which you've likely seen in basic usage), the library creates unified state/control vectors and applies JAX's `vmap` to evaluate dynamics and non-convex constraints across decision nodes in parallel.

## Processing Pipeline

The transformation from symbolic expressions to vectorized execution happens in several stages:

1. **Symbolic Preprocessing**: Augmentation with time state, CTCS states, and time dilation (covered in basic usage)
2. **Unification**: Individual State/Control objects combined into monolithic vectors
3. **JAX Lowering**: Symbolic expressions compiled to executable JAX functions (dynamics and non-convex constraints)
4. **Vectorization**: `vmap` applied to batch dynamics and constraint computations across decision nodes

Understanding this pipeline is useful for performance optimization, debugging shape mismatches, and extending the library.

## Stage 1: Symbolic Problem Definition

Starting from a typical problem definition with individual states and controls:

```python
import openscvx as ox
import numpy as np

# Individual state components
position = ox.State("position", shape=(2,))
velocity = ox.State("velocity", shape=(1,))

# Control
theta = ox.Control("theta", shape=(1,))

# Dynamics per state
dynamics = {
    "position": ox.Concat(velocity[0] * ox.Sin(theta[0]), -velocity[0] * ox.Cos(theta[0])),
    "velocity": 9.81 * ox.Cos(theta[0]),
}
```

At this stage, each state/control is independent with its own shape, and dynamics are symbolic expressions without any notion of batching or decision nodes.

## Stage 2: Symbolic Preprocessing and Augmentation

During `TrajOptProblem` construction (in `preprocess_symbolic_problem`), the symbolic problem is augmented:

```python
problem = TrajOptProblem(
    dynamics=dynamics,
    states=[position, velocity],
    controls=[theta],
    N=10,
    time=ox.Time(initial=0.0, final=2.0),
)
```

Internally, additional states and controls are added:
- Time state (if not user-provided)
- CTCS augmented states for path constraints
- Time dilation control for time-optimal problems

After augmentation: `states_aug = [position, velocity, time, ...]` and `controls_aug = [theta, _time_dilation]`, with corresponding dynamics for all augmented states.

## Stage 3: Unification

The augmented states and controls are combined into unified vectors (in `lower_symbolic_expressions`):

```python
x_unified: UnifiedState = unify_states(states_aug)
u_unified: UnifiedControl = unify_controls(controls_aug)
```

The unification process (in `openscvx/symbolic/unified.py`) sorts variables (user-defined first, then augmented), concatenates properties (bounds, guesses, etc.), and assigns each State/Control a slice for indexing into the unified vector.

### Unified Vector Shapes

For a problem with `N` decision nodes:

```python
x_unified.shape = (n_x,)          # Sum of all state dimensions
u_unified.shape = (n_u,)          # Sum of all control dimensions
x_unified.guess.shape = (N, n_x)  # State trajectory
u_unified.guess.shape = (N, n_u)  # Control trajectory
```

**Concrete example** (brachistochrone with N=10, no CTCS constraints):
```python
x_unified.shape = (4,)        # position(2) + velocity(1) + time(1)
u_unified.shape = (2,)        # theta(1) + _time_dilation(1)
x_unified.guess.shape = (10, 4)
u_unified.guess.shape = (10, 2)
```

Each original State/Control retains a slice for extraction:
```python
position._slice = slice(0, 2)
velocity._slice = slice(2, 3)
time._slice = slice(3, 4)

# Extract during evaluation:
position_value = x_unified[position._slice]  # (2,)
```

## Stage 4: JAX Lowering

Symbolic expressions for dynamics and non-convex constraints are converted to executable JAX functions (in `openscvx/symbolic/lower.py`). Convex constraints are lowered to CVXPy separately.

### Dynamics Lowering

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

**Dynamics Function Signature (Before Vmap):**

```python
def f(x: Array, u: Array, node: int, params: dict) -> Array:
    """Compute state derivative at a single decision node.

    Args:
        x: State vector at this node, shape (n_x,)
        u: Control vector at this node, shape (n_u,)
        node: Node index (0 to N-1), used for time-varying behavior
        params: Dictionary of problem parameters

    Returns:
        State derivative dx/dt, shape (n_x,)
    """
    ...
```

Jacobians have similar signatures:

```python
A(x, u, node, params) -> Array[n_x, n_x]  # df/dx
B(x, u, node, params) -> Array[n_x, n_u]  # df/du
```

### Non-Convex Constraint Lowering

Non-convex nodal constraints that are to be lowered to JAX follow the same pattern:

```python
# Convert symbolic constraint expressions to JAX functions (lower.py:312)
constraints_nodal_fns = lower_to_jax(constraints_nodal)

# Create LoweredNodalConstraint objects with Jacobians (lower.py:315-326)
for i, fn in enumerate(constraints_nodal_fns):
    constraint = LoweredNodalConstraint(
        func=fn,                          # Constraint function
        grad_g_x=jacfwd(fn, argnums=0),  # Jacobian dg/dx
        grad_g_u=jacfwd(fn, argnums=1),  # Jacobian dg/du
        nodes=constraints_nodal[i].nodes, # Node indices where constraint applies
    )
```

**Constraint Function Signature (Before Vmap):**

```python
def g(x: Array, u: Array, node: int, params: dict) -> float:
    """Evaluate constraint at a single decision node.

    Args:
        x: State vector at this node, shape (n_x,)
        u: Control vector at this node, shape (n_u,)
        node: Node index, used for time-varying constraints
        params: Dictionary of problem parameters

    Returns:
        Constraint value (scalar)
    """
    ...
```

Constraint Jacobians:

```python
grad_g_x(x, u, node, params) -> Array[n_x]  # dg/dx
grad_g_u(x, u, node, params) -> Array[n_u]  # dg/du
```

### Cross-Node Constraint Lowering

Cross-node constraints relate variables across multiple trajectory nodes (e.g., rate limits, multi-step dependencies). Unlike regular nodal constraints that evaluate at single nodes, cross-node constraints require access to the full trajectory.

**Key Difference**: Regular nodal constraints have signature `(x, u, node, params)` and are vmapped across nodes. Cross-node constraints have signature `(X, U, params)` where `X` and `U` are full trajectories.

Cross-node constraints are defined using the `.node()` method with **relative indexing**:

```python
import openscvx as ox

position = ox.State("position", shape=(2,))

# Rate limit: distance between consecutive nodes
pos_k = position.node('k')      # Position at current node
pos_k_prev = position.node('k-1')  # Position at previous node

step_distance = ox.linalg.Norm(pos_k - pos_k_prev, ord=2)
rate_limit = (step_distance <= max_step).at([1, 2, 3, ..., N-1])
```

The relative indexing supports patterns like:
- `'k'` - current node
- `'k-1'` - previous node
- `'k+1'` - next node
- `'k-2'`, `'k+3'`, etc.

During lowering (in `openscvx/symbolic/lower.py:464-523`), constraints are separated into regular and cross-node:

```python
# Detect cross-node constraints by presence of NodeReference
for constraint in constraints_nodal:
    if _contains_node_reference(constraint.constraint):
        cross_node_constraints.append(constraint)
    else:
        regular_constraints.append(constraint)
```

For each cross-node constraint, the lowering process:

1. **Lowers the expression to JAX** using `JaxLowerer` (which handles `NodeReference` nodes)
2. **Wraps the function** to evaluate at multiple nodes along the trajectory
3. **Computes trajectory-level Jacobians** using automatic differentiation

```python
# Lower constraint expression
constraint_fn = lower_to_jax(constraint_expr)

# Create trajectory-level wrapper (internal helper function)
wrapped_fn = _create_cross_node_wrapper(constraint_fn, references, is_relative, eval_nodes)

# Compute Jacobians for full trajectory
grad_g_X = jacfwd(wrapped_fn, argnums=0)  # dg/dX - shape (M, N, n_x)
grad_g_U = jacfwd(wrapped_fn, argnums=1)  # dg/dU - shape (M, N, n_u)

# Create CrossNodeConstraintLowered object
cross_node_lowered = CrossNodeConstraintLowered(
    func=wrapped_fn,
    grad_g_X=grad_g_X,
    grad_g_U=grad_g_U,
    eval_nodes=eval_nodes,  # List of nodes where constraint is evaluated
)
```

**Cross-Node Constraint Function Signature:**

```python
def g_cross(X: Array, U: Array, params: dict) -> Array:
    """Evaluate cross-node constraint at multiple nodes simultaneously.

    Args:
        X: Full state trajectory, shape (N, n_x)
        U: Full control trajectory, shape (N, n_u)
        params: Dictionary of problem parameters

    Returns:
        Constraint residuals, shape (M,) where M = len(eval_nodes)
    """
    ...
```

**Cross-Node Constraint Jacobians:**

```python
grad_g_X(X, U, params) -> Array[M, N, n_x]  # dg/dX - Jacobian wrt all states
grad_g_U(X, U, params) -> Array[M, N, n_u]  # dg/dU - Jacobian wrt all controls
```

The Jacobian shapes deserve special attention:
- **M**: Number of evaluation points (e.g., for rate limit at nodes [1, 2, ..., N-1], M = N-1)
- **N**: Total number of trajectory nodes
- **Sparsity**: These Jacobians are dense arrays but typically very sparse. For a rate limit `x[k] - x[k-1]`, only 2 out of N nodes have non-zero derivatives.

**Example: Rate Limit Jacobian Structure**

For constraint `||position.node('k') - position.node('k-1')|| <= r` evaluated at node k=5:

```python
grad_g_X[0, :, :]  # Jacobian for this constraint evaluation
# Only non-zero at:
#   grad_g_X[0, 5, :] = ∂g/∂position[5]    # Derivative wrt current node
#   grad_g_X[0, 4, :] = ∂g/∂position[4]    # Derivative wrt previous node
# All other grad_g_X[0, j, :] for j ≠ 4,5 are zero
```

## Stage 5: Vectorization with Vmap

Finally, both dynamics and constraints are vectorized to operate on decision nodes simultaneously. This enables efficient parallel evaluation on GPU/TPU hardware.

### Dynamics Vectorization

Dynamics functions are vmapped to process all intervals in parallel (in `openscvx/trajoptproblem.py:356-358`):

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

**Dynamics Vmap Configuration: `in_axes=(0, 0, 0, None)`**

This means:
- **Axis 0 of x**: Batch over states at different intervals
- **Axis 0 of u**: Batch over controls at different intervals
- **Axis 0 of node**: Batch over node indices
- **None for params**: Shared parameters (not batched)

**Dynamics Signature (After Vmap):**

```python
def f_vmapped(x_batch: Array, u_batch: Array, nodes: Array, params: dict) -> Array:
    """Compute state derivatives at all intervals simultaneously.

    Args:
        x_batch: States at interval starts, shape (N-1, n_x)
        u_batch: Controls at interval starts, shape (N-1, n_u)
        nodes: Node indices, shape (N-1,) - typically jnp.arange(0, N-1)
        params: Dictionary of problem parameters (shared across all nodes)

    Returns:
        State derivatives at all intervals, shape (N-1, n_x)
    """
    ...
```

Jacobians after vmap:

```python
A_vmapped(x_batch, u_batch, nodes, params) -> Array[N-1, n_x, n_x]
B_vmapped(x_batch, u_batch, nodes, params) -> Array[N-1, n_x, n_u]
```

**Why N-1 instead of N?**

Trajectory discretization operates on **intervals** between consecutive decision nodes:
- **N decision nodes**: Including initial and final states (e.g., nodes 0, 1, 2, ..., 9 for N=10)
- **N-1 intervals**: Between consecutive nodes (e.g., intervals [0→1], [1→2], ..., [8→9] for N=10)
- **Dynamics evaluation**: At the start of each interval, giving N-1 evaluations

This is why vmapped dynamics process batches of size `(N-1, ...)` rather than `(N, ...)`.

### Constraint Vectorization

Non-convex nodal constraints are also vectorized, but with a key difference (in `openscvx/symbolic/lower.py:320-323`):

```python
# Vectorize constraint functions (during JAX lowering)
constraint = LoweredNodalConstraint(
    func=jax.vmap(fn, in_axes=(0, 0, None, None)),
    grad_g_x=jax.vmap(jacfwd(fn, argnums=0), in_axes=(0, 0, None, None)),
    grad_g_u=jax.vmap(jacfwd(fn, argnums=1), in_axes=(0, 0, None, None)),
    nodes=constraint.nodes,  # List of specific node indices where constraint applies
)
```

**Constraint Vmap Configuration: `in_axes=(0, 0, None, None)`**

Note the key difference from dynamics:
- **Axis 0 of x**: Batch over states
- **Axis 0 of u**: Batch over controls
- **None for node**: Node index is **not batched** (same value for all evaluations in a batch)
- **None for params**: Shared parameters (not batched)

**Why the difference?** Constraints are only evaluated at specific nodes (e.g., a collision avoidance constraint might only apply at nodes [2, 5, 7]). The constraint is vmapped to handle multiple constraint evaluations in parallel, but each evaluation receives the same `node` value since it's evaluating the same logical constraint at potentially different states/controls.

**Constraint Signature (After Vmap):**

```python
def g_vmapped(x_batch: Array, u_batch: Array, node: int, params: dict) -> Array:
    """Evaluate constraint at multiple state/control pairs simultaneously.

    Args:
        x_batch: State vectors, shape (batch_size, n_x)
        u_batch: Control vectors, shape (batch_size, n_u)
        node: Single node index (broadcast to all evaluations)
        params: Dictionary of problem parameters (shared across all evaluations)

    Returns:
        Constraint values, shape (batch_size,)
    """
    ...
```

Constraint Jacobians after vmap:

```python
grad_g_x_vmapped(x_batch, u_batch, node, params) -> Array[batch_size, n_x]
grad_g_u_vmapped(x_batch, u_batch, node, params) -> Array[batch_size, n_u]
```

When constraints are evaluated in practice:

```python
# Extract states/controls at nodes where constraint applies
x_batch = x[constraint.nodes]  # Shape: (len(nodes), n_x)
u_batch = u[constraint.nodes]  # Shape: (len(nodes), n_u)

# Evaluate constraint at all specified nodes
g_values = constraint.func(x_batch, u_batch, node_idx, params)  # Shape: (len(nodes),)
```

### Cross-Node Constraint Vectorization

Cross-node constraints are **not vmapped** in the traditional sense because they already operate on full trajectory arrays. Instead, they use a custom wrapper that evaluates the constraint pattern at multiple nodes.

**Key Difference from Regular Constraints:**

| Aspect | Regular Nodal Constraints | Cross-Node Constraints |
|--------|--------------------------|------------------------|
| **Input Shape** | Single-node vectors (n_x,), (n_u,) | Full trajectories (N, n_x), (N, n_u) |
| **Vectorization** | `jax.vmap` with `in_axes=(0, 0, None, None)` | Custom wrapper (no vmap needed) |
| **Signature** | `(x, u, node, params)` → scalar | `(X, U, params)` → (M,) |
| **Jacobians** | `(n_x,)`, `(n_u,)` | `(M, N, n_x)`, `(M, N, n_u)` |

**Cross-Node Constraint "Vectorization" via Wrapper:**

The `_create_cross_node_wrapper` function (in `openscvx/symbolic/lower.py:84-145`) is an internal helper that wraps the lowered constraint to evaluate at multiple trajectory nodes:

```python
def _create_cross_node_wrapper(constraint_fn, references, is_relative, eval_nodes):
    """Wrap constraint to evaluate at multiple nodes.

    For relative indexing (e.g., position.node('k') - position.node('k-1')):
    - eval_nodes specifies where to evaluate (e.g., [1, 2, 3, ..., N-1])
    - At each eval_node k, the pattern shifts: x[k] - x[k-1]
    """
    if is_relative:
        def trajectory_constraint(X, U, params):
            residuals = []
            for eval_idx in eval_nodes:
                # Evaluate constraint pattern at this node
                # NodeReference extracts x[k+offset] based on eval_idx
                residual = constraint_fn(X, U, eval_idx, params)
                residuals.append(residual)
            return jnp.stack(residuals, axis=0)  # Shape: (M,)
        return trajectory_constraint
```

**Cross-Node Constraint Signature (No Vmap Applied):**

```python
def g_cross(X: Array, U: Array, params: dict) -> Array:
    """Evaluate cross-node constraint - already operates on full trajectories.

    Args:
        X: Full state trajectory, shape (N, n_x)
        U: Full control trajectory, shape (N, n_u)
        params: Dictionary of problem parameters

    Returns:
        Constraint residuals at all evaluation points, shape (M,)
        where M = len(eval_nodes)
    """
    ...
```

**Cross-Node Constraint Jacobians (No Vmap Applied):**

```python
grad_g_X(X, U, params) -> Array[M, N, n_x]  # Full trajectory Jacobian
grad_g_U(X, U, params) -> Array[M, N, n_u]  # Full trajectory Jacobian
```

**Why No Vmap?**

Cross-node constraints need simultaneous access to multiple nodes (e.g., `x[k]` and `x[k-1]`), which isn't compatible with vmapping over single-node slices. Instead:

1. The constraint function receives full trajectories `X` and `U`
2. `NodeReference` nodes extract specific slices: `X[k+offset, :]`
3. The wrapper evaluates this pattern at each node in `eval_nodes`

**Example Evaluation:**

```python
# Rate limit constraint: ||position[k] - position[k-1]|| <= max_step
# Defined with: position.node('k') - position.node('k-1')
# Evaluated at nodes [1, 2, 3, ..., N-1]

# During SCP iteration with current trajectory guess:
X = trajectory_guess  # Shape: (N, n_x) - full state trajectory
U = control_guess     # Shape: (N, n_u) - full control trajectory

# Evaluate cross-node constraint
residuals = constraint.func(X, U, params)  # Shape: (N-1,)
# residuals[0] = ||position[1] - position[0]|| - max_step
# residuals[1] = ||position[2] - position[1]|| - max_step
# ...

# Compute trajectory-level Jacobians
grad_X = constraint.grad_g_X(X, U, params)  # Shape: (N-1, N, n_x)
grad_U = constraint.grad_g_U(X, U, params)  # Shape: (N-1, N, n_u)

# Sparse structure example: grad_X[i, :, :] for residuals[i]
# Only non-zero at nodes i and i+1 (for k and k-1 pattern)
```

**Performance Note:**

Cross-node constraints operate on full trajectories and use JAX-native vectorization for efficient evaluation. The constraint function and its Jacobians are JIT-compiled, enabling efficient execution on GPU/TPU hardware.

## Usage in Discretization

The vmapped dynamics functions are called during discretization (in `openscvx/discretization.py:73-86`):

```python
# Setup batch inputs
x = V[:, :n_x]                          # Shape: (N-1, n_x) - States at interval starts
u = u[: x.shape[0]]                     # Shape: (N-1, n_u) - Controls (includes time dilation)
nodes = jnp.arange(0, N-1)              # Shape: (N-1,) - Node indices

# Extract time dilation (last control dimension)
s = u[:, -1]                            # Shape: (N-1,) - Time dilation values

# Call vmapped dynamics - evaluates all intervals in parallel
# Note: dynamics receive u[:, :-1] (vehicle controls only, excluding time dilation)
f = state_dot(x, u[:, :-1], nodes, params)  # Shape: (N-1, n_x)
dfdx = A(x, u[:, :-1], nodes, params)       # Shape: (N-1, n_x, n_x)
dfdu_veh = B(x, u[:, :-1], nodes, params)   # Shape: (N-1, n_x, n_u-1)

# Build full control Jacobian including time dilation
dfdu = jnp.zeros((x.shape[0], n_x, n_u))
dfdu = dfdu.at[:, :, :-1].set(s[:, None, None] * dfdu_veh)  # Vehicle control derivatives
dfdu = dfdu.at[:, :, -1].set(f)                              # Time dilation derivative = f
```

**Why exclude time dilation from dynamics?** Time dilation is a meta-control that scales the entire dynamics (used for time-optimal problems). The actual vehicle dynamics are defined without it, and time dilation is applied as a scaling factor during discretization. This is why `n_u-1` appears in the vehicle dynamics Jacobians.

**Example with N=10:** This single call evaluates dynamics at all 9 intervals simultaneously, leveraging JAX's efficient vectorization on GPU/TPU.

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
| **JAX Functions (Pre-Vmap)** | **Dynamics:** | | |
| | `f(x, u, node, params)` | Input: `(n_x,), (n_u,), scalar, dict` | Input: `(4,), (2,), scalar, dict` |
| | | Output: `(n_x,)` | Output: `(4,)` - Single state derivative |
| | `A(x, u, node, params)` | Output: `(n_x, n_x)` | Output: `(4, 4)` - Jacobian df/dx |
| | `B(x, u, node, params)` | Output: `(n_x, n_u)` | Output: `(4, 2)` - Jacobian df/du |
| | **Constraints:** | | |
| | `g(x, u, node, params)` | Input: `(n_x,), (n_u,), scalar, dict` | Input: `(4,), (2,), scalar, dict` |
| | | Output: `scalar` | Output: `scalar` - Single constraint value |
| | `grad_g_x(x, u, node, params)` | Output: `(n_x,)` | Output: `(4,)` - Gradient dg/dx |
| | `grad_g_u(x, u, node, params)` | Output: `(n_u,)` | Output: `(2,)` - Gradient dg/du |
| | | | |
| **JAX Functions (Post-Vmap)** | **Dynamics:** | | |
| | `f(x, u, nodes, params)` | Input: `(N-1, n_x), (N-1, n_u), (N-1,), dict` | Input: `(9, 4), (9, 2), (9,), dict` |
| | | Output: `(N-1, n_x)` | Output: `(9, 4)` - Derivatives at 9 intervals |
| | `A(x, u, nodes, params)` | Output: `(N-1, n_x, n_x)` | Output: `(9, 4, 4)` - Jacobians at 9 intervals |
| | `B(x, u, nodes, params)` | Output: `(N-1, n_x, n_u)` | Output: `(9, 4, 2)` - Jacobians at 9 intervals |
| | **Constraints:** | | |
| | `g(x, u, node, params)` | Input: `(M, n_x), (M, n_u), scalar, dict` | Input: `(3, 4), (3, 2), scalar, dict` |
| | | Output: `(M,)` | Output: `(3,)` - M=3 constraint evaluations |
| | `grad_g_x(x, u, node, params)` | Output: `(M, n_x)` | Output: `(3, 4)` - Gradients at M nodes |
| | `grad_g_u(x, u, node, params)` | Output: `(M, n_u)` | Output: `(3, 2)` - Gradients at M nodes |
| | **Cross-Node Constraints:** | | |
| | `g_cross(X, U, params)` | Input: `(N, n_x), (N, n_u), dict` | Input: `(10, 4), (10, 2), dict` |
| | | Output: `(M,)` | Output: `(9,)` - Rate limit at N-1 nodes |
| | `grad_g_X(X, U, params)` | Output: `(M, N, n_x)` | Output: `(9, 10, 4)` - Trajectory Jacobian |
| | `grad_g_U(X, U, params)` | Output: `(M, N, n_u)` | Output: `(9, 10, 2)` - Trajectory Jacobian |
| | | **Note:** Jacobians are dense but sparse | **Sparsity:** Only 2 nodes per row non-zero |

## Performance Implications

**Why This Architecture?**

1. **GPU/TPU Acceleration**: Vmapping enables SIMD parallelism across nodes for both dynamics and constraints
2. **JIT Compilation**: JAX compiles vmapped functions once, not per-node
3. **Automatic Differentiation**: Jacobians and gradients computed automatically via `jacfwd`
4. **Reduced Python Overhead**: Single JAX call instead of Python loops for evaluation

## Implementation Files Reference

| **File** | **Function/Class** | **Purpose** |
|----------|-------------------|-------------|
| `examples/abstract/brachistochrone.py` | — | Example problem definition |
| `examples/abstract/brachistochrone_rate_limit.py` | — | Example with cross-node rate limiting |
| `openscvx/trajoptproblem.py` | `TrajOptProblem.__init__` | Orchestrates preprocessing pipeline |
| `openscvx/symbolic/builder.py` | `preprocess_symbolic_problem` | Augments states/controls/dynamics |
| `openscvx/symbolic/lower.py` | `lower_symbolic_expressions` | Unification and JAX lowering for dynamics/constraints |
| `openscvx/symbolic/lower.py:464-523` | Cross-node constraint lowering | Separates and lowers cross-node constraints |
| `openscvx/symbolic/lower.py:84-145` | `_create_cross_node_wrapper` | Wraps constraints for trajectory-level evaluation |
| `openscvx/symbolic/expr/expr.py:218-313` | `NodeReference` class | Enables `.node('k')` syntax for cross-node refs |
| `openscvx/symbolic/lowerers/jax.py:938-1044` | `JaxLowerer._visit_node_reference` | Lowers NodeReference to JAX array indexing |
| `openscvx/constraints/cross_node.py` | `CrossNodeConstraintLowered` | Container for lowered cross-node constraints |
| `openscvx/symbolic/unified.py` | `unify_states`, `unify_controls` | Combines individual variables |
| `openscvx/trajoptproblem.py:356-358` | `initialize` | Applies vmap to dynamics |
| `openscvx/trajoptproblem.py:372-375` | `initialize` | JIT compiles cross-node constraints |
| `openscvx/ocp.py:91-116` | `create_cvxpy_variables` | Creates CVXPy variables for cross-node constraints |
| `openscvx/ocp.py:333-350` | `OptimalControlProblem` | Adds linearized cross-node constraints to OCP |
| `openscvx/ptr.py:289-301` | `PTR_subproblem` | Updates cross-node constraint parameters |
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

## Common Developer Pitfalls

1. **Confusing nodes vs intervals**: Discretization operates on N-1 intervals between N nodes, so vmapped dynamics have batch size (N-1, ...), while constraints evaluate at specific nodes (batch size M where M = number of nodes where constraint applies)
2. **Forgetting augmented dimensions**: `n_x` and `n_u` include auto-added states/controls (time, CTCS augmented states, time dilation)
3. **Parameter mutability**: The `params` dict is shared across all evaluations - don't modify it during dynamics or constraint evaluation
4. **Node index usage**: The `node` parameter enables time-varying behavior (e.g., time-dependent constraints), not for indexing into trajectory arrays
5. **Constraint vs dynamics vmap axes**: Constraints use `in_axes=(0, 0, None, None)` (node not batched), while dynamics use `in_axes=(0, 0, 0, None)` (node batched across intervals)
6. **Cross-node constraint signature confusion**: Regular nodal constraints use `(x, u, node, params)` while cross-node constraints use `(X, U, params)` - don't mix them up
7. **Cross-node Jacobian sparsity**: Cross-node Jacobians have shape `(M, N, n_x)` but are typically very sparse (e.g., rate limits only couple 2 nodes). Be aware of memory usage for large N
8. **Mixing relative and absolute node references**: Cannot mix `.node('k')` and `.node(0)` in the same constraint expression - use one indexing mode consistently

## See Also

- [Basic Problem Setup](../Usage/basic_problem_setup.md) - How to define problems
- [API: State](../Usage/api_state.md) - State class documentation
- [API: Control](../Usage/api_control.md) - Control class documentation
- [API: TrajOptProblem](../Usage/api_trajoptproblem.md) - Main problem class
- [Discretization](../Overview/discretization.md) - How discretization works in OpenSCvx
