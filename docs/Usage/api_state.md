# State

State represents a dynamic state variable in a trajectory optimization problem. Unlike control inputs, states evolve according to dynamics constraints and can have boundary conditions specified at the initial and final time points. Like all Variables, States also support min/max bounds and initial trajectory guesses to help guide the optimization solver toward good solutions.

::: openscvx.symbolic.expr.state.State
    options:
      show_source: false
      show_root_heading: true

## Boundary Conditions

States support four types of boundary conditions at initial and final time points. Each element of a multi-dimensional state can have different boundary condition types. Boundary conditions are specified using either a simple number (defaults to "fixed") or a tuple of (type, value).

### BoundaryType Enum

::: openscvx.symbolic.expr.state.BoundaryType
    options:
      show_source: false
      show_root_heading: true

### Boundary Condition Types

- **fixed**: State value is constrained to a specific value (use plain number or tuple `("fixed", value)`)
- **free**: State value is optimized within bounds, initialized at the given value (use tuple `("free", value)`)
- **minimize**: Adds objective term to minimize the state value (use tuple `("minimize", value)`)
- **maximize**: Adds objective term to maximize the state value (use tuple `("maximize", value)`)
