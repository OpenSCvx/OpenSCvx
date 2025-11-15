# Examples

OpenSCvx comes with a comprehensive set of examples demonstrating various trajectory optimization problems. These examples are located in the `examples/` folder and cover different applications and complexity levels.

## Running Examples

See `examples/params/` folder for several example trajectory optimization problems.
To run a problem simply run any of the examples directly, for example:

```sh
python3 examples/params/brachistochrone.py
```

and adjust the plotting as needed.

## Example Categories

### Abstract Problems
- **Brachistochrone**: Classic minimum-time problem with gravity
- **3DoF Rocket Landing**: Rocket landing with fuel optimization

### Drone Applications
- **Obstacle Avoidance**: 6DoF drone navigating around obstacles
- **Line-of-Sight Guidance**: Drone maintaining line-of-sight to keypoints while passing through gates
- **Cinema View Planning**: Camera drone planning for cinematic shots
- **Drone Racing**: High-speed racing through gates

### Ground Vehicle Applications
- **Dubins Car**: Simple 2D vehicle with turning constraints
- **Dubins Car with Waypoints**: Navigation through multiple waypoints

### Real-time Applications
- **Real-time Drone Racing**: Live optimization during flight
- **Real-time Obstacle Avoidance**: Dynamic obstacle avoidance

## Creating Your Own Problems

Check out the problem definitions inside `examples/params` to see how to define your own problems. Each example demonstrates:

- State and control variable definition
- Dynamics specification
- Constraint formulation
- Problem instantiation and solving
- Results visualization

## Example Structure

Most examples follow this structure:

1. **Imports**: Import necessary OpenSCvx modules
2. **Problem Setup**: Define parameters, state, and control variables
3. **Dynamics**: Specify the system dynamics
4. **Constraints**: Define path and boundary constraints
5. **Problem Instantiation**: Create and configure the TrajOptProblem
6. **Solving**: Run the optimization
7. **Visualization**: Plot and analyze results

## Interactive Examples

For interactive 3D visualization, many examples include GUI plotting functions:

```python
from examples.plotting import plot_animation_pyqtgraph

# After solving the problem
plot_animation_pyqtgraph(results).show()
```

## Performance Tips

- Start with simpler examples to understand the workflow
- Use the provided initial guesses as starting points
- Adjust SCP weights based on your specific problem
- Consider using CVXPYGen for faster performance on smaller problems 