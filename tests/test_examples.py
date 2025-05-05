import pytest

import jax

from examples.params.cinema_vp import problem as cinema_vp_problem
from examples.params.cinema_vp import plotting_dict as cinema_vp_plotting_dict
from examples.params.dr_vp import problem as dr_vp_problem
from examples.params.dr_vp import plotting_dict as dr_vp_plotting_dict
from examples.params.obstacle_avoidance import problem as obstacle_avoidance_problem
from examples.params.obstacle_avoidance import plotting_dict as obstacle_avoidance_plotting_dict
from examples.params.dr_vp_nodal import problem as dr_vp_polytope_problem
from examples.params.dr_vp_nodal import plotting_dict as dr_vp_polytope_plotting_dict
from examples.plotting import plot_camera_animation, plot_animation, plot_scp_animation


def test_obstacle_avoidance():
    # This test is specific to the obstacle avoidance problem
    problem = obstacle_avoidance_problem
    problem.initialize()
    result = problem.solve()
    result = problem.post_process(result)

    result.update(obstacle_avoidance_plotting_dict)

    plot_animation(result, problem.params)
    plot_scp_animation(result, problem.params)
    
    # Assuming PTR_main returns a dictionary
    output_dict = result
    
    scp_iters = len(result['discretization_history'])
    prop_constr_vio = result['x_full'][:,-1][-1]
    sol_constr_vio = result['x'][:,-1][-1]
    prop_cost = result['x_full'][:,-2][-1]
    sol_cost = result['x'][:,-2][-1]

    assert sol_cost < 2.0, "Problem failed with solution cost"
    assert prop_cost < 2.0, "Problem failed with propagated cost"
    assert sol_constr_vio < 1e-3, "Problem failed with solution constraint violation"
    assert prop_constr_vio < 1e-3, "Problem failed with propagated constraint violation"
    assert scp_iters < 10, "Problem took more then expected iterations"
    assert output_dict['converged'], "Problem failed with output"

    assert problem.timing_init < 6.0, "Problem took more then expected initialization time"
    assert problem.timing_solve < 0.2, "Problem took more then expected solve time"
    assert problem.timing_post < 0.2, "Problem took more then expected post process time"
    
    # Clean up jax memory usage
    jax.clear_caches()

def test_dr_vp_nodal():
    # This test is specific to the dr_vp_nodal problem
    problem = dr_vp_polytope_problem
    problem.params.dis.custom_integrator = False
    problem.initialize()
    result = problem.solve()
    result = problem.post_process(result)

    result.update(dr_vp_plotting_dict)

    plot_animation(result, problem.params)
    plot_camera_animation(result, problem.params)
    plot_scp_animation(result, problem.params)
    
    # Assuming PTR_main returns a dictionary
    output_dict = result
    prop_constr_vio = result['x_full'][:,-1][-1]
    sol_constr_vio = result['x'][:,-1][-1]
    prop_cost = result['x_full'][:,-2][-1]
    sol_cost = result['x'][:,-2][-1]

    assert sol_cost < 30.0, "Problem failed with solution cost"
    assert prop_cost < 30.0, "Problem failed with propagated cost"
    assert sol_constr_vio < 1e-3, "Problem failed with solution constraint violation"
    assert prop_constr_vio < 1e-3, "Problem failed with propagated constraint violation"
    assert output_dict['converged'], "DR VP Nodal Process failed with output"

    assert problem.timing_init < 20.0, "Problem took more then expected initialization time"
    assert problem.timing_solve < 2.0, "Problem took more then expected solve time"
    assert problem.timing_post < 0.5, "Problem took more then expected post process time"
    
    # Clean up jax memory usage
    jax.clear_caches()

def test_dr_vp():
    # This test is specific to the dr_vp problem
    problem = dr_vp_problem
    problem.params.dis.custom_integrator = False
    problem.initialize()
    result = problem.solve()
    result = problem.post_process(result)

    result.update(dr_vp_plotting_dict)

    plot_animation(result, problem.params)
    plot_camera_animation(result, problem.params)
    plot_scp_animation(result, problem.params)
    
    # Assuming PTR_main returns a dictionary
    output_dict = result
    prop_constr_vio = result['x_full'][:,-1][-1]
    sol_constr_vio = result['x'][:,-1][-1]
    prop_cost = result['x_full'][:,-2][-1]
    sol_cost = result['x'][:,-2][-1]
    
    assert sol_cost < 45.0, "Problem failed with solution cost"
    assert prop_cost < 45.0, "Problem failed with propagated cost"
    assert sol_constr_vio < 1e0, "Problem failed with solution constraint violation"
    assert prop_constr_vio < 1e0, "Problem failed with propagated constraint violation"
    assert output_dict['converged'], "DR VP Process failed with output"

    assert problem.timing_init < 25.0, "Problem took more then expected initialization time"
    assert problem.timing_solve < 2.0, "Problem took more then expected solve time"
    assert problem.timing_post < 0.5, "Problem took more then expected post process time"
    
    # Clean up jax memory usage
    jax.clear_caches()

def test_cinema_vp():
    # This test is specific to the cinema_vp problem
    problem = cinema_vp_problem
    problem.params.dis.custom_integrator = False
    problem.initialize()
    result = problem.solve()
    result = problem.post_process(result)

    result.update(cinema_vp_plotting_dict)

    plot_animation(result, problem.params)
    plot_camera_animation(result, problem.params)
    plot_scp_animation(result, problem.params)
    
    # Assuming PTR_main returns a dictionary
    output_dict = result
    prop_constr_vio = result['x_full'][:,-1][-1]
    sol_constr_vio = result['x'][:,-1][-1]
    prop_cost = result['x_full'][:,-3][-1]
    sol_cost = result['x'][:,-3][-1]
    
    assert sol_cost < 400.0, "Problem failed with solution cost"
    assert prop_cost < 400.0, "Problem failed with propagated cost"
    assert sol_constr_vio < 1E0, "Problem failed with solution constraint violation"
    assert prop_constr_vio < 1e0, "Problem failed with propagated constraint violation"
    assert output_dict['converged'], "Cinema VP Process failed with output"

    assert problem.timing_init < 8.0, "Problem took more then expected initialization time"
    assert problem.timing_solve < 0.5, "Problem took more then expected solve time"
    assert problem.timing_post < 0.5, "Problem took more then expected post process time"
    
    # Clean up jax memory usage
    jax.clear_caches()
