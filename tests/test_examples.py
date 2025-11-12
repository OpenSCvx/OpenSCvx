import os
import platform

import jax
import pytest

from examples.drone.cinema_vp import plotting_dict as cinema_vp_plotting_dict
from examples.drone.cinema_vp import problem as cinema_vp_problem
from examples.drone.dr_vp import plotting_dict as dr_vp_plotting_dict
from examples.drone.dr_vp import problem as dr_vp_problem
from examples.drone.dr_vp_nodal import plotting_dict as dr_vp_polytope_plotting_dict
from examples.drone.dr_vp_nodal import problem as dr_vp_polytope_problem
from examples.drone.obstacle_avoidance import (
    plotting_dict as obstacle_avoidance_plotting_dict,
)
from examples.drone.obstacle_avoidance import problem as obstacle_avoidance_problem
from examples.plotting import plot_animation, plot_camera_animation, plot_scp_animation

# Import pyqtgraph testing helpers
from tests.test_pyqtgraph_helpers import run_pyqtgraph_function_headless

CI_OS = os.getenv("RUNNER_OS", platform.system())

OS_MULTIPLIER = {
    "Windows": 1.5,  # give Windows twice as much time
    "Linux": 1.0,
    "Darwin": 1.0,
}.get(CI_OS, 1.0)

TEST_CASES = {
    "obstacle_avoidance": {
        "problem": obstacle_avoidance_problem,
        "plotting_dict": obstacle_avoidance_plotting_dict,
        "plot_funcs": [plot_animation, plot_scp_animation],
        "pyqtgraph_funcs": ["plot_animation_pyqtgraph", "plot_scp_animation_pyqtgraph"],
        "cost_idx": -2,
        "vio_idx": -1,
        "max_cost": 2.0,
        "max_vio": 1e-3,
        "max_iters": 10,
        "timing": {"init": 20.0, "solve": 0.5, "post": 5.0},
        # no custom integrator flag
    },
    "dr_vp_nodal": {
        "problem": dr_vp_polytope_problem,
        "plotting_dict": dr_vp_polytope_plotting_dict,
        "plot_funcs": [plot_animation, plot_camera_animation, plot_scp_animation],
        "pyqtgraph_funcs": [
            "plot_animation_pyqtgraph",
            "plot_camera_animation_pyqtgraph",
            "plot_scp_animation_pyqtgraph",
        ],
        "cost_idx": -2,
        "vio_idx": -1,
        "max_cost": 30.0,
        "max_vio": -1,
        "timing": {"init": 35.0, "solve": 2.0, "post": 6.0},
        "pre_init": [
            lambda p: setattr(p.settings.dis, "custom_integrator", False),
            lambda p: setattr(p.settings.dev, "printing", False),
        ],
    },
    "dr_vp": {
        "problem": dr_vp_problem,
        "plotting_dict": dr_vp_plotting_dict,
        "plot_funcs": [plot_animation, plot_camera_animation, plot_scp_animation],
        "pyqtgraph_funcs": [
            "plot_animation_pyqtgraph",
            "plot_camera_animation_pyqtgraph",
            "plot_scp_animation_pyqtgraph",
        ],
        "cost_idx": -2,
        "vio_idx": -1,
        "max_cost": 45.0,
        "max_vio": 1.0,
        "timing": {"init": 60.0, "solve": 2.0, "post": 6.0},
        "pre_init": [
            lambda p: setattr(p.settings.dis, "custom_integrator", False),
            lambda p: setattr(p.settings.dev, "printing", False),
        ],
    },
    "cinema_vp": {
        "problem": cinema_vp_problem,
        "plotting_dict": cinema_vp_plotting_dict,
        "plot_funcs": [plot_animation, plot_camera_animation, plot_scp_animation],
        "pyqtgraph_funcs": [
            "plot_animation_pyqtgraph",
            "plot_camera_animation_pyqtgraph",
            "plot_scp_animation_pyqtgraph",
        ],
        "cost_idx": -3,
        "vio_idx": -1,
        "max_cost": 400.0,
        "max_vio": 1.0,
        "timing": {"init": 20.0, "solve": 1.1, "post": 5.0},
        "pre_init": [
            lambda p: setattr(p.settings.dis, "custom_integrator", False),
            lambda p: setattr(p.settings.dev, "printing", False),
        ],
    },
    # "brachistochrone": {
    #     "problem": brachistochrone_problem,
    #     "plotting_dict": {},
    #     "plot_funcs": [],
    #     "pyqtgraph_funcs": [],
    #     "cost_idx": -2,
    #     "vio_idx": -1,
    #     "max_cost": 1.81,
    #     "max_vio": 1.0,
    #     "max_iters": 5,
    #     "timing": {"init": 1E2, "solve": 0.01, "post": 5.0},
    #     "pre_init": [
    #         lambda p: setattr(p.settings.cvx, "solver", "qocogen"),
    #         lambda p: setattr(p.settings.cvx, "cvxpygen", True),
    #         lambda p: setattr(p.settings.cvx, "cvxpygen_override", True),
    #     ],
    # },
}

for conf in TEST_CASES.values():
    base = conf["timing"]
    conf["timing"] = {phase: base[phase] * OS_MULTIPLIER for phase in base}


@pytest.mark.parametrize("name,conf", TEST_CASES.items(), ids=list(TEST_CASES))
def test_example_problem(name, conf):
    problem = conf["problem"]
    # apply any pre‐init hooks
    if "pre_init" in conf:
        hooks = conf["pre_init"]
        # normalize to list
        if callable(hooks):
            hooks = [hooks]
        for fn in hooks:
            fn(problem)

    problem.initialize()
    result = problem.solve()
    result = problem.post_process(result)

    # merge in plotting metadata and run plots
    result.update(conf["plotting_dict"])
    for fn in conf["plot_funcs"]:
        fn(result, problem.settings)

    # Test pyqtgraph functions if specified
    if conf.get("pyqtgraph_funcs"):
        try:
            from examples.plotting import (
                plot_animation_pyqtgraph,
                plot_camera_animation_pyqtgraph,
                plot_scp_animation_pyqtgraph,
            )

            # Map function names to actual functions
            pyqtgraph_func_map = {
                "plot_animation_pyqtgraph": plot_animation_pyqtgraph,
                "plot_scp_animation_pyqtgraph": plot_scp_animation_pyqtgraph,
                "plot_camera_animation_pyqtgraph": plot_camera_animation_pyqtgraph,
            }

            # Test each pyqtgraph function
            for func_name in conf["pyqtgraph_funcs"]:
                if func_name in pyqtgraph_func_map:
                    func = pyqtgraph_func_map[func_name]
                    success = run_pyqtgraph_function_headless(func, result, problem.settings)
                    if not success:
                        print(f"Warning: pyqtgraph function {func_name} failed for {name}")

        except ImportError as e:
            # Skip pyqtgraph tests if not available
            print(f"Skipping pyqtgraph tests for {name}: {e}")
            # Don't fail the test if GUI packages are not available
            pass
        except Exception as e:
            print(f"Error testing pyqtgraph functions for {name}: {e}")
            # Don't fail the test for GUI-related errors
            pass

    # extract metrics
    scp_iters = len(result["discretization_history"])
    sol_cost = result.x.guess[:, conf["cost_idx"]][-1]
    prop_cost = result.x_full[:, conf["cost_idx"]][-1]
    sol_constr_vio = result.x.guess[:, conf["vio_idx"]][-1]
    prop_constr_vio = result.x_full[:, conf["vio_idx"]][-1]

    # assertions
    if conf["max_cost"] > 0.0:
        assert sol_cost < conf["max_cost"], "Problem failed with solution cost"
        assert prop_cost < conf["max_cost"], "Problem failed with propagated cost"
    if conf["max_vio"] > 0.0:
        assert sol_constr_vio < conf["max_vio"], "Problem failed with solution constraint violation"
        assert prop_constr_vio < conf["max_vio"], (
            "Problem failed with propagated constraint violation"
        )
    if "max_iters" in conf and conf["max_iters"] > 0:
        assert scp_iters < conf["max_iters"], "Problem took more then expected iterations"
    assert result["converged"], "Problem failed with output"

    # timing checks
    t = conf["timing"]
    assert problem.timing_init < t["init"], "Problem took more then expected initialization time"
    assert problem.timing_solve < t["solve"], "Problem took more then expected solve time"
    assert problem.timing_post < t["post"], "Problem took more then expected post process time"

    # clean up
    jax.clear_caches()
