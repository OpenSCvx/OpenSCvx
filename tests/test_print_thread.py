"""Re-initializing a Problem replaces its printing thread instead of adding one."""

from tests.solvers._iteration_callback_helpers import build_brachistochrone


def test_reinitialize_stops_previous_print_thread():
    prob = build_brachistochrone("cvxpy", n=4, k_max=1)
    prob.settings.dev.printing = True
    prob.initialize()
    first = prob.print_thread
    prob.initialize()
    assert not first.is_alive()
    assert prob.print_thread.is_alive()
