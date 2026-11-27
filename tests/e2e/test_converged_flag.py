"""``converged`` comes from the convergence test, not the iteration counter."""

from tests.solvers._iteration_callback_helpers import build_brachistochrone


def test_converged_on_last_allowed_iteration():
    prob = build_brachistochrone("cvxpy", n=4, k_max=50)
    prob.initialize()
    first = prob.solve()
    assert first.converged
    n_iters = int(prob.state.k) - 1

    # Re-run capped at exactly the iteration that converges: ``k`` ends at
    # ``max_iters + 1``, which the old ``k <= k_max`` flag reported as False.
    prob.reset()
    capped = prob.solve(max_iters=n_iters)
    assert int(prob.state.k) == n_iters + 1
    assert capped.converged
    assert prob.post_process().converged
