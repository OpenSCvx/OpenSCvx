"""Pytest session configuration for the OpenSCvx test suite.

Optional third-party dependencies are declared at test sites with a plain
per-package marker (``pytest.mark.qpax``, ``pytest.mark.mjx``, ...) whose name
matches the extra in ``pyproject.toml [project.optional-dependencies]``. The
collection hook below derives everything else: the umbrella ``extras`` marker
(so CI can select all optional-dependency tests at once) and a skip when the
package is not installed. Tests under ``tests/e2e/`` are auto-marked ``e2e``.
"""

import importlib.util
from pathlib import Path

import pytest


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


# Marker name == extra name in pyproject.toml [project.optional-dependencies].
OPTIONAL_DEPS = {
    "mjx": lambda: _installed("mujoco") and _installed("mujoco.mjx"),
    "qpax": lambda: _installed("qpax"),
    "cvxpygen": lambda: _installed("cvxpygen") and _installed("qocogen"),
    "lie": lambda: _installed("jaxlie"),
    "moreau": lambda: _installed("moreau"),
    # Not a pip extra: ICORE is not on PyPI yet, so it is used from a clone on
    # PYTHONPATH (see openscvx.solvers.icore_ptr_solver).
    "icore": lambda: _installed("icore"),
}

# Install hint shown in the skip reason; defaults to the pip extra.
_INSTALL_HINTS = {
    "icore": "clone https://github.com/sametuzun781/ICORE onto PYTHONPATH",
}

_E2E_DIR = Path(__file__).parent / "e2e"


def pytest_collection_modifyitems(config, items):
    available = {dep: probe() for dep, probe in OPTIONAL_DEPS.items()}
    for item in items:
        if _E2E_DIR in item.path.parents:
            item.add_marker(pytest.mark.e2e)
        for dep, ok in available.items():
            if item.get_closest_marker(dep):
                item.add_marker(pytest.mark.extras)
                if not ok:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"requires optional dependency '{dep}' "
                            f"({_INSTALL_HINTS.get(dep, f'pip install openscvx[{dep}]')})"
                        )
                    )
