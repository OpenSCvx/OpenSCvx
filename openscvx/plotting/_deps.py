"""Fail fast when the plotting extra is not installed.

Imported by :mod:`openscvx.plotting` before Plotly, Matplotlib, or Viser.
``import openscvx`` does not import this module, so solving a problem does not
require the plotting stack.
"""

from importlib.util import find_spec

_REQUIRED = ("matplotlib", "plotly", "viser")

_missing = [name for name in _REQUIRED if find_spec(name) is None]
if _missing:
    missing = ", ".join(_missing)
    raise ImportError(
        "openscvx.plotting requires the optional plotting extra "
        f"(missing {missing}). Install it with: pip install 'openscvx[plotting]'"
    )
