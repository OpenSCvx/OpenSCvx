"""Tests for variable nodes.

This module tests variable node types: Variable, State, Control.
Tests cover:
- Node creation and properties
- Bounds and constraints
- Slice assignment and usage
- Lowering to JAX (with slices)
- Lowering to CVXPY (with variable mapping)
"""

import numpy as np
import pytest
