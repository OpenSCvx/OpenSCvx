"""Tests for array operation nodes.

This module tests array operation nodes: Index, Concat, Stack, Hstack, Vstack.
Tests cover:
- Node creation and indexing/slicing operations
- Concatenation and stacking operations
- Shape inference
- Lowering to JAX
- Lowering to CVXPY
- Canonicalization patterns
"""

import numpy as np
import pytest
