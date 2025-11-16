"""Tests for spatial/6DOF operation nodes.

This module tests spatial operation nodes for aerospace and robotics applications:
- QDCM: Quaternion to Direction Cosine Matrix
- SSMP: 4×4 skew-symmetric matrix for quaternion dynamics
- SSM: 3×3 skew-symmetric matrix for cross products

Tests cover:
- Node creation and mathematical correctness
- Integration in 6DOF rigid body dynamics
- Lowering to JAX
"""

import numpy as np
import pytest
