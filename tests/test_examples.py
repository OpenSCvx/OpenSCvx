"""
Automatically discover and test all examples in the examples/ directory.

This test discovers all Python files in examples/ that define a 'problem' variable
and validates that they converge successfully.
"""

import importlib.util
import sys
from pathlib import Path

import jax
import pytest

IGNORED_FILES = ["__init__.py", "plotting.py"]


def discover_examples():
    """Discover all runnable examples in the examples/ directory."""
    examples_dir = Path(__file__).parent.parent / "examples"
    discovered = {}

    # Find all .py files in examples/
    for py_file in examples_dir.rglob("*.py"):
        # Skip non-example files
        if py_file.name in IGNORED_FILES:
            continue

        # Skip realtime examples (require special event loop handling)
        if "realtime" in py_file.parts:
            continue

        # Get relative path for naming
        rel_path = py_file.relative_to(examples_dir)
        module_name = str(rel_path.with_suffix("")).replace("/", ".")

        try:
            # Import the example module
            spec = importlib.util.spec_from_file_location(f"examples.{module_name}", py_file)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"examples.{module_name}"] = module
            spec.loader.exec_module(module)

            # Only include if it has a 'problem' attribute
            if hasattr(module, "problem"):
                test_name = module_name.replace(".", "_")
                discovered[test_name] = {
                    "problem": module.problem,
                    "path": str(rel_path),
                }

        except Exception as e:
            # Skip files that can't be imported
            print(f"Warning: Could not import {rel_path}: {e}")
            continue

    return discovered


# Discover examples at module load time
DISCOVERED_EXAMPLES = discover_examples()


@pytest.mark.integration
@pytest.mark.parametrize(
    "name,metadata", DISCOVERED_EXAMPLES.items(), ids=list(DISCOVERED_EXAMPLES.keys())
)
def test_example(name, metadata):
    """
    Test that a discovered example converges successfully.

    Each example is run through:
    1. problem.initialize()
    2. problem.solve()
    3. problem.post_process()
    4. Assert convergence
    """
    problem = metadata["problem"]

    # Disable printing for cleaner test output
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False

    # Disable custom integrator for stability (used in some drone examples)
    if hasattr(problem.settings, "dis"):
        problem.settings.dis.custom_integrator = False

    # Run the optimization pipeline
    problem.initialize()
    result = problem.solve()
    result = problem.post_process(result)

    # Check convergence
    assert result["converged"], f"Example {name} ({metadata['path']}) failed to converge"

    # Clean up JAX caches
    jax.clear_caches()


def test_discovery_report():
    """Report discovered examples."""
    print(f"\nDiscovered {len(DISCOVERED_EXAMPLES)} examples for integration testing:")
    for name, metadata in sorted(DISCOVERED_EXAMPLES.items()):
        print(f"  - {name:40s} ({metadata['path']})")
    assert len(DISCOVERED_EXAMPLES) > 0, "No examples were discovered!"
