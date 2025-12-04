"""Cache management for compiled solvers.

This module provides utilities for managing the cache directory where compiled
JAX solvers are stored. The cache location follows platform conventions:

- **Linux**: ``~/.cache/openscvx/``
- **macOS**: ``~/Library/Caches/openscvx/``
- **Windows**: ``%LOCALAPPDATA%/openscvx/Cache/``

The cache location can be overridden by setting the ``OPENSCVX_CACHE_DIR``
environment variable.

Example:
    Get the cache directory::

        import openscvx as ox
        print(ox.get_cache_dir())  # /home/user/.cache/openscvx

    Clear all cached solvers::

        import openscvx as ox
        ox.clear_cache()

    Check cache size::

        import openscvx as ox
        size_mb = ox.get_cache_size() / (1024 * 1024)
        print(f"Cache size: {size_mb:.1f} MB")
"""

import os
import shutil
import sys
from pathlib import Path


def get_cache_dir() -> Path:
    """Get the cache directory for compiled solvers.

    The cache location is determined in the following order:
    1. ``OPENSCVX_CACHE_DIR`` environment variable (if set)
    2. Platform-specific default:
       - Linux: ``~/.cache/openscvx/``
       - macOS: ``~/Library/Caches/openscvx/``
       - Windows: ``%LOCALAPPDATA%/openscvx/Cache/``

    Returns:
        Path to the cache directory (may not exist yet)
    """
    # Check environment variable override
    env_dir = os.environ.get("OPENSCVX_CACHE_DIR")
    if env_dir:
        return Path(env_dir)

    # Platform-specific defaults
    if sys.platform == "darwin":
        # macOS: ~/Library/Caches/openscvx/
        return Path.home() / "Library" / "Caches" / "openscvx"
    elif sys.platform == "win32":
        # Windows: %LOCALAPPDATA%/openscvx/Cache/
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "openscvx" / "Cache"
        else:
            # Fallback if LOCALAPPDATA not set
            return Path.home() / "AppData" / "Local" / "openscvx" / "Cache"
    else:
        # Linux and others: follow XDG Base Directory Specification
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            return Path(xdg_cache) / "openscvx"
        else:
            return Path.home() / ".cache" / "openscvx"


def clear_cache() -> int:
    """Clear all cached compiled solvers.

    Removes all files in the cache directory. The directory itself is
    preserved but emptied.

    Returns:
        Number of files deleted

    Example:
        Clear the cache::

            import openscvx as ox
            deleted = ox.clear_cache()
            print(f"Deleted {deleted} cached files")
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return 0

    count = 0
    for item in cache_dir.iterdir():
        if item.is_file():
            item.unlink()
            count += 1
        elif item.is_dir():
            shutil.rmtree(item)
            count += 1

    return count


def get_cache_size() -> int:
    """Get the total size of the cache in bytes.

    Returns:
        Total size of all files in the cache directory in bytes.
        Returns 0 if the cache directory doesn't exist.

    Example:
        Check cache size in megabytes::

            import openscvx as ox
            size_mb = ox.get_cache_size() / (1024 * 1024)
            print(f"Cache size: {size_mb:.1f} MB")
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return 0

    total = 0
    for item in cache_dir.rglob("*"):
        if item.is_file():
            total += item.stat().st_size

    return total
