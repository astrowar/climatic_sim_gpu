"""Native solver package.

Prefer the compiled CUDA extension (if available) built into
`cuda_mod/build/lib/cuda_fem_solver.so`. If the compiled extension
isn't found or fails to import, fall back to the pure-Python shim
implemented in :mod:`native_solver.native_solver`.

This module exposes a stable API: ``init``, ``simulation_init``,
``simulation_step``, ``get_surface_temp`` and ``SphereFEMSolver``.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

# Attempt to load compiled extension from repo build directory
_impl: Any | None = None
_so_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'cuda_mod', 'build', 'lib'))
if os.path.isdir(_so_dir):
	if _so_dir not in sys.path:
		sys.path.insert(0, _so_dir)
	try:
		_impl = importlib.import_module('cuda_fem_solver')
	except Exception:
		_impl = None

if _impl is None:
	# fallback to Python implementation
	from .native_solver import init, simulation_init, simulation_step, get_surface_temp, NativeFEMSolver
else:
	# expose compiled functions
	init = getattr(_impl, 'init')
	simulation_init = getattr(_impl, 'simulation_init')
	simulation_step = getattr(_impl, 'simulation_step')
	get_surface_temp = getattr(_impl, 'get_surface_temp')

	# Always expose the Python `NativeFEMSolver` class for users who
	# want a pure-Python fallback or to instantiate the CPU solver.
	from .native_solver import NativeFEMSolver

# Backwards compatibility: some callers still import `SphereFEMSolver`.
# Provide a deprecated alias pointing to `NativeFEMSolver`.
import warnings
SphereFEMSolver = NativeFEMSolver
warnings.warn("SphereFEMSolver is deprecated; use NativeFEMSolver instead", DeprecationWarning, stacklevel=2)

__all__ = ["init", "simulation_init", "simulation_step", "get_surface_temp", "NativeFEMSolver", "SphereFEMSolver"]
