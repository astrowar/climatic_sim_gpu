"""Initial Python-based native FEM solver shim.

Provides a simple CPU implementation exposing the minimal API
used by the CUDA example: `init()`, `simulation_init(...)`,
`simulation_step(dt)` and `get_surface_temp()`.

This is a lightweight, single-file reference implementation intended
for development and testing until a compiled native/accelerated solver
is available.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple
from typing import Dict

try:
    # Reuse the shared mesh conversion utility from repository root
    from convert_cubed_sphere_to_fem import convert_cubed_sphere_to_fem
except Exception:
    convert_cubed_sphere_to_fem = None


class NativeFEMSolver:
    """CUDA-backed adapter that provides the high-level solver API.

    This class mirrors the previous `SphereFEMSolver` behavior but
    delegates computation to the compiled `cuda_fem_solver` extension.
    The constructor accepts the cubed-sphere `grid_points` structure and
    uses `convert_cubed_sphere_to_fem` to build global node coordinates
    and element connectivity before initializing the compiled solver.
    """

    def __init__(self, grid_points: Dict, elevation_data: Dict = None):
        if convert_cubed_sphere_to_fem is None:
            raise RuntimeError("convert_cubed_sphere_to_fem not available")

        self.global_nodes, self.connectivity, self.node_mapping = convert_cubed_sphere_to_fem(grid_points)
        node_coords = self.global_nodes.copy()
        self.n_nodes = len(node_coords)
        self.n_elements = len(self.connectivity)

        # Prepare arrays expected by the compiled extension
        import numpy as _np
        coords = _np.asarray(node_coords, dtype=_np.float32)
        conn = _np.asarray(self.connectivity, dtype=_np.int32)

        # Initialize compiled solver
        simulation_init(coords, conn, 5)

        # Runtime metadata expected by callers
        self.time = 0.0
        self.simulation_type = None
        self.sim_params = {}
        # Build node->elements mapping for element->node->node-value interpolation
        self.node_to_elems = [[] for _ in range(self.n_nodes)]
        for ei, elem in enumerate(self.connectivity):
            for nid in map(int, elem):
                if 0 <= nid < self.n_nodes:
                    self.node_to_elems[nid].append(ei)

    def get_element_count(self) -> int:
        return self.n_elements

    def get_node_count(self) -> int:
        return self.n_nodes

    def setup_simulation(self, sim_type: str, heavy_computation: bool = False, computation_delay: float = 0.0, **params):
        self.simulation_type = sim_type
        self.sim_params = params

    def update_simulation(self, dt: float = 0.016) -> Dict:
        simulation_step(float(dt))
        import numpy as _np
        surf = get_surface_temp()
        if isinstance(surf, _np.ndarray):
            arr = surf
        else:
            arr = _np.array(surf)[0]
        scalars = self.interpolate_to_nodes(arr)
        return {
            'scalars': scalars,
            'vectors': None,
            'pressure': None,
            'vertical_motion': None
        }

    def interpolate_to_nodes(self, global_values: np.ndarray = None) -> Dict:
        import numpy as _np
        values_to_map = global_values
        if values_to_map is None:
            surf = get_surface_temp()
            values_to_map = _np.array(surf)[0]
        # values_to_map may be node-centered (len == n_nodes) or
        # element-centered (len == n_elements). Handle both cases.
        is_node_centered = (values_to_map.shape[0] == self.n_nodes)
        is_elem_centered = (values_to_map.shape[0] == self.n_elements)
        if not (is_node_centered or is_elem_centered):
            raise RuntimeError(f"Unexpected surface values length: {values_to_map.shape[0]}")

        node_values = {}
        for face_id in range(6):
            face_key = f'face_{face_id}'
            face_node_indices = self.node_mapping[face_key]
            rows, cols = face_node_indices.shape
            face_node_values = _np.zeros((rows, cols), dtype=_np.float64)
            for i in range(rows):
                for j in range(cols):
                    global_node_id = int(face_node_indices[i, j])
                    if is_node_centered:
                        face_node_values[i, j] = float(values_to_map[global_node_id])
                    else:
                        # average element-centered values of elements touching this node
                        elems = self.node_to_elems[global_node_id]
                        if not elems:
                            face_node_values[i, j] = 0.0
                        else:
                            face_node_values[i, j] = float(_np.mean(values_to_map[elems]))
            node_values[face_key] = face_node_values
        return node_values


 


# Note: `SphereFEMSolver` adapter intentionally removed — only
# `NativeFEMSolver` is provided by this module. High-level adapters
# or application glue that previously imported `SphereFEMSolver` should
# be updated to construct and use `NativeFEMSolver` directly or use the
# compiled `cuda_fem_solver` extension.


# Require a compiled CUDA extension at import time.
# The project uses a compiled `cuda_fem_solver` for the runtime API.
# If it's not available, raise immediately so callers know the module
# cannot function without the native extension.
import importlib, os, sys
_so_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'cuda_mod', 'build', 'lib'))
if os.path.isdir(_so_dir):
    if _so_dir not in sys.path:
        sys.path.insert(0, _so_dir)
    try:
        _impl = importlib.import_module('cuda_fem_solver')
    except Exception as e:
        raise RuntimeError(f"cuda_fem_solver extension import failed: {e}")
else:
    raise RuntimeError(f"cuda_fem_solver extension directory not found: {_so_dir}")

# Bind the module-level API directly to the compiled extension
init = getattr(_impl, 'init')
simulation_init = getattr(_impl, 'simulation_init')
simulation_step = getattr(_impl, 'simulation_step')
get_surface_temp = getattr(_impl, 'get_surface_temp')

try:
    from convert_cubed_sphere_to_fem import convert_cubed_sphere_to_fem
except Exception:
    convert_cubed_sphere_to_fem = None
__all__ = ["init", "simulation_init", "simulation_step", "get_surface_temp", "NativeFEMSolver"]
