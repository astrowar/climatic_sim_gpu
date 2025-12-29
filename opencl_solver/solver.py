import pyopencl as cl
import numpy as np
import os
import sys
from typing import Dict, Tuple

# Ensure we can import from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from convert_cubed_sphere_to_fem import convert_cubed_sphere_to_fem
except ImportError:
    print("Warning: convert_cubed_sphere_to_fem not found. Solver will fail.")
    convert_cubed_sphere_to_fem = None

from .climatic_data import ClimaticModelData

class OpenCLFEMSolver:
    """
    OpenCL-based FEM Solver for Climatic Simulation.
    Compatible with the interface expected by main.py (SphereFEMSolver).
    """
    def __init__(self, grid_points: Dict, elevation_data: Dict = None):
        if convert_cubed_sphere_to_fem is None:
            raise RuntimeError("convert_cubed_sphere_to_fem not available")

        # 1. Initialize OpenCL Context
        try:
            # Prefer GPU
            platform = cl.get_platforms()[0]
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if not devices:
                devices = platform.get_devices() # Fallback to any
            
            self.device = devices[0]
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)
            print(f"OpenCL Solver initialized on: {self.device.name}")
        except Exception as e:
            print(f"Failed to initialize OpenCL: {e}")
            raise

        # 2. Mesh Conversion (Host Side)
        self.global_nodes, self.connectivity, self.node_mapping = convert_cubed_sphere_to_fem(grid_points)
        self.n_nodes = len(self.global_nodes)
        self.n_elements = len(self.connectivity)
        
        # 3. Data Initialization
        self.n_layers = 5
        self.data = ClimaticModelData(self.context, self.n_elements, self.n_layers)
        
        # 4. Upload Mesh to Device
        mf = cl.mem_flags
        # Nodes (n_nodes, 3) -> flattened
        nodes_np = np.array(self.global_nodes, dtype=np.float32).flatten()
        self.d_nodes = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nodes_np)
        
        # Connectivity (n_elements, 4) -> flattened
        conn_np = np.array(self.connectivity, dtype=np.int32).flatten()
        self.d_connectivity = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=conn_np)

        # 5. Load and Compile Kernels
        self.program = self._load_program()
        
        # 6. Initialize Simulation State
        self._init_state(elevation_data)

        # 7. Pre-compute Node-to-Element mapping for interpolation (Host side for now)
        self.node_to_elems = [[] for _ in range(self.n_nodes)]
        for ei, elem in enumerate(self.connectivity):
            for nid in map(int, elem):
                if 0 <= nid < self.n_nodes:
                    self.node_to_elems[nid].append(ei)
                    
        # Simulation metadata
        self.time = 0.0
        self.simulation_type = None
        self.sim_params = {}

    def _load_program(self):
        kernel_path = os.path.join(os.path.dirname(__file__), 'kernels.cl')
        with open(kernel_path, 'r') as f:
            code = f.read()
        return cl.Program(self.context, code).build()

    def _init_state(self, elevation_data):
        # Run initialization kernel
        self.program.init_state(self.queue, (self.n_elements,), None,
                                self.data.surface_temp,
                                self.data.temp,
                                np.int32(self.n_elements),
                                np.int32(self.n_layers))
        self.queue.finish()

    def get_element_count(self) -> int:
        return self.n_elements

    def get_node_count(self) -> int:
        return self.n_nodes

    def setup_simulation(self, sim_type: str, heavy_computation: bool = False, computation_delay: float = 0.0, **params):
        self.simulation_type = sim_type
        self.sim_params = params
        # Could re-configure kernels here based on sim_type

    def update_simulation(self, dt: float = 0.016) -> Dict:
        # 1. Run Simulation Step Kernel
        self.program.simulation_step(self.queue, (self.n_elements,), None,
                                     self.data.surface_temp,
                                     self.data.temp,
                                     np.float32(dt),
                                     np.int32(self.n_elements),
                                     np.int32(self.n_layers))
        
        # 2. Read back surface temperature for visualization
        # (In a full GPU pipeline, we would map this directly to GL, but here we copy back)
        surface_temp_host = np.empty(self.n_elements, dtype=np.float32)
        cl.enqueue_copy(self.queue, surface_temp_host, self.data.surface_temp)
        
        # 3. Interpolate to nodes (Host side for compatibility)
        scalars = self.interpolate_to_nodes(surface_temp_host)
        
        self.time += dt
        
        return {
            'scalars': scalars,
            'vectors': None,
            'pressure': None,
            'vertical_motion': None
        }

    def interpolate_to_nodes(self, element_values: np.ndarray) -> Dict:
        """
        Interpolates element-centered values to nodes and formats for the visualizer.
        """
        # This logic is similar to NativeFEMSolver to ensure compatibility
        
        # Map element values to nodes (averaging)
        node_values_flat = np.zeros(self.n_nodes, dtype=np.float64)
        
        # Simple average of elements connected to node
        # TODO: Move this to a kernel for performance!
        for nid in range(self.n_nodes):
            elems = self.node_to_elems[nid]
            if elems:
                # mean of values[elems]
                val_sum = 0.0
                for ei in elems:
                    val_sum += element_values[ei]
                node_values_flat[nid] = val_sum / len(elems)
            else:
                node_values_flat[nid] = 0.0

        # Map global nodes back to cubed-sphere faces for visualization
        node_values = {}
        for face_id in range(6):
            face_key = f'face_{face_id}'
            face_node_indices = self.node_mapping[face_key]
            rows, cols = face_node_indices.shape
            face_node_values = np.zeros((rows, cols), dtype=np.float64)
            
            for i in range(rows):
                for j in range(cols):
                    global_node_id = int(face_node_indices[i, j])
                    face_node_values[i, j] = node_values_flat[global_node_id]
            
            node_values[face_key] = face_node_values
            
        return node_values
