import pyopencl as cl
import numpy as np
import os
import sys
import time
from typing import Dict, Tuple

# Ensure we can import from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from convert_cubed_sphere_to_fem import convert_cubed_sphere_to_fem
except ImportError:
    print("Warning: convert_cubed_sphere_to_fem not found. Solver will fail.")
    convert_cubed_sphere_to_fem = None

from .climatic_data import ClimaticModelData
from python_solver.simulation_params import EARTH_RADIUS

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
        
        # Create Physics Nodes (Scaled to Earth Radius)
        # Check if we need to scale (if radius is ~1.0)
        p0 = self.global_nodes[0]
        r0 = np.sqrt(p0[0]**2 + p0[1]**2 + p0[2]**2)
        
        physics_nodes = np.array(self.global_nodes, dtype=np.float32)
        if r0 < 1000.0:
            print(f"[Solver] Scaling mesh from R={r0:.2f} to Earth Radius ({EARTH_RADIUS:.0f} m)")
            physics_nodes *= (EARTH_RADIUS / r0)
            
        # Nodes (n_nodes, 3) -> flattened
        nodes_np = physics_nodes.flatten()
        self.d_nodes = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nodes_np)
        
        # Connectivity (n_elements, 4) -> flattened
        conn_np = np.array(self.connectivity, dtype=np.int32).flatten()
        self.d_connectivity = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=conn_np)

        # Pre-compute Latitudes for Elements
        # Y is up in this mesh generator
        nodes_reshaped = np.array(self.global_nodes, dtype=np.float32) # (n_nodes, 3)
        element_latitudes = np.zeros(self.n_elements, dtype=np.float32)
        
        for i in range(self.n_elements):
            # Get nodes for this element
            node_indices = self.connectivity[i]
            # Average position (centroid)
            centroid = np.mean(nodes_reshaped[node_indices], axis=0)
            # Normalize to get direction
            length = np.linalg.norm(centroid)
            if length > 1e-6:
                y = centroid[1] / length
                # Clamp for safety
                y = max(-1.0, min(1.0, y))
                element_latitudes[i] = np.arcsin(y)
            else:
                element_latitudes[i] = 0.0
                
        self.d_latitudes = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=element_latitudes)

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
        self.step_counter = 0

    def _load_program(self):
        kernel_dir = os.path.join(os.path.dirname(__file__), 'kernels')
        sources = []
        # Load all kernel files
        kernel_files = ['initialization.cl', 'radiation.cl', 'dynamics.cl', 'advection.cl', 'pressure.cl', 'physics.cl']
        for name in kernel_files:
            path = os.path.join(kernel_dir, name)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    sources.append(f.read())
            else:
                print(f"Warning: Kernel file {name} not found.")
        
        full_source = "\n".join(sources)
        program = cl.Program(self.context, full_source).build()
        
        # Cache kernels to avoid RepeatedKernelRetrieval warning
        self.kernels = {
            'init_state': cl.Kernel(program, 'init_state'),
            'compute_radiation': cl.Kernel(program, 'compute_radiation'),
            'compute_dynamics': cl.Kernel(program, 'compute_dynamics'),
            'compute_advection': cl.Kernel(program, 'compute_advection'),
            'compute_pressure': cl.Kernel(program, 'compute_pressure'),
            'compute_physics': cl.Kernel(program, 'compute_physics')
        }
        return program

    def _init_state(self, elevation_data):
        # Run initialization kernel on IN buffers
        self.kernels['init_state'](self.queue, (self.n_elements,), None,
                                self.data.surface_temp_in,
                                self.data.temp_in,
                                self.data.u_wind_in,
                                self.data.v_wind_in,
                                self.data.q_vap,
                                self.data.pressure,
                                self.data.rho,
                                self.data.surface_albedo,
                                self.data.surface_type,
                                self.d_latitudes,
                                np.int32(self.n_elements),
                                np.int32(self.n_layers))
        
        # Initialize OUT buffers with same data
        cl.enqueue_copy(self.queue, self.data.surface_temp_out, self.data.surface_temp_in)
        cl.enqueue_copy(self.queue, self.data.temp_out, self.data.temp_in)
        cl.enqueue_copy(self.queue, self.data.u_wind_out, self.data.u_wind_in)
        cl.enqueue_copy(self.queue, self.data.v_wind_out, self.data.v_wind_in)
        
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
        # 0. Physics (Albedo/Snow): Update surface properties based on current state
        self.kernels['compute_physics'](self.queue, (self.n_elements,), None,
                                     self.data.surface_temp_in,
                                     self.data.surface_albedo,
                                     np.int32(self.n_elements))

        # 1. Radiation (Physics): temp_in -> temp_out (heating)
        # Note: We use temp_in as input and write to temp_out. 
        # Ideally we should accumulate tendencies, but for now we update sequentially.
        self.kernels['compute_radiation'](self.queue, (self.n_elements,), None,
                                       self.data.temp_in,
                                       self.data.temp_out,
                                       self.data.surface_temp_in,
                                       self.data.surface_temp_out,
                                       self.data.surface_albedo,
                                       self.d_latitudes,
                                       np.float32(dt),
                                       np.int32(self.n_elements),
                                       np.int32(self.n_layers))
        
        # 2. Dynamics (Momentum): u_in, v_in -> u_out, v_out
        # Sub-stepping for stability (2 steps of dt/2)
        n_substeps = 2
        dt_dyn = dt / n_substeps
        
        for i in range(n_substeps):
            self.kernels['compute_dynamics'](self.queue, (self.n_elements,), None,
                                          self.data.temp_out, # Use updated temp
                                          self.data.pressure,
                                          self.data.u_wind_in,
                                          self.data.v_wind_in,
                                          self.data.u_wind_out,
                                          self.data.v_wind_out,
                                          self.d_latitudes,
                                          np.float32(dt_dyn),
                                          np.int32(self.n_elements),
                                          np.int32(self.n_layers))
            
            # Swap wind buffers for next sub-step (Ping-Pong)
            if i < n_substeps - 1:
                 self.data.u_wind_in, self.data.u_wind_out = self.data.u_wind_out, self.data.u_wind_in
                 self.data.v_wind_in, self.data.v_wind_out = self.data.v_wind_out, self.data.v_wind_in

        # 3. Advection: Transport temp_out using u_out, v_out
        # (Placeholder: currently just copy)
        self.kernels['compute_advection'](self.queue, (self.n_elements,), None,
                                       self.data.temp_out,
                                       self.data.temp_in, # Write back to IN (or use another buffer)
                                       self.data.u_wind_out,
                                       self.data.v_wind_out,
                                       self.d_connectivity,
                                       np.float32(dt),
                                       np.int32(self.n_elements),
                                       np.int32(self.n_layers))
        
        # 4. Pressure Update (Hydrostatic)
        self.kernels['compute_pressure'](self.queue, (self.n_elements,), None,
                                      self.data.temp_in, # Use latest temp
                                      self.data.pressure,
                                      np.int32(self.n_elements),
                                      np.int32(self.n_layers))
        
        # 5. Swap Buffers
        self.data.swap_buffers()
        
        # Wait for GPU
        self.queue.finish()
        t_gpu = time.time()
        
        # 6. Read back surface temperature for visualization
        surface_temp_host = np.empty(self.n_elements, dtype=np.float32)
        cl.enqueue_copy(self.queue, surface_temp_host, self.data.surface_temp_in)
        
        # Read back winds (u, v) for visualization (Layer 0 - Surface)
        # We need to read u_wind_in and v_wind_in (since we swapped)
        # But u_wind is 3D (n_elements * n_layers). We only want layer 0.
        # Since data is [element * n_layers + layer], layer 0 is at stride n_layers.
        # Reading strided data from OpenCL buffer is tricky with enqueue_copy directly.
        # It's easier to read the whole buffer and slice in numpy, or write a kernel to extract.
        # For simplicity/performance balance, let's read the whole buffer for now (it's not huge).
        
        u_wind_host = np.empty(self.n_elements * self.n_layers, dtype=np.float32)
        v_wind_host = np.empty(self.n_elements * self.n_layers, dtype=np.float32)
        cl.enqueue_copy(self.queue, u_wind_host, self.data.u_wind_in)
        cl.enqueue_copy(self.queue, v_wind_host, self.data.v_wind_in)
        
        # Extract surface layer (layer 0)
        # Index: gid * n_layers + 0
        u_surf = u_wind_host[0::self.n_layers]
        v_surf = v_wind_host[0::self.n_layers]
        
        # Read back pressure (Surface Pressure - Layer 0)
        pressure_host = np.empty(self.n_elements * self.n_layers, dtype=np.float32)
        cl.enqueue_copy(self.queue, pressure_host, self.data.pressure)
        pressure_surf = pressure_host[0::self.n_layers]
        
        # Read back albedo
        albedo_host = np.empty(self.n_elements, dtype=np.float32)
        cl.enqueue_copy(self.queue, albedo_host, self.data.surface_albedo)
        
        # Calculate Vertical Motion (Convergence/Divergence proxy)
        # Simple approximation: High pressure -> Sinking (Divergence), Low pressure -> Rising (Convergence)
        # Or use temperature: Hot -> Rising, Cold -> Sinking
        # Let's use Temperature anomaly for now as a simple proxy for vertical motion
        t_mean = np.mean(surface_temp_host)
        vertical_motion_surf = surface_temp_host - t_mean
        
        t_readback = time.time()
        
        # Log min/max temperatures
        self.step_counter += 1
        if self.step_counter % 20 == 0:
            t_min = np.min(surface_temp_host)
            t_max = np.max(surface_temp_host)
            t_avg = np.mean(surface_temp_host)
            # Conversion to Celsius
            t_min_c = t_min - 273.15
            t_max_c = t_max - 273.15
            t_avg_c = t_avg - 273.15
            
            # Wind Speed Statistics
            # Speed = sqrt(u^2 + v^2)
            wind_speed = np.sqrt(u_surf**2 + v_surf**2)
            w_avg_kmh = np.mean(wind_speed) * 3.6
            w_max_kmh = np.max(wind_speed) * 3.6
            
            print(f"[Sim t={self.time:.1f}s] Temp (C): Min={t_min_c:.1f}, Avg={t_avg_c:.1f}, Max={t_max_c:.1f} | Wind (km/h): Avg={w_avg_kmh:.1f}, Max={w_max_kmh:.1f}")

        # 7. Interpolate to nodes
        scalars = self.interpolate_to_nodes(surface_temp_host)
        pressure_nodes = self.interpolate_to_nodes(pressure_surf)
        vertical_nodes, vertical_flat = self.interpolate_to_nodes(vertical_motion_surf, return_flat=True)
        albedo_nodes = self.interpolate_to_nodes(albedo_host)
        
        t_interp_scalar = time.time()
        
        # Interpolate vectors to nodes
        # We need to map local (u,v) to global (x,y,z) vectors on the sphere surface
        # This is non-trivial because u,v are in local tangent space (East, North).
        # For visualization, we need global Cartesian vectors.
        vectors = self.interpolate_vectors_to_nodes(u_surf, v_surf)
        
        t_interp_vector = time.time()
        
        if self.step_counter % 200 == 0:
             print(f"[Perf] Total: {(t_interp_vector - t_start)*1000:.1f}ms | GPU: {(t_gpu - t_start)*1000:.1f}ms | Read: {(t_readback - t_gpu)*1000:.1f}ms | IntScal: {(t_interp_scalar - t_readback)*1000:.1f}ms | IntVec: {(t_interp_vector - t_interp_scalar)*1000:.1f}ms")

        self.time += dt
        
        return {
            'scalars': scalars,
            'vectors': vectors,
            'pressure': pressure_nodes,
            'vertical_motion': vertical_nodes,
            'vertical_motion_flat': vertical_flat,
            'albedo': albedo_nodes
        }

    def interpolate_vectors_to_nodes(self, u_elem, v_elem):
        """
        Interpolates element-centered (u,v) winds to node-centered global (vx, vy, vz) vectors.
        """
        # 1. Interpolate u, v to nodes (scalar interpolation)
        u_nodes = np.zeros(self.n_nodes)
        v_nodes = np.zeros(self.n_nodes)
        
        for nid in range(self.n_nodes):
            elems = self.node_to_elems[nid]
            if elems:
                u_sum = 0.0
                v_sum = 0.0
                for ei in elems:
                    u_sum += u_elem[ei]
                    v_sum += v_elem[ei]
                u_nodes[nid] = u_sum / len(elems)
                v_nodes[nid] = v_sum / len(elems)
        
        # 2. Convert local (u,v) at node to global (vx, vy, vz)
        # u is East-West (zonal), v is North-South (meridional)
        # We need basis vectors at each node.
        
        global_vectors = np.zeros((self.n_nodes, 3), dtype=np.float32)
        nodes = np.array(self.global_nodes)
        
        for i in range(self.n_nodes):
            p = nodes[i] # Position vector (x, y, z)
            r = np.linalg.norm(p)
            if r < 1e-6: continue
            
            # Normal vector (Up)
            up = p / r
            
            # North vector (tangent pointing towards +Y axis projected on plane)
            # N = (0,1,0) - (Up . (0,1,0)) * Up
            # Normalized
            north_raw = np.array([0.0, 1.0, 0.0]) - np.dot(up, np.array([0.0, 1.0, 0.0])) * up
            if np.linalg.norm(north_raw) < 1e-6:
                # At poles, North is undefined/singular. 
                # Arbitrary choice or handle specifically.
                north = np.array([1.0, 0.0, 0.0]) 
            else:
                north = north_raw / np.linalg.norm(north_raw)
            
            # East vector = North x Up
            east = np.cross(north, up)
            
            # Global Vector = u * East + v * North
            u_val = u_nodes[i]
            v_val = v_nodes[i]
            
            global_vectors[i] = u_val * east + v_val * north
            
        return global_vectors

    def interpolate_to_nodes(self, element_values: np.ndarray, return_flat: bool = False):
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
            
        if return_flat:
            return node_values, node_values_flat
        return node_values
