import pyopencl as cl
import numpy as np

class ClimaticModelData:
    """
    Manages OpenCL buffers for the climatic simulation data.
    Mirrors the structure of ClimaticModelData in cuda_mod/src/climatic_data.hpp
    """
    def __init__(self, context, n_elements, n_layers):
        self.context = context
        self.n_elements = n_elements
        self.n_layers = n_layers
        
        mf = cl.mem_flags
        
        # Helper to create buffers
        def create_buffer(count, dtype=np.float32):
            size_bytes = count * np.dtype(dtype).itemsize
            return cl.Buffer(context, mf.READ_WRITE, size=size_bytes)

        # 2. Atmospheric State (n_elements * n_layers)
        size_3d = n_elements * n_layers
        self.temp = create_buffer(size_3d)      # Temperature (K)
        self.q_vap = create_buffer(size_3d)     # Specific humidity (kg/kg)
        self.u_wind = create_buffer(size_3d)    # Horizontal velocity U (m/s)
        self.v_wind = create_buffer(size_3d)    # Horizontal velocity V (m/s)
        self.rho = create_buffer(size_3d)       # Density (kg/m^3)
        self.pressure = create_buffer(size_3d)  # Pressure (Pa)

        # 3. Surface State (n_elements)
        size_2d = n_elements
        self.surface_temp = create_buffer(size_2d)
        self.surface_albedo = create_buffer(size_2d)
        self.surface_type = create_buffer(size_2d, dtype=np.int32) # 0: Ocean, 1: Land, 2: Ice

        # 4. Flux Fields
        self.flux_sw = create_buffer(size_2d)       # Shortwave flux
        self.flux_lw = create_buffer(size_2d)       # Longwave flux
        self.precipitation = create_buffer(size_2d) # Precipitation accumulation
