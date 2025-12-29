import pyopencl as cl
import numpy as np

# Helper to create buffers
def create_buffer(context, count, dtype=np.float32):
            mf = cl.mem_flags
            size_bytes = count * np.dtype(dtype).itemsize
            return cl.Buffer(context, mf.READ_WRITE, size=size_bytes)


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

        # 2. Atmospheric State (n_elements * n_layers)
        size_3d = n_elements * n_layers
        
        # Double buffering for time-stepping fields
        self.temp_in = create_buffer(context, size_3d)
        self.temp_out = create_buffer(context, size_3d)
        
        self.u_wind_in = create_buffer(context, size_3d)
        self.u_wind_out = create_buffer(context, size_3d)
        
        self.v_wind_in = create_buffer(context, size_3d)
        self.v_wind_out = create_buffer(context, size_3d)
        
        self.q_vap = create_buffer(context, size_3d)     # Specific humidity (kg/kg)
        self.rho = create_buffer(context, size_3d)       # Density (kg/m^3)
        self.pressure = create_buffer(context, size_3d)  # Pressure (Pa)

        # 3. Surface State (n_elements)
        size_2d = n_elements
        self.surface_temp_in = create_buffer(context, size_2d)
        self.surface_temp_out = create_buffer(context, size_2d)
        
        self.surface_albedo = create_buffer(context, size_2d)
        self.surface_type = create_buffer(context, size_2d, dtype=np.int32) # 0: Ocean, 1: Land, 2: Ice

        # 4. Flux Fields
        self.flux_sw = create_buffer(context, size_2d)       # Shortwave flux
        self.flux_lw = create_buffer(context, size_2d)       # Longwave flux
        self.precipitation = create_buffer(context, size_2d) # Precipitation accumulation

    def swap_buffers(self):
        """Swaps input and output buffers."""
        self.temp_in, self.temp_out = self.temp_out, self.temp_in
        self.u_wind_in, self.u_wind_out = self.u_wind_out, self.u_wind_in
        self.v_wind_in, self.v_wind_out = self.v_wind_out, self.v_wind_in
        self.surface_temp_in, self.surface_temp_out = self.surface_temp_out, self.surface_temp_in

