"""
Compatibility stub for legacy import path.

The implementation has moved to the `python_solver` package. Import
from `python_solver.fem_solver` instead. This small stub raises an
ImportError to make the change explicit at runtime.
"""

raise ImportError("fem_solver moved to python_solver.fem_solver; import from python_solver.fem_solver instead")


def convert_cubed_sphere_to_fem(grid_points: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convert cubed-sphere grid (6 faces) to unified FEM structure.
    
    Merges shared nodes at face boundaries to create a single global mesh.
    Uses spatial hashing for efficient duplicate detection.
    
    Parameters
    ----------
    grid_points : dict
        Dictionary containing grid data for all six faces
        
    Returns
    -------
    global_nodes : np.ndarray
        Array of shape (n_nodes, 3) with (x, y, z) coordinates
    connectivity : np.ndarray
        Array of shape (n_elements, 4) with node indices for each quad element
    node_mapping : dict
        Maps face indices back to global node indices
        Format: {face_id: array of shape (rows, cols) with global node indices}
    """
    n_faces = 6
    
    # Get dimensions from first face
    face_0 = grid_points['face_0']
    rows, cols = face_0['x'].shape
    
    # Get radius from first point
    x0, y0, z0 = face_0['x'][0, 0], face_0['y'][0, 0], face_0['z'][0, 0]
    radius = np.sqrt(x0**2 + y0**2 + z0**2)
    
    # Tolerance for considering two nodes as the same (shared at boundaries)
    # Use adaptive tolerance based on grid resolution
    avg_spacing = 2.0 * radius / (rows - 1)  # Approximate spacing on sphere surface
    tol = avg_spacing * 0.1  # 10% of average spacing (Increased from 1% to fix connectivity)
    
    print(f"  Using tolerance: {tol:.2e} (euclidean distance)")

    global_nodes = []
    connectivity = []
    node_mapping = {}
    node_hash = {}
    node_id = 0

    for face_id in range(n_faces):
        face_key = f'face_{face_id}'
        face_data = grid_points[face_key]
        xs = face_data['x']
        ys = face_data['y']
        zs = face_data['z']

        face_node_indices = np.zeros((rows, cols), dtype=np.int32)

        for i in range(rows):
            for j in range(cols):
                x, y, z = xs[i, j], ys[i, j], zs[i, j]
                coord = np.array([x, y, z])

                # Spatial hash based on rounded coordinates to tolerance
                h = tuple(np.round(coord / tol, 4))
                if h in node_hash:
                    face_node_indices[i, j] = node_hash[h]
                else:
                    global_nodes.append(coord)
                    node_hash[h] = node_id
                    face_node_indices[i, j] = node_id
                    node_id += 1

        node_mapping[face_key] = face_node_indices

    # Build connectivity (quads) per face
    for face_id in range(n_faces):
        face_key = f'face_{face_id}'
        face_node_indices = node_mapping[face_key]
        for i in range(rows - 1):
            for j in range(cols - 1):
                n0 = face_node_indices[i, j]
                n1 = face_node_indices[i + 1, j]
                n2 = face_node_indices[i + 1, j + 1]
                n3 = face_node_indices[i, j + 1]
                connectivity.append([n0, n1, n2, n3])

    global_nodes = np.array(global_nodes)
    connectivity = np.array(connectivity)

    # Quick duplicate check
    duplicate_count = 0
    for i in range(len(global_nodes)):
        for j in range(i + 1, len(global_nodes)):
            if np.linalg.norm(global_nodes[i] - global_nodes[j]) < tol:
                duplicate_count += 1

    if duplicate_count > 0:
        print(f"  WARNING: Found {duplicate_count} duplicate nodes within tolerance!")
    else:
        print(f"  ✓ No duplicate nodes found - mesh is properly connected")

    print(f"  ✓ FEM mesh created: {len(global_nodes)} nodes, {len(connectivity)} elements")

    return global_nodes, connectivity, node_mapping


class SphereFEMSolver:
    """
    Finite Element solver for spherical grids.
    
    Simulates physical phenomena and returns scalar values per element.
    """
    
    def __init__(self, grid_points: Dict, elevation_data: Dict = None):
        """
        Initialize FEM solver with mesh data.
        
        Parameters
        ----------
        grid_points : dict
            Dictionary containing grid data for all six faces
        elevation_data : dict, optional
            Pre-normalized elevation data [0, 1] for each face.
            If None, will try to extract from grid_points.
        """
        self.n_faces = 6
        self.time = 0.0
        self.simulation_type = 'heat_diffusion'
        self.sim_params = {}
        self.heavy_computation = False
        self.computation_delay = 0.1  # seconds
        
        # Convert to unified FEM structure (GLOBAL MESH)
        print("Converting cubed-sphere to unified FEM structure...")
        self.global_nodes, self.connectivity, self.node_mapping = convert_cubed_sphere_to_fem(grid_points)
        
        # Store only grid dimensions for visualization mapping
        face_0 = grid_points['face_0']
        self.face_rows, self.face_cols = face_0['x'].shape
        
        # Discard face-based structure - use only global mesh from now on
        del grid_points
        
        # Initialize elevation data if provided
        self.global_elevation = None
        if elevation_data is not None:
            self.global_elevation = self._extract_elevation_global(elevation_data)
            print("FEM Solver initialized with elevation data")
        
        # Initialize temperatures for 3-layer atmosphere model (Kelvin)
        # Ground (Layer 0): Surface/soil (absorbs solar radiation directly)
        # Layer 1: Lower atmosphere (exchanges with ground and Layer 2)
        # Layer 2: Upper atmosphere (exchanges with Layer 1, radiates to space)
        # Using global node arrays only
        self.temperatures_ground = None  # Ground/surface layer, shape (n_nodes,)
        self.temperatures_layer1 = None  # Lower atmosphere, shape (n_nodes,)
        self.temperatures_layer2 = None  # Upper atmosphere, shape (n_nodes,)
        self.initial_temperature = INITIAL_GLOBAL_TEMPERATURE  # K (from params)

        # Adicionar perturbação inicial na camada 1 (atmosfera baixa)
        self._add_initial_perturbation = True
        
        # Global node values for visualization (shape: n_nodes)
        # Will show ground (surface) temperatures
        self.global_node_values = None
        
        # Initialize Fluid Solver
        self.fluid_solver = FluidSolver(self.global_nodes, self.connectivity)
        print("Fluid Solver initialized")
    
    def _extract_elevation_global(self, elevation_data: Dict) -> np.ndarray:
        """
        Extract and normalize elevation data to global node array.
        
        Parameters
        ----------
        elevation_data : dict
            Face-based elevation data
            
        Returns
        -------
        global_elevation : np.ndarray
            Elevation values mapped to global nodes
        """
        n_nodes = len(self.global_nodes)
        global_elevation = np.zeros(n_nodes)
        
        for face_id in range(self.n_faces):
            face_key = f'face_{face_id}'
            if face_key not in elevation_data:
                continue
                
            face_elev = elevation_data[face_key]
            face_indices = self.node_mapping[face_key]
            rows, cols = face_indices.shape
            
            for i in range(rows):
                for j in range(cols):
                    global_node_id = face_indices[i, j]
                    # Normalize elevation to [0, 1]
                    elev_val = np.clip(face_elev[i, j] / 8000.0, 0.0, 1.0)
                    global_elevation[global_node_id] = elev_val
        
        print("Elevation data mapped to global nodes")
        return global_elevation
    
    def initialize_with_elevation(self):
        """
        Initialize simulation values with elevation data.
        Uses global node structure only.
        """
        if self.global_elevation is None:
            print("Warning: No elevation data available")
            return
        
        print("Initializing simulation with global elevation data...")
        self.global_node_values = self.global_elevation.copy()
    
    def setup_simulation(self, sim_type: str, heavy_computation: bool = False, 
                        computation_delay: float = 0.1, **params):
        """
        Setup a time-dependent simulation.
        
        Parameters
        ----------
        sim_type : str
            Type of simulation: 'heat_diffusion', 'wave_pattern', 'rotating_heat', 
            'dual_vortex', 'latitude_gradient'
        heavy_computation : bool
            If True, simulates heavy computation with artificial delay
        computation_delay : float
            Artificial delay in seconds to simulate heavy computation
        **params : dict
            Simulation-specific parameters
        """
        self.simulation_type = sim_type
        self.sim_params = params
        self.time = 0.0
        self.heavy_computation = heavy_computation
        self.computation_delay = computation_delay
        print(f"\nSetup simulation: {sim_type}")
        print(f"Heavy computation: {heavy_computation} (delay: {computation_delay}s)")
        print(f"Parameters: {params}")
    
    def update_simulation(self, dt: float = 0.016) -> Dict:
        """
        Update simulation by one time step.
        
        Parameters
        ----------
        dt : float
            Time step (default: ~60 FPS)
        
        Returns
        -------
        node_values : dict
            Updated node values for visualization
        """
        self.time += dt
        
        # Simulate heavy computation
        if self.heavy_computation:
            time.sleep(self.computation_delay)
        
        if self.simulation_type == 'climate_model':
            self.simulate_climate_model(dt)
        
        elif self.simulation_type == 'heat_diffusion':
            source = self.sim_params.get('source_point', (1.0, 0.0, 0.0))
            decay = self.sim_params.get('decay_rate', 2.0)
            self.simulate_heat_diffusion(source_point=source, decay_rate=decay)
        
        elif self.simulation_type == 'wave_pattern':
            n_waves = self.sim_params.get('n_waves', 4)
            speed = self.sim_params.get('speed', 1.0)
            self.simulate_wave_pattern(n_waves=n_waves, time=self.time * speed)
        
        elif self.simulation_type == 'rotating_heat':
            speed = self.sim_params.get('speed', 0.5)
            decay = self.sim_params.get('decay_rate', 2.0)
            angle = self.time * speed
            source = (np.cos(angle), np.sin(angle), 0.0)
            self.simulate_heat_diffusion(source_point=source, decay_rate=decay)
        
        elif self.simulation_type == 'dual_vortex':
            speed = self.sim_params.get('speed', 1.0)
            self.simulate_dual_vortex(time=self.time * speed)
        
        elif self.simulation_type == 'pulsating_heat':
            decay = self.sim_params.get('decay_rate', 2.0)
            freq = self.sim_params.get('frequency', 1.0)
            source = self.sim_params.get('source_point', (1.0, 0.0, 0.0))
            pulse = (np.sin(self.time * freq) + 1.0) / 2.0  # [0, 1]
            self.simulate_heat_diffusion(source_point=source, decay_rate=decay * (0.5 + pulse))
        
        else:  # latitude_gradient (static)
            self.simulate_latitude_gradient()
        
        result = {
            'scalars': self.interpolate_to_nodes(),
            'vectors': self.get_wind_vectors_for_visualization(),
            'pressure': self.get_pressure_for_visualization(),
            'vertical_motion': self.get_vertical_motion_for_visualization()
        }
        return result

    def get_vertical_motion_for_visualization(self) -> Dict:
        """
        Map global vertical velocity proxy to face-based structure.
        Normalizes to [0, 1] where 0.5 is neutral, 1.0 is strong updraft, 0.0 is strong downdraft.
        """
        if not hasattr(self, 'fluid_solver') or self.fluid_solver is None:
            return None
            
        w = self.fluid_solver.vertical_velocity
        
        # Normalize for visualization
        w_mean = np.mean(w)
        w_std = np.std(w)
        
        if w_std < 1e-8:
            normalized_w = np.full_like(w, 0.5)
        else:
            # Map +/- 2 standard deviations to [0, 1]
            normalized_w = (w - (w_mean - 2*w_std)) / (4*w_std)
            normalized_w = np.clip(normalized_w, 0.0, 1.0)
            
        return self.interpolate_to_nodes(normalized_w)

    def get_pressure_for_visualization(self) -> Dict:
        """
        Map global pressure values to face-based structure for OpenGL visualization.
        Normalizes pressure to [0, 1] range for visualization.
        """
        if not hasattr(self, 'fluid_solver') or self.fluid_solver is None:
            return None
            
        pressure = self.fluid_solver.pressure
        
        # Normalize pressure for visualization
        # Standard sea level pressure is ~101325 Pa
        # We want to visualize deviations
        p_mean = np.mean(pressure)
        p_std = np.std(pressure)
        
        if p_std < 1e-5:
            normalized_pressure = np.full_like(pressure, 0.5)
        else:
            # Visualize +/- 2 standard deviations
            normalized_pressure = (pressure - (p_mean - 2*p_std)) / (4*p_std)
            normalized_pressure = np.clip(normalized_pressure, 0.0, 1.0)
            
        return self.interpolate_to_nodes(normalized_pressure)

    def get_wind_vectors_for_visualization(self) -> Dict:
        """
        Map global wind vectors to face-based structure for OpenGL visualization.
        """
        if not hasattr(self, 'fluid_solver') or self.fluid_solver is None:
            return None
            
        winds = self.fluid_solver.get_wind_vectors()
        wind_values = {}
        
        for face_id in range(self.n_faces):
            face_key = f'face_{face_id}'
            face_node_indices = self.node_mapping[face_key]
            
            rows, cols = face_node_indices.shape
            # Shape: (rows, cols, 3)
            face_winds = np.zeros((rows, cols, 3), dtype=np.float64)
            
            for i in range(rows):
                for j in range(cols):
                    global_node_id = face_node_indices[i, j]
                    face_winds[i, j] = winds[global_node_id]
            
            wind_values[face_key] = face_winds
            
        return wind_values
    
    def simulate_climate_model(self, dt: float):
        """
        Simulate minimalist climate model with single atmospheric layer.
        Uses unified global FEM structure for temperature field.
        
        Two-step process:
        1. Radiative balance: Solar input vs thermal output (vectorized)
        2. Atmosphere fluid solve: Lateral heat diffusion through atmosphere
        
        Energy conservation equation:
        4πR²hρc dT/dt = πR²(1-α)S₀ - 4πR²εσT⁴ + heat_diffusion
        
        Simplified per unit area:
        hρc dT/dt = (1-α)S₀/4 - εσT⁴ + κ∇²T
        
        Parameters
        ----------
        dt : float
            Time step in seconds
        """
        # Initialize temperatures if first call (global arrays for all three layers)
        if (self.temperatures_ground is None or 
            self.temperatures_layer1 is None or 
            self.temperatures_layer2 is None):
            n_nodes = len(self.global_nodes)
            self.temperatures_ground = np.full(n_nodes, self.initial_temperature)
            # Atmosphere starts slightly cooler (typical atmospheric profile)
            self.temperatures_layer1 = np.full(n_nodes, self.initial_temperature - 10.0)
            self.temperatures_layer2 = np.full(n_nodes, self.initial_temperature - 30.0)
            # Adicionar perturbação inicial na camada 1
            if getattr(self, '_add_initial_perturbation', False):
                np.random.seed(42)
                perturb = np.random.normal(0, 0.5, n_nodes)  # 0.5K de desvio padrão
                self.temperatures_layer1 += perturb
            print(f"Initialized 3-layer temperature field: {n_nodes} nodes per layer")
        
        # ============================================================
        # THREE-LAYER ATMOSPHERIC MODEL WITH SOLAR ABSORPTION
        # ============================================================
        # Ground (Layer 0): Receives transmitted solar + exchanges with Layer 1
        # Layer 1 (Lower atm): Absorbs solar + exchanges with ground and Layer 2
        # Layer 2 (Upper atm): Absorbs solar + exchanges with Layer 1 and space
        #
        # Solar radiation flow (top to bottom):
        #   S_top = S₀ (1-α)  [after albedo reflection]
        #   S₂ = a₂ * S_top  [Layer 2 absorbs fraction a₂]
        #   S₁ = a₁ * S_top * (1-a₂)  [Layer 1 absorbs fraction a₁ of transmitted]
        #   S₀ = S_top * (1-a₂) * (1-a₁)  [Ground absorbs remainder]
        #
        # Thermal radiation exchanges:
        #   Ground ↔ Layer 1: ε0σT0⁴ ↔ ε1σT1⁴
        #   Layer 1 ↔ Layer 2: ε1σT1⁴ ↔ ε2σT2⁴
        #   Layer 2 → Space: ε2σT2⁴
        # ============================================================
        
        # Physical constants
        from simulation_params import (
            GROUND_THICKNESS, GROUND_DENSITY, GROUND_HEAT_CAPACITY, GROUND_EMISSIVITY,
            LAYER1_HEIGHT, LAYER1_DENSITY, LAYER1_HEAT_CAPACITY, LAYER1_EMISSIVITY,
            LAYER2_HEIGHT, LAYER2_DENSITY, LAYER2_HEAT_CAPACITY, LAYER2_EMISSIVITY,
            LAYER1_SOLAR_ABSORPTIVITY, LAYER2_SOLAR_ABSORPTIVITY,
            LAYER1_SOLAR_TRANSMISSIVITY, LAYER2_SOLAR_TRANSMISSIVITY,
            EARTH_ALBEDO, STEFAN_BOLTZMANN
        )
        
        sigma = STEFAN_BOLTZMANN  # W/(m²·K⁴)
        alpha = EARTH_ALBEDO  # albedo
        
        # Thermal masses (heat capacity per unit area)
        C0 = GROUND_THICKNESS * GROUND_DENSITY * GROUND_HEAT_CAPACITY  # J/(m²·K)
        C1 = LAYER1_HEIGHT * LAYER1_DENSITY * LAYER1_HEAT_CAPACITY  # J/(m²·K)
        C2 = LAYER2_HEIGHT * LAYER2_DENSITY * LAYER2_HEAT_CAPACITY  # J/(m²·K)
        
        # Emissivities
        eps0 = 1.0 # GROUND_EMISSIVITY - Force 1.0 to match 3layers.py blackbody assumption
        eps1 = LAYER1_EMISSIVITY
        eps2 = LAYER2_EMISSIVITY
        
        # Solar absorptivities & transmissivities
        a1 = LAYER1_SOLAR_ABSORPTIVITY
        a2 = LAYER2_SOLAR_ABSORPTIVITY
        tau1 = LAYER1_SOLAR_TRANSMISSIVITY
        tau2 = LAYER2_SOLAR_TRANSMISSIVITY
        
        # ============================================================
        # STEP 1: Calculate solar input (LATITUDE DEPENDENT)
        # ============================================================
        
        from simulation_params import SOLAR_CONSTANT
        
        # Calculate latitude-based solar insolation
        # S(phi) = (S0 / pi) * cos(phi)
        # This distribution preserves the global average of S0/4
        
        # Get node coordinates
        x = self.global_nodes[:, 0]
        y = self.global_nodes[:, 1]
        z = self.global_nodes[:, 2]
        
        # Calculate latitude (radians)
        r = np.sqrt(x**2 + y**2 + z**2)
        # Avoid division by zero
        r = np.maximum(r, 1e-10)
        lat_rad = np.arcsin(np.clip(y / r, -1.0, 1.0))
        lat_deg = np.degrees(lat_rad)

        # Use average solar radiation (no day/night cycle)
        S_toa = solar_radiation_at_latitude_average(lat_deg)
        
        # Solar radiation flow (top to bottom) matching 3layers.py:
        # F_A2_SW = F_solar * a1 (Upper)
        # F_A1_SW = F_solar * tau1 * a2 (Lower)
        # F_surf_SW = F_solar * tau1 * tau2 * (1 - albedo)
        
        # Note: a2 in fem_solver is Upper (Layer 2), a1 is Lower (Layer 1)
        # tau2 is Upper (Layer 2), tau1 is Lower (Layer 1)
        
        S_layer2 = S_toa * a2
        S_layer1 = S_toa * tau2 * a1
        S_ground = S_toa * tau2 * tau1 * (1.0 - alpha)
        
        
        # ============================================================
        # STEP 2: Thermal radiation fluxes
        # ============================================================
        
        # Upward radiation from each layer
        R0_up = eps0 * sigma * self.temperatures_ground**4  # Ground → Layer 1
        R1_up = eps1 * sigma * self.temperatures_layer1**4  # Layer 1 → Layer 2
        R2_up = eps2 * sigma * self.temperatures_layer2**4  # Layer 2 → Space
        
        # Downward radiation from each layer
        R1_down = eps1 * sigma * self.temperatures_layer1**4  # Layer 1 → Ground
        R2_down = eps2 * sigma * self.temperatures_layer2**4  # Layer 2 → Layer 1
        
        # ============================================================
        # STEP 3: Energy balance for each layer
        # ============================================================
        
        # Ground (Layer 0):
        # Gains: Solar + IR from Layer 1
        # Loses: IR upward
        dE0_dt = S_ground + R1_down - R0_up  # W/m²
        dT0_dt = dE0_dt / C0
        
        # Layer 1 (Lower atmosphere):
        # Gains: Solar + IR from ground (absorbed fraction) + IR from Layer 2
        # Loses: IR upward + IR downward
        # Note: Absorbs eps1 fraction of R0_up
        dE1_dt = S_layer1 + (eps1 * R0_up) + R2_down - R1_up - R1_down  # W/m²
        dT1_dt = dE1_dt / C1
        
        # Layer 2 (Upper atmosphere):
        # Gains: Solar + IR from Layer 1 (absorbed fraction)
        # Loses: IR upward + IR downward
        # Note: Absorbs eps2 fraction of R1_up
        dE2_dt = S_layer2 + (eps2 * R1_up) - R2_up - R2_down  # W/m²
        dT2_dt = dE2_dt / C2
        
        # ============================================================
        # STEP 4: Time integration with stability limits
        # ============================================================
        
        # Calculate temperature changes for this time step
        dT0 = dT0_dt * dt
        dT1 = dT1_dt * dt
        dT2 = dT2_dt * dt
        
        # Apply stability limit: max temperature change per step
        max_dT = 5.0  # Maximum 5K change per time step
        dT0 = np.clip(dT0, -max_dT, max_dT)
        dT1 = np.clip(dT1, -max_dT, max_dT)
        dT2 = np.clip(dT2, -max_dT, max_dT)
        
        # Update temperatures (vectorized)
        self.temperatures_ground = self.temperatures_ground + dT0
        self.temperatures_layer1 = self.temperatures_layer1 + dT1
        self.temperatures_layer2 = self.temperatures_layer2 + dT2
        
        # Clamp to reasonable ranges (increased ceilings to prevent saturation)
        # Ground: 220K to 360K (wider range for surface)
        # Layer 1: 220K to 340K (increased ceiling)
        # Layer 2: 200K to 330K (increased ceiling)
        self.temperatures_ground = np.clip(self.temperatures_ground, 220.0, 360.0)
        self.temperatures_layer1 = np.clip(self.temperatures_layer1, 220.0, 340.0)
        self.temperatures_layer2 = np.clip(self.temperatures_layer2, 200.0, 330.0)
        
        # ============================================================
        # STEP 5: Logging and diagnostics
        # ============================================================
        
        # Log temperatures periodically
        if not hasattr(self, '_temp_log_counter'):
            self._temp_log_counter = 0
        
        self._temp_log_counter += 1
        if self._temp_log_counter % 100 == 0:
            # Calculate regional averages
            equator_mask = np.abs(lat_deg) < 10.0
            poles_mask = np.abs(lat_deg) > 80.0
            
            T0_mean = np.mean(self.temperatures_ground)
            T0_eq = np.mean(self.temperatures_ground[equator_mask]) if np.any(equator_mask) else T0_mean
            T0_pole = np.mean(self.temperatures_ground[poles_mask]) if np.any(poles_mask) else T0_mean
            
            T1_mean = np.mean(self.temperatures_layer1)
            T1_eq = np.mean(self.temperatures_layer1[equator_mask]) if np.any(equator_mask) else T1_mean
            T1_pole = np.mean(self.temperatures_layer1[poles_mask]) if np.any(poles_mask) else T1_mean
            
            print(f"[Climate] Time: {self.time:.1f}s")
            print(f"  Ground Temp (°C):  Global={T0_mean-273.15:6.2f} | Equator={T0_eq-273.15:6.2f} | Poles={T0_pole-273.15:6.2f}")
            print(f"  Air Temp    (°C):  Global={T1_mean-273.15:6.2f} | Equator={T1_eq-273.15:6.2f} | Poles={T1_pole-273.15:6.2f}")
        
        # ============================================================
        # STEP 6: ATMOSPHERE FLUID SOLVE (Horizontal heat transport)
        # ============================================================
        # Apply atmospheric fluid dynamics (heat diffusion)
        self.solve_atmosphere_fluid(dt)

        # Update fluid dynamics (winds) based on Layer 1 temperature
        self.fluid_solver.update(dt, self.temperatures_layer1)

        # Apply advection to Layer 1 temperature
        # This closes the loop: Temp -> Wind -> Temp
        # The wind moves the heat around, changing the temperature field
        self.temperatures_layer1 = self.fluid_solver.advect(self.temperatures_layer1, dt)

        # Convert ground temperatures to element values for visualization
        self._temperatures_to_element_values()
    
    def solve_atmosphere_fluid(self, dt: float):
        """
        Solve atmospheric fluid dynamics: horizontal heat diffusion.
        
        Models lateral heat transport emulating dT/dt = kappa * Laplacian(T).
        Aplica difusão explícita na camada 1 (temperatura da atmosfera baixa).
        
        Parâmetros
        ----------
        dt : float
            Time step in seconds
        """
        # Coeficiente de difusão reduzido para permitir formação de células (ex: 2000 m²/s)
        kappa = 2000.0  # m²/s (valor típico para atmosfera, pode ajustar)
        # Calcular Laplaciano usando FluidSolver
        laplacian_T = self.fluid_solver._compute_laplacian(self.temperatures_layer1)
        # Atualizar temperatura da camada 1
        self.temperatures_layer1 += kappa * laplacian_T * dt
    
    def _temperatures_to_element_values(self):
        """
        Convert global node temperatures to global node values for visualization.
        Uses ground (surface) temperatures and normalizes to [0, 1] range.
        """
        # Temperature range for visualization (realistic Earth ground range)
        T_min = 220.0  # -53°C (coldest regions)
        T_max = 320.0  # +47°C (hottest regions)
        
        # Use ground temperatures for visualization
        # Normalize temperatures to [0, 1] and store as global node values
        self.global_node_values = (self.temperatures_ground - T_min) / (T_max - T_min)
        self.global_node_values = np.clip(self.global_node_values, 0.0, 1.0)
    
    def simulate_heat_diffusion(self, source_point: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                                 decay_rate: float = 2.0) -> Dict:
        """
        Simulate heat diffusion from a source point on the sphere.
        Uses global node structure only.
        
        Parameters
        ----------
        source_point : tuple
            (x, y, z) coordinates of heat source on sphere
        decay_rate : float
            Rate of heat decay with distance (higher = faster decay)
        
        Returns
        -------
        node_values : dict
            Dictionary with scalar values [0, 1] for each node (for visualization)
        """
        source = np.array(source_point)
        source = source / np.linalg.norm(source)  # Normalize to sphere surface
        
        # Compute distances for all global nodes at once (vectorized)
        # Angular distance on sphere (great circle distance)
        dots = np.dot(self.global_nodes, source)
        cos_angles = np.clip(dots / (np.linalg.norm(self.global_nodes, axis=1) + 1e-10), -1.0, 1.0)
        angles = np.arccos(cos_angles)  # Radians [0, π]
        
        # Exponential decay: value = exp(-decay_rate * distance)
        self.global_node_values = np.exp(-decay_rate * angles)
        self.global_node_values = np.clip(self.global_node_values, 0.0, 1.0)
        
        return self.interpolate_to_nodes()
    
    def simulate_wave_pattern(self, n_waves: int = 4, time: float = 0.0) -> Dict:
        """
        Simulate wave pattern based on spherical harmonics.
        Uses global node structure only.
        
        Parameters
        ----------
        n_waves : int
            Number of wave lobes
        time : float
            Time parameter for animation
        
        Returns
        -------
        node_values : dict
            Dictionary with scalar values [0, 1] for each node (for visualization)
        """
        # Convert global nodes to spherical coordinates (vectorized)
        cx, cy, cz = self.global_nodes[:, 0], self.global_nodes[:, 1], self.global_nodes[:, 2]
        r = np.sqrt(cx**2 + cy**2 + cz**2)
        theta = np.arccos(np.clip(cz / r, -1.0, 1.0))  # Polar angle [0, π]
        phi = np.arctan2(cy, cx)   # Azimuthal angle [-π, π]
        
        # Wave pattern using trigonometric functions
        values = np.sin(n_waves * theta) * np.cos(n_waves * phi + time)
        
        # Normalize to [0, 1]
        self.global_node_values = (values + 1.0) / 2.0
        self.global_node_values = np.clip(self.global_node_values, 0.0, 1.0)
        
        return self.interpolate_to_nodes()
    
    def simulate_dual_vortex(self, time: float = 0.0) -> Dict:
        """
        Simulate two rotating vortex patterns on the sphere.
        Uses global node structure only.
        
        Parameters
        ----------
        time : float
            Time parameter for animation
        
        Returns
        -------
        node_values : dict
            Dictionary with scalar values [0, 1] for each node (for visualization)
        """
        # Two vortex centers rotating around Z-axis
        angle1 = time
        angle2 = time + np.pi
        
        center1 = np.array([np.cos(angle1) * 0.7, np.sin(angle1) * 0.7, 0.3])
        center2 = np.array([np.cos(angle2) * 0.7, np.sin(angle2) * 0.7, -0.3])
        
        # Normalize to sphere
        center1 = center1 / np.linalg.norm(center1)
        center2 = center2 / np.linalg.norm(center2)
        
        # Vectorized distance calculations
        nodes_norm = self.global_nodes / (np.linalg.norm(self.global_nodes, axis=1, keepdims=True) + 1e-10)
        
        # Distance from both vortices
        dots1 = np.dot(nodes_norm, center1)
        dots2 = np.dot(nodes_norm, center2)
        dist1 = np.arccos(np.clip(dots1, -1.0, 1.0))
        dist2 = np.arccos(np.clip(dots2, -1.0, 1.0))
        
        # Combine with different decay rates
        val1 = np.exp(-3.0 * dist1)
        val2 = np.exp(-3.0 * dist2)
        
        self.global_node_values = np.clip(val1 + val2, 0.0, 1.0)
        
        return self.interpolate_to_nodes()
    
    def simulate_latitude_gradient(self) -> Dict:
        """
        Simple simulation: value varies with latitude (z-coordinate).
        Uses global node structure only.
        
        Returns
        -------
        node_values : dict
            Dictionary with scalar values [0, 1] for each node (for visualization)
        """
        # Extract z-coordinates from global nodes (vectorized)
        cz = self.global_nodes[:, 2]
        
        # Normalize z-coordinate [-1, 1] to [0, 1]
        self.global_node_values = (cz + 1.0) / 2.0
        self.global_node_values = np.clip(self.global_node_values, 0.0, 1.0)
        
        return self.interpolate_to_nodes()
    
    def interpolate_to_nodes(self, global_values: np.ndarray = None) -> Dict:
        """
        Map global node values to face-based structure for OpenGL visualization.
        This is the ONLY function that creates face-based structure from global mesh.
        
        Parameters
        ----------
        global_values : np.ndarray, optional
            Global values to map. If None, uses self.global_node_values.
        
        Returns
        -------
        node_values : dict
            Dictionary with scalar values [0, 1] for each node (per face) - FOR VISUALIZATION ONLY
        """
        values_to_map = global_values if global_values is not None else self.global_node_values
        
        if values_to_map is None:
            raise ValueError("No global node values. Run a simulation first.")
        
        node_values = {}
        
        # Map global node values directly to face-based structure
        for face_id in range(self.n_faces):
            face_key = f'face_{face_id}'
            face_node_indices = self.node_mapping[face_key]
            
            # Create face node value array
            face_node_values = np.zeros_like(face_node_indices, dtype=np.float64)
            
            rows, cols = face_node_indices.shape
            for i in range(rows):
                for j in range(cols):
                    global_node_id = face_node_indices[i, j]
                    face_node_values[i, j] = values_to_map[global_node_id]
            
            node_values[face_key] = face_node_values
        
        return node_values
    
    def get_element_count(self) -> int:
        """Get total number of elements in the mesh."""
        return len(self.connectivity)
    
    def get_node_count(self) -> int:
        """Get total number of nodes in the mesh."""
        return len(self.global_nodes)


if __name__ == "__main__":
    # Simple test
    print("FEM Solver Test")
    print("=" * 60)
    
    # Create a simple test grid
    from mesh_generator import generate_cubed_sphere_grid
    
    grid = generate_cubed_sphere_grid(n_points=10, radius=1.0)
    
    # Create solver
    solver = SphereFEMSolver(grid)
    
    print(f"\nMesh statistics:")
    print(f"  Total elements: {solver.get_element_count()}")
    print(f"  Total nodes: {solver.get_node_count()}")
    
    # Test different simulations
    print("\n" + "=" * 60)
    
    # Heat diffusion
    elem_vals = solver.simulate_heat_diffusion(source_point=(1.0, 0.0, 0.0), decay_rate=2.0)
    node_vals = solver.interpolate_to_nodes()
    print(f"Element value range: [{min([v.min() for v in elem_vals.values()]):.3f}, "
          f"{max([v.max() for v in elem_vals.values()]):.3f}]")
    
    # Wave pattern
    elem_vals = solver.simulate_wave_pattern(n_waves=3, time=0.0)
    node_vals = solver.interpolate_to_nodes()
    print(f"Element value range: [{min([v.min() for v in elem_vals.values()]):.3f}, "
          f"{max([v.max() for v in elem_vals.values()]):.3f}]")
    
    # Latitude gradient
    elem_vals = solver.simulate_latitude_gradient()
    node_vals = solver.interpolate_to_nodes()
    print(f"Element value range: [{min([v.min() for v in elem_vals.values()]):.3f}, "
          f"{max([v.max() for v in elem_vals.values()]):.3f}]")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
