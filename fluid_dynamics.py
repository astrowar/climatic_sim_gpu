
import numpy as np
from typing import Dict, List, Tuple
from simulation_params import EARTH_RADIUS, EARTH_ANGULAR_VELOCITY

class FluidSolver:
    """
    Simulates atmospheric fluid dynamics (winds and advection) on a spherical mesh.
    
    Physics Model:
    - Pressure derived from Temperature (Ideal Gas Law / Hydrostatic)
    - Winds driven by Pressure Gradient Force (PGF) and Coriolis Force
    - Temperature advected by Winds
    """
    
    def __init__(self, nodes: np.ndarray, connectivity: List[List[int]]):
        """
        Initialize fluid solver.
        
        Parameters
        ----------
        nodes : np.ndarray
            Global node coordinates (N, 3)
        connectivity : list
            Element connectivity list
        """
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.connectivity = connectivity
        
        # State variables
        self.velocity = np.zeros((self.n_nodes, 3))  # 3D vector (vx, vy, vz)
        self.pressure = np.zeros(self.n_nodes)
        self.density = np.full(self.n_nodes, 1.2) # kg/m³ - Initialize with constant density
        self.vertical_velocity = np.zeros(self.n_nodes) # Proxy for updrafts/downdrafts
        
        # Physical constants
        self.R_gas = 287.0  # J/(kg·K) - Specific gas constant for dry air
        # Rotation rate from parameters
        self.omega = EARTH_ANGULAR_VELOCITY
        self.drag_coeff = 5.0e-6 # Reduced friction to allow geostrophic balance
        self.drag_coeff_quad = 1.0e-7 # Reduced for more fluid motion
        
        # Precompute neighbor graph for gradient calculations
        self.neighbors = self._build_neighbor_graph()
        self._precompute_geometry()
        
    def _build_neighbor_graph(self) -> List[List[int]]:
        """Build adjacency list for all nodes."""
        neighbors = [set() for _ in range(self.n_nodes)]
        for element in self.connectivity:
            for i in range(len(element)):
                u = element[i]
                for j in range(len(element)):
                    if i == j: continue
                    v = element[j]
                    neighbors[u].add(v)
        return [list(s) for s in neighbors]

    def _precompute_geometry(self):
        """Precompute edge data for vectorized operations."""
        # Convert neighbor graph to edge list for vectorization
        sources = []
        targets = []
        
        for i in range(self.n_nodes):
            for j in self.neighbors[i]:
                sources.append(i)
                targets.append(j)
        
        self.edge_sources = np.array(sources, dtype=np.int32)
        self.edge_targets = np.array(targets, dtype=np.int32)
        
        # Precompute edge vectors and weights
        pos_i = self.nodes[self.edge_sources]
        pos_j = self.nodes[self.edge_targets]
        
        edge_vectors = pos_j - pos_i
        distances_sq = np.sum(edge_vectors**2, axis=1)
        distances = np.sqrt(distances_sq)
        
        # Avoid division by zero
        mask = distances > 1e-10
        
        # Normalized direction: r_ij / |r_ij|
        self.edge_dirs = np.zeros_like(edge_vectors)
        self.edge_dirs[mask] = edge_vectors[mask] / distances[mask][:, np.newaxis]
        
        # Scale distances to real world (meters)
        # The grid is on a unit sphere, so we multiply by Earth radius
        real_distances = distances * EARTH_RADIUS
        real_distances_sq = real_distances**2
        
        # Weight for gradient: 1 / |r_ij|
        # Formula: (val_j - val_i) * disp / dist_sq = (val_j - val_i) * dir / dist
        self.edge_weights = np.zeros(len(sources))
        self.edge_weights[mask] = 1.0 / real_distances[mask]
        
        # Weight for Laplacian: 1 / |r_ij|^2
        self.laplacian_weights = np.zeros(len(sources))
        self.laplacian_weights[mask] = 1.0 / real_distances_sq[mask]
        
        # Precompute neighbor counts for smoothing
        self.neighbor_counts = np.zeros(self.n_nodes)
        np.add.at(self.neighbor_counts, self.edge_sources, 1)
        self.neighbor_counts[self.neighbor_counts == 0] = 1
        
        # Precompute gradient normalization factor based on valence
        # Valence 3 (Corners) -> Factor 1.0
        # Valence 4 (Faces/Edges) -> Factor 0.5
        self.gradient_scale = np.where(self.neighbor_counts <= 3, 1.0, 0.5)

    def _smooth_field(self, field: np.ndarray, factor: float = 0.1) -> np.ndarray:
        """
        Apply simple neighbor averaging to smooth the field.
        field_new = (1-factor)*field + factor*avg(neighbors)
        """
        neighbor_sum = np.zeros_like(field)
        np.add.at(neighbor_sum, self.edge_sources, field[self.edge_targets])
        neighbor_avg = neighbor_sum / self.neighbor_counts
        return (1.0 - factor) * field + factor * neighbor_avg

    def update(self, dt: float, temperatures: np.ndarray):
        """
        Update fluid state (winds) based on temperature field.
        
        Parameters
        ----------
        dt : float
            Time step
        temperatures : np.ndarray
            Temperature field (N,)
        """
        # 0. Update Density (Mass Conservation / Continuity Equation)
        # d(rho)/dt = -div(rho * v)
        # Mass flows out of divergent regions (equator) and into convergent regions (poles)
        
        # Compute mass flux vector: J = rho * v
        # rho is (N,), v is (N,3) -> J is (N,3)
        mass_flux = self.velocity * self.density[:, np.newaxis]
        
        # Compute divergence of mass flux
        divergence = self._compute_divergence(mass_flux)
        
        # Vertical velocity proxy: Convergence -> Updraft (+), Divergence -> Downdraft (-)
        # We use a smoothed version of -divergence
        self.vertical_velocity = -divergence / self.density
        self.vertical_velocity = self._smooth_field(self.vertical_velocity, factor=0.1)
        
        # Compute Laplacian of density for diffusion (smoothing)
        # This prevents checkerboard patterns and numerical instability
        laplacian_rho = self._compute_laplacian(self.density)
        diffusion_coeff = 20000.0 # m²/s - Reduced artificial viscosity to allow smaller cells
        
        # Thermal Buoyancy Term (Simulates vertical convection)
        # Hot air rises -> Mass loss at surface -> Density decreases
        # Cold air sinks -> Mass gain at surface -> Density increases
        # This creates Thermal Lows at equator and Thermal Highs at poles
        T_mean = np.mean(temperatures)
        buoyancy_coeff = 2.0e-4  # Reduced to prevent instability
        thermal_forcing = -buoyancy_coeff * (temperatures - T_mean) * self.density

        # Update density
        # d(rho)/dt = -div(rho*v) + nu * laplacian(rho) + thermal_forcing
        self.density += (-divergence + diffusion_coeff * laplacian_rho + thermal_forcing) * dt
        
        # Apply explicit smoothing filter to remove checkerboard artifacts
        # Reduced smoothing to preserve gradients for Ferrel/Polar cells
        self.density = self._smooth_field(self.density, factor=0.01)
        
        self.density = np.clip(self.density, 0.1, 10.0) # Prevent vacuum or black hole
        
        # 1. Compute Pressure Field: P = rho * R * T
        # Now pressure depends on both Temperature AND Density
        self.pressure = self.density * self.R_gas * temperatures
        
        # Smooth pressure field as well to prevent gradient spikes
        self.pressure = self._smooth_field(self.pressure, factor=0.01)
        
        # 2. Compute Pressure Gradient Force (PGF)
        # F_pg = -(1/rho) * grad(P)
        # Use local density for acceleration
        grad_p = self._compute_gradient(self.pressure)
        pgf = -grad_p / self.density[:, np.newaxis]
        
        # 3. Update Velocity (Momentum Equation)
        # dv/dt = PGF + Coriolis + Friction
        
        # Coriolis Force: F_c = -2 * Omega x v
        # Earth rotation vector (along Y axis in OpenGL, Z in physics - check convention)
        # Based on previous fix, Y is North/South axis.
        earth_axis = np.array([0.0, 1.0, 0.0]) 
        rotation_vec = earth_axis * self.omega
        
        # We need to update velocity iteratively or explicitly
        # v_new = v + (PGF + Coriolis + Friction) * dt
        
        # Vectorized cross product for Coriolis
        # np.cross works on last axis
        coriolis = -2.0 * np.cross(rotation_vec, self.velocity)
        
        # Friction / Drag
        # Linear drag (Rayleigh friction) + Quadratic drag (Aerodynamic)
        # F_drag = -k1*v - k2*|v|*v
        speed = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        friction = -(self.drag_coeff + self.drag_coeff_quad * speed) * self.velocity
        
        acceleration = pgf + coriolis + friction
        
        self.velocity += acceleration * dt
        
        # Limit maximum velocity to prevent numerical explosion
        max_velocity = 150.0 # m/s (Jet streams can reach ~100-150 m/s)
        current_speed = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        # Avoid division by zero
        current_speed = np.maximum(current_speed, 1e-10)
        
        # Apply limit where speed exceeds max
        mask = current_speed > max_velocity
        # We need to broadcast the scaling factor
        scale_factor = np.ones_like(current_speed)
        scale_factor[mask] = max_velocity / current_speed[mask]
        
        self.velocity *= scale_factor
        
        # 3.5 Advection of Momentum (Inertia)
        # This makes the wind "carry" its own speed, creating swirls and vortices
        self.velocity = self.advect_vector(self.velocity, dt)
        
        # Apply momentum diffusion (viscosity) via smoothing
        # This helps stabilize the velocity field
        # Reduced factor to preserve more detail in the flow
        vx = self._smooth_field(self.velocity[:, 0], factor=0.005)
        vy = self._smooth_field(self.velocity[:, 1], factor=0.005)
        vz = self._smooth_field(self.velocity[:, 2], factor=0.005)
        self.velocity = np.stack([vx, vy, vz], axis=1)
        
        # 4. Project velocity to tangent plane (enforce flow on sphere surface)
        self._project_velocity_tangent()
        
        # 5. Limit max velocity (stability)
        # Increased limit to 150 m/s (Jet streams can reach ~110 m/s)
        max_vel = 150.0 
        speed = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        mask = speed > max_vel
        
        # Only apply if there are velocities exceeding limit
        if np.any(mask):
            # speed[mask] is 1D (k,), need (k,1) to broadcast with (k,3)
            scaling_factor = (max_vel / speed[mask])[:, np.newaxis]
            self.velocity[mask.flatten()] *= scaling_factor

    def _compute_divergence(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Compute divergence of a vector field on the sphere (Vectorized).
        """
        div = np.zeros(self.n_nodes)
        
        # Vectorized calculation
        vec_i = vector_field[self.edge_sources]
        vec_j = vector_field[self.edge_targets]
        
        # dot(vec_j - vec_i, edge_dir) / dist
        # = dot(vec_j - vec_i, edge_dir) * weight
        
        vec_diff = vec_j - vec_i
        
        # Dot product along last axis (N, 3) . (N, 3) -> (N,)
        dot_products = np.sum(vec_diff * self.edge_dirs, axis=1)
        
        contributions = dot_products * self.edge_weights
        
        # Accumulate
        np.add.at(div, self.edge_sources, contributions)
        
        # Normalize based on valence
        div *= self.gradient_scale
                
        return div

    def _compute_laplacian(self, scalar_field: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian of a scalar field (Vectorized).
        L(phi) = sum( (phi_j - phi_i) / dist_sq )
        """
        lap = np.zeros(self.n_nodes)
        
        val_i = scalar_field[self.edge_sources]
        val_j = scalar_field[self.edge_targets]
        
        diff = val_j - val_i
        
        # Contribution: (val_j - val_i) / dist^2
        contributions = diff * self.laplacian_weights
        
        # Accumulate
        np.add.at(lap, self.edge_sources, contributions)
        
        return lap

    def _compute_gradient(self, scalar_field: np.ndarray) -> np.ndarray:
        """
        Compute gradient of a scalar field on the sphere (Vectorized).
        Returns vector field (N, 3).
        """
        grad = np.zeros((self.n_nodes, 3))
        
        # Vectorized calculation
        val_i = scalar_field[self.edge_sources]
        val_j = scalar_field[self.edge_targets]
        diff = val_j - val_i
        
        # Contribution: diff * weight * dir
        # (val_j - val_i) * dir / dist
        contributions = (diff * self.edge_weights)[:, np.newaxis] * self.edge_dirs
        
        # Accumulate
        np.add.at(grad, self.edge_sources, contributions)
        
        # Normalize based on valence
        grad *= self.gradient_scale[:, np.newaxis]
        
        return grad

    def advect(self, scalar_field: np.ndarray, dt: float) -> np.ndarray:
        """
        Advect a scalar field by the current velocity field.
        dT/dt = -v . grad(T)
        Returns the updated scalar field.
        """
        # Compute gradient of scalar field
        grad = self._compute_gradient(scalar_field)
        
        # Gradient is already normalized by _compute_gradient
        
        # Compute dot product v . grad(T)
        # v is (N,3), grad is (N,3) -> (N,)
        advection_term = np.sum(self.velocity * grad, axis=1)
        
        # Update: T_new = T_old - (v . grad(T)) * dt
        # Add numerical diffusion for stability? For now, pure advection.
        new_field = scalar_field - advection_term * dt
        
        return new_field

    def advect_vector(self, vector_field: np.ndarray, dt: float) -> np.ndarray:
        """
        Advect a vector field (like velocity itself).
        dv/dt = -(v . grad)v
        """
        new_vector = np.zeros_like(vector_field)
        for i in range(3):
            new_vector[:, i] = self.advect(vector_field[:, i], dt)
        return new_vector

    def _project_velocity_tangent(self):
        """Remove radial component of velocity to keep flow on sphere."""
        # Radial vector is just the normalized position vector for a unit sphere
        # Assuming nodes are on unit sphere or close to it
        normals = self.nodes / (np.linalg.norm(self.nodes, axis=1, keepdims=True) + 1e-10)
        
        # v_tangent = v - (v . n) * n
        dots = np.sum(self.velocity * normals, axis=1, keepdims=True)
        self.velocity -= dots * normals

    def get_wind_vectors(self) -> np.ndarray:
        """Return current wind velocity vectors."""
        return self.velocity
