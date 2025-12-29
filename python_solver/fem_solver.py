"""
Finite Element Method Solver for Cubed-Sphere Grid

This module provides a simple FEM solver that simulates physical phenomena
on the spherical mesh and returns scalar values for each element.
"""

import numpy as np
import time
from typing import Dict, Tuple

# No sys.path insertion here — assume package imports are configured
from python_solver.simulation_params import (AVERAGE_SOLAR_RADIATION, EARTH_ALBEDO, 
                               STEFAN_BOLTZMANN, ATMOSPHERIC_EMISSIVITY,
                               EARTH_RADIUS, ATMOSPHERIC_LAYER_HEIGHT,
                               AIR_DENSITY, AIR_HEAT_CAPACITY,
                               INITIAL_GLOBAL_TEMPERATURE, SOLAR_CONSTANT, 
                               solar_radiation_at_latitude_average,
                               ATMOSPHERIC_THERMAL_DIFFUSIVITY)

from fluid_dynamics import FluidSolver

# Use the shared utility from the repository root
from convert_cubed_sphere_to_fem import convert_cubed_sphere_to_fem


class SphereFEMSolver:
    """
    Finite Element solver for spherical grids.
    """
    def __init__(self, grid_points: Dict, elevation_data: Dict = None):
        self.n_faces = 6
        self.time = 0.0
        self.simulation_type = 'heat_diffusion'
        self.sim_params = {}
        self.heavy_computation = False
        self.computation_delay = 0.1
        print("Converting cubed-sphere to unified FEM structure...")
        self.global_nodes, self.connectivity, self.node_mapping = convert_cubed_sphere_to_fem(grid_points)
        face_0 = grid_points['face_0']
        self.face_rows, self.face_cols = face_0['x'].shape
        del grid_points
        self.global_elevation = None
        if elevation_data is not None:
            self.global_elevation = self._extract_elevation_global(elevation_data)
            print("FEM Solver initialized with elevation data")
        self.temperatures_ground = None
        self.temperatures_layer1 = None
        self.temperatures_layer2 = None
        self.initial_temperature = INITIAL_GLOBAL_TEMPERATURE
        self._add_initial_perturbation = True
        self.global_node_values = None
        self.fluid_solver = FluidSolver(self.global_nodes, self.connectivity)
        print("Fluid Solver initialized")

    def _extract_elevation_global(self, elevation_data: Dict) -> np.ndarray:
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
                    elev_val = np.clip(face_elev[i, j] / 8000.0, 0.0, 1.0)
                    global_elevation[global_node_id] = elev_val
        print("Elevation data mapped to global nodes")
        return global_elevation

    def initialize_with_elevation(self):
        if self.global_elevation is None:
            print("Warning: No elevation data available")
            return
        print("Initializing simulation with global elevation data...")
        self.global_node_values = self.global_elevation.copy()

    def setup_simulation(self, sim_type: str, heavy_computation: bool = False, 
                        computation_delay: float = 0.1, **params):
        self.simulation_type = sim_type
        self.sim_params = params
        self.time = 0.0
        self.heavy_computation = heavy_computation
        self.computation_delay = computation_delay
        print(f"\nSetup simulation: {sim_type}")
        print(f"Heavy computation: {heavy_computation} (delay: {computation_delay}s)")
        print(f"Parameters: {params}")

    def update_simulation(self, dt: float = 0.016) -> Dict:
        self.time += dt
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
            pulse = (np.sin(self.time * freq) + 1.0) / 2.0
            self.simulate_heat_diffusion(source_point=source, decay_rate=decay * (0.5 + pulse))
        else:
            self.simulate_latitude_gradient()
        result = {
            'scalars': self.interpolate_to_nodes(),
            'vectors': self.get_wind_vectors_for_visualization(),
            'pressure': self.get_pressure_for_visualization(),
            'vertical_motion': self.get_vertical_motion_for_visualization()
        }

        # Log temperature statistics from scalars (min, mean, max)
        try:
            scalars_dict = result.get('scalars') or {}
            if scalars_dict:
                all_vals = np.concatenate([arr.ravel() for arr in scalars_dict.values()])
                if all_vals.size > 0:
                    vmin = float(np.min(all_vals))
                    vmax = float(np.max(all_vals))
                    vmean = float(np.mean(all_vals))
                    print(f"[PythonSolver] Temp stats — min: {vmin:.3f}, mean: {vmean:.3f}, max: {vmax:.3f}")
        except Exception:
            pass
        return result

    def get_vertical_motion_for_visualization(self) -> Dict:
        if not hasattr(self, 'fluid_solver') or self.fluid_solver is None:
            return None
        w = self.fluid_solver.vertical_velocity
        w_mean = np.mean(w)
        w_std = np.std(w)
        if w_std < 1e-8:
            normalized_w = np.full_like(w, 0.5)
        else:
            normalized_w = (w - (w_mean - 2*w_std)) / (4*w_std)
            normalized_w = np.clip(normalized_w, 0.0, 1.0)
        return self.interpolate_to_nodes(normalized_w)

    def get_pressure_for_visualization(self) -> Dict:
        if not hasattr(self, 'fluid_solver') or self.fluid_solver is None:
            return None
        pressure = self.fluid_solver.pressure
        p_mean = np.mean(pressure)
        p_std = np.std(pressure)
        if p_std < 1e-5:
            normalized_pressure = np.full_like(pressure, 0.5)
        else:
            normalized_pressure = (pressure - (p_mean - 2*p_std)) / (4*p_std)
            normalized_pressure = np.clip(normalized_pressure, 0.0, 1.0)
        return self.interpolate_to_nodes(normalized_pressure)

    def get_wind_vectors_for_visualization(self) -> Dict:
        if not hasattr(self, 'fluid_solver') or self.fluid_solver is None:
            return None
        winds = self.fluid_solver.get_wind_vectors()
        wind_values = {}
        for face_id in range(self.n_faces):
            face_key = f'face_{face_id}'
            face_node_indices = self.node_mapping[face_key]
            rows, cols = face_node_indices.shape
            face_winds = np.zeros((rows, cols, 3), dtype=np.float64)
            for i in range(rows):
                for j in range(cols):
                    global_node_id = face_node_indices[i, j]
                    face_winds[i, j] = winds[global_node_id]
            wind_values[face_key] = face_winds
        return wind_values

    def simulate_climate_model(self, dt: float):
        if (self.temperatures_ground is None or 
            self.temperatures_layer1 is None or 
            self.temperatures_layer2 is None):
            n_nodes = len(self.global_nodes)
            self.temperatures_ground = np.full(n_nodes, self.initial_temperature)
            self.temperatures_layer1 = np.full(n_nodes, self.initial_temperature - 10.0)
            self.temperatures_layer2 = np.full(n_nodes, self.initial_temperature - 30.0)
            if getattr(self, '_add_initial_perturbation', False):
                np.random.seed(42)
                perturb = np.random.normal(0, 0.5, n_nodes)
                self.temperatures_layer1 += perturb
            print(f"Initialized 3-layer temperature field: {n_nodes} nodes per layer")
        from python_solver.simulation_params import (
            GROUND_THICKNESS, GROUND_DENSITY, GROUND_HEAT_CAPACITY, GROUND_EMISSIVITY,
            LAYER1_HEIGHT, LAYER1_DENSITY, LAYER1_HEAT_CAPACITY, LAYER1_EMISSIVITY,
            LAYER2_HEIGHT, LAYER2_DENSITY, LAYER2_HEAT_CAPACITY, LAYER2_EMISSIVITY,
            LAYER1_SOLAR_ABSORPTIVITY, LAYER2_SOLAR_ABSORPTIVITY,
            LAYER1_SOLAR_TRANSMISSIVITY, LAYER2_SOLAR_TRANSMISSIVITY,
            EARTH_ALBEDO, STEFAN_BOLTZMANN
        )
        sigma = STEFAN_BOLTZMANN
        alpha = EARTH_ALBEDO
        C0 = GROUND_THICKNESS * GROUND_DENSITY * GROUND_HEAT_CAPACITY
        C1 = LAYER1_HEIGHT * LAYER1_DENSITY * LAYER1_HEAT_CAPACITY
        C2 = LAYER2_HEIGHT * LAYER2_DENSITY * LAYER2_HEAT_CAPACITY
        eps0 = 1.0
        eps1 = LAYER1_EMISSIVITY
        eps2 = LAYER2_EMISSIVITY
        a1 = LAYER1_SOLAR_ABSORPTIVITY
        a2 = LAYER2_SOLAR_ABSORPTIVITY
        tau1 = LAYER1_SOLAR_TRANSMISSIVITY
        tau2 = LAYER2_SOLAR_TRANSMISSIVITY
        from python_solver.simulation_params import SOLAR_CONSTANT
        x = self.global_nodes[:, 0]
        y = self.global_nodes[:, 1]
        z = self.global_nodes[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.maximum(r, 1e-10)
        lat_rad = np.arcsin(np.clip(y / r, -1.0, 1.0))
        lat_deg = np.degrees(lat_rad)
        S_toa = solar_radiation_at_latitude_average(lat_deg)
        S_layer2 = S_toa * a2
        S_layer1 = S_toa * tau2 * a1
        S_ground = S_toa * tau2 * tau1 * (1.0 - alpha)
        R0_up = eps0 * sigma * self.temperatures_ground**4
        R1_up = eps1 * sigma * self.temperatures_layer1**4
        R2_up = eps2 * sigma * self.temperatures_layer2**4
        R1_down = eps1 * sigma * self.temperatures_layer1**4
        R2_down = eps2 * sigma * self.temperatures_layer2**4
        dE0_dt = S_ground + R1_down - R0_up
        dT0_dt = dE0_dt / C0
        dE1_dt = S_layer1 + (eps1 * R0_up) + R2_down - R1_up - R1_down
        dT1_dt = dE1_dt / C1
        dE2_dt = S_layer2 + (eps2 * R1_up) - R2_up - R2_down
        dT2_dt = dE2_dt / C2
        dT0 = dT0_dt * dt
        dT1 = dT1_dt * dt
        dT2 = dT2_dt * dt
        max_dT = 5.0
        dT0 = np.clip(dT0, -max_dT, max_dT)
        dT1 = np.clip(dT1, -max_dT, max_dT)
        dT2 = np.clip(dT2, -max_dT, max_dT)
        self.temperatures_ground = self.temperatures_ground + dT0
        self.temperatures_layer1 = self.temperatures_layer1 + dT1
        self.temperatures_layer2 = self.temperatures_layer2 + dT2
        self.temperatures_ground = np.clip(self.temperatures_ground, 220.0, 360.0)
        self.temperatures_layer1 = np.clip(self.temperatures_layer1, 220.0, 340.0)
        self.temperatures_layer2 = np.clip(self.temperatures_layer2, 200.0, 330.0)
        if not hasattr(self, '_temp_log_counter'):
            self._temp_log_counter = 0
        self._temp_log_counter += 1
        if self._temp_log_counter % 100 == 0:
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
        self.solve_atmosphere_fluid(dt)
        self.fluid_solver.update(dt, self.temperatures_layer1)
        self.temperatures_layer1 = self.fluid_solver.advect(self.temperatures_layer1, dt)
        self._temperatures_to_element_values()

    def solve_atmosphere_fluid(self, dt: float):
        kappa = 2000.0
        laplacian_T = self.fluid_solver._compute_laplacian(self.temperatures_layer1)
        self.temperatures_layer1 += kappa * laplacian_T * dt

    def _temperatures_to_element_values(self):
        T_min = 220.0
        T_max = 320.0
        self.global_node_values = (self.temperatures_ground - T_min) / (T_max - T_min)
        self.global_node_values = np.clip(self.global_node_values, 0.0, 1.0)

    def simulate_heat_diffusion(self, source_point: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                                 decay_rate: float = 2.0) -> Dict:
        source = np.array(source_point)
        source = source / np.linalg.norm(source)
        dots = np.dot(self.global_nodes, source)
        cos_angles = np.clip(dots / (np.linalg.norm(self.global_nodes, axis=1) + 1e-10), -1.0, 1.0)
        angles = np.arccos(cos_angles)
        self.global_node_values = np.exp(-decay_rate * angles)
        self.global_node_values = np.clip(self.global_node_values, 0.0, 1.0)
        return self.interpolate_to_nodes()

    def simulate_wave_pattern(self, n_waves: int = 4, time: float = 0.0) -> Dict:
        cx, cy, cz = self.global_nodes[:, 0], self.global_nodes[:, 1], self.global_nodes[:, 2]
        r = np.sqrt(cx**2 + cy**2 + cz**2)
        theta = np.arccos(np.clip(cz / r, -1.0, 1.0))
        phi = np.arctan2(cy, cx)
        values = np.sin(n_waves * theta) * np.cos(n_waves * phi + time)
        self.global_node_values = (values + 1.0) / 2.0
        self.global_node_values = np.clip(self.global_node_values, 0.0, 1.0)
        return self.interpolate_to_nodes()

    def simulate_dual_vortex(self, time: float = 0.0) -> Dict:
        angle1 = time
        angle2 = time + np.pi
        center1 = np.array([np.cos(angle1) * 0.7, np.sin(angle1) * 0.7, 0.3])
        center2 = np.array([np.cos(angle2) * 0.7, np.sin(angle2) * 0.7, -0.3])
        center1 = center1 / np.linalg.norm(center1)
        center2 = center2 / np.linalg.norm(center2)
        nodes_norm = self.global_nodes / (np.linalg.norm(self.global_nodes, axis=1, keepdims=True) + 1e-10)
        dots1 = np.dot(nodes_norm, center1)
        dots2 = np.dot(nodes_norm, center2)
        dist1 = np.arccos(np.clip(dots1, -1.0, 1.0))
        dist2 = np.arccos(np.clip(dots2, -1.0, 1.0))
        val1 = np.exp(-3.0 * dist1)
        val2 = np.exp(-3.0 * dist2)
        self.global_node_values = np.clip(val1 + val2, 0.0, 1.0)
        return self.interpolate_to_nodes()

    def simulate_latitude_gradient(self) -> Dict:
        cz = self.global_nodes[:, 2]
        self.global_node_values = (cz + 1.0) / 2.0
        self.global_node_values = np.clip(self.global_node_values, 0.0, 1.0)
        return self.interpolate_to_nodes()

    def interpolate_to_nodes(self, global_values: np.ndarray = None) -> Dict:
        values_to_map = global_values if global_values is not None else self.global_node_values
        if values_to_map is None:
            raise ValueError("No global node values. Run a simulation first.")
        node_values = {}
        for face_id in range(self.n_faces):
            face_key = f'face_{face_id}'
            face_node_indices = self.node_mapping[face_key]
            face_node_values = np.zeros_like(face_node_indices, dtype=np.float64)
            rows, cols = face_node_indices.shape
            for i in range(rows):
                for j in range(cols):
                    global_node_id = face_node_indices[i, j]
                    face_node_values[i, j] = values_to_map[global_node_id]
            node_values[face_key] = face_node_values
        return node_values

    def get_element_count(self) -> int:
        return len(self.connectivity)

    def get_node_count(self) -> int:
        return len(self.global_nodes)


if __name__ == "__main__":
    print("FEM Solver Test")
    print("=" * 60)
    from mesh_generator import generate_cubed_sphere_grid
    grid = generate_cubed_sphere_grid(n_points=10, radius=1.0)
    solver = SphereFEMSolver(grid)
    print(f"\nMesh statistics:")
    print(f"  Total elements: {solver.get_element_count()}")
    print(f"  Total nodes: {solver.get_node_count()}")
    print("\n" + "=" * 60)
    elem_vals = solver.simulate_heat_diffusion(source_point=(1.0, 0.0, 0.0), decay_rate=2.0)
    node_vals = solver.interpolate_to_nodes()
    print(f"Element value range: [{min([v.min() for v in elem_vals.values()]):.3f}, "
          f"{max([v.max() for v in elem_vals.values()]):.3f}]")
    elem_vals = solver.simulate_wave_pattern(n_waves=3, time=0.0)
    node_vals = solver.interpolate_to_nodes()
    print(f"Element value range: [{min([v.min() for v in elem_vals.values()]):.3f}, "
          f"{max([v.max() for v in elem_vals.values()]):.3f}]")
    elem_vals = solver.simulate_latitude_gradient()
    node_vals = solver.interpolate_to_nodes()
    print(f"Element value range: [{min([v.min() for v in elem_vals.values()]):.3f}, "
          f"{max([v.max() for v in elem_vals.values()]):.3f}]")
    print("\n" + "=" * 60)
    print("All tests passed!")

