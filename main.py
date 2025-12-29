"""
Main script for Cubed-Sphere Grid with FEM Simulation

This is the entry point for running the spherical grid visualization
with time-dependent FEM simulations.
"""

from mesh_generator import generate_cubed_sphere_grid, mesh_relaxation, add_elevation_data, normalize_elevation_data
from mesh_quality import compute_mesh_quality
from sphere_viewer import OpenGLSphereViewer
from native_solver import SphereFEMSolver
from elevation_reader import ElevationReader
import os


def main(mesh_density: int = 15):
    """
    Main function to run the OpenGL viewer.
    
    Parameters
    ----------
    mesh_density : int
        Mesh density (number of points per face edge).
        Recommended values:
        - 5-10: Low density (fast, coarse)
        - 15-20: Medium density (balanced)
        - 25-40: High density (slow, detailed)
        - 50+: Very high density (very slow, very detailed)
    """
    print("="*60)
    print("Cubed-Sphere Grid with Threaded FEM Simulation")
    print("="*60)
    
    # ========== MESH GENERATION ==========
    resolution = mesh_density
    print(f"\nMesh density: {resolution}")
    print(f"Generating cubed-sphere grid ({resolution}x{resolution} per face)...")
    grid_points = generate_cubed_sphere_grid(n_points=resolution, radius=1.0)
    
    print(f"Grid generated: {resolution}x{resolution} points per face")
    print(f"Total points: {resolution * resolution * 6}")
    
    # ========== MESH QUALITY - INITIAL ==========
    print("\nInitial mesh quality:")
    initial_metrics = compute_mesh_quality(grid_points)
    print(f"  Mean aspect ratio: {initial_metrics['mean_aspect_ratio']:.4f}")
    print(f"  Max aspect ratio:  {initial_metrics['max_aspect_ratio']:.4f}")
    
    # ========== ELEVATION DATA ==========
    elevation_reader = None
    elevation_file = "etopo_land.png"
    if os.path.exists(elevation_file):
        print(f"\nLoading elevation data from {elevation_file}...")
        try:
            elevation_reader = ElevationReader(elevation_file)
            grid_points = add_elevation_data(grid_points, elevation_reader)
        except Exception as e:
            print(f"Warning: Could not load elevation data: {e}")
            print("Continuing without elevation data...")
    else:
        print(f"\nWarning: {elevation_file} not found.")
        print("Continuing without elevation data (will use face colors)...")
    
    # ========== MESH RELAXATION ==========
    iterations = 10
    omega = 0.5
    grid_points = mesh_relaxation(grid_points, iterations=iterations, omega=omega, 
                                  radius=1.0, verbose=True)
    
    # ========== MESH QUALITY - FINAL ==========
    print("Final mesh quality:")
    final_metrics = compute_mesh_quality(grid_points)
    print(f"  Mean aspect ratio: {final_metrics['mean_aspect_ratio']:.4f}")
    print(f"  Max aspect ratio:  {final_metrics['max_aspect_ratio']:.4f}")
    
    improvement = ((initial_metrics['mean_aspect_ratio'] - final_metrics['mean_aspect_ratio']) 
                   / initial_metrics['mean_aspect_ratio'] * 100)
    print(f"\nMesh quality improvement: {improvement:.2f}%")
    
    # ========== FEM SIMULATION SETUP ==========
    print("\n" + "="*60)
    print("Setting up Heavy Time-Dependent FEM Simulation")
    print("="*60)
    
    # Normalize elevation data for FEM (constant topography)
    elevation_data_normalized = None
    if elevation_reader is not None:
        elevation_data_normalized = normalize_elevation_data(grid_points)
    
    solver = SphereFEMSolver(grid_points, elevation_data=elevation_data_normalized)
    print(f"\nMesh statistics:")
    print(f"  Total elements: {solver.get_element_count()}")
    print(f"  Total nodes: {solver.get_node_count()}")
    
    # Simulation configuration
    USE_THREADS = True          # Enable threaded simulation
    HEAVY_COMPUTATION = False   # Simulate expensive computation (disabled for climate model)
    COMPUTATION_DELAY = 0.0     # Delay per simulation step (seconds)
    SIMULATION_DT = 0.01       # Time step in seconds (300s = 5 minutes)
    
    print(f"\nSimulation mode:")
    print(f"  Time step: {SIMULATION_DT:.1f}s ({SIMULATION_DT/60:.1f} minutes)")
    print(f"  Threaded: {USE_THREADS}")
    print(f"  Heavy computation: {HEAVY_COMPUTATION}")
    print(f"  Computation delay: {COMPUTATION_DELAY}s per step")
    
    # Choose simulation type
    solver.setup_simulation('climate_model', 
                           heavy_computation=HEAVY_COMPUTATION,
                           computation_delay=COMPUTATION_DELAY)
    
 
    
    # Get initial values
    node_values = solver.update_simulation(0.0)
    
    # ========== VISUALIZATION ==========
    print("\nLaunching OpenGL viewer...")
    if USE_THREADS:
        print("Simulation will run in separate thread (decoupled from rendering)")
        print("Rendering should remain smooth even with heavy computation")
    else:
        print("Simulation runs in main thread (coupled with rendering)")
        print("Heavy computation will affect render FPS")
    print("\nPress 'A' to toggle animation on/off")
    print("Press 'C' to toggle FEM colors on/off")
    
    viewer = OpenGLSphereViewer(width=1200, height=900)
    viewer.load_grid_data(grid_points, 
                         node_values=node_values, 
                         fem_solver=solver,
                         use_threads=USE_THREADS,
                         elevation_reader=elevation_reader,
                         simulation_dt=SIMULATION_DT)
    viewer.run()


if __name__ == "__main__":
    # ========== MESH DENSITY CONFIGURATION ==========
    # Control mesh density here:
    # - Low (5-10): Fast, coarse mesh
    # - Medium (15-20): Balanced (recommended)
    # - High (25-40): Slow, fine mesh
    # - Very high (50+): Very slow, very fine mesh
    MESH_DENSITY = 40
    
    # ========== TIME STEP CONFIGURATION ==========
    # Simulation time step (seconds per iteration)
    # Smaller values = more stable but slower simulation
    # Larger values = faster but may become unstable
    # Recommended: 100-600 seconds (1.7-10 minutes)
    # Current default is set inside main() function
    
    main(mesh_density=MESH_DENSITY)
