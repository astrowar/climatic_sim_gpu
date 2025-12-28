# Solar Climate Simulation

A 3D atmospheric circulation model on a spherical grid with real-time visualization. This simulation models Earth's climate system including solar radiation, atmospheric layers, fluid dynamics, and wind patterns.

## Features

### Physics Model
- **3-Layer Atmospheric Model**: Ground, lower atmosphere, and upper atmosphere with radiative transfer
- **Solar Radiation**: Latitude-dependent insolation with diurnal cycle (day/night)
- **Fluid Dynamics**: Navier-Stokes equations with:
  - Pressure Gradient Force
  - Coriolis Effect (Earth rotation)
  - Advection (heat and momentum transport)
  - Friction/viscosity
- **Mass Conservation**: Density evolution with continuity equation
- **Vertical Motion**: Convergence/divergence proxy for updrafts and downdrafts

### Visualization
- **Interactive 3D Sphere**: OpenGL rendering with VBO optimization
- **Wind Vectors**: Surface and upper-atmosphere winds with directional arrows
- **Multiple Scalar Modes**:
  - Temperature field
  - Pressure field
  - Vertical motion (updrafts/downdrafts)
- **Color-Coded Layers**: Visual distinction between atmospheric circulation cells
- **Real-time Animation**: Threaded simulation decoupled from rendering

### Grid System
- **Cubed-Sphere Grid**: Uniform mesh avoiding polar singularities
- **Adaptive Resolution**: Configurable mesh density (5-50+ points per face)
- **Node Fusion**: Automatic merging of shared boundary nodes
- **Elevation Data Support**: Optional terrain visualization

## Installation

### Requirements
- Python 3.8+
- Windows, Linux, or macOS

### Setup (PowerShell/Terminal)
```powershell
# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- **PyOpenGL**: 3D graphics rendering
- **GLFW**: Window management and input
- **NumPy**: Numerical computations
- **Pillow**: Image processing (elevation data)

## Usage

### Basic Execution
```powershell
python main.py
```

### Configuration
Edit `main.py` to adjust parameters:

```python
# Mesh density (higher = more detailed but slower)
MESH_DENSITY = 50  # Recommended: 15-50

# Simulation time step (seconds per iteration)
SIMULATION_DT = 300.0  # 5 minutes (recommended: 100-600s)
```

### Controls
| Key | Action |
|-----|--------|
| **Mouse Drag** | Rotate view |
| **Mouse Wheel** | Zoom in/out |
| **W** | Toggle wireframe |
| **F** | Toggle filled faces |
| **C** | Toggle FEM color overlay |
| **M** | Cycle scalar mode (Temp → Pressure → Vertical) |
| **V** | Toggle wind vectors |
| **A** | Toggle animation on/off |
| **R** | Reset view |
| **ESC** | Exit |

## Project Structure

```
solar_sim/
├── main.py                  # Entry point
├── fem_solver.py            # FEM climate solver
├── fluid_dynamics.py        # Atmospheric fluid dynamics (Navier-Stokes)
├── mesh_generator.py        # Cubed-sphere grid generation
├── mesh_quality.py          # Mesh metrics and validation
├── sphere_viewer.py         # OpenGL 3D visualization
├── simulation_params.py     # Physical constants and parameters
├── elevation_reader.py      # Terrain data loader
├── requirements.txt         # Python dependencies
├── legacy/                  # Archive of test/demo scripts
└── README.md               # This file
```

## Physics Model Details

### Atmospheric Layers
1. **Ground (Layer 0)**: Surface with thermal mass, absorbs transmitted solar radiation
2. **Lower Atmosphere (Layer 1)**: Where winds are calculated and visualized
3. **Upper Atmosphere (Layer 2)**: Absorbs incoming solar radiation, radiates to space

### Energy Balance
Solar input varies with:
- **Latitude**: Cosine law ($S \propto \cos(\phi)$)
- **Time of day**: Rotating sun creates diurnal cycle
- **Atmospheric absorption**: 3-layer radiative transfer

Thermal radiation exchanges between layers follow Stefan-Boltzmann law.

### Wind Dynamics
The fluid solver computes wind from:

$$\frac{d\mathbf{v}}{dt} = -\frac{1}{\rho}\nabla P - 2\boldsymbol{\Omega} \times \mathbf{v} - k\mathbf{v} + \nu\nabla^2\mathbf{v}$$

Where:
- $\nabla P$: Pressure gradient (drives wind from hot to cold)
- $2\boldsymbol{\Omega} \times \mathbf{v}$: Coriolis force (deflects wind)
- $k\mathbf{v}$: Friction (dissipates energy)
- $\nu\nabla^2\mathbf{v}$: Viscosity (smooths flow)

### Numerical Methods
- **Time Integration**: Explicit Euler with adaptive time stepping
- **Spatial Discretization**: Finite Element Method on cubed-sphere
- **Stabilization**: 
  - Artificial diffusion for density/pressure
  - Neighbor averaging for smoothing
  - Velocity projection to sphere tangent
- **Parallelization**: Vectorized NumPy operations

## Performance Tips

1. **Lower mesh density** (15-30) for faster simulation
2. **Increase time step** (300-600s) to skip ahead faster
3. **Disable wind vectors** (press 'V') if rendering is slow
4. **Use threaded mode** (enabled by default) to decouple simulation from rendering

## Expected Behavior

### Circulation Patterns
With proper parameters, you should observe:
- **Hadley Cell** (0°-30° latitude): Rising air at equator, sinking at ~30°
- **Ferrel Cell** (30°-60°): Mid-latitude circulation
- **Polar Cell** (60°-90°): Cold polar circulation

### Wind Patterns
- **Trade Winds**: Easterlies near equator
- **Westerlies**: Mid-latitude winds
- **Polar Easterlies**: High-latitude winds

### Temperature Distribution
- Warmest at equator (solar input maximum)
- Coolest at poles (low solar angle)
- Day/night temperature variation

## Troubleshooting

### Simulation Instability
- Reduce `SIMULATION_DT` in `main.py`
- Check that friction coefficients are not too low
- Increase smoothing factors in `fluid_dynamics.py`

### Slow Performance
- Decrease `MESH_DENSITY`
- Disable elevation data loading
- Close other GPU-intensive applications

### No Wind Vectors Visible
- Press 'V' to enable
- Ensure `SIMULATION_DT` is reasonable (100-600s)
- Check that simulation is running (press 'A')

## Development

### Adding New Features
- Modify physics in `fem_solver.py` or `fluid_dynamics.py`
- Adjust visualization in `sphere_viewer.py`
- Change constants in `simulation_params.py`

### Legacy Code
Test scripts and demos are archived in `legacy/` folder:
- `3layers.py`: Original reference model
- `test_*.py`: Unit tests
- `verify_climate.py`: Model validation

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

Based on atmospheric circulation models and finite element methods for geophysical fluid dynamics.
