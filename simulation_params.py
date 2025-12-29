"""
Simulation Parameters

This module contains physical and numerical parameters for the solar/climate simulation.
"""

# ============================================================
# SOLAR RADIATION PARAMETERS
# ============================================================

# Solar constant at top of atmosphere (W/m²)
# This is the average solar energy received per square meter at Earth's distance from the Sun
import math


SOLAR_CONSTANT = 1367.0  # W/m²

# Average solar radiation received by Earth (accounting for spherical geometry and day/night)
# Formula: SOLAR_CONSTANT / 4 (sphere surface area vs cross-section)
AVERAGE_SOLAR_RADIATION = 1367.0 / 4.0  # W/m² = 341.75 W/m²

# Albedo (reflectivity) - fraction of solar radiation reflected back to space
# Earth's average albedo
EARTH_ALBEDO = 0.30  # Earth's actual value ~0.30-0.31

# Net solar radiation absorbed by Earth (after albedo)
NET_SOLAR_RADIATION = AVERAGE_SOLAR_RADIATION * (1.0 - EARTH_ALBEDO)  # ~239 W/m²


# ============================================================
# ATMOSPHERIC PARAMETERS
# ============================================================

# Greenhouse effect parameters
GREENHOUSE_FACTOR = 0.6  # Fraction of IR radiation trapped by atmosphere

# Atmospheric transmissivity
ATMOSPHERIC_TRANSMISSIVITY = 0.7  # Fraction of solar radiation reaching surface

# ============================================================
# THREE-LAYER ATMOSPHERIC MODEL WITH GROUND
# ============================================================
# Ground (Layer 0): Surface/soil (absorbs solar radiation directly)
# Layer 1: Lower atmosphere (0-5 km)
# Layer 2: Upper atmosphere (5-15 km)
#
# Solar radiation flow:
# Sun → Layer 2 (absorbs fraction) → Layer 1 (absorbs fraction) → Ground (absorbs remainder)
#
# Thermal radiation flow:
# Ground ↔ Layer 1 ↔ Layer 2 → Space

# ============================================================
# SOLAR ABSORPTIVITY (fraction of incident solar absorbed by each layer)
# ============================================================
# Solar energy passes through layers from top to bottom
# Atmosphere is mostly transparent to visible light - solar absorption is minimal
# Ground absorbs most solar radiation directly
# Based on validated 3-layer model: tau1=0.90, tau2=0.85, a1=a2=0.05

LAYER2_SOLAR_TRANSMISSIVITY = 0.90  # Upper atmosphere (tau1 from 3layers.py)
LAYER1_SOLAR_TRANSMISSIVITY = 0.85  # Lower atmosphere (tau2 from 3layers.py)

LAYER2_SOLAR_ABSORPTIVITY = 0.05  # Upper atmosphere (a1 from 3layers.py)
LAYER1_SOLAR_ABSORPTIVITY = 0.05  # Lower atmosphere (a2 from 3layers.py)
# Ground receives: tau1 * tau2 * (1-albedo) ≈ 0.90 * 0.85 * 0.70 = 53.6%
# Transmission: ~76.5% reaches ground, ~10% absorbed by atmosphere

# ============================================================
# GROUND LAYER (Layer 0)
# ============================================================
GROUND_THICKNESS = 1.0  # m (effective thermal depth of soil/surface)
GROUND_DENSITY = 2000.0  # kg/m³ (soil/rock density)
GROUND_HEAT_CAPACITY = 800.0  # J/(kg⋅K) (soil specific heat)
GROUND_EMISSIVITY = 0.95  # High emissivity (land/ocean surface)

# ============================================================
# ATMOSPHERIC LAYER 1 (Lower atmosphere)
# ============================================================
LAYER1_HEIGHT = 5000.0  # m (5 km)
LAYER1_DENSITY = 1.2  # kg/m³
LAYER1_HEAT_CAPACITY = 1000.0  # J/(kg⋅K) specific heat at constant pressure
LAYER1_EMISSIVITY = 0.85  # eps1 from validated 3layers.py model

# ============================================================
# ATMOSPHERIC LAYER 2 (Upper atmosphere)
# ============================================================
LAYER2_HEIGHT = 10000.0  # m (10 km, represents 5-15 km altitude)
LAYER2_DENSITY = 0.6  # kg/m³ (lower density at higher altitude)
LAYER2_HEAT_CAPACITY = 1000.0  # J/(kg⋅K)
LAYER2_EMISSIVITY = 0.75  # eps2 from validated 3layers.py model

# ============================================================
# LEGACY PARAMETERS (for backward compatibility)
# ============================================================
ATMOSPHERIC_LAYER_HEIGHT = LAYER1_HEIGHT  # m
AIR_DENSITY = LAYER1_DENSITY  # kg/m³
AIR_HEAT_CAPACITY = LAYER1_HEAT_CAPACITY  # J/(kg⋅K)
ATMOSPHERIC_EMISSIVITY = LAYER1_EMISSIVITY  # dimensionless (0-1)


# ============================================================
# THERMAL PARAMETERS
# ============================================================

# Initial global temperature for climate model (based on 3layers.py)
INITIAL_GLOBAL_TEMPERATURE = 288.0  # K (~15°C) - Earth's actual average

# Stefan-Boltzmann constant (W⋅m⁻²⋅K⁻⁴)
STEFAN_BOLTZMANN = 5.67e-8

# Specific heat capacity of Earth's surface (approximation)
# Weighted average of land and ocean
SURFACE_HEAT_CAPACITY = 2.0e6  # J/(m²⋅K)

# Thermal diffusivity (m²/s)
THERMAL_DIFFUSIVITY = 1.0e-6

# Atmospheric horizontal thermal diffusivity (m²/s)
# Controls lateral heat transport in atmosphere
ATMOSPHERIC_THERMAL_DIFFUSIVITY = 1.0e5  # m²/s (much higher than surface)


# ============================================================
# GEOGRAPHICAL PARAMETERS
# ============================================================

# Earth radius (meters)
EARTH_RADIUS = 6371000.0  # m (6371 km)

# Maximum elevation (Mount Everest)
MAX_ELEVATION = 8849.0  # m

# Minimum elevation (Dead Sea)
MIN_ELEVATION = -430.0  # m

# Ocean depth (not used for elevation but for reference)
AVERAGE_OCEAN_DEPTH = 3688.0  # m


# ============================================================
# TEMPORAL PARAMETERS
# ============================================================

# Simulation time step (seconds)
# Default: 1 hour
TIME_STEP = 3600.0  # seconds

# Earth rotation period (seconds)
EARTH_ROTATION_PERIOD = 86400.0  # 24 hours

# Earth angular velocity (rad/s)
# Real Earth: 7.29e-5 rad/s
# Increased for simulation to force Hadley cell breakup on coarse grids
EARTH_ANGULAR_VELOCITY = 7.29e-5  # rad/s

# Earth orbital period (seconds)
EARTH_ORBITAL_PERIOD = 31557600.0  # 365.25 days


# ============================================================
# NUMERICAL PARAMETERS
# ============================================================

# Convergence tolerance for iterative solvers
CONVERGENCE_TOLERANCE = 1.0e-6

# Maximum iterations for iterative solvers
MAX_ITERATIONS = 1000

# Relaxation factor for iterative methods (0 < omega <= 1)
RELAXATION_FACTOR = 0.5


# ============================================================
# VISUALIZATION PARAMETERS
# ============================================================

# Temperature range for color mapping (Kelvin)
TEMP_MIN = 233.0  # -40°C (cold regions)
TEMP_MAX = 313.0  # +40°C (hot regions)

# Temperature range in Celsius for reference
TEMP_MIN_CELSIUS = TEMP_MIN - 273.15  # -40°C
TEMP_MAX_CELSIUS = TEMP_MAX - 273.15  # +40°C


# ============================================================
# SIMULATION MODES
# ============================================================

class SimulationMode:
    """Available simulation modes."""
    CLIMATE_MODEL = 'climate_model'  # Physical climate simulation
    SOLAR_HEATING = 'solar_heating'
    ROTATING_HEAT = 'rotating_heat'
    WAVE_PATTERN = 'wave_pattern'
    DUAL_VORTEX = 'dual_vortex'
    PULSATING_HEAT = 'pulsating_heat'
    LATITUDE_GRADIENT = 'latitude_gradient'
    HEAT_DIFFUSION = 'heat_diffusion'


# Default simulation mode
DEFAULT_SIMULATION_MODE = SimulationMode.CLIMATE_MODEL


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def celsius_to_kelvin(temp_c):
    """Convert temperature from Celsius to Kelvin."""
    return temp_c + 273.15


def kelvin_to_celsius(temp_k):
    """Convert temperature from Kelvin to Celsius."""
    return temp_k - 273.15


def solar_radiation_at_latitude(latitude_degrees):
    """
    Calculate average solar radiation at a given latitude.
    
    Parameters
    ----------
    latitude_degrees : float
        Latitude in degrees (-90 to +90)
        
    Returns
    -------
    radiation : float
        Solar radiation in W/m²
    """
    import numpy as np
    
    # Convert to radians
    lat_rad = np.radians(latitude_degrees)
    
    # Simplified model: radiation decreases with cosine of latitude
    # Maximum at equator, minimum at poles
    radiation = AVERAGE_SOLAR_RADIATION * np.cos(lat_rad)
    
    return max(0.0, radiation)


 
 
def solar_radiation_at_latitude_average(lat_deg):
    """
    Irradiância solar média ANUAL (W/m²)
    sem atmosfera e sem albedo.
    
    Parameters
    ----------
    lat_deg : float or np.ndarray
        Latitude(s) in degrees. Can be scalar or array.
        
    Returns
    -------
    radiation : float or np.ndarray
        Solar radiation in W/m². Same shape as input.
    """
    import numpy as np
    lat = np.radians(lat_deg)
    return (SOLAR_CONSTANT / 4.0) * (1.0 + 0.5 * np.cos(lat)**2)




def print_parameters():
    """Print all simulation parameters."""
    print("\n" + "="*60)
    print("SIMULATION PARAMETERS")
    print("="*60)
    
    print("\nSolar Radiation:")
    print(f"  Solar constant:           {SOLAR_CONSTANT:.1f} W/m²")
    print(f"  Average solar radiation:  {AVERAGE_SOLAR_RADIATION:.1f} W/m²")
    print(f"  Earth albedo:             {EARTH_ALBEDO:.2f}")
    print(f"  Net absorbed radiation:   {NET_SOLAR_RADIATION:.1f} W/m²")
    
    print("\nThermal:")
    print(f"  Surface heat capacity:    {SURFACE_HEAT_CAPACITY:.2e} J/(m²⋅K)")
    print(f"  Thermal diffusivity:      {THERMAL_DIFFUSIVITY:.2e} m²/s")
    
    print("\nGeography:")
    print(f"  Earth radius:             {EARTH_RADIUS/1000:.1f} km")
    print(f"  Max elevation:            {MAX_ELEVATION:.1f} m")
    
    print("\nTemporal:")
    print(f"  Time step:                {TIME_STEP/3600:.1f} hours")
    print(f"  Earth rotation period:    {EARTH_ROTATION_PERIOD/3600:.1f} hours")
    
    print("\nVisualization:")
    print(f"  Temperature range:        {TEMP_MIN_CELSIUS:.1f}°C to {TEMP_MAX_CELSIUS:.1f}°C")
    print(f"  Default mode:             {DEFAULT_SIMULATION_MODE}")
    
    print("="*60)


if __name__ == "__main__":
    # Test the parameters
    print_parameters()
    
    # Test solar radiation at different latitudes
    print("\nSolar Radiation by Latitude:")
    print("-" * 40)
    for lat in [0, 30, 45, 60, 90]:
        rad = solar_radiation_at_latitude(lat)
        print(f"  {lat:3d}°: {rad:6.1f} W/m²")
