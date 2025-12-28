"""
Cubed-Sphere Mesh Generator using Gnomonic Projection

This module provides functions to generate spherical grids using the
cubed-sphere approach with gnomonic projection.
"""

import numpy as np
from typing import Dict


def cartesian_to_latlon(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to latitude/longitude.
    
    In OpenGL convention:
    - Y axis points up (north pole = +Y, south pole = -Y)
    - X and Z define the equatorial plane
    - Longitude 0° is along +X axis
    
    Parameters
    ----------
    x, y, z : float or ndarray
        Cartesian coordinates on unit sphere (OpenGL convention)
        
    Returns
    -------
    lat, lon : float or ndarray
        Latitude and longitude in degrees
    """
    # Latitude: angle from equatorial plane (Y is the polar axis)
    lat = np.degrees(np.arcsin(np.clip(y, -1.0, 1.0)))
    
    # Longitude: angle in xz plane (X=0°, Z=90°)
    # Negate to match standard East/West convention
    lon = -np.degrees(np.arctan2(z, x))
    
    return lat, lon


def add_elevation_data(grid_points: Dict, elevation_reader) -> Dict:
    """
    Add elevation data to grid points.
    
    Parameters
    ----------
    grid_points : dict
        Dictionary containing grid data for all six faces
    elevation_reader : ElevationReader
        Elevation reader instance
        
    Returns
    -------
    grid_points : dict
        Dictionary with added 'elevation' field for each face
    """
    print("Adding elevation data to grid...")
    
    for face_id in range(6):
        face_key = f'face_{face_id}'
        face_data = grid_points[face_key]
        
        xs = face_data['x']
        ys = face_data['y']
        zs = face_data['z']
        
        rows, cols = xs.shape
        elevations = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                x, y, z = xs[i, j], ys[i, j], zs[i, j]
                
                # Convert to lat/lon
                lat, lon = cartesian_to_latlon(x, y, z)
                
                # Get elevation (None if water)
                elevation = elevation_reader.get_elevation_at_point(lat, lon)
                
                # Store elevation (use -1 for water)
                elevations[i, j] = elevation if elevation is not None else -1.0
        
        grid_points[face_key]['elevation'] = elevations
    
    print("Elevation data added to grid")
    return grid_points


def normalize_elevation_data(grid_points: Dict) -> Dict:
    """
    Extract and normalize elevation data from grid for FEM simulation.
    
    Parameters
    ----------
    grid_points : dict
        Dictionary containing grid data with 'elevation' field
        
    Returns
    -------
    elevation_data : dict or None
        Normalized elevation data [0, 1] for each face, or None if not available
    """
    if not grid_points or 'face_0' not in grid_points:
        return None
    
    if 'elevation' not in grid_points['face_0']:
        return None
    
    elevation_data = {}
    
    for face_id in range(6):
        face_key = f'face_{face_id}'
        elevation = grid_points[face_key]['elevation']
        
        # Normalize elevation to [0, 1]
        # Water (elevation < 0) -> 0
        # Land (0 to 8000m) -> 0 to 1
        normalized = np.clip(elevation / 8000.0, 0.0, 1.0)
        
        elevation_data[face_key] = normalized
    
    return elevation_data


def generate_cubed_sphere_grid(n_points: int = 15, radius: float = 1.0, a: float = 1.0) -> Dict:
    """
    Generate cubed-sphere grid using gnomonic projection.
    
    Parameters
    ----------
    n_points : int
        Number of grid points per face dimension
    radius : float
        Sphere radius
    a : float
        Half-width of cube face
    
    Returns
    -------
    grid_points : dict
        Dictionary containing grid data for all six faces
    """
    coords = np.linspace(-a, a, n_points)
    x_local, y_local = np.meshgrid(coords, coords)
    
    grid_points = {}
    
    # Define transformations for each face
    face_transforms = [
        # +X face (right)
        lambda x, y: (np.ones_like(x) * a, x, y),
        # -X face (left)
        lambda x, y: (-np.ones_like(x) * a, -x, y),
        # +Y face (front)
        lambda x, y: (-x, np.ones_like(x) * a, y),
        # -Y face (back)
        lambda x, y: (x, -np.ones_like(x) * a, y),
        # +Z face (top)
        lambda x, y: (x, y, np.ones_like(x) * a),
        # -Z face (bottom)
        lambda x, y: (x, -y, -np.ones_like(x) * a),
    ]
    
    for face_id, transform in enumerate(face_transforms):
        # Get cube coordinates
        xc, yc, zc = transform(x_local, y_local)
        
        # Project onto sphere (gnomonic projection)
        r = np.sqrt(xc**2 + yc**2 + zc**2)
        xs = radius * xc / r
        ys = radius * yc / r
        zs = radius * zc / r
        
        grid_points[f'face_{face_id}'] = {
            'x': xs,
            'y': ys,
            'z': zs
        }
    
    return grid_points


def mesh_relaxation(grid_points: Dict, iterations: int = 50, omega: float = 0.5, 
                   radius: float = 1.0, verbose: bool = True) -> Dict:
    """
    Apply Laplacian smoothing to improve mesh quality while keeping boundary nodes fixed.
    
    This function relaxes the interior nodes of each face to make elements more regular
    and "square-like", while keeping edge/corner nodes fixed to maintain continuity
    between faces.
    
    Parameters
    ----------
    grid_points : dict
        Dictionary containing grid data for all six faces
    iterations : int
        Number of relaxation iterations
    omega : float
        Relaxation factor (0 < omega <= 1). Lower values = slower but more stable
    radius : float
        Sphere radius for reprojection
    verbose : bool
        If True, print progress messages
    
    Returns
    -------
    grid_points : dict
        Relaxed grid data
    """
    if verbose:
        print(f"\nApplying mesh relaxation ({iterations} iterations, omega={omega})...")
    
    for face_id in range(6):
        face_data = grid_points[f'face_{face_id}']
        xs = face_data['x'].copy()
        ys = face_data['y'].copy()
        zs = face_data['z'].copy()
        
        rows, cols = xs.shape
        
        for iteration in range(iterations):
            xs_new = xs.copy()
            ys_new = ys.copy()
            zs_new = zs.copy()
            
            # Relax interior nodes only (keep boundary fixed)
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    # Compute average position of neighbors (4-connectivity)
                    x_avg = (xs[i-1, j] + xs[i+1, j] + xs[i, j-1] + xs[i, j+1]) / 4.0
                    y_avg = (ys[i-1, j] + ys[i+1, j] + ys[i, j-1] + ys[i, j+1]) / 4.0
                    z_avg = (zs[i-1, j] + zs[i+1, j] + zs[i, j-1] + zs[i, j+1]) / 4.0
                    
                    # Apply relaxation with omega factor
                    x_relaxed = xs[i, j] + omega * (x_avg - xs[i, j])
                    y_relaxed = ys[i, j] + omega * (y_avg - ys[i, j])
                    z_relaxed = zs[i, j] + omega * (z_avg - zs[i, j])
                    
                    # Project back onto sphere
                    r = np.sqrt(x_relaxed**2 + y_relaxed**2 + z_relaxed**2)
                    xs_new[i, j] = radius * x_relaxed / r
                    ys_new[i, j] = radius * y_relaxed / r
                    zs_new[i, j] = radius * z_relaxed / r
            
            xs = xs_new
            ys = ys_new
            zs = zs_new
            
            # Print progress every 10 iterations
            if verbose and (iteration + 1) % 10 == 0:
                print(f"  Face {face_id}: iteration {iteration + 1}/{iterations}")
        
        # Update grid data
        grid_points[f'face_{face_id}']['x'] = xs
        grid_points[f'face_{face_id}']['y'] = ys
        grid_points[f'face_{face_id}']['z'] = zs
    
    if verbose:
        print("Mesh relaxation complete!\n")
    
    return grid_points
