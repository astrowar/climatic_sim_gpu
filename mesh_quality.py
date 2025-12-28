"""
Mesh Quality Analysis

This module provides functions to compute mesh quality metrics
for cubed-sphere grids.
"""

import numpy as np
from typing import Dict


def compute_mesh_quality(grid_points: Dict) -> Dict:
    """
    Compute mesh quality metrics (aspect ratio, skewness, etc.)
    
    Parameters
    ----------
    grid_points : dict
        Dictionary containing grid data
    
    Returns
    -------
    metrics : dict
        Dictionary with quality metrics
    """
    aspect_ratios = []
    
    for face_id in range(6):
        face_data = grid_points[f'face_{face_id}']
        xs = face_data['x']
        ys = face_data['y']
        zs = face_data['z']
        
        rows, cols = xs.shape
        
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Get quad vertices
                p0 = np.array([xs[i, j], ys[i, j], zs[i, j]])
                p1 = np.array([xs[i, j+1], ys[i, j+1], zs[i, j+1]])
                p2 = np.array([xs[i+1, j+1], ys[i+1, j+1], zs[i+1, j+1]])
                p3 = np.array([xs[i+1, j], ys[i+1, j], zs[i+1, j]])
                
                # Compute edge lengths
                edge1 = np.linalg.norm(p1 - p0)
                edge2 = np.linalg.norm(p2 - p1)
                edge3 = np.linalg.norm(p3 - p2)
                edge4 = np.linalg.norm(p0 - p3)
                
                # Aspect ratio: max_edge / min_edge
                edges = [edge1, edge2, edge3, edge4]
                aspect_ratio = max(edges) / (min(edges) + 1e-10)
                aspect_ratios.append(aspect_ratio)
    
    metrics = {
        'mean_aspect_ratio': np.mean(aspect_ratios),
        'max_aspect_ratio': np.max(aspect_ratios),
        'min_aspect_ratio': np.min(aspect_ratios),
        'std_aspect_ratio': np.std(aspect_ratios)
    }
    
    return metrics


def print_mesh_statistics(grid_points: Dict, label: str = "Mesh"):
    """
    Print mesh statistics.
    
    Parameters
    ----------
    grid_points : dict
        Dictionary containing grid data
    label : str
        Label for the statistics output
    """
    # Count nodes and elements
    total_nodes = 0
    total_elements = 0
    
    for face_id in range(6):
        face_data = grid_points[f'face_{face_id}']
        rows, cols = face_data['x'].shape
        total_nodes += rows * cols
        total_elements += (rows - 1) * (cols - 1)
    
    print(f"\n{label} Statistics:")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Total elements: {total_elements}")
    
    # Compute quality metrics
    metrics = compute_mesh_quality(grid_points)
    print(f"\nQuality Metrics:")
    print(f"  Mean aspect ratio: {metrics['mean_aspect_ratio']:.4f}")
    print(f"  Max aspect ratio:  {metrics['max_aspect_ratio']:.4f}")
    print(f"  Min aspect ratio:  {metrics['min_aspect_ratio']:.4f}")
    print(f"  Std aspect ratio:  {metrics['std_aspect_ratio']:.4f}")
