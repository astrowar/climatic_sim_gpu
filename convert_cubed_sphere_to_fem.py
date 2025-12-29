"""Shared utility: convert cubed-sphere grid into FEM global mesh.

This module exposes `convert_cubed_sphere_to_fem(grid_points)` which
is used by both the Python and native solver adapters.
"""

import numpy as np
from typing import Dict, Tuple


def convert_cubed_sphere_to_fem(grid_points: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convert cubed-sphere grid (6 faces) to unified FEM structure.

    Merges shared nodes at face boundaries to create a single global mesh.
    Uses spatial hashing for efficient duplicate detection.
    """
    n_faces = 6

    # Get dimensions from first face
    face_0 = grid_points['face_0']
    rows, cols = face_0['x'].shape

    # Get radius from first point
    x0, y0, z0 = face_0['x'][0, 0], face_0['y'][0, 0], face_0['z'][0, 0]
    radius = np.sqrt(x0**2 + y0**2 + z0**2)

    # Tolerance for considering two nodes as the same (shared at boundaries)
    # Estimate mean element size from grid spacing on unit sphere
    avg_spacing = 2.0 * radius / (rows - 1)
    element_mean_size = avg_spacing
    # Use 1/10 of the element mean size as fusion tolerance
    tol = element_mean_size * 0.1

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

    # Quick duplicate check (vectorized): compute pairwise squared distances
    # and count pairs within tolerance using the upper-triangle only.
    if len(global_nodes) > 1:
        coords = np.asarray(global_nodes, dtype=np.float64)
        # pairwise squared distances (N x N) via broadcasting
        diffs = coords[:, None, :] - coords[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diffs, diffs)
        iu = np.triu_indices(d2.shape[0], k=1)
        duplicate_count = int(np.count_nonzero(d2[iu] < (tol * tol)))
    else:
        duplicate_count = 0

    if duplicate_count > 0:
        print(f"  WARNING: Found {duplicate_count} duplicate nodes within tolerance!")
    else:
        print(f"  ✓ No duplicate nodes found - mesh is properly connected")

    print(f"  ✓ FEM mesh created: {len(global_nodes)} nodes, {len(connectivity)} elements")

    return global_nodes, connectivity, node_mapping
