#!/usr/bin/env python3
"""
Predict connectivity matrix from three orthogonal views using trained model.
Takes a seed, generates solid and connectivity matrices, predicts output, and visualizes.
"""

import numpy as np
import matplotlib
# Only use non-interactive backend if explicitly saving without display
# matplotlib.use('Agg')  # Non-interactive backend for solid generation
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import subprocess
import os

# Add keras import
try:
    import keras
    import tensorflow as tf
except ImportError:
    print("Error: Keras not found. Please install with: pip install keras")
    sys.exit(1)


# Custom functions for model loading (in case model was saved with custom layers)
def copy_coordinates(inputs):
    """
    Copy first 4 columns (coordinates) from input to output, replacing predicted values.
    This ensures coordinates pass through the model unaltered.
    """
    predicted, top_view = inputs
    # Extract first 4 columns from top view: [:, :, :4, :]
    coords = top_view[:, :, :4, :]  # (batch, height, 4, 1)
    # Extract remaining columns from predicted: [:, :, 4:, :]
    connectivity = predicted[:, :, 4:, :]  # (batch, height, remaining_cols, 1)
    # Concatenate: [coords, connectivity]
    return tf.concat([coords, connectivity], axis=2)


def load_model(model_path):
    """Load the trained connectivity prediction model."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    # Load model without compiling, but provide custom objects for Lambda layers
    custom_objects = {
        'copy_coordinates': copy_coordinates,
    }
    try:
        model = keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    except Exception as e:
        # If loading with custom objects fails, try without (for old models)
        print(f"Note: Loading without custom objects (old model format)")
        model = keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully (inference mode)")
    return model


def reconstruct_vertices_from_views(top_matrix, front_matrix, side_matrix, tolerance=1e-5):
    """
    Reconstruct 3D vertices from three view matrices (reverse engineering).
    
    Algorithm:
    1. Extract all (x,y) from top view - columns 1,2
    2. Extract all (x,z) from front view - columns 1,2 
    3. Combine by matching x coordinates
    4. Filter by side view
    
    Returns:
        reconstructed_vertices: Array of 3D vertices (M, 3)
    """
    print("\n=== RECONSTRUCTING 3D VERTICES FROM VIEWS ===")
    
    # Extract 2D coordinates from each view (columns 1, 2)
    top_xy = top_matrix[:, 1:3]  # (x, y)
    front_xz = front_matrix[:, 1:3]  # (x, z)
    side_yz = side_matrix[:, 1:3]  # (y, z)
    
    print(f"  Top view: {len(top_xy)} vertices (x,y)")
    print(f"  Front view: {len(front_xz)} vertices (x,z)")
    print(f"  Side view: {len(side_yz)} vertices (y,z)")
    
    # Combine top (x,y) with front (x,z) by matching x coordinates
    reconstructed_vertices = []
    for x_top, y_top in top_xy:
        for x_front, z_front in front_xz:
            if abs(x_top - x_front) < tolerance:
                vertex = [x_top, y_top, z_front]
                reconstructed_vertices.append(vertex)
    
    # Remove duplicates
    unique_vertices = []
    for v in reconstructed_vertices:
        is_duplicate = False
        for existing in unique_vertices:
            if (abs(v[0] - existing[0]) < tolerance and 
                abs(v[1] - existing[1]) < tolerance and 
                abs(v[2] - existing[2]) < tolerance):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_vertices.append(v)
    
    # Filter by side view projections
    valid_vertices = []
    for vertex in unique_vertices:
        y, z = vertex[1], vertex[2]
        for y_side, z_side in side_yz:
            if abs(y - y_side) < tolerance and abs(z - z_side) < tolerance:
                valid_vertices.append(vertex)
                break
    
    reconstructed_vertices = np.array(valid_vertices)
    print(f"  Reconstructed: {len(reconstructed_vertices)} unique 3D vertices")
    
    return reconstructed_vertices


def build_view_to_reconstructed_mapping(reconstructed_vertices, view_matrix, 
                                       view_type, tolerance=1e-5):
    """
    Build mapping from view rows to reconstructed vertex indices.
    
    Args:
        reconstructed_vertices: (M, 3) array of 3D vertices
        view_matrix: (N, N+4) view connectivity matrix
        view_type: 'top', 'front', or 'side'
        
    Returns:
        Dict mapping view_row → list of reconstructed vertex indices
    """
    # Determine projection axes for this view
    if view_type == 'top':
        projection_axes = [0, 1]  # x, y
    elif view_type == 'front':
        projection_axes = [0, 2]  # x, z
    else:  # side
        projection_axes = [1, 2]  # y, z
    
    view_to_recon = {}
    
    for view_row in range(view_matrix.shape[0]):
        # Get 2D projection from view (columns 1, 2)
        view_proj = view_matrix[view_row, 1:3]
        
        # Find all reconstructed vertices that project to this position
        matching_recon_verts = []
        for recon_idx, vertex_3d in enumerate(reconstructed_vertices):
            recon_proj = np.array([vertex_3d[projection_axes[0]], 
                                  vertex_3d[projection_axes[1]]])
            
            if np.allclose(view_proj, recon_proj, atol=tolerance):
                matching_recon_verts.append(recon_idx)
        
        view_to_recon[view_row] = matching_recon_verts
    
    return view_to_recon


def expand_view_with_reconstruction(view_matrix, view_to_recon_mapping, 
                                   n_reconstructed, view_name='view', 
                                   reconstructed_vertices=None):
    """
    Expand sparse view connectivity to full reconstructed vertex space.
    
    Args:
        view_matrix: Sparse connectivity matrix (N, N+4)
        view_to_recon_mapping: Dict mapping view_row → list of reconstructed vertex indices
        n_reconstructed: Total number of reconstructed vertices
        view_name: Name of view for debug output
        reconstructed_vertices: (M, 3) array of reconstructed 3D vertices
    
    Returns:
        Full connectivity matrix (M, M+4) for reconstructed vertices
    """
    n_sparse = view_matrix.shape[0]
    
    # Create full matrix structure
    full_matrix = np.zeros((n_reconstructed, n_reconstructed + 4), dtype=np.float32)
    
    # Fill columns 0-3 with reconstructed vertex information
    if reconstructed_vertices is not None:
        for i in range(n_reconstructed):
            full_matrix[i, 0] = i  # Vertex index
            full_matrix[i, 1] = reconstructed_vertices[i, 0]  # x
            full_matrix[i, 2] = reconstructed_vertices[i, 1]  # y
            full_matrix[i, 3] = reconstructed_vertices[i, 2]  # z
    
    # Propagate edges using the mapping
    # SPARSE view matrix has structure: [idx, x, y, conn0, conn1, ...]
    # So connectivity to vertex j is at column 3+j (not 4+j!)
    for view_i in range(n_sparse):
        for view_j in range(n_sparse):
            conn_col = 3 + view_j  # Sparse views use column 3+j
            
            if conn_col < view_matrix.shape[1]:
                edge_value = view_matrix[view_i, conn_col]
                
                if edge_value > 0:
                    # Add edges between ALL reconstructed vertices in both groups
                    recon_verts_i = view_to_recon_mapping.get(view_i, [])
                    recon_verts_j = view_to_recon_mapping.get(view_j, [])
                    
                    for vi in recon_verts_i:
                        for vj in recon_verts_j:
                            # EXPANDED matrix has structure: [idx, x, y, z, conn0, conn1, ...]
                            # So connectivity to vertex vj is at column 4+vj
                            conn_idx = 4 + vj
                            if conn_idx < full_matrix.shape[1]:
                                if full_matrix[vi, conn_idx] == 0:
                                    full_matrix[vi, conn_idx] = edge_value
                                    full_matrix[vj, 4 + vi] = edge_value
    
    return full_matrix


def expand_views_to_reconstructed(top, front, side):
    """
    Expand sparse view matrices to include all reconstructed vertices.
    This matches the training data preparation process.
    
    Args:
        top, front, side: Sparse view matrices (N, N+4) or (N, N+4, 11)
        
    Returns:
        Expanded matrices (M, M+4) where M is number of reconstructed vertices
    """
    print("\n=== EXPANDING VIEWS TO RECONSTRUCTED VERTICES ===")
    
    # Convert 3D to 2D if needed (extract layer 0)
    if len(top.shape) == 3:
        top_2d = top[:, :, 0]
        front_2d = front[:, :, 0]
        side_2d = side[:, :, 0]
    else:
        top_2d = top
        front_2d = front
        side_2d = side
    
    # Step 1: Reconstruct 3D vertices
    reconstructed_vertices = reconstruct_vertices_from_views(top_2d, front_2d, side_2d)
    n_reconstructed = len(reconstructed_vertices)
    
    # Step 2: Build mappings from view rows to reconstructed vertices
    print("\n  Building view-to-reconstructed mappings...")
    top_mapping = build_view_to_reconstructed_mapping(reconstructed_vertices, top_2d, 'top')
    front_mapping = build_view_to_reconstructed_mapping(reconstructed_vertices, front_2d, 'front')
    side_mapping = build_view_to_reconstructed_mapping(reconstructed_vertices, side_2d, 'side')
    
    # Step 3: Expand each view to full reconstructed space
    print(f"\n  Expanding views to {n_reconstructed} reconstructed vertices...")
    top_expanded = expand_view_with_reconstruction(
        top_2d, top_mapping, n_reconstructed, 'top', reconstructed_vertices
    )
    front_expanded = expand_view_with_reconstruction(
        front_2d, front_mapping, n_reconstructed, 'front', reconstructed_vertices
    )
    side_expanded = expand_view_with_reconstruction(
        side_2d, side_mapping, n_reconstructed, 'side', reconstructed_vertices
    )
    
    print(f"  Expanded shapes:")
    print(f"    Top: {top_2d.shape} → {top_expanded.shape}")
    print(f"    Front: {front_2d.shape} → {front_expanded.shape}")
    print(f"    Side: {side_2d.shape} → {side_expanded.shape}")
    
    return top_expanded, front_expanded, side_expanded, reconstructed_vertices


def generate_solid_and_matrices(seed, output_dir="Output"):
    """
    Generate solid and extract connectivity matrices using Build_Solid.py.
    
    Args:
        seed: Random seed for solid generation
        output_dir: Directory where Build_Solid.py saves output files
        
    Returns:
        tuple: (top, front, side) connectivity matrices, each shape (N, N+4, 11)
    """
    print(f"\nGenerating solid for seed {seed}...")
    
    # Run Build_Solid.py to generate the solid and matrices
    cmd = [
        sys.executable,
        "Build_Solid.py",
        "--seed", str(seed),
        "--no-graphics",
        "--no-lettering",
        "--quiet",
        "--output-dir", output_dir
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print("Solid generation complete")
    except subprocess.CalledProcessError as e:
        print(f"Error running Build_Solid.py: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    
    # Load the generated connectivity matrices
    matrix_file = Path(output_dir) / f"connectivity_matrices_seed_{seed}.npz"
    
    if not matrix_file.exists():
        raise FileNotFoundError(
            f"Connectivity matrices not generated: {matrix_file}\n"
            f"Build_Solid.py may have failed."
        )
    
    print(f"Loading connectivity matrices from: {matrix_file}")
    data = np.load(matrix_file, allow_pickle=True)
    
    top = data['top_view_matrix']
    front = data['front_view_matrix']
    side = data['side_view_matrix']
    
    print(f"  Top:   {top.shape}")
    print(f"  Front: {front.shape}")
    print(f"  Side:  {side.shape}")
    
    return top, front, side


def load_connectivity_matrices(seed, data_dir="."):
    """
    Load the three orthogonal view connectivity matrices for a given seed.
    
    First tries to load from .npy files, then from .npz file.
    
    Returns:
        tuple: (top, front, side) connectivity matrices, each shape (N, N+4, 11)
    """
    data_path = Path(data_dir)
    
    # Try individual .npy files first
    top_file = data_path / f"seed_{seed}_top_connectivity.npy"
    front_file = data_path / f"seed_{seed}_front_connectivity.npy"
    side_file = data_path / f"seed_{seed}_side_connectivity.npy"
    
    if top_file.exists() and front_file.exists() and side_file.exists():
        print(f"\nLoading pre-saved connectivity matrices for seed {seed}...")
        top = np.load(top_file)
        front = np.load(front_file)
        side = np.load(side_file)
        
        print(f"  Top:   {top.shape}")
        print(f"  Front: {front.shape}")
        print(f"  Side:  {side.shape}")
        
        return top, front, side
    
    # Try .npz file from Build_Solid.py output
    npz_file = data_path / f"connectivity_matrices_seed_{seed}.npz"
    
    if npz_file.exists():
        print(f"\nLoading connectivity matrices from: {npz_file}")
        data = np.load(npz_file, allow_pickle=True)
        
        top = data['top_view_matrix']
        front = data['front_view_matrix']
        side = data['side_view_matrix']
        
        print(f"  Top:   {top.shape}")
        print(f"  Front: {front.shape}")
        print(f"  Side:  {side.shape}")
        
        return top, front, side
    
    # Files don't exist - need to generate them
    raise FileNotFoundError(
        f"Connectivity matrices not found for seed {seed}.\n"
        f"Will attempt to generate them using Build_Solid.py."
    )


def pad_to_size(matrix, target_size=200):
    """
    Pad connectivity matrix to target size (200, 204, 11).
    
    Args:
        matrix: Original matrix of shape (N, N+4) or (N, N+4, 11)
        target_size: Target first dimension (default 200)
    
    Returns:
        Padded matrix of shape (target_size, target_size+4, 11)
    """
    n = matrix.shape[0]
    
    # Check if matrix is 2D or 3D
    if len(matrix.shape) == 2:
        # 2D matrix (N, N+4) - expand to 3D by adding connectivity layer
        # Create 3D matrix with 11 layers, copy data to layer 0 (connectivity)
        matrix_3d = np.zeros((n, matrix.shape[1], 11), dtype=matrix.dtype)
        matrix_3d[:, :, 0] = matrix  # Put connectivity in layer 0
        matrix = matrix_3d
    
    if n >= target_size:
        return matrix[:target_size, :target_size+4, :]
    
    # Create padded matrix
    padded = np.zeros((target_size, target_size + 4, 11), dtype=matrix.dtype)
    padded[:n, :matrix.shape[1], :] = matrix
    
    return padded


def plot_padded_input_matrices(top_padded, front_padded, side_padded, seed):
    """
    Plot the PADDED input matrices (200, 204, 11) as they are fed to the model.
    Uses same logic as prepare_training_data.py for consistency.
    
    Args:
        top_padded: (200, 204, 11) padded top view
        front_padded: (200, 204, 11) padded front view
        side_padded: (200, 204, 11) padded side view
        seed: Seed number for title
    """
    import matplotlib.pyplot as plt
    
    # Extract connectivity layer (layer 0)
    top_matrix = top_padded[:, :, 0]  # (200, 204)
    front_matrix = front_padded[:, :, 0]  # (200, 204)
    side_matrix = side_padded[:, :, 0]  # (200, 204)
    
    # After expansion, all matrices have [idx, x, y, z, conn...]
    # We need to extract the correct 2D projection for each view
    
    # Determine actual number of vertices (non-zero rows)
    n_top = np.count_nonzero(np.any(top_matrix != 0, axis=1))
    n_front = np.count_nonzero(np.any(front_matrix != 0, axis=1))
    n_side = np.count_nonzero(np.any(side_matrix != 0, axis=1))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Padded Input Matrices (Seed {seed}) - Size (200, 204, 11)', fontsize=14, fontweight='bold')
    
    # Top view - project to x,y plane (columns 1, 2)
    ax = axes[0]
    ax.set_title('Top View (Padded)', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot vertices (only non-zero rows)
    for i in range(n_top):
        x, y = top_matrix[i, 1], top_matrix[i, 2]
        if x != 0 or y != 0:  # Skip padding zeros
            ax.plot(x, y, 'ko', markersize=8)
            ax.text(x, y, f'{i}', fontsize=8, ha='right', va='bottom', color='blue')
    
    # Plot edges using column 4+j for connectivity
    edge_count = 0
    for i in range(n_top):
        x_i, y_i = top_matrix[i, 1], top_matrix[i, 2]
        if x_i == 0 and y_i == 0:
            continue
        for j in range(i+1, n_top):
            col = 4 + j
            if col < top_matrix.shape[1] and top_matrix[i, col] > 0:
                x_j, y_j = top_matrix[j, 1], top_matrix[j, 2]
                if x_j != 0 or y_j != 0:
                    ax.plot([x_i, x_j], [y_i, y_j], 'r-', linewidth=1, alpha=0.6)
                    edge_count += 1
    
    ax.set_title(f'Top View (Padded) - {edge_count} edges', fontsize=12)
    
    # Front view - project to x,z plane (columns 1, 3)
    ax = axes[1]
    ax.set_title('Front View (Padded)', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot vertices - use columns 1 (x) and 3 (z)
    for i in range(n_front):
        x, z = front_matrix[i, 1], front_matrix[i, 3]
        if x != 0 or z != 0:
            ax.plot(x, z, 'ko', markersize=8)
            ax.text(x, z, f'{i}', fontsize=8, ha='right', va='bottom', color='blue')
    
    # Plot edges
    edge_count = 0
    for i in range(n_front):
        x_i, z_i = front_matrix[i, 1], front_matrix[i, 3]
        if x_i == 0 and z_i == 0:
            continue
        for j in range(i+1, n_front):
            col = 4 + j
            if col < front_matrix.shape[1] and front_matrix[i, col] > 0:
                x_j, z_j = front_matrix[j, 1], front_matrix[j, 3]
                if x_j != 0 or z_j != 0:
                    ax.plot([x_i, x_j], [z_i, z_j], 'g-', linewidth=1, alpha=0.6)
                    edge_count += 1
    
    ax.set_title(f'Front View (Padded) - {edge_count} edges', fontsize=12)
    
    # Side view - project to y,z plane (columns 2, 3)
    ax = axes[2]
    ax.set_title('Side View (Padded)', fontsize=12)
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot vertices - use columns 2 (y) and 3 (z)
    for i in range(n_side):
        y, z = side_matrix[i, 2], side_matrix[i, 3]
        if y != 0 or z != 0:
            ax.plot(y, z, 'ko', markersize=8)
            ax.text(y, z, f'{i}', fontsize=8, ha='right', va='bottom', color='blue')
    
    # Plot edges
    edge_count = 0
    for i in range(n_side):
        y_i, z_i = side_matrix[i, 2], side_matrix[i, 3]
        if y_i == 0 and z_i == 0:
            continue
        for j in range(i+1, n_side):
            col = 4 + j
            if col < side_matrix.shape[1] and side_matrix[i, col] > 0:
                y_j, z_j = side_matrix[j, 2], side_matrix[j, 3]
                if y_j != 0 or z_j != 0:
                    ax.plot([y_i, y_j], [z_i, z_j], 'b-', linewidth=1, alpha=0.6)
                    edge_count += 1
    
    ax.set_title(f'Side View (Padded) - {edge_count} edges', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def predict_connectivity(model, top, front, side):
    """
    Use model to predict output connectivity matrix from three views.
    
    Args:
        model: Trained Keras model
        top, front, side: Connectivity matrices, shape (N, N+4) or (N, N+4, 11)
    
    Returns:
        Predicted connectivity matrix, shape (N, N+4, 1)
    """
    # Pad matrices to model input size
    top_padded = pad_to_size(top)
    front_padded = pad_to_size(front)
    side_padded = pad_to_size(side)
    
    # Extract only layer 0 (connectivity layer) and add channel dimension
    # Model expects (batch, height, width, 1) - single channel input
    X1 = top_padded[:, :, 0:1]  # (200, 204, 1) - only layer 0
    X2 = front_padded[:, :, 0:1]
    X3 = side_padded[:, :, 0:1]
    
    # Add batch dimension
    X1 = X1[np.newaxis, :, :, :]  # (1, 200, 204, 1)
    X2 = X2[np.newaxis, :, :, :]
    X3 = X3[np.newaxis, :, :, :]
    
    print("\nRunning model prediction...")
    print(f"  Input shapes: {X1.shape}, {X2.shape}, {X3.shape}")
    
    # Predict
    prediction = model.predict([X1, X2, X3], verbose=0)
    
    # Remove batch dimension and channel dimension
    predicted_matrix = prediction[0]  # (200, 204, 1)
    
    # Get original size from input
    original_size = top.shape[0]
    predicted_matrix = predicted_matrix[:original_size, :original_size+4, :]
    
    print(f"  Output shape: {predicted_matrix.shape}")
    print(f"  Raw prediction range: [{predicted_matrix.min():.6f}, {predicted_matrix.max():.6f}]")
    print(f"  Raw prediction mean: {predicted_matrix.mean():.6f}")
    
    return predicted_matrix


def extract_edges_from_top_view(top_matrix, threshold=1.5):
    """
    Extract edges and vertex coordinates from TOP VIEW matrix.
    After expansion, structure is: [idx, x, y, z, conn0, conn1, ...]
    
    Args:
        top_matrix: Shape (N, N+4) or (N, N+4, 11) - top view connectivity matrix
                   Column 0: vertex index
                   Columns 1-3: x, y, z coordinates
                   Columns 4+j: connectivity to vertex j
        threshold: Minimum connectivity value to consider an edge
    
    Returns:
        tuple: (edges, vertices_2d)
            edges: List of (i, j, weight) tuples
            vertices_2d: Array of shape (N, 2) with x, y coordinates
    """
    print(f"\nextract_edges_from_top_view: Input shape = {top_matrix.shape}")
    
    # Handle 3D matrices (N, N+4, 11) - extract connectivity layer 0
    if len(top_matrix.shape) == 3:
        print(f"  Converting 3D matrix to 2D (using layer 0)")
        top_matrix = top_matrix[:, :, 0]
    
    # Determine actual number of vertices (non-zero rows) - same logic as plot function
    n_actual = np.count_nonzero(np.any(top_matrix != 0, axis=1))
    print(f"  Matrix shape: {top_matrix.shape}, Actual vertices: {n_actual}")
    
    edges = []
    
    # Extract x, y coordinates for top view projection (columns 1, 2)
    vertices_2d = top_matrix[:, 1:3]  # (N, 2) - x, y only
    print(f"  Vertices shape: {vertices_2d.shape}")
    if n_actual > 0:
        print(f"  Vertex range: X[{vertices_2d[:n_actual, 0].min():.3f}, {vertices_2d[:n_actual, 0].max():.3f}], Y[{vertices_2d[:n_actual, 1].min():.3f}, {vertices_2d[:n_actual, 1].max():.3f}]")
    
    # Find edges where connectivity > threshold
    # After expansion: connectivity to vertex j is at column 4+j
    for i in range(n_actual):
        x_i, y_i = top_matrix[i, 1], top_matrix[i, 2]
        if x_i == 0 and y_i == 0:  # Skip if vertex is at origin (unlikely but check)
            continue
        for j in range(i + 1, n_actual):  # Upper triangle, only real vertices
            col = 4 + j  # After expansion: [idx, x, y, z, conn0, conn1, ...]
            if col < top_matrix.shape[1]:
                conn_val = top_matrix[i, col]
                if conn_val > threshold:
                    edges.append((i, j, conn_val))
    
    print(f"  Found {len(edges)} edges with connectivity > {threshold}")
    return edges, vertices_2d


def extract_edges_and_vertices(connectivity_matrix, threshold=1.5):
    """
    Extract edges and vertex coordinates from FULL 3D connectivity matrix.
    For original matrices, connectivity is in the UPPER triangle.
    
    Args:
        connectivity_matrix: Shape (N, N+4, 1) or (N, N+4) or (N, N+4, 11)
                           Columns 0-3: [index, x, y, z]
                           Columns 4+: connectivity values
        threshold: Minimum connectivity value to consider an edge
    
    Returns:
        tuple: (edges, vertices, indices)
            edges: List of (i, j, weight) tuples
            vertices: Array of shape (N, 3) with x, y, z coordinates
            indices: Array of vertex indices from column 0
    """
    print(f"\n{'='*80}")
    print("EXTRACT_EDGES_AND_VERTICES DEBUG")
    print(f"{'='*80}")
    print(f"Input shape: {connectivity_matrix.shape}")
    print(f"Threshold: {threshold}")
    
    n = connectivity_matrix.shape[0]
    edges = []
    
    # Extract connectivity values
    # Columns 0-3: [index, x, y, z]
    # Columns 4+j: connectivity to vertex j (starts at column 4)
    if len(connectivity_matrix.shape) == 3:
        # 3D matrix (N, N+4, C) - squeeze out channel dimension or use first layer
        # Extract vertex coordinates from columns 1-3
        vertices = connectivity_matrix[:, 1:4, 0]  # (N, 3) - x, y, z
        indices = connectivity_matrix[:, 0, 0]  # vertex indices
        print(f"\nUsing 3D matrix - extracting layer 0")
        print(f"Vertices from columns 1-3: shape {vertices.shape}")
        print(f"Connectivity starts at column 4")
        # Connectivity is in columns 4+j for vertex j
        # matrix[i, 4+j, 0] = connectivity from vertex i to vertex j
    else:
        # 2D matrix (N, N+4)
        vertices = connectivity_matrix[:, 1:4]  # (N, 3) - x, y, z
        indices = connectivity_matrix[:, 0]  # vertex indices
        print(f"\nUsing 2D matrix")
        print(f"Vertices from columns 1-3: shape {vertices.shape}")
        print(f"Connectivity starts at column 4")
    
    print(f"\nFirst 5 vertices:")
    for i in range(min(5, len(vertices))):
        print(f"  V{int(indices[i])}: X={vertices[i,0]:.3f}, Y={vertices[i,1]:.3f}, Z={vertices[i,2]:.3f}")
    
    # Find edges where connectivity > threshold
    # Access connectivity at column 4+j for vertex j
    print(f"\nSearching for edges (connectivity > {threshold})...")
    for i in range(n):
        for j in range(i + 1, n):  # Only check upper triangle to avoid duplicates
            col = 4 + j  # COLUMN 4+j for connectivity to vertex j
            if col < connectivity_matrix.shape[1]:
                conn_val = connectivity_matrix[i, col, 0] if len(connectivity_matrix.shape) == 3 else connectivity_matrix[i, col]
                if conn_val > threshold:
                    edges.append((i, j, conn_val))
                    if len(edges) <= 5:  # Show first 5 edges
                        print(f"  Edge {i}-{j}: connectivity={conn_val:.3f} (matrix[{i}, {col}])")
    
    print(f"\nFound {len(edges)} edges with connectivity > {threshold}")
    print(f"Vertex coordinates extracted: {vertices.shape}")
    print(f"Vertex coordinate range: X[{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}], Y[{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}], Z[{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")
    print(f"{'='*80}\n")
    
    return edges, vertices, indices


def check_edge_exists_in_views(i, j, top_matrix, front_matrix, side_matrix, threshold=1.0):
    """
    Check if an edge (i,j) should be included based on visibility in views.
    Include if:
    - Visible in all 3 views, OR
    - Visible in 2 views AND perpendicular to the third view (dot product with view normal < tolerance)
    
    Args:
        i, j: Vertex indices
        top_matrix: Top view matrix
        front_matrix: Front view matrix
        side_matrix: Side view matrix
        threshold: Minimum connectivity value to consider edge visible
        
    Returns:
        tuple: (exists: bool, visibility_count: int, top_conn, front_conn, side_conn)
    """
    # Extract 2D matrices if 3D
    if len(top_matrix.shape) == 3:
        top_2d = top_matrix[:, :, 0]
        front_2d = front_matrix[:, :, 0]
        side_2d = side_matrix[:, :, 0]
    else:
        top_2d = top_matrix
        front_2d = front_matrix
        side_2d = side_matrix
    
    # Check connectivity at column 4+j for vertex i
    col = 4 + j
    
    # Get connectivity values
    top_conn = top_2d[i, col] if i < top_2d.shape[0] and col < top_2d.shape[1] else 0
    front_conn = front_2d[i, col] if i < front_2d.shape[0] and col < front_2d.shape[1] else 0
    side_conn = side_2d[i, col] if i < side_2d.shape[0] and col < side_2d.shape[1] else 0
    
    # Count visibility - value >= 1 means edge is visible (increment count by 1, not by the cell value)
    visibility_count = 0
    visible_in_top = top_conn >= 1.0
    visible_in_front = front_conn >= 1.0
    visible_in_side = side_conn >= 1.0
    
    if visible_in_top:
        visibility_count += 1
    if visible_in_front:
        visibility_count += 1
    if visible_in_side:
        visibility_count += 1
    
    # If visible in all 3 views, include
    if visibility_count == 3:
        return True, visibility_count, top_conn, front_conn, side_conn
    
    # If visible in 2 views, check if perpendicular to third
    if visibility_count == 2:
        # Get vertex coordinates
        vi_coords = top_2d[i, 1:4] if i < top_2d.shape[0] else [0, 0, 0]
        vj_coords = top_2d[j, 1:4] if j < top_2d.shape[0] else [0, 0, 0]
        
        # Edge vector
        edge_vec = [vj_coords[0] - vi_coords[0], 
                   vj_coords[1] - vi_coords[1], 
                   vj_coords[2] - vi_coords[2]]
        
        # Normalize edge vector
        import math
        edge_length = math.sqrt(edge_vec[0]**2 + edge_vec[1]**2 + edge_vec[2]**2)
        if edge_length > 0.001:  # Avoid division by zero
            edge_vec_norm = [edge_vec[0]/edge_length, edge_vec[1]/edge_length, edge_vec[2]/edge_length]
        else:
            return False, visibility_count, top_conn, front_conn, side_conn
        
        # View normals: Top=(0,0,1), Front=(0,1,0), Side=(1,0,0)
        # Absolute dot product should be close to 1 (within tolerance) if perpendicular to view plane
        tolerance = 0.1
        
        # If not visible in top, check if perpendicular to top plane (parallel to z-axis)
        if not visible_in_top:
            dot_top = abs(edge_vec_norm[2])  # abs(dot with (0,0,1))
            if abs(dot_top - 1.0) < tolerance:  # Close to 1
                return True, visibility_count, top_conn, front_conn, side_conn
        
        # If not visible in front, check if perpendicular to front plane (parallel to y-axis)
        if not visible_in_front:
            dot_front = abs(edge_vec_norm[1])  # abs(dot with (0,1,0))
            if abs(dot_front - 1.0) < tolerance:  # Close to 1
                return True, visibility_count, top_conn, front_conn, side_conn
        
        # If not visible in side, check if perpendicular to side plane (parallel to x-axis)
        if not visible_in_side:
            dot_side = abs(edge_vec_norm[0])  # abs(dot with (1,0,0))
            if abs(dot_side - 1.0) < tolerance:  # Close to 1
                return True, visibility_count, top_conn, front_conn, side_conn
    
    # Otherwise, exclude
    return False, visibility_count, top_conn, front_conn, side_conn


def extract_edges_from_predicted_topview(predicted_matrix, input_padded, threshold=1.5, 
                                         top_matrix=None, front_matrix=None, side_matrix=None,
                                         filter_by_views=True):
    """
    Extract edges from predicted connectivity matrix for top view visualization.
    Uses the input matrix for vertex coordinates since predicted matrix only has connectivity.
    Optionally filters edges to only include those that exist in the input views.
    
    Args:
        predicted_matrix: Shape (N, N+4, 1) predicted connectivity values
        input_padded: Shape (200, 204, 11) original padded input with coordinates
        threshold: Minimum connectivity value to consider an edge
        top_matrix: Optional top view matrix for filtering
        front_matrix: Optional front view matrix for filtering
        side_matrix: Optional side view matrix for filtering
        filter_by_views: If True, only include edges that exist in the views
    
    Returns:
        tuple: (edges, vertices_2d)
            edges: List of (i, j, weight) tuples
            vertices_2d: Array of shape (N, 2) with x, y coordinates from input
    """
    print(f"\nextract_edges_from_predicted_topview: Predicted shape = {predicted_matrix.shape}, Input shape = {input_padded.shape}")
    if filter_by_views:
        print(f"  View filtering: ENABLED (checking edges exist in top/front/side views)")
    else:
        print(f"  View filtering: DISABLED")
    
    # Extract connectivity layer from input for coordinates
    if len(input_padded.shape) == 3:
        input_matrix = input_padded[:, :, 0]  # (200, 204)
    else:
        input_matrix = input_padded
    
    # Determine actual number of vertices from input (non-zero rows)
    n_actual = np.count_nonzero(np.any(input_matrix != 0, axis=1))
    n_pred = predicted_matrix.shape[0]
    print(f"  Input has {n_actual} actual vertices, predicted matrix size: {n_pred}")
    
    # Extract coordinates from input matrix (columns 1, 2)
    vertices_2d = input_matrix[:, 1:3]  # (200, 204) - x, y only
    
    # Handle 3D predicted matrix
    if len(predicted_matrix.shape) == 3:
        pred_matrix_2d = predicted_matrix[:, :, 0]  # (N, N+4)
    else:
        pred_matrix_2d = predicted_matrix
    
    edges = []
    filtered_count = 0
    debug_first_filtered = []
    debug_first_included = []
    
    # Extract edges from predicted connectivity
    # Only loop over actual vertices (minimum of n_actual and n_pred)
    n_vertices = min(n_actual, n_pred)
    print(f"  Extracting edges from {n_vertices} vertices")
    
    # IMPORTANT: Predicted matrix has same format as training Y: [idx, x, y, z, conn0, conn1, ...]
    # Column 4+j = connectivity to vertex j (same as solid and training matrices)
    for i in range(n_vertices):
        x_i, y_i = input_matrix[i, 1], input_matrix[i, 2]
        if x_i == 0 and y_i == 0:
            continue
        for j in range(i + 1, n_vertices):  # Upper triangle
            # Predicted output: column 4+j is connectivity to vertex j
            col = 4 + j
            if col < pred_matrix_2d.shape[1]:
                conn_val = pred_matrix_2d[i, col]
                if conn_val > threshold:
                    # Check if edge exists in views (if filtering enabled)
                    if filter_by_views and top_matrix is not None and front_matrix is not None and side_matrix is not None:
                        # Get connectivity values from all views for debugging
                        top_2d = top_matrix[:, :, 0] if len(top_matrix.shape) == 3 else top_matrix
                        front_2d = front_matrix[:, :, 0] if len(front_matrix.shape) == 3 else front_matrix
                        side_2d = side_matrix[:, :, 0] if len(side_matrix.shape) == 3 else side_matrix
                        
                        top_conn = top_2d[i, col] if i < top_2d.shape[0] and col < top_2d.shape[1] else 0
                        front_conn = front_2d[i, col] if i < front_2d.shape[0] and col < front_2d.shape[1] else 0
                        side_conn = side_2d[i, col] if i < side_2d.shape[0] and col < side_2d.shape[1] else 0
                        
                        # Check if edge exists in views with visibility count
                        exists, vis_count, top_c, front_c, side_c = check_edge_exists_in_views(
                            i, j, top_matrix, front_matrix, side_matrix, threshold=1.0)
                        
                        if exists:
                            edges.append((i, j, conn_val))
                            if len(debug_first_included) < 5:
                                # Get vertex coordinates for printing
                                vi_coords = input_matrix[i, 1:4]  # x, y, z
                                vj_coords = input_matrix[j, 1:4]
                                debug_first_included.append((i, j, conn_val, vis_count, top_c, front_c, side_c, vi_coords, vj_coords))
                        else:
                            filtered_count += 1
                            if len(debug_first_filtered) < 5:
                                vi_coords = input_matrix[i, 1:4]
                                vj_coords = input_matrix[j, 1:4]
                                debug_first_filtered.append((i, j, conn_val, vis_count, top_c, front_c, side_c, vi_coords, vj_coords))
                    else:
                        # No filtering, include all edges above threshold
                        edges.append((i, j, conn_val))
    
    print(f"  Found {len(edges)} predicted edges with connectivity > {threshold}")
    if filter_by_views and filtered_count > 0:
        print(f"  Filtered out {filtered_count} edges not present in views")
        if debug_first_filtered:
            print(f"\n  First 5 FILTERED edges:")
            for i, j, pred_val, vis_count, top_val, front_val, side_val, vi_coords, vj_coords in debug_first_filtered:
                print(f"    V{i} -> V{j}: pred={pred_val:.3f}, visibility={vis_count}")
                print(f"      V{i}: ({vi_coords[0]:.2f}, {vi_coords[1]:.2f}, {vi_coords[2]:.2f})")
                print(f"      V{j}: ({vj_coords[0]:.2f}, {vj_coords[1]:.2f}, {vj_coords[2]:.2f})")
                print(f"      Top={top_val:.1f}, Front={front_val:.1f}, Side={side_val:.1f}")
    elif filter_by_views and filtered_count == 0:
        print(f"  WARNING: No edges were filtered - all {len(edges)} edges exist in views")
    
    if debug_first_included:
        print(f"\n  First 5 INCLUDED edges:")
        for i, j, pred_val, vis_count, top_val, front_val, side_val, vi_coords, vj_coords in debug_first_included:
            print(f"    V{i} -> V{j}: pred={pred_val:.3f}, visibility={vis_count}")
            print(f"      V{i}: ({vi_coords[0]:.2f}, {vi_coords[1]:.2f}, {vi_coords[2]:.2f})")
            print(f"      V{j}: ({vj_coords[0]:.2f}, {vj_coords[1]:.2f}, {vj_coords[2]:.2f})")
            print(f"      Top={top_val:.1f}, Front={front_val:.1f}, Side={side_val:.1f}")
    if edges:
        weights = [w for _, _, w in edges]
        print(f"  Connectivity range: [{min(weights):.3f}, {max(weights):.3f}]")
    
    return edges, vertices_2d


def extract_edges_from_predicted(predicted_matrix, original_vertices, threshold=1.5, view_to_solid_mapping=None):
    """
    Extract edges from predicted connectivity using original vertex positions.
    Uses upper triangular matrix (i < j).
    
    Args:
        predicted_matrix: Shape (N, N+4, 1) predicted connectivity values
        original_vertices: Array of shape (N, 3) with actual x, y, z coordinates
        threshold: Minimum connectivity value to consider an edge
        view_to_solid_mapping: Array mapping view vertex indices to solid vertex indices
    
    Returns:
        list: List of (i, j, weight) tuples representing edges (using SOLID vertex indices)
    """
    print(f"\nextract_edges_from_predicted: Predicted shape = {predicted_matrix.shape}, Original vertices = {original_vertices.shape}")
    
    # Handle 3D predicted matrix
    if len(predicted_matrix.shape) == 3:
        pred_matrix_2d = predicted_matrix[:, :, 0]  # (N, N+4)
    else:
        pred_matrix_2d = predicted_matrix
    
    # The predicted matrix uses VIEW vertex indices (0-27 for top view)
    # We need to map these to SOLID vertex indices (0-45)
    if view_to_solid_mapping is None:
        print("  WARNING: No view_to_solid_mapping provided, assuming direct mapping")
        n_view = min(pred_matrix_2d.shape[0], len(original_vertices))
        view_to_solid_mapping = np.arange(n_view)
    else:
        n_view = len(view_to_solid_mapping)
        print(f"  Using view_to_solid_mapping for {n_view} view vertices")
    
    edges = []
    
    # Check matrix symmetry using correct columns
    # IMPORTANT: Predicted matrix has same format as training Y: [idx, x, y, z, conn0, conn1, ...]
    # Column 4+j = connectivity to vertex j (same as solid matrix)
    upper_vals = []
    lower_vals = []
    for i in range(n_view):
        for j in range(i + 1, n_view):
            col_ij = 4 + j
            col_ji = 4 + i
            if col_ij < pred_matrix_2d.shape[1] and col_ji < pred_matrix_2d.shape[1]:
                val_ij = pred_matrix_2d[i, col_ij]  # i's connection to j (view indices)
                val_ji = pred_matrix_2d[j, col_ji]  # j's connection to i
                upper_vals.append(val_ij)
                lower_vals.append(val_ji)
    
    upper_vals = np.array(upper_vals)
    lower_vals = np.array(lower_vals)
    diff = np.abs(upper_vals - lower_vals)
    print(f"\nMatrix symmetry check (column 4+j for vertex j in view coords):")
    print(f"  Mean absolute difference: {diff.mean():.6f}")
    print(f"  Max absolute difference: {diff.max():.6f}")
    
    # Print prediction statistics
    print(f"\nPredicted connectivity statistics (column 4+j for vertex j):")
    print(f"  Min: {upper_vals.min():.6f}")
    print(f"  Max: {upper_vals.max():.6f}")
    print(f"  Mean: {upper_vals.mean():.6f}")
    print(f"  Std: {upper_vals.std():.6f}")
    print(f"  Values > threshold ({threshold}): {np.sum(upper_vals > threshold)}")
    
    # Find edges where connectivity > threshold
    # Use upper triangle to avoid duplicates
    # Convert from view indices to solid indices
    for i_view in range(n_view):
        for j_view in range(i_view + 1, n_view):
            col = 4 + j_view
            if col < pred_matrix_2d.shape[1]:
                conn_val = pred_matrix_2d[i_view, col]
                if conn_val > threshold:
                    # Map from view indices to solid indices
                    i_solid = view_to_solid_mapping[i_view]
                    j_solid = view_to_solid_mapping[j_view]
                    edges.append((i_solid, j_solid, conn_val))
    
    print(f"\nFound {len(edges)} predicted edges with connectivity > {threshold}")
    if edges:
        weights = [w for _, _, w in edges]
        print(f"  Min connectivity: {min(weights):.3f}")
        print(f"  Max connectivity: {max(weights):.3f}")
        print(f"  Mean connectivity: {np.mean(weights):.3f}")
    
    return edges


def plot_connectivity_comparison(pred_edges, pred_vertices, 
                                 orig_edges=None, orig_vertices=None,
                                 top_view_edges=None, top_view_vertices=None,
                                 pred_topview_edges=None, pred_topview_vertices=None,
                                 training_edges=None, view_to_solid_mapping=None,
                                 seed=None):
    """
    Plot connectivity graph comparison: predicted vs original vs training.
    Creates both 3D and 2D (top view) projections.
    
    Args:
        pred_edges: List of (i, j, weight) tuples for predicted (solid indices)
        pred_vertices: Array of shape (N, 3) with x, y, z coordinates for predicted
        orig_edges: List of (i, j, weight) tuples for original (optional)
        orig_vertices: Array of shape (N, 3) with x, y, z coordinates for original (optional)
        training_edges: List of (i, j, weight) tuples for training Y (view indices)
        view_to_solid_mapping: Mapping from view to solid indices
        seed: Seed number for title
    
    Returns:
        fig: matplotlib figure object
    """
    if not pred_edges and not orig_edges:
        print("No edges to plot!")
        return None
    
    num_pred_vertices = len(pred_vertices)
    num_orig_vertices = len(orig_vertices) if orig_vertices is not None else 0
    
    # If we have training edges, plot 3 columns: input top view, training target, predicted
    # to compare the three directly
    if training_edges is not None and view_to_solid_mapping is not None:
        fig = plt.figure(figsize=(24, 8))
        ncols = 3
        
        # Training edges are already in solid indices (extracted from training Y matrix)
        # No conversion needed
        training_edges_solid = training_edges
    elif orig_edges is not None and orig_vertices is not None:
        fig = plt.figure(figsize=(16, 14))
        nrows = 2
        ncols = 2
        training_edges_solid = None
    else:
        fig = plt.figure(figsize=(16, 7))
        nrows = 1
        ncols = 2
        training_edges_solid = None
    
    # Plot three columns when comparing with training data
    if training_edges_solid is not None:
        # Column 1: Input Top View (from seed 30 data)
        ax1 = fig.add_subplot(1, 3, 1)
        
        if top_view_edges is not None and top_view_vertices is not None:
            # Plot input top view
            for i, j, weight in top_view_edges:
                x_coords = [top_view_vertices[i, 0], top_view_vertices[j, 0]]
                y_coords = [top_view_vertices[i, 1], top_view_vertices[j, 1]]
                ax1.plot(x_coords, y_coords, 'b-', alpha=0.5, linewidth=1)
            
            ax1.scatter(top_view_vertices[:, 0], top_view_vertices[:, 1], 
                       c='cyan', s=50, zorder=5, edgecolors='black', linewidth=1)
            
            for i in range(len(top_view_vertices)):
                if top_view_vertices[i, 0] != 0 or top_view_vertices[i, 1] != 0:
                    ax1.text(top_view_vertices[i, 0], top_view_vertices[i, 1], str(i), 
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_aspect('equal')
        ax1.set_title(f'INPUT: Top View Matrix (Seed {seed})\\n{len(top_view_edges) if top_view_edges else 0} edges', 
                     fontsize=12, color='blue', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Column 2: Training Target (what model should output)
        ax2 = fig.add_subplot(1, 3, 2)
        
        for i, j, weight in training_edges_solid:
            x_coords = [orig_vertices[i, 0], orig_vertices[j, 0]]
            y_coords = [orig_vertices[i, 1], orig_vertices[j, 1]]
            ax2.plot(x_coords, y_coords, 'm-', alpha=0.5, linewidth=1)
        
        ax2.scatter(orig_vertices[:, 0], orig_vertices[:, 1], 
                   c='yellow', s=50, zorder=5, edgecolors='black', linewidth=1)
        
        for i in range(num_orig_vertices):
            ax2.text(orig_vertices[i, 0], orig_vertices[i, 1], str(i), 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_aspect('equal')
        ax2.set_title(f'TRAINING TARGET: Full Solid (from training data)\\n{len(training_edges_solid)} edges', 
                     fontsize=12, color='magenta', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Column 3: Predicted Output
        ax3 = fig.add_subplot(1, 3, 3)
        
        for i, j, weight in pred_edges:
            x_coords = [pred_vertices[i, 0], pred_vertices[j, 0]]
            y_coords = [pred_vertices[i, 1], pred_vertices[j, 1]]
            ax3.plot(x_coords, y_coords, 'r-', alpha=0.5, linewidth=1)
        
        ax3.scatter(pred_vertices[:, 0], pred_vertices[:, 1], 
                   c='lightcoral', s=50, zorder=5, edgecolors='black', linewidth=1)
        
        for i in range(num_pred_vertices):
            ax3.text(pred_vertices[i, 0], pred_vertices[i, 1], str(i), 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_aspect('equal')
        ax3.set_title(f'PREDICTED OUTPUT: Full Solid (from model)\\n{len(pred_edges)} edges', 
                     fontsize=12, color='red', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        title = f'Top View Comparison (Seed {seed})' if seed else 'Top View Comparison'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    # Otherwise use original 2-row layout for non-training comparison
    nrows = 2 if orig_edges is not None else 1
    ncols = 2
    
    # Use filtered edges if available for both 3D and 2D views
    edges_to_plot_3d = pred_topview_edges if pred_topview_edges is not None else pred_edges
    
    # Plot predicted connectivity (top row)
    ax1 = fig.add_subplot(nrows, ncols, 1, projection='3d')
    
    # Draw predicted edges in 3D (use filtered if available)
    for i, j, weight in edges_to_plot_3d:
        x_coords = [pred_vertices[i, 0], pred_vertices[j, 0]]
        y_coords = [pred_vertices[i, 1], pred_vertices[j, 1]]
        z_coords = [pred_vertices[i, 2], pred_vertices[j, 2]]
        ax1.plot(x_coords, y_coords, z_coords, 'b-', alpha=0.5, linewidth=1)
    
    # Draw ALL vertices in 3D
    ax1.scatter(pred_vertices[:, 0], pred_vertices[:, 1], pred_vertices[:, 2], 
               c='red', s=50, zorder=5, edgecolors='black', linewidth=1)
    
    # Label all vertices in 3D
    for i in range(num_pred_vertices):
        ax1.text(pred_vertices[i, 0], pred_vertices[i, 1], pred_vertices[i, 2], str(i), 
                fontsize=8, fontweight='bold')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    filter_label = " (filtered)" if pred_topview_edges is not None else ""
    ax1.set_title(f'Predicted Full Solid - {len(edges_to_plot_3d)} edges{filter_label}', fontsize=12, color='blue')
    
    # 2D top view (x-y plane) for predicted
    ax2 = fig.add_subplot(nrows, 2, 2)
    
    # Use same filtered edges as 3D view
    edges_to_plot = edges_to_plot_3d
    
    for i, j, weight in edges_to_plot:
        x_coords = [pred_vertices[i, 0], pred_vertices[j, 0]]
        y_coords = [pred_vertices[i, 1], pred_vertices[j, 1]]
        ax2.plot(x_coords, y_coords, 'b-', alpha=0.5, linewidth=1)
    
    # Draw all vertices in 2D (x-y projection)
    ax2.scatter(pred_vertices[:, 0], pred_vertices[:, 1], 
               c='red', s=50, zorder=5, edgecolors='black', linewidth=1)
    
    # Label all vertices in 2D
    for i in range(num_pred_vertices):
        ax2.text(pred_vertices[i, 0], pred_vertices[i, 1], str(i), 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    edge_count_label = len(edges_to_plot)
    filter_status = " (filtered)" if pred_topview_edges is not None else ""
    ax2.set_title(f'Predicted Top View (X-Y) - {edge_count_label} edges{filter_status}', fontsize=12, color='blue')
    ax2.grid(True, alpha=0.3)
    
    # Add info box for predicted
    info_text = f'Vertices: {num_pred_vertices}\nEdges: {len(edges_to_plot)}'
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Plot training connectivity (middle row) if available
    if training_edges_solid is not None:
        row_offset = 2  # Training goes in row 2 (middle)
        
        ax_train3d = fig.add_subplot(nrows, 2, row_offset*2+1, projection='3d')
        
        # Show training edges in 3D (using solid indices)
        for i, j, weight in training_edges_solid:
            x_coords = [orig_vertices[i, 0], orig_vertices[j, 0]]
            y_coords = [orig_vertices[i, 1], orig_vertices[j, 1]]
            z_coords = [orig_vertices[i, 2], orig_vertices[j, 2]]
            ax_train3d.plot(x_coords, y_coords, z_coords, 'm-', alpha=0.5, linewidth=1)
        
        # Draw all vertices
        ax_train3d.scatter(orig_vertices[:, 0], orig_vertices[:, 1], orig_vertices[:, 2], 
                          c='yellow', s=50, zorder=5, edgecolors='black', linewidth=1)
        
        # Label vertices
        for i in range(num_orig_vertices):
            ax_train3d.text(orig_vertices[i, 0], orig_vertices[i, 1], orig_vertices[i, 2], str(i), 
                           fontsize=8, fontweight='bold')
        
        ax_train3d.set_xlabel('X')
        ax_train3d.set_ylabel('Y')
        ax_train3d.set_zlabel('Z')
        ax_train3d.set_title(f'Training Target (Full Solid) - {len(training_edges_solid)} edges', 
                            fontsize=12, color='magenta')
        
        # 2D projection of training
        ax_train2d = fig.add_subplot(nrows, 2, row_offset*2+2)
        
        for i, j, weight in training_edges_solid:
            x_coords = [orig_vertices[i, 0], orig_vertices[j, 0]]
            y_coords = [orig_vertices[i, 1], orig_vertices[j, 1]]
            ax_train2d.plot(x_coords, y_coords, 'm-', alpha=0.5, linewidth=1)
        
        ax_train2d.scatter(orig_vertices[:, 0], orig_vertices[:, 1], 
                          c='yellow', s=50, zorder=5, edgecolors='black', linewidth=1)
        
        for i in range(num_orig_vertices):
            ax_train2d.text(orig_vertices[i, 0], orig_vertices[i, 1], str(i), 
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax_train2d.set_xlabel('X')
        ax_train2d.set_ylabel('Y')
        ax_train2d.set_aspect('equal')
        ax_train2d.set_title(f'Training Target (X-Y) - {len(training_edges_solid)} edges', 
                            fontsize=12, color='magenta')
        ax_train2d.grid(True, alpha=0.3)
        
        info_text = f'Vertices: {num_orig_vertices}\nEdges: {len(training_edges_solid)}'
        ax_train2d.text(0.02, 0.98, info_text, transform=ax_train2d.transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
    
    # Plot original connectivity (bottom row) if available
    if orig_edges is not None and orig_vertices is not None:
        row_offset = 3 if training_edges_solid is not None else 2
        ax3 = fig.add_subplot(nrows, 2, (row_offset-1)*2+1, projection='3d')
        
        # Show FULL 3D solid connectivity (ground truth / solution)
        # This is what the model is trying to predict
        for i, j, weight in orig_edges:
            x_coords = [orig_vertices[i, 0], orig_vertices[j, 0]]
            y_coords = [orig_vertices[i, 1], orig_vertices[j, 1]]
            z_coords = [orig_vertices[i, 2], orig_vertices[j, 2]]
            ax3.plot(x_coords, y_coords, z_coords, 'g-', alpha=0.5, linewidth=1)
        
        # Draw all vertices in 3D
        ax3.scatter(orig_vertices[:, 0], orig_vertices[:, 1], orig_vertices[:, 2], 
                   c='orange', s=50, zorder=5, edgecolors='black', linewidth=1)
        
        # Label all vertices
        for i in range(num_orig_vertices):
            ax3.text(orig_vertices[i, 0], orig_vertices[i, 1], orig_vertices[i, 2], str(i), 
                    fontsize=8, fontweight='bold')
        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title(f'Original Full Solid - {len(orig_edges)} edges', fontsize=12, color='green')
        
        # 2D projection for original (bottom right)
        ax4 = fig.add_subplot(nrows, 2, (row_offset-1)*2+2)
        
        # Show FULL solid in 2D (same edges as 3D, just x-y projection)
        for i, j, weight in orig_edges:
            x_coords = [orig_vertices[i, 0], orig_vertices[j, 0]]
            y_coords = [orig_vertices[i, 1], orig_vertices[j, 1]]
            ax4.plot(x_coords, y_coords, 'g-', alpha=0.5, linewidth=1)
        
        ax4.scatter(orig_vertices[:, 0], orig_vertices[:, 1], 
                   c='orange', s=50, zorder=5, edgecolors='black', linewidth=1)
        
        for i in range(num_orig_vertices):
            ax4.text(orig_vertices[i, 0], orig_vertices[i, 1], str(i), 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax4.set_title(f'Original Full Solid (X-Y) - {len(orig_edges)} edges', fontsize=12, color='green')
        
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        
        # Add info box for original
        info_text = f'Vertices: {num_orig_vertices}\nEdges: {len(orig_edges)}'
        ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    title = f'Connectivity Comparison (Seed {seed})' if seed else 'Connectivity Comparison'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def save_plot(fig, output_file):
    """Save plot to file."""
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")


def load_original_connectivity(seed, data_dir="Output"):
    """
    Load the original solid connectivity matrix.
    
    Returns:
        connectivity_matrix: Shape (N, N+4) original connectivity matrix
    """
    matrix_file = Path(data_dir) / f"solid_connectivity_matrix_seed_{seed}.npy"
    
    if not matrix_file.exists():
        raise FileNotFoundError(f"Original connectivity matrix not found: {matrix_file}")
    
    print(f"\nLoading original connectivity matrix from: {matrix_file}")
    matrix = np.load(matrix_file)
    print(f"  Shape: {matrix.shape}")
    
    return matrix


def load_training_data_for_seed(seed, training_data_file):
    """
    Load the training data (padded inputs and output) for a specific seed.
    
    Args:
        seed: Seed number
        training_data_file: Path to training data .npz file
    
    Returns:
        tuple: (X1, X2, X3, Y) padded matrices if seed found, None otherwise
    """
    if not Path(training_data_file).exists():
        print(f"Training data file not found: {training_data_file}")
        return None
    
    print(f"\nLoading training data from: {training_data_file}")
    data = np.load(training_data_file)
    
    # Find seed in training data
    if 'seeds' not in data:
        print("No seed information in training data")
        return None
    
    seeds = data['seeds']
    if seed not in seeds:
        print(f"Seed {seed} not found in training data")
        print(f"Available seeds: {seeds}")
        return None
    
    # Find index of seed
    seed_idx = np.where(seeds == seed)[0][0]
    print(f"Found seed {seed} at index {seed_idx}")
    
    # Training data has 5 augmentations per seed
    # Get the first augmentation (original, not rotated)
    sample_idx = seed_idx * 5  # First augmentation for this seed
    
    X1 = data['X1'][sample_idx]  # Top view
    X2 = data['X2'][sample_idx]  # Front view
    X3 = data['X3'][sample_idx]  # Side view
    Y = data['Y'][sample_idx]    # Output (padded solid connectivity)
    
    print(f"Loaded training sample {sample_idx}:")
    print(f"  X1 (top): {X1.shape}")
    print(f"  X2 (front): {X2.shape}")
    print(f"  X3 (side): {X3.shape}")
    print(f"  Y (output): {Y.shape}")
    print(f"  Y range: [{Y.min():.3f}, {Y.max():.3f}]")
    
    return X1, X2, X3, Y


def main():
    parser = argparse.ArgumentParser(
        description='Predict connectivity matrix from three views using trained model'
    )
    parser.add_argument('seed', type=int, help='Seed number for solid')
    parser.add_argument('--model', default='NN/best_connectivity_model.h5',
                       help='Path to trained model (default: NN/best_connectivity_model.h5)')
    parser.add_argument('--data-dir', default='Output',
                       help='Directory containing connectivity matrices (default: Output)')
    parser.add_argument('--generate', action='store_true',
                       help='Generate solid and matrices if not found (requires pyocc environment)')
    parser.add_argument('--threshold', type=float, default=1.5,
                       help='Connectivity threshold for edge detection (default: 1.5)')
    parser.add_argument('--compare', action='store_true',
                       help='Load and compare with original solid connectivity matrix')
    parser.add_argument('--output', default=None,
                       help='Output plot filename (default: seed_{seed}_prediction.png)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip displaying plot (only save)')
    parser.add_argument('--training-data', default=None,
                       help='Path to training data .npz file for comparison')
    
    args = parser.parse_args()
    
    try:
        # Load model
        model = load_model(args.model)
        
        # Load or generate connectivity matrices
        try:
            top, front, side = load_connectivity_matrices(args.seed, args.data_dir)
        except FileNotFoundError as e:
            if args.generate:
                print(f"\n{e}")
                print("Generating connectivity matrices...")
                top, front, side = generate_solid_and_matrices(args.seed, args.data_dir)
            else:
                print(f"\n{e}")
                print("\nUse --generate flag to automatically generate the solid and matrices.")
                print("Example: python predict_connectivity.py 42 --generate")
                sys.exit(1)
        
        # EXPAND views to include all reconstructed vertices (like training data)
        top_expanded, front_expanded, side_expanded, reconstructed_verts = expand_views_to_reconstructed(top, front, side)
        
        print(f"\n=== EXPANDED VIEW COORDINATE CHECK ===")
        print(f"First 5 vertices in expanded top view:")
        for i in range(min(5, top_expanded.shape[0])):
            print(f"  V{i}: idx={top_expanded[i,0]:.0f}, x={top_expanded[i,1]:.3f}, y={top_expanded[i,2]:.3f}, z={top_expanded[i,3]:.3f}")
        
        # Pad matrices to model input size
        print("\n=== PADDING INPUT MATRICES ===")
        top_padded = pad_to_size(top_expanded, target_size=200)
        front_padded = pad_to_size(front_expanded, target_size=200)
        side_padded = pad_to_size(side_expanded, target_size=200)
        print(f"Padded shapes: {top_padded.shape}, {front_padded.shape}, {side_padded.shape}")
        
        # Plot padded input matrices to verify they are correct
        print("\n=== VISUALIZING PADDED INPUT MATRICES ===")
        plot_padded_input_matrices(top_padded, front_padded, side_padded, args.seed)
        
        # Predict output connectivity
        predicted = predict_connectivity(model, top_padded, front_padded, side_padded)
        
        # Save predicted matrix
        output_matrix_file = f"seed_{args.seed}_predicted_connectivity.npy"
        np.save(output_matrix_file, predicted)
        print(f"\nPredicted connectivity matrix saved to: {output_matrix_file}")
        
        # Load training data if provided
        training_Y = None
        training_edges = None
        if args.training_data:
            training_result = load_training_data_for_seed(args.seed, args.training_data)
            if training_result is not None:
                train_X1, train_X2, train_X3, training_Y = training_result
                print(f"\n=== COMPARING PREDICTED VS TRAINING OUTPUT ===")
                print(f"Predicted shape: {predicted.shape}")
                print(f"Training Y shape: {training_Y.shape}")
                
                # Add channel dimension to training_Y if needed
                if len(training_Y.shape) == 2:
                    training_Y = training_Y[:, :, np.newaxis]
                
                # Compare statistics
                pred_flat = predicted.flatten()
                train_flat = training_Y.flatten()
                diff = np.abs(pred_flat - train_flat)
                print(f"\nPrediction vs Training:")
                print(f"  Mean absolute difference: {diff.mean():.6f}")
                print(f"  Max absolute difference: {diff.max():.6f}")
                print(f"  Correlation: {np.corrcoef(pred_flat, train_flat)[0,1]:.6f}")
                
                # Compare first 4 columns (coordinates) for first 5 rows
                print(f"\n=== COMPARING COORDINATES (Columns 0-3) ===")
                pred_2d = predicted[:, :, 0] if len(predicted.shape) == 3 else predicted
                train_2d = training_Y[:, :, 0] if len(training_Y.shape) == 3 else training_Y
                print(f"First 5 rows, columns 0-3 [idx, x, y, z]:")
                print(f"\n{'Row':<5} {'Source':<10} {'Col 0 (idx)':<15} {'Col 1 (x)':<15} {'Col 2 (y)':<15} {'Col 3 (z)':<15}")
                print(f"{'-'*80}")
                for i in range(5):
                    print(f"{i:<5} {'Training':<10} {train_2d[i, 0]:<15.3f} {train_2d[i, 1]:<15.3f} {train_2d[i, 2]:<15.3f} {train_2d[i, 3]:<15.3f}")
                    print(f"{i:<5} {'Predicted':<10} {pred_2d[i, 0]:<15.3f} {pred_2d[i, 1]:<15.3f} {pred_2d[i, 2]:<15.3f} {pred_2d[i, 3]:<15.3f}")
                    print(f"{i:<5} {'Difference':<10} {abs(train_2d[i, 0] - pred_2d[i, 0]):<15.3f} {abs(train_2d[i, 1] - pred_2d[i, 1]):<15.3f} {abs(train_2d[i, 2] - pred_2d[i, 2]):<15.3f} {abs(train_2d[i, 3] - pred_2d[i, 3]):<15.3f}")
                    print()
                
                # Extract edges from training Y for comparison
                print(f"\n=== EXTRACTING EDGES FROM TRAINING OUTPUT ===")
                # Training Y has shape (200, 204) with structure [idx, x, y, z, conn...]
                # Connectivity to vertex j is at column 4+j (same as solid matrix)
                training_edges = []
                n_train = training_Y.shape[0]
                train_matrix_2d = training_Y[:, :, 0] if len(training_Y.shape) == 3 else training_Y
                
                # Extract edges using SOLID matrix format (column 4+j for connectivity)
                for i in range(n_train):
                    for j in range(i + 1, n_train):
                        col = 4 + j  # Training Y uses column 4+j for connectivity to vertex j
                        if col < train_matrix_2d.shape[1]:
                            conn_val = train_matrix_2d[i, col]
                            if conn_val > args.threshold:
                                # Store as solid indices (training Y already uses solid vertex numbering)
                                training_edges.append((i, j, conn_val))
                
                print(f"Found {len(training_edges)} edges in training Y with connectivity > {args.threshold}")
                
                # DEBUG: Print vertex info for training target
                print(f"\n=== TRAINING TARGET VERTEX INFO ===")
                print(f"Training Y uses SOLID vertex indices (like original solid matrix)")
                print(f"Training Y structure: [idx, x, y, z, conn0, conn1, ...]")
                print(f"First 10 training edges (solid indices):")
                for idx, (i, j, w) in enumerate(training_edges[:10]):
                    x_i, y_i, z_i = train_matrix_2d[i, 1], train_matrix_2d[i, 2], train_matrix_2d[i, 3]
                    x_j, y_j, z_j = train_matrix_2d[j, 1], train_matrix_2d[j, 2], train_matrix_2d[j, 3]
                    print(f"  Edge {idx}: V{i} ({x_i:.1f},{y_i:.1f},{z_i:.1f}) - V{j} ({x_j:.1f},{y_j:.1f},{z_j:.1f}), weight={w:.3f}")
        
        # Load original connectivity matrix to get vertex positions
        print("\n=== LOADING ORIGINAL FOR VERTEX POSITIONS ===")
        try:
            original = load_original_connectivity(args.seed, args.data_dir)
            
            # DEBUG: Show matrix structure
            print("\n" + "="*80)
            print("ORIGINAL CONNECTIVITY MATRIX DEBUG INFO")
            print("="*80)
            print(f"Shape: {original.shape}")
            print(f"\nFirst 5 rows, first 10 columns:")
            if len(original.shape) == 3:
                print(original[:5, :10, 0])
            else:
                print(original[:5, :10])
            print(f"\nColumn meanings:")
            print(f"  Column 0: Vertex index")
            print(f"  Column 1: X coordinate")
            print(f"  Column 2: Y coordinate")
            print(f"  Column 3: Z coordinate")
            print(f"  Column 4+j: Connectivity to vertex j (4+0=vertex 0, 4+1=vertex 1, etc.)")
            print(f"\nVertex coordinates (first 5):")
            if len(original.shape) == 3:
                coords = original[:5, 1:4, 0]
            else:
                coords = original[:5, 1:4]
            for i in range(min(5, len(coords))):
                print(f"  Vertex {i}: X={coords[i,0]:.3f}, Y={coords[i,1]:.3f}, Z={coords[i,2]:.3f}")
            print("="*80)
            
            orig_edges, orig_vertices, orig_indices = extract_edges_and_vertices(original, threshold=args.threshold)
            
            # Extract vertices from predicted output (columns 1-3: x, y, z)
            print("\n=== EXTRACTING VERTICES FROM PREDICTED OUTPUT ===")
            pred_2d = predicted[:, :, 0] if len(predicted.shape) == 3 else predicted
            
            # Count ALL rows with non-zero coordinates (not just consecutive ones)
            # The matrix may have zeros scattered throughout, not just at the end
            non_zero_mask = np.any(pred_2d[:, 1:4] != 0, axis=1)
            n_pred = np.sum(non_zero_mask)
            
            # Get the maximum index of non-zero rows to determine range
            non_zero_indices = np.where(non_zero_mask)[0]
            if len(non_zero_indices) > 0:
                max_vertex_idx = non_zero_indices[-1] + 1
                pred_vertices = pred_2d[:max_vertex_idx, 1:4]  # Extract x,y,z coordinates
                n_pred = max_vertex_idx
            else:
                pred_vertices = np.array([])
                n_pred = 0
            
            print(f"Found {n_pred} vertices in predicted output (up to index {max_vertex_idx-1 if n_pred > 0 else 0})")
            print(f"First 5 predicted vertices:")
            for i in range(min(5, n_pred)):
                print(f"  V{i}: ({pred_vertices[i, 0]:.1f}, {pred_vertices[i, 1]:.1f}, {pred_vertices[i, 2]:.1f})")
            print(f"Last 5 predicted vertices:")
            for i in range(max(0, n_pred-5), n_pred):
                print(f"  V{i}: ({pred_vertices[i, 0]:.1f}, {pred_vertices[i, 1]:.1f}, {pred_vertices[i, 2]:.1f})")
            
            # VERIFY: Check that matrix row index matches vertex index in column 0
            print("\n=== VERIFYING PREDICTED MATRIX: ROW INDEX vs VERTEX INDEX ===")
            pred_2d = predicted[:, :, 0] if len(predicted.shape) == 3 else predicted
            print("Checking first 10 rows:")
            for i in range(min(10, n_pred)):
                row_idx = i
                vertex_idx_col0 = int(pred_2d[i, 0])
                coords = (pred_2d[i, 1], pred_2d[i, 2], pred_2d[i, 3])
                match = "✓" if row_idx == vertex_idx_col0 else "✗"
                print(f"  Row {row_idx}: col[0]={vertex_idx_col0} {match}, coords=({coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f})")
            
            # Use predicted vertices directly (no mapping needed - indices match)
            predicted_to_solid = np.arange(n_pred)  # Identity mapping
            
            # Create mapping from view vertices to solid vertices (for top view input)
            # Match by X,Y coordinates
            print("\n=== CREATING VIEW TO SOLID VERTEX MAPPING (for input) ===")
            view_to_solid = []
            for i in range(len(top_padded)):
                if np.any(top_padded[i, :, 0] != 0):  # Non-zero row
                    view_x, view_y = top_padded[i, 1, 0], top_padded[i, 2, 0]
                    # Find matching solid vertex
                    for j in range(len(orig_vertices)):
                        solid_x, solid_y = orig_vertices[j, 0], orig_vertices[j, 1]
                        if abs(view_x - solid_x) < 0.01 and abs(view_y - solid_y) < 0.01:
                            view_to_solid.append(j)
                            if len(view_to_solid) <= 10:  # Show first 10
                                print(f"  View V{i} (X={view_x:.1f}, Y={view_y:.1f}) -> Solid V{j} (X={solid_x:.1f}, Y={solid_y:.1f})")
                            break
            view_to_solid = np.array(view_to_solid)
            print(f"Mapped {len(view_to_solid)} view vertices to solid vertices")
            
            # DEBUG: Print mapping and vertex coordinates
            print(f"\n=== INPUT TOP VIEW VERTEX INFO ===")
            print(f"First 10 vertices in top view (view index -> solid index):")
            for i in range(min(10, len(view_to_solid))):
                view_idx = i
                solid_idx = view_to_solid[i]
                view_x, view_y = top_padded[i, 1, 0], top_padded[i, 2, 0]
                solid_x, solid_y, solid_z = orig_vertices[solid_idx]
                print(f"  View V{view_idx} -> Solid V{solid_idx}: XY=({view_x:.1f}, {view_y:.1f}) -> XYZ=({solid_x:.1f}, {solid_y:.1f}, {solid_z:.1f})")
            
            # VERIFY: Check original solid matrix
            print("\n=== VERIFYING ORIGINAL SOLID MATRIX: ROW INDEX vs VERTEX INDEX ===")
            orig_2d = original[:, :, 0] if len(original.shape) == 3 else original
            print("Checking first 10 rows:")
            for i in range(min(10, len(orig_vertices))):
                row_idx = i
                vertex_idx_col0 = int(orig_2d[i, 0])
                coords = orig_vertices[i]
                match = "✓" if row_idx == vertex_idx_col0 else "✗"
                print(f"  Row {row_idx}: col[0]={vertex_idx_col0} {match}, coords=({coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f})")
            
            # VERIFY: Check training matrix if available
            if training_Y is not None:
                print("\n=== VERIFYING TRAINING TARGET MATRIX: ROW INDEX vs VERTEX INDEX ===")
                train_2d = training_Y[:, :, 0] if len(training_Y.shape) == 3 else training_Y
                n_train = np.count_nonzero(np.any(train_2d != 0, axis=1))
                print(f"Checking first 10 rows (out of {n_train} non-zero):")
                for i in range(min(10, n_train)):
                    row_idx = i
                    vertex_idx_col0 = int(train_2d[i, 0])
                    coords = (train_2d[i, 1], train_2d[i, 2], train_2d[i, 3])
                    match = "✓" if row_idx == vertex_idx_col0 else "✗"
                    print(f"  Row {row_idx}: col[0]={vertex_idx_col0} {match}, coords=({coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f})")
            
            # CRITICAL ISSUE: Compare vertex ordering
            print("\n" + "="*80)
            print("⚠️  CRITICAL: VERTEX ORDERING MISMATCH")
            print("="*80)
            print("Predicted matrix uses RECONSTRUCTED vertex ordering (48 vertices)")
            print("Original/Training matrix uses ORIGINAL SOLID vertex ordering (46 vertices)")
            print("\nSame vertex index refers to DIFFERENT physical vertices:")
            print("\n{:<15} {:<30} {:<30}".format("Vertex Index", "Predicted Coords", "Original Coords"))
            print("-" * 80)
            for i in range(min(10, n_pred, len(orig_vertices))):
                pred_coord = f"({pred_2d[i, 1]:.1f}, {pred_2d[i, 2]:.1f}, {pred_2d[i, 3]:.1f})"
                orig_coord = f"({orig_2d[i, 1]:.1f}, {orig_2d[i, 2]:.1f}, {orig_2d[i, 3]:.1f})"
                mismatch = "✗" if pred_2d[i, 1] != orig_2d[i, 1] or pred_2d[i, 2] != orig_2d[i, 2] or pred_2d[i, 3] != orig_2d[i, 3] else "✓"
                print(f"{i:<15} {pred_coord:<30} {orig_coord:<30} {mismatch}")
            
            print("\nThis means edges are indexed differently!")
            print("Example: Edge (0,1) in predicted matrix connects DIFFERENT vertices than edge (0,1) in original matrix")
            print("="*80)
            
            # Extract predicted edges using predicted vertex coordinates
            print("\n=== PREDICTED CONNECTIVITY (using predicted coordinates) ===")
            # Don't use mapping - edges should reference pred_vertices directly with indices 0,1,2,...
            # So we extract edges from the predicted matrix without any mapping
            pred_edges = []
            pred_2d_for_edges = predicted[:, :, 0] if len(predicted.shape) == 3 else predicted
            for i in range(n_pred):
                for j in range(i + 1, n_pred):
                    col = 4 + j
                    if col < pred_2d_for_edges.shape[1]:
                        conn_val = pred_2d_for_edges[i, col]
                        if conn_val > args.threshold:
                            pred_edges.append((i, j, conn_val))
            
            print(f"Found {len(pred_edges)} predicted edges")
            if pred_edges:
                print(f"First 10 predicted edges:")
                for idx, (i, j, w) in enumerate(pred_edges[:10]):
                    print(f"  Edge {idx}: V{i} ({pred_vertices[i, 0]:.1f},{pred_vertices[i, 1]:.1f},{pred_vertices[i, 2]:.1f}) - V{j} ({pred_vertices[j, 0]:.1f},{pred_vertices[j, 1]:.1f},{pred_vertices[j, 2]:.1f}), weight={w:.3f}")
                
                # Check for vertices with x > 32
                high_x_edges = [(i, j, w) for i, j, w in pred_edges if pred_vertices[i, 0] > 32 or pred_vertices[j, 0] > 32]
                print(f"\nEdges involving vertices with x>32: {len(high_x_edges)}")
                if high_x_edges:
                    print(f"First 5 high-x edges:")
                    for idx, (i, j, w) in enumerate(high_x_edges[:5]):
                        print(f"  Edge: V{i} ({pred_vertices[i, 0]:.1f},{pred_vertices[i, 1]:.1f},{pred_vertices[i, 2]:.1f}) - V{j} ({pred_vertices[j, 0]:.1f},{pred_vertices[j, 1]:.1f},{pred_vertices[j, 2]:.1f}), weight={w:.3f}")
            
            # Also extract predicted top view edges for visualization
            print("\n=== EXTRACTING PREDICTED TOP VIEW ===")
            pred_topview_edges, pred_topview_vertices = extract_edges_from_predicted_topview(
                predicted, top_padded, threshold=args.threshold,
                top_matrix=top_padded, front_matrix=front_padded, side_matrix=side_padded,
                filter_by_views=True
            )
            
            # Print summary statistics
            total_pred_edges = len(pred_edges)
            filtered_pred_edges = len(pred_topview_edges)
            dropped_edges = total_pred_edges - filtered_pred_edges
            print(f"\n{'='*60}")
            print(f"EDGE FILTERING STATISTICS")
            print(f"{'='*60}")
            print(f"Total edges predicted (threshold > {args.threshold}): {total_pred_edges}")
            print(f"Edges after filtering (exist in views):     {filtered_pred_edges}")
            print(f"Edges dropped (not in views):               {dropped_edges}")
            if total_pred_edges > 0:
                print(f"Percentage kept:                             {100*filtered_pred_edges/total_pred_edges:.1f}%")
                print(f"Percentage dropped:                          {100*dropped_edges/total_pred_edges:.1f}%")
            print(f"{'='*60}")
            
            # Also load top view matrix for accurate comparison
            print("\n=== LOADING TOP VIEW MATRIX ===")
            top_view_edges = None
            top_view_vertices = None
            try:
                # Use the padded top matrix to match what was shown in the first plot
                # Use threshold=0 to show ALL edges like in the first plot (which uses > 0)
                top_edges, top_verts = extract_edges_from_top_view(top_padded, threshold=0.0)
                top_view_edges = top_edges
                top_view_vertices = top_verts
            except Exception as e:
                print(f"Could not extract top view edges: {e}")
                import traceback
                traceback.print_exc()
            
            # Show comparison statistics
            if args.compare:
                print("\n=== COMPARISON ===")
                print(f"Number of vertices: {len(orig_vertices)}")
                
                # Determine which vertices are visible in predicted matrix
                n_visible = predicted.shape[0]
                print(f"Vertices visible in views: {n_visible} (out of {len(orig_vertices)})")
                
                # Filter original edges to only include those between visible vertices
                orig_edges_visible = [(i, j, w) for i, j, w in orig_edges if i < n_visible and j < n_visible]
                
                print(f"Predicted edges: {len(pred_edges)}, Original edges: {len(orig_edges)} (total), {len(orig_edges_visible)} (visible only)")
                
                # Calculate edge accuracy only on visible vertices
                pred_set = set((i, j) for i, j, _ in pred_edges)
                orig_set_visible = set((i, j) for i, j, _ in orig_edges_visible)
                
                correct = len(pred_set & orig_set_visible)
                precision = correct / len(pred_set) if pred_set else 0
                recall = correct / len(orig_set_visible) if orig_set_visible else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"Correctly predicted edges: {correct}")
                print(f"Precision: {precision:.3f}")
                print(f"Recall: {recall:.3f}")
                print(f"F1 Score: {f1:.3f}")
                
                if len(orig_edges) > len(orig_edges_visible):
                    print(f"\nNote: {len(orig_edges) - len(orig_edges_visible)} edges involve vertices not visible in any view (cannot be reconstructed)")
            else:
                # If not comparing, don't show original edges
                orig_edges = None
                orig_vertices = None
                
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("Cannot visualize without original vertex positions.")
            print("The predicted matrix contains connectivity values, not vertex coordinates.")
            sys.exit(1)
        
        # Create plot
        print(f"\n{'='*60}")
        print(f"PLOTTING EDGES")
        print(f"{'='*60}")
        print(f"3D view (left):  {len(pred_topview_edges)} edges (FILTERED)")
        print(f"Top view (right): {len(pred_topview_edges)} edges (FILTERED)")
        print(f"{'='*60}")
        
        fig = plot_connectivity_comparison(
            pred_edges, 
            pred_vertices,
            orig_edges,
            orig_vertices,
            top_view_edges=top_view_edges,
            top_view_vertices=top_view_vertices,
            pred_topview_edges=pred_topview_edges,
            pred_topview_vertices=pred_topview_vertices,
            training_edges=training_edges if training_Y is not None else None,
            view_to_solid_mapping=view_to_solid,
            seed=args.seed
        )
        
        # DEBUG: Print what's being plotted
        if training_edges is not None:
            print(f"\n=== PLOTTING COMPARISON ===")
            print(f"Column 1 (Input): {len(top_view_edges)} edges from top_view_edges")
            if top_view_vertices is not None:
                print(f"  First 3 vertices: {top_view_vertices[:3]}")
            print(f"Column 2 (Training): {len(training_edges)} edges (already in solid indices)")
            print(f"  Using {len(orig_vertices)} solid vertices")
            print(f"Column 3 (Predicted): {len(pred_edges)} edges (already in solid indices)")
            print(f"  Using {len(pred_vertices)} solid vertices")
        
        # Save plot
        if fig:
            if args.output is None:
                suffix = "_comparison" if args.compare else "_prediction"
                args.output = f"seed_{args.seed}{suffix}.png"
            save_plot(fig, args.output)
            
            # Show plot
            if not args.no_plot:
                plt.show()
        
        # Plot predicted connectivity in 3D
        print("\n=== PLOTTING PREDICTED 3D CONNECTIVITY ===")
        fig_3d = plt.figure(figsize=(14, 7))
        
        # Use filtered edges for this plot too
        edges_for_3d_plot = pred_topview_edges if pred_topview_edges is not None else pred_edges
        filter_label_3d = " (filtered)" if pred_topview_edges is not None else ""
        
        # Left: Predicted 3D connectivity
        ax1 = fig_3d.add_subplot(1, 2, 1, projection='3d')
        ax1.set_title(f'Predicted Connectivity - {len(edges_for_3d_plot)} edges{filter_label_3d}', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Plot predicted edges (filtered)
        for i, j, weight in edges_for_3d_plot:
            x_coords = [pred_vertices[i, 0], pred_vertices[j, 0]]
            y_coords = [pred_vertices[i, 1], pred_vertices[j, 1]]
            z_coords = [pred_vertices[i, 2], pred_vertices[j, 2]]
            ax1.plot(x_coords, y_coords, z_coords, 'b-', alpha=0.5, linewidth=1)
        
        # Plot vertices
        ax1.scatter(pred_vertices[:, 0], pred_vertices[:, 1], pred_vertices[:, 2], 
                   c='blue', s=50, zorder=5, edgecolors='black', linewidth=1)
        
        # Right: Original solid for comparison
        if orig_edges is not None and orig_vertices is not None:
            ax2 = fig_3d.add_subplot(1, 2, 2, projection='3d')
            ax2.set_title(f'Original Solid - {len(orig_edges)} edges', fontsize=12, fontweight='bold')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            # Plot original edges
            for i, j, weight in orig_edges:
                x_coords = [orig_vertices[i, 0], orig_vertices[j, 0]]
                y_coords = [orig_vertices[i, 1], orig_vertices[j, 1]]
                z_coords = [orig_vertices[i, 2], orig_vertices[j, 2]]
                ax2.plot(x_coords, y_coords, z_coords, 'g-', alpha=0.5, linewidth=1)
            
            # Plot vertices
            ax2.scatter(orig_vertices[:, 0], orig_vertices[:, 1], orig_vertices[:, 2], 
                       c='green', s=50, zorder=5, edgecolors='black', linewidth=1)
        
        plt.tight_layout()
        
        # Save 3D plot
        plot_3d_file = f"seed_{args.seed}_predicted_3d.png"
        plt.savefig(plot_3d_file, dpi=150, bbox_inches='tight')
        print(f"3D plot saved to: {plot_3d_file}")
        
        # Show 3D plot
        if not args.no_plot:
            plt.show()
        
        print("\nPrediction complete!")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
