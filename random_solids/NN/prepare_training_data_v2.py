#!/usr/bin/env python3
"""
Improved training data preparation with explicit reconstruction algorithm.
"""

import numpy as np
import sys
import os
import argparse

# Constants
MAX_VERTICES = 100  # Maximum number of vertices to pad to
MATRIX_WIDTH = MAX_VERTICES + 4  # Columns: idx, x, y, z + connectivity


def pad_matrix_to_fixed_size(matrix, target_size=MAX_VERTICES):
    """Pad matrix to fixed size."""
    n, width = matrix.shape
    padded = np.zeros((target_size, target_size + 4), dtype=np.float32)
    
    # Copy existing data
    padded[:n, :width] = matrix
    
    return padded

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_view_matrices(seed, input_dir="Output"):
    """Load view matrices from npz file."""
    filename = os.path.join(input_dir, f"connectivity_matrices_seed_{seed}.npz")
    data = np.load(filename)
    
    top_matrix = data['top_view_matrix']
    front_matrix = data['front_view_matrix']
    side_matrix = data['side_view_matrix']
    all_vertices = data['all_vertices']
    
    return top_matrix, front_matrix, side_matrix, all_vertices

def load_solid_matrix(seed, input_dir="Output"):
    """Load solid connectivity matrix."""
    filename = os.path.join(input_dir, f"solid_connectivity_matrix_seed_{seed}.npy")
    solid_matrix = np.load(filename)
    return solid_matrix

def reconstruct_vertices(top_matrix, front_matrix, side_matrix, tolerance=0.1):
    """
    Reconstruct 3D vertices following exact algorithm:
    1. Get unique x values from top view
    2. For each x, get unique z values from front view  
    3. Combine (x,y) from top with z from front
    4. Verify (y,z) exists in side view
    """
    print("\n[RECONSTRUCTION] Step-by-step vertex reconstruction:")
    
    # Step 1: Extract all (x,y) pairs from top view
    # Top view sparse format: [idx, x, y, connectivity...]
    top_xy = []
    for i in range(top_matrix.shape[0]):
        x, y = top_matrix[i, 1], top_matrix[i, 2]  # columns 1, 2
        top_xy.append((x, y))
    print(f"  Step 1: Extracted {len(top_xy)} (x,y) pairs from top view")
    
    # Step 2: Extract all (x,z) pairs from front view
    # Front view sparse format: [idx, x, z, connectivity...]
    front_xz = []
    for i in range(front_matrix.shape[0]):
        x, z = front_matrix[i, 1], front_matrix[i, 2]  # columns 1, 2
        front_xz.append((x, z))
    print(f"  Step 2: Extracted {len(front_xz)} (x,z) pairs from front view")
    
    # Step 3: Extract all (y,z) pairs from side view
    # Side view sparse format: [idx, y, z, connectivity...]
    side_yz = []
    for i in range(side_matrix.shape[0]):
        y, z = side_matrix[i, 1], side_matrix[i, 2]  # columns 1, 2
        side_yz.append((y, z))
    print(f"  Step 3: Extracted {len(side_yz)} (y,z) pairs from side view")
    
    # Step 4: For each unique x, find all (y,z) combinations
    # Get unique x values
    unique_x = set()
    for x, y in top_xy:
        unique_x.add(round(x / tolerance) * tolerance)
    unique_x = sorted(unique_x)
    print(f"  Step 4: Found {len(unique_x)} unique x values")
    
    # Step 5: Build candidate vertices by combining views
    candidates = []
    for x, y in top_xy:
        # Find z values from front view that match this x
        for x_front, z_front in front_xz:
            if abs(x - x_front) < tolerance:
                # Create candidate (x, y, z)
                candidate = (x, y, z_front)
                candidates.append(candidate)
    
    print(f"  Step 5: Created {len(candidates)} candidate (x,y,z) vertices")
    
    # Step 6: Verify each candidate against side view
    verified = []
    rejected = 0
    for x, y, z in candidates:
        # Check if (y, z) exists in side view
        found = False
        for y_side, z_side in side_yz:
            if abs(y - y_side) < tolerance and abs(z - z_side) < tolerance:
                found = True
                break
        
        if found:
            verified.append([x, y, z])
        else:
            rejected += 1
    
    print(f"  Step 6: Verified {len(verified)} vertices, rejected {rejected}")
    
    # Step 7: Remove duplicates
    unique_vertices = []
    for v in verified:
        is_duplicate = False
        for existing in unique_vertices:
            if (abs(v[0] - existing[0]) < tolerance and 
                abs(v[1] - existing[1]) < tolerance and 
                abs(v[2] - existing[2]) < tolerance):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_vertices.append(v)
    
    print(f"  Step 7: After deduplication: {len(unique_vertices)} unique vertices")
    
    reconstructed = np.array(unique_vertices, dtype=np.float32)
    return reconstructed

def build_mapping(reconstructed_vertices, view_matrix, proj_axes, tolerance=0.1):
    """
    Build mapping from sparse view rows to reconstructed vertex indices.
    
    Args:
        reconstructed_vertices: (M, 3) array of 3D vertices
        view_matrix: Sparse view matrix (N, N+3)
        proj_axes: Tuple (axis1, axis2) for projection
        tolerance: Matching tolerance
    
    Returns:
        Dict mapping view_row -> list of reconstructed indices
    """
    mapping = {}
    
    for view_row in range(view_matrix.shape[0]):
        # Get 2D coordinates from view matrix (columns 1, 2)
        view_coord = np.array([view_matrix[view_row, 1], view_matrix[view_row, 2]])
        
        # Find all reconstructed vertices that project to this coordinate
        matching = []
        for recon_idx, vertex_3d in enumerate(reconstructed_vertices):
            recon_proj = np.array([vertex_3d[proj_axes[0]], vertex_3d[proj_axes[1]]])
            
            if abs(view_coord[0] - recon_proj[0]) < tolerance and abs(view_coord[1] - recon_proj[1]) < tolerance:
                matching.append(recon_idx)
        
        mapping[view_row] = matching
    
    return mapping

def expand_view_connectivity(view_matrix, view_to_recon, n_reconstructed, reconstructed_vertices):
    """
    Expand sparse view connectivity to reconstructed vertex space.
    For each edge (i,j) in sparse matrix:
      - Find all reconstructed vertices matching row i
      - Find all reconstructed vertices matching row j
      - Copy edge to ALL combinations
    """
    n_sparse = view_matrix.shape[0]
    
    # Create expanded matrix: (n_reconstructed, n_reconstructed+4)
    expanded = np.zeros((n_reconstructed, n_reconstructed + 4), dtype=np.float32)
    
    # Fill vertex information
    for i in range(n_reconstructed):
        expanded[i, 0] = i  # index
        expanded[i, 1] = reconstructed_vertices[i, 0]  # x
        expanded[i, 2] = reconstructed_vertices[i, 1]  # y
        expanded[i, 3] = reconstructed_vertices[i, 2]  # z
    
    # Expand edges
    edges_added = 0
    for i in range(n_sparse):
        for j in range(n_sparse):
            # Check if edge exists in sparse matrix (column 3+j)
            col_sparse = 3 + j
            if col_sparse < view_matrix.shape[1]:
                edge_value = view_matrix[i, col_sparse]
                
                if edge_value > 0:
                    # Get all reconstructed vertices for rows i and j
                    recon_i_list = view_to_recon.get(i, [])
                    recon_j_list = view_to_recon.get(j, [])
                    
                    # Copy edge to ALL combinations
                    for recon_i in recon_i_list:
                        for recon_j in recon_j_list:
                            col_expanded = 4 + recon_j
                            if col_expanded < expanded.shape[1]:
                                if expanded[recon_i, col_expanded] == 0:
                                    expanded[recon_i, col_expanded] = edge_value
                                    expanded[recon_j, 4 + recon_i] = edge_value
                                    edges_added += 1
    
    print(f"    Expanded: {edges_added} edges added")
    return expanded

def count_edges(matrix):
    """Count edges in connectivity matrix."""
    n = matrix.shape[0]
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            col = 4 + j
            if col < matrix.shape[1] and matrix[i, col] > 0:
                count += 1
    return count
def count_edges_in_solid(solid_matrix):
    """Count edges in 3D solid matrix (shape N, N+4, 11)."""
    n = solid_matrix.shape[0]
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            col = 4 + j
            if col < solid_matrix.shape[1] and solid_matrix[i, col, 0] > 0:
                count += 1
    return count
def prepare_sample(seed, tolerance=0.1, input_dir="Output"):
    """Prepare one training sample."""
    print(f"\n{'='*70}")
    print(f"PREPARING SAMPLE FOR SEED {seed}")
    print(f"{'='*70}")
    
    # Load data
    top_matrix, front_matrix, side_matrix, all_vertices = load_view_matrices(seed, input_dir)
    solid_matrix = load_solid_matrix(seed, input_dir)
    
    print(f"\nOriginal data:")
    print(f"  Solid: {solid_matrix.shape[0]} vertices")
    print(f"  Top view: {top_matrix.shape}")
    print(f"  Front view: {front_matrix.shape}")
    print(f"  Side view: {side_matrix.shape}")
    
    # Reconstruct vertices
    reconstructed = reconstruct_vertices(top_matrix, front_matrix, side_matrix, tolerance)
    n_recon = len(reconstructed)
    
    print(f"\nReconstructed {n_recon} vertices")
    print(f"Original solid has {solid_matrix.shape[0]} vertices")
    
    # Build mappings
    print("\n[MAPPING] Building view-to-reconstructed mappings...")
    top_to_recon = build_mapping(reconstructed, top_matrix, (0, 1), tolerance)
    front_to_recon = build_mapping(reconstructed, front_matrix, (0, 2), tolerance)
    side_to_recon = build_mapping(reconstructed, side_matrix, (1, 2), tolerance)
    
    # Show mappings
    print(f"\nTop view mapping (showing first 10):")
    for i in range(min(10, top_matrix.shape[0])):
        print(f"  Row {i} -> Reconstructed vertices {top_to_recon[i]}")
    
    # Expand views
    print("\n[EXPANSION] Expanding sparse views to reconstructed space...")
    
    print(f"  Top view:")
    top_expanded = expand_view_connectivity(top_matrix, top_to_recon, n_recon, reconstructed)
    top_edges = count_edges(top_expanded)
    print(f"    Result: {top_edges} edges")
    
    print(f"  Front view:")
    front_expanded = expand_view_connectivity(front_matrix, front_to_recon, n_recon, reconstructed)
    front_edges = count_edges(front_expanded)
    print(f"    Result: {front_edges} edges")
    
    print(f"  Side view:")
    side_expanded = expand_view_connectivity(side_matrix, side_to_recon, n_recon, reconstructed)
    side_edges = count_edges(side_expanded)
    print(f"    Result: {side_edges} edges")
    
    # Create Y from original solid
    print("\n[Y TARGET] Creating Y from RECONSTRUCTED vertices (same as X)...")
    n_solid = solid_matrix.shape[0]
    
    # Y should use RECONSTRUCTED vertices, not original solid vertices!
    Y = np.zeros((n_recon, n_recon + 4), dtype=np.float32)
    
    # Fill coordinates from RECONSTRUCTED vertices (same as X1/X2/X3)
    for i in range(n_recon):
        Y[i, 0] = i
        Y[i, 1] = reconstructed[i, 0]  # x
        Y[i, 2] = reconstructed[i, 1]  # y
        Y[i, 3] = reconstructed[i, 2]  # z
    
    # Build reverse mapping: solid vertex index -> list of reconstructed vertex indices
    solid_to_recon = {}
    for recon_idx in range(n_recon):
        recon_vertex = reconstructed[recon_idx]
        # Find matching solid vertex (they should have same coordinates)
        for solid_idx in range(n_solid):
            solid_vertex = all_vertices[solid_idx]
            if np.allclose(recon_vertex, solid_vertex, atol=tolerance):
                if solid_idx not in solid_to_recon:
                    solid_to_recon[solid_idx] = []
                solid_to_recon[solid_idx].append(recon_idx)
                break
    
    print(f"  Mapping solid {n_solid} vertices to reconstructed {n_recon} vertices")
    print(f"  Solid vertices mapped: {len(solid_to_recon)}")
    
    # Map solid connectivity to reconstructed space
    edges_mapped = 0
    for i_solid in range(n_solid):
        for j_solid in range(n_solid):
            conn_val = solid_matrix[i_solid, 4+j_solid, 0]
            if conn_val > 0 and i_solid in solid_to_recon and j_solid in solid_to_recon:
                # Map edge to all reconstructed vertex combinations
                for i_recon in solid_to_recon[i_solid]:
                    for j_recon in solid_to_recon[j_solid]:
                        if i_recon != j_recon:  # No self-loops
                            Y[i_recon, 4 + j_recon] = conn_val
                            if i_recon < j_recon:  # Count unique edges
                                edges_mapped += 1
    
    y_edges = count_edges(Y)
    solid_edges = count_edges_in_solid(solid_matrix)
    print(f"  Y: {n_recon} vertices (reconstructed), {y_edges} edges")
    print(f"  Original solid: {n_solid} vertices, {solid_edges} edges")
    print(f"  Edges mapped from solid to reconstructed: {edges_mapped}")
    print(f"  ✓ X1/X2/X3/Y all use SAME {n_recon} reconstructed vertices!")
    
    # Save for visualization
    output = {
        'reconstructed_vertices': reconstructed,
        'top_expanded': top_expanded,
        'front_expanded': front_expanded,
        'side_expanded': side_expanded,
        'Y': Y,
        'all_vertices': all_vertices
    }
    
    filename = f'reconstructed_seed_{seed}.npz'
    np.savez(filename, **output)
    print(f"\nSaved to: {filename}")
    
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare training data from view matrices')
    parser.add_argument(
        '--seeds', type=str, required=True,
        help='Comma-separated list of seeds to process (e.g., "5,10,15,30")'
    )
    parser.add_argument(
        '--tolerance', type=float, default=0.1,
        help='Tolerance for vertex matching (default: 0.1)'
    )
    parser.add_argument(
        '--augment', type=int, default=0,
        help='Number of augmentations per sample (default: 0)'
    )
    parser.add_argument(
        '--input-dir', type=str, default='Output',
        help='Input directory containing connectivity matrices (default: Output)'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output file path for training data (e.g., NN/training_data.npz)'
    )
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    print(f"\n{'='*70}")
    print(f"PREPARING TRAINING DATA")
    print(f"{'='*70}")
    print(f"Seeds: {len(seeds)}")
    print(f"Tolerance: {args.tolerance}")
    print(f"Augmentation: {args.augment}x per seed")
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output}")
    print(f"Total samples: {len(seeds) * max(1, args.augment)}")
    print(f"{'='*70}\n")
    
    # Process all seeds and collect samples
    all_X1 = []  # Top view matrices
    all_X2 = []  # Front view matrices
    all_X3 = []  # Side view matrices
    all_Y = []   # Solid connectivity matrices (targets)
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n[{i}/{len(seeds)}] Processing seed {seed}...")
        try:
            result = prepare_sample(seed, args.tolerance, input_dir=args.input_dir)
            
            # Extract matrices from result
            X1 = result['top_expanded']  # Top view
            X2 = result['front_expanded']  # Front view
            X3 = result['side_expanded']  # Side view
            Y = result['Y']  # Solid connectivity
            
            # Pad to fixed size
            X1_padded = pad_matrix_to_fixed_size(X1, MAX_VERTICES)
            X2_padded = pad_matrix_to_fixed_size(X2, MAX_VERTICES)
            X3_padded = pad_matrix_to_fixed_size(X3, MAX_VERTICES)
            Y_padded = pad_matrix_to_fixed_size(Y, MAX_VERTICES)
            
            # Add original sample
            all_X1.append(X1_padded)
            all_X2.append(X2_padded)
            all_X3.append(X3_padded)
            all_Y.append(Y_padded)
            
            # Apply augmentation if requested
            if args.augment > 0:
                for aug in range(args.augment):
                    # Simple augmentation: shuffle rows (vertices)
                    n_rows = X1.shape[0]
                    perm = np.random.permutation(n_rows)
                    
                    # Apply same permutation to all matrices, then pad
                    X1_aug = pad_matrix_to_fixed_size(X1[perm], MAX_VERTICES)
                    X2_aug = pad_matrix_to_fixed_size(X2[perm], MAX_VERTICES)
                    X3_aug = pad_matrix_to_fixed_size(X3[perm], MAX_VERTICES)
                    Y_aug = pad_matrix_to_fixed_size(Y[perm], MAX_VERTICES)
                    
                    all_X1.append(X1_aug)
                    all_X2.append(X2_aug)
                    all_X3.append(X3_aug)
                    all_Y.append(Y_aug)
            
            print(f"  ✓ Seed {seed}: {1 + args.augment} samples added")
            
        except Exception as e:
            print(f"  ✗ Seed {seed} failed: {e}")
            continue
    
    # Stack all samples into arrays
    print(f"\n{'='*70}")
    print("CREATING TRAINING DATA ARRAYS")
    print(f"{'='*70}")
    
    X1_array = np.array(all_X1)
    X2_array = np.array(all_X2)
    X3_array = np.array(all_X3)
    Y_array = np.array(all_Y)
    
    print(f"Final shapes:")
    print(f"  X1 (top view): {X1_array.shape}")
    print(f"  X2 (front view): {X2_array.shape}")
    print(f"  X3 (side view): {X3_array.shape}")
    print(f"  Y (solid): {Y_array.shape}")
    
    # Save to output file
    print(f"\nSaving to: {args.output}")
    np.savez(args.output, X1=X1_array, X2=X2_array, X3=X3_array, Y=Y_array)
    print(f"✓ Training data saved successfully!")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total samples: {len(all_X1)}")
    print(f"Seeds processed: {len(seeds)}")
    print(f"Augmentation: {args.augment}x per seed")
    print(f"Output file: {args.output}")
