#!/usr/bin/env python3
"""
Connectivity Matrix Viewer
Visualizes connectivity matrices from top, front, side views and merged matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def project_vertex_to_view(vertex, view_normal):
    """Project 3D vertex to 2D view using coordinate dropping.
    Matches the projection method in Reconstruct_Solid.py.
    """
    normal = np.array(view_normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    
    # Use coordinate dropping for standard orthogonal views
    if np.allclose(normal, [0, 0, 1], atol=1e-3):  # Top view
        return np.array([vertex[0], vertex[1]])  # Drop Z, keep X,Y
    elif np.allclose(normal, [0, -1, 0], atol=1e-3):  # Front view
        return np.array([vertex[0], vertex[2]])  # Drop Y, keep X,Z
    elif np.allclose(normal, [1, 0, 0], atol=1e-3):  # Side view
        return np.array([vertex[1], vertex[2]])  # Drop X, keep Y,Z
    else:
        raise ValueError(f"Unsupported projection normal: {normal}")


def find_matching_row(proj, matrix, tol=0.1):
    """Find row in matrix matching the 2D projection.
    Matrix format: n × (n+3) where columns are [index, x, y, connectivity...].
    """
    for i in range(matrix.shape[0]):
        row_proj = matrix[i, 1:3]  # Columns 1-2 are x,y coordinates
        if np.allclose(proj, row_proj, atol=tol):
            return i
    return None


def build_connectivity_matrices(vertices, top_matrix, front_matrix,
                                side_matrix):
    """Build square connectivity matrices from projection matrices.
    Matrix format: n_proj × (n_proj+3) where columns are
    [index, x, y, connectivity...].
    Each row represents a unique 2D projection with connectivity indexed
    by projection row number, not vertex index.
    """
    N = len(vertices)
    top_conn = np.zeros((N, N), dtype=int)
    front_conn = np.zeros((N, N), dtype=int)
    side_conn = np.zeros((N, N), dtype=int)
    
    # Project all vertices to each view and find their projection indices
    for i in range(N):
        vertex_i = vertices[i]
        tp_i = project_vertex_to_view(vertex_i, [0, 0, 1])
        fp_i = project_vertex_to_view(vertex_i, [0, -1, 0])
        sp_i = project_vertex_to_view(vertex_i, [1, 0, 0])
        
        top_idx_i = find_matching_row(tp_i, top_matrix)
        front_idx_i = find_matching_row(fp_i, front_matrix)
        side_idx_i = find_matching_row(sp_i, side_matrix)
        
        for j in range(N):
            vertex_j = vertices[j]
            tp_j = project_vertex_to_view(vertex_j, [0, 0, 1])
            fp_j = project_vertex_to_view(vertex_j, [0, -1, 0])
            sp_j = project_vertex_to_view(vertex_j, [1, 0, 0])
            
            top_idx_j = find_matching_row(tp_j, top_matrix)
            front_idx_j = find_matching_row(fp_j, front_matrix)
            side_idx_j = find_matching_row(sp_j, side_matrix)
            
            # Get connectivity between projection indices
            # Connectivity stored at column 3 + projection_index
            if top_idx_i is not None and top_idx_j is not None:
                top_conn[i, j] = int(top_matrix[top_idx_i, 3 + top_idx_j])
            
            if front_idx_i is not None and front_idx_j is not None:
                front_conn[i, j] = int(front_matrix[front_idx_i,
                                                     3 + front_idx_j])
            
            if side_idx_i is not None and side_idx_j is not None:
                side_conn[i, j] = int(side_matrix[side_idx_i,
                                                   3 + side_idx_j])
    
    return top_conn, front_conn, side_conn


def load_connectivity_matrices(seed):
    """Load connectivity matrices for a given seed."""
    npz_path = Path(f"Output/connectivity_matrices_seed_{seed}.npz")
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Connectivity matrices not found: {npz_path}")
    
    data = np.load(npz_path)
    
    # Load projection matrices and vertices
    vertices = data['all_vertices']
    top_matrix = data['top_view_matrix']
    front_matrix = data['front_view_matrix']
    side_matrix = data['side_view_matrix']
    
    # Build connectivity matrices
    top_conn, front_conn, side_conn = build_connectivity_matrices(
        vertices, top_matrix, front_matrix, side_matrix
    )
    
    # Build merged connectivity matrix
    N = vertices.shape[0]
    merged = np.zeros((N, N), dtype=int)
    
    for i in range(N):
        for j in range(N):
            merged[i, j] = top_conn[i, j] + front_conn[i, j] + side_conn[i, j]
    
    return {
        'vertices': vertices,
        'top_view': top_conn,
        'front_view': front_conn,
        'side_view': side_conn,
        'merged': merged
    }


def visualize_connectivity_matrix(ax, matrix, title, vertices=None):
    """Visualize a single connectivity matrix."""
    N = matrix.shape[0]
    
    # Create visualization matrix
    vis_matrix = np.zeros((N, N))
    
    # Fill with connectivity values
    for i in range(N):
        for j in range(N):
            if i == j:
                vis_matrix[i, j] = -1  # Diagonal
            else:
                vis_matrix[i, j] = matrix[i, j]
    
    # Plot
    im = ax.imshow(vis_matrix, cmap='RdYlGn', vmin=-1, vmax=3, 
                   interpolation='nearest', aspect='auto')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Vertex Index')
    ax.set_ylabel('Vertex Index')
    
    # Set ticks
    tick_interval = max(1, N // 10)
    ax.set_xticks(range(0, N, tick_interval))
    ax.set_yticks(range(0, N, tick_interval))
    
    # Add grid
    ax.grid(True, which='both', color='black', linewidth=0.5, alpha=0.3)
    
    return im


def print_connectivity_statistics(matrices):
    """Print statistics about connectivity matrices."""
    print("\n" + "="*70)
    print("CONNECTIVITY MATRIX STATISTICS")
    print("="*70)
    
    N = matrices['vertices'].shape[0]
    print(f"\nNumber of vertices: {N}")
    
    for view_name in ['top_view', 'front_view', 'side_view', 'merged']:
        matrix = matrices[view_name]
        
        # Count edges by connectivity value
        edges_conn_0 = 0
        edges_conn_1 = 0
        edges_conn_2 = 0
        edges_conn_3 = 0
        
        for i in range(N):
            for j in range(i+1, N):
                val = matrix[i, j]
                if val == 0:
                    edges_conn_0 += 1
                elif val == 1:
                    edges_conn_1 += 1
                elif val == 2:
                    edges_conn_2 += 1
                elif val == 3:
                    edges_conn_3 += 1
        
        total_edges = edges_conn_1 + edges_conn_2 + edges_conn_3
        
        print(f"\n{view_name.replace('_', ' ').title()}:")
        if view_name == 'merged':
            print(f"  Edges with conn=3 (all 3 views): {edges_conn_3}")
            print(f"  Edges with conn=2 (2 views): {edges_conn_2}")
            print(f"  Edges with conn=1 (1 view): {edges_conn_1}")
            print(f"  Total edges: {total_edges}")
            print(f"  Non-edges: {edges_conn_0}")
        else:
            print(f"  Edges (conn=1): {edges_conn_1}")
            print(f"  Non-edges (conn=0): {edges_conn_0}")


def find_specific_edges(matrices, vertex_indices):
    """Check connectivity for specific vertex pairs."""
    print("\n" + "="*70)
    print(f"CONNECTIVITY FOR SPECIFIC VERTICES: {vertex_indices}")
    print("="*70)
    
    N = len(vertex_indices)
    merged = matrices['merged']
    top = matrices['top_view']
    front = matrices['front_view']
    side = matrices['side_view']
    vertices = matrices['vertices']
    
    print("\nVertex Coordinates:")
    for v_idx in vertex_indices:
        v = vertices[v_idx]
        print(f"  Vertex {v_idx}: [{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}]")
    
    print("\nEdge Connectivity:")
    print(f"{'Edge':<12} {'Top':<6} {'Front':<6} {'Side':<6} {'Merged':<7}")
    print("-" * 45)
    
    for i in range(N):
        for j in range(i+1, N):
            v_i = vertex_indices[i]
            v_j = vertex_indices[j]
            
            edge_str = f"{v_i}-{v_j}"
            top_val = top[v_i, v_j]
            front_val = front[v_i, v_j]
            side_val = side[v_i, v_j]
            merged_val = merged[v_i, v_j]
            
            print(f"{edge_str:<12} {top_val:<6} {front_val:<6} {side_val:<6} {merged_val:<7}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize connectivity matrices for a given seed'
    )
    parser.add_argument('--seed', type=int, default=2,
                       help='Seed number (default: 2)')
    parser.add_argument('--vertices', type=int, nargs='+', default=None,
                       help='Specific vertex indices to check connectivity')
    parser.add_argument('--save', action='store_true',
                       help='Save plot to file instead of displaying')
    
    args = parser.parse_args()
    
    # Load matrices
    print(f"Loading connectivity matrices for seed {args.seed}...")
    matrices = load_connectivity_matrices(args.seed)
    
    # Print all vertices first
    vertices = matrices['vertices']
    print("\n" + "="*70)
    print("ALL VERTICES")
    print("="*70)
    for i, v in enumerate(vertices):
        print(f"  Vertex {i:2d}: [{v[0]:7.2f}, {v[1]:7.2f}, {v[2]:7.2f}]")
    
    # Print statistics
    print_connectivity_statistics(matrices)
    
    # Check specific vertices if provided
    if args.vertices:
        find_specific_edges(matrices, args.vertices)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Connectivity Matrices - Seed {args.seed}', 
                 fontsize=16, fontweight='bold')
    
    # Plot each matrix
    im1 = visualize_connectivity_matrix(axes[0, 0], matrices['top_view'], 
                                        'Top View')
    im2 = visualize_connectivity_matrix(axes[0, 1], matrices['front_view'], 
                                        'Front View')
    im3 = visualize_connectivity_matrix(axes[1, 0], matrices['side_view'], 
                                        'Side View')
    im4 = visualize_connectivity_matrix(axes[1, 1], matrices['merged'], 
                                        'Merged (conn=0,1,2,3)')
    
    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cbar1.set_label('Connectivity')
    cbar1.set_ticks([-1, 0, 1])
    cbar1.set_ticklabels(['Diagonal', 'No Edge', 'Edge'])
    
    cbar2 = plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar2.set_label('Connectivity')
    cbar2.set_ticks([-1, 0, 1])
    cbar2.set_ticklabels(['Diagonal', 'No Edge', 'Edge'])
    
    cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar3.set_label('Connectivity')
    cbar3.set_ticks([-1, 0, 1])
    cbar3.set_ticklabels(['Diagonal', 'No Edge', 'Edge'])
    
    cbar4 = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar4.set_label('Merged Connectivity')
    cbar4.set_ticks([-1, 0, 1, 2, 3])
    cbar4.set_ticklabels(['Diagonal', 'No Edge', '1 View', '2 Views', '3 Views'])
    
    plt.tight_layout()
    
    if args.save:
        output_path = f'connectivity_matrices_seed_{args.seed}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
