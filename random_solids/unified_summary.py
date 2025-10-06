"""
Create a unified summary array for solid topology.

Structure:
- Column 0: Vertex index
- Columns 1-3: 3D coordinates (x, y, z)
- Columns 4-5: Top view projection (x, y)
- Columns 6-7: Front view projection (x, z)
- Columns 8-9: Side view projection (y, z)
- Columns 10+: Adjacency matrix (n x n) for edges

If edge exists between vertex i and j:
  matrix[i, 10+j] = 1
  matrix[j, 10+i] = 1 (symmetric)
"""

import numpy as np


def create_unified_summary(vertices, edges, view_normals=None):
    """
    Create a unified summary array with all vertex and edge information.
    
    Args:
        vertices: Array of 3D vertices [n, 3]
        edges: List of edge tuples (v1_idx, v2_idx)
        view_normals: Optional dict with 'top', 'front', 'side' normals
        
    Returns:
        summary_array: Array of shape [n, 10+n] where:
            - Col 0: Vertex index
            - Cols 1-3: 3D coordinates (x, y, z)
            - Cols 4-5: Top view projection (x, y)
            - Cols 6-7: Front view projection (x, z)
            - Cols 8-9: Side view projection (y, z)
            - Cols 10+: Adjacency matrix for edges
    """
    n_vertices = len(vertices)
    n_cols = 10 + n_vertices
    
    # Initialize array
    summary = np.zeros((n_vertices, n_cols))
    
    # Column 0: Vertex indices
    summary[:, 0] = np.arange(n_vertices)
    
    # Columns 1-3: 3D coordinates
    summary[:, 1:4] = vertices
    
    # Columns 4-5: Top view projection (x, y) - looking down z-axis
    summary[:, 4] = vertices[:, 0]  # x
    summary[:, 5] = vertices[:, 1]  # y
    
    # Columns 6-7: Front view projection (x, z) - looking down y-axis
    summary[:, 6] = vertices[:, 0]  # x
    summary[:, 7] = vertices[:, 2]  # z
    
    # Columns 8-9: Side view projection (y, z) - looking down x-axis
    summary[:, 8] = vertices[:, 1]  # y
    summary[:, 9] = vertices[:, 2]  # z
    
    # Columns 10+: Adjacency matrix for edges
    for v1_idx, v2_idx in edges:
        if v1_idx < n_vertices and v2_idx < n_vertices:
            summary[v1_idx, 10 + v2_idx] = 1
            summary[v2_idx, 10 + v1_idx] = 1  # Symmetric
    
    return summary


def print_summary_info(summary):
    """Print information about the unified summary array."""
    n_vertices = summary.shape[0]
    n_cols = summary.shape[1]
    
    print("=" * 70)
    print("UNIFIED SUMMARY ARRAY")
    print("=" * 70)
    print(f"Shape: {summary.shape}")
    print(f"Vertices: {n_vertices}")
    print(f"Total columns: {n_cols}")
    print()
    print("Column structure:")
    print("  [0]      : Vertex index")
    print("  [1-3]    : 3D coordinates (x, y, z)")
    print("  [4-5]    : Top view projection (x, y)")
    print("  [6-7]    : Front view projection (x, z)")
    print("  [8-9]    : Side view projection (y, z)")
    print(f"  [10-{n_cols-1}] : Adjacency matrix ({n_vertices}×{n_vertices})")
    print()
    
    # Count edges
    adjacency = summary[:, 10:]
    n_edges = int(np.sum(adjacency) / 2)  # Divide by 2 (symmetric)
    print(f"Total edges in adjacency matrix: {n_edges}")
    print("=" * 70)


def save_summary_to_file(summary, filename="unified_summary.txt"):
    """Save the unified summary to a text file."""
    n_vertices = summary.shape[0]
    
    with open(filename, 'w') as f:
        f.write("UNIFIED SUMMARY ARRAY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Column Structure:\n")
        f.write("  [0]      : Vertex index\n")
        f.write("  [1-3]    : 3D coordinates (x, y, z)\n")
        f.write("  [4-5]    : Top view projection (x, y)\n")
        f.write("  [6-7]    : Front view projection (x, z)\n")
        f.write("  [8-9]    : Side view projection (y, z)\n")
        f.write(f"  [10-{summary.shape[1]-1}] : Adjacency matrix "
                f"({n_vertices}×{n_vertices})\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("VERTEX DATA (Columns 0-9)\n")
        f.write("=" * 70 + "\n")
        f.write("Idx |      3D Coords (x,y,z)      | "
                "Top(x,y) | Front(x,z) | Side(y,z)\n")
        f.write("-" * 70 + "\n")
        
        for i in range(n_vertices):
            idx = int(summary[i, 0])
            x, y, z = summary[i, 1], summary[i, 2], summary[i, 3]
            top_x, top_y = summary[i, 4], summary[i, 5]
            front_x, front_z = summary[i, 6], summary[i, 7]
            side_y, side_z = summary[i, 8], summary[i, 9]
            
            f.write(f"{idx:3d} | ({x:7.3f},{y:7.3f},{z:7.3f}) | "
                    f"({top_x:6.2f},{top_y:6.2f}) | "
                    f"({front_x:6.2f},{front_z:6.2f}) | "
                    f"({side_y:6.2f},{side_z:6.2f})\n")
        
        # Adjacency matrix section
        f.write("\n" + "=" * 70 + "\n")
        f.write("ADJACENCY MATRIX (Columns 10+)\n")
        f.write("=" * 70 + "\n")
        f.write("Edges represented as 1 in matrix[i, 10+j] and "
                "matrix[j, 10+i]\n\n")
        
        adjacency = summary[:, 10:]
        edge_count = 0
        
        f.write("Edge List:\n")
        f.write("-" * 70 + "\n")
        
        # Extract edges from adjacency matrix
        for i in range(n_vertices):
            for j in range(i+1, n_vertices):
                if adjacency[i, j] == 1:
                    edge_count += 1
                    v1 = summary[i, 1:4]
                    v2 = summary[j, 1:4]
                    f.write(f"Edge {edge_count:3d}: V{i:2d} -- V{j:2d}  "
                            f"({v1[0]:7.3f},{v1[1]:7.3f},{v1[2]:7.3f}) → "
                            f"({v2[0]:7.3f},{v2[1]:7.3f},{v2[2]:7.3f})\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Total edges: {edge_count}\n")
        f.write("=" * 70 + "\n")


def save_summary_to_numpy(summary, filename="unified_summary.npy"):
    """Save the unified summary as a NumPy binary file."""
    np.save(filename, summary)
    print(f"Saved unified summary to {filename}")


def load_summary_from_numpy(filename="unified_summary.npy"):
    """Load the unified summary from a NumPy binary file."""
    return np.load(filename)


def visualize_adjacency_matrix(summary, filename="adjacency_matrix.png"):
    """
    Visualize the adjacency matrix as a heatmap.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        
        adjacency = summary[:, 10:]
        n = adjacency.shape[0]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(adjacency, cmap='binary', interpolation='nearest')
        
        ax.set_xlabel('Vertex Index')
        ax.set_ylabel('Vertex Index')
        ax.set_title(f'Adjacency Matrix ({n}×{n} vertices)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add grid
        ax.set_xticks(np.arange(0, n, 5))
        ax.set_yticks(np.arange(0, n, 5))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Saved adjacency matrix visualization to {filename}")
        plt.close()
        
    except ImportError:
        print("Matplotlib not available, skipping visualization")


if __name__ == "__main__":
    # Example usage
    print("This module provides functions to create unified summary arrays.")
    print("Import and use in your main script.")
