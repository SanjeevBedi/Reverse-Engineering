"""
Demonstration script showing how to use the unified summary array.
"""

import numpy as np


def load_and_analyze_unified_summary(filename="unified_summary.npy"):
    """Load and analyze the unified summary array."""
    
    print("=" * 70)
    print("UNIFIED SUMMARY ARRAY ANALYSIS")
    print("=" * 70)
    
    # Load the array
    summary = np.load(filename)
    n_vertices = summary.shape[0]
    n_cols = summary.shape[1]
    
    print(f"\nArray shape: {summary.shape}")
    print(f"Vertices: {n_vertices}")
    print(f"Total columns: {n_cols}")
    
    # Analyze structure
    print("\n" + "-" * 70)
    print("STRUCTURE VERIFICATION")
    print("-" * 70)
    
    # Check vertex indices
    indices = summary[:, 0]
    print(f"Vertex indices: 0 to {int(indices[-1])}")
    print(f"  All sequential: {np.all(indices == np.arange(n_vertices))}")
    
    # Check 3D coordinates
    coords_3d = summary[:, 1:4]
    print(f"\n3D Coordinates:")
    print(f"  X range: [{coords_3d[:, 0].min():.3f}, "
          f"{coords_3d[:, 0].max():.3f}]")
    print(f"  Y range: [{coords_3d[:, 1].min():.3f}, "
          f"{coords_3d[:, 1].max():.3f}]")
    print(f"  Z range: [{coords_3d[:, 2].min():.3f}, "
          f"{coords_3d[:, 2].max():.3f}]")
    
    # Check projections
    top_proj = summary[:, 4:6]
    front_proj = summary[:, 6:8]
    side_proj = summary[:, 8:10]
    
    print(f"\nTop view projection (x,y):")
    print(f"  Matches 3D x: {np.allclose(top_proj[:, 0], coords_3d[:, 0])}")
    print(f"  Matches 3D y: {np.allclose(top_proj[:, 1], coords_3d[:, 1])}")
    
    print(f"\nFront view projection (x,z):")
    print(f"  Matches 3D x: {np.allclose(front_proj[:, 0], coords_3d[:, 0])}")
    print(f"  Matches 3D z: {np.allclose(front_proj[:, 1], coords_3d[:, 2])}")
    
    print(f"\nSide view projection (y,z):")
    print(f"  Matches 3D y: {np.allclose(side_proj[:, 0], coords_3d[:, 1])}")
    print(f"  Matches 3D z: {np.allclose(side_proj[:, 1], coords_3d[:, 2])}")
    
    # Analyze adjacency matrix
    print("\n" + "-" * 70)
    print("ADJACENCY MATRIX ANALYSIS")
    print("-" * 70)
    
    adjacency = summary[:, 10:]
    
    # Check symmetry
    is_symmetric = np.allclose(adjacency, adjacency.T)
    print(f"Matrix is symmetric: {is_symmetric}")
    
    # Count edges
    n_edges = int(np.sum(adjacency) / 2)
    print(f"Total edges: {n_edges}")
    
    # Degree distribution (number of edges per vertex)
    degrees = np.sum(adjacency, axis=1).astype(int)
    print(f"\nVertex degree statistics:")
    print(f"  Min degree: {degrees.min()}")
    print(f"  Max degree: {degrees.max()}")
    print(f"  Mean degree: {degrees.mean():.2f}")
    
    # Find vertices with specific degrees
    print(f"\nDegree distribution:")
    unique_degrees = np.unique(degrees)
    for deg in unique_degrees:
        count = np.sum(degrees == deg)
        print(f"  Degree {deg}: {count} vertices")
    
    # Example: Find all neighbors of vertex 0
    print("\n" + "-" * 70)
    print("EXAMPLE QUERIES")
    print("-" * 70)
    
    vertex_id = 0
    neighbors = np.where(adjacency[vertex_id] == 1)[0]
    print(f"\nVertex {vertex_id} has {len(neighbors)} neighbors: {neighbors}")
    print(f"Vertex {vertex_id} coordinates: "
          f"({coords_3d[vertex_id, 0]:.3f}, "
          f"{coords_3d[vertex_id, 1]:.3f}, "
          f"{coords_3d[vertex_id, 2]:.3f})")
    
    for n in neighbors:
        print(f"  → Vertex {n}: ({coords_3d[n, 0]:.3f}, "
              f"{coords_3d[n, 1]:.3f}, {coords_3d[n, 2]:.3f})")
    
    # Check if specific edge exists
    print("\n" + "-" * 70)
    print("EDGE EXISTENCE QUERIES")
    print("-" * 70)
    
    test_pairs = [(0, 10), (0, 32), (5, 20)]
    for v1, v2 in test_pairs:
        if v1 < n_vertices and v2 < n_vertices:
            exists = adjacency[v1, v2] == 1
            status = "EXISTS" if exists else "DOES NOT EXIST"
            print(f"Edge V{v1} -- V{v2}: {status}")
    
    print("\n" + "=" * 70)
    
    return summary


def extract_edges_from_summary(summary):
    """Extract list of edges from unified summary array."""
    n_vertices = summary.shape[0]
    adjacency = summary[:, 10:]
    
    edges = []
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if adjacency[i, j] == 1:
                v1 = summary[i, 1:4]
                v2 = summary[j, 1:4]
                edges.append((i, j, v1, v2))
    
    return edges


def find_shortest_path_bfs(summary, start_vertex, end_vertex):
    """
    Find shortest path between two vertices using BFS.
    Returns list of vertex indices forming the path.
    """
    from collections import deque
    
    n_vertices = summary.shape[0]
    adjacency = summary[:, 10:]
    
    if start_vertex == end_vertex:
        return [start_vertex]
    
    # BFS
    visited = {start_vertex}
    queue = deque([(start_vertex, [start_vertex])])
    
    while queue:
        current, path = queue.popleft()
        
        # Get neighbors
        neighbors = np.where(adjacency[current] == 1)[0]
        
        for neighbor in neighbors:
            if neighbor == end_vertex:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None  # No path found


if __name__ == "__main__":
    # Load and analyze the unified summary
    summary = load_and_analyze_unified_summary("unified_summary.npy")
    
    # Demonstrate edge extraction
    print("\n" + "=" * 70)
    print("EXTRACTING EDGES FROM SUMMARY")
    print("=" * 70)
    edges = extract_edges_from_summary(summary)
    print(f"Extracted {len(edges)} edges from adjacency matrix")
    print("\nFirst 5 edges:")
    for i, (v1, v2, coord1, coord2) in enumerate(edges[:5], 1):
        print(f"  {i}. V{v1} -- V{v2}: "
              f"({coord1[0]:.2f},{coord1[1]:.2f},{coord1[2]:.2f}) → "
              f"({coord2[0]:.2f},{coord2[1]:.2f},{coord2[2]:.2f})")
    
    # Demonstrate path finding
    print("\n" + "=" * 70)
    print("SHORTEST PATH FINDING")
    print("=" * 70)
    
    start, end = 0, 13
    path = find_shortest_path_bfs(summary, start, end)
    
    if path:
        print(f"\nShortest path from V{start} to V{end}:")
        print(f"  Path length: {len(path) - 1} edges")
        print(f"  Path: {' → '.join(f'V{v}' for v in path)}")
        
        print("\nPath coordinates:")
        for v in path:
            coords = summary[v, 1:4]
            print(f"  V{v}: ({coords[0]:.3f}, {coords[1]:.3f}, "
                  f"{coords[2]:.3f})")
    else:
        print(f"\nNo path found from V{start} to V{end}")
    
    print("\n" + "=" * 70)
