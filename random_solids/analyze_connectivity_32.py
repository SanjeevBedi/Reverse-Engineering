#!/usr/bin/env python3
"""
Analyze connectivity matrix for seed 32 to understand which edges exist
"""

import numpy as np
import sys

def main():
    seed = 32
    
    # Load connectivity matrices
    filename = f"Output/connectivity_matrices_seed_{seed}.npz"
    data = np.load(filename, allow_pickle=True)
    
    top_matrix = data['top_view_matrix']
    front_matrix = data['front_view_matrix']
    side_matrix = data['side_view_matrix']
    all_vertices = data['all_vertices']
    
    print("="*70)
    print(f"CONNECTIVITY MATRIX ANALYSIS FOR SEED {seed}")
    print("="*70)
    
    print(f"\nTop view: {top_matrix.shape[0]} vertices")
    print(f"Front view: {front_matrix.shape[0]} vertices") 
    print(f"Side view: {side_matrix.shape[0]} vertices")
    print(f"All vertices: {len(all_vertices)}")
    
    # Count edges in each view
    def count_edges(matrix):
        n = matrix.shape[0]
        if matrix.shape[1] < 3 + n:
            return 0
        conn = matrix[:, 3:]
        edges = np.sum(conn > 0) // 2
        return edges
    
    top_edges = count_edges(top_matrix)
    front_edges = count_edges(front_matrix)
    side_edges = count_edges(side_matrix)
    
    print(f"\nEdges in Top view: {top_edges}")
    print(f"Edges in Front view: {front_edges}")
    print(f"Edges in Side view: {side_edges}")
    
    print("\n" + "="*70)
    print("This explains why some faces are missing:")
    print("Faces that are nearly parallel to a view direction will have")
    print("few or no edges visible in that view, resulting in conn < 3")
    print("="*70)

if __name__ == "__main__":
    main()
