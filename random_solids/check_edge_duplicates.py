"""
Script to check for duplicate edges in the reconstructed edge list
"""
import sys
from collections import Counter

def parse_edges_from_output():
    """Parse edges from the program output"""
    import subprocess
    
    # Run the main program and capture output
    cmd = [
        '/opt/anaconda3/envs/pyocc/bin/python',
        'V6_current.py',
        '--seed', '17',
        '--normal', '1,1,1'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse the reconstructed_edges.txt file instead
    edges = []
    try:
        with open('reconstructed_edges.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('Edge') and 'V' in line and '--' in line:
                    # Format: "Edge  1: V 0 -- V10"
                    try:
                        # Split by colon to get the vertex part
                        vertex_part = line.split(':', 1)[1].strip()
                        # Split by --
                        parts = vertex_part.split('--')
                        # Extract numbers after 'V'
                        v1_str = parts[0].strip().replace('V', '').strip()
                        v2_str = parts[1].strip().replace('V', '').strip()
                        v1 = int(v1_str)
                        v2 = int(v2_str)
                        
                        # Normalize: always store as (min, max)
                        edge = tuple(sorted([v1, v2]))
                        edges.append(edge)
                    except Exception as e:
                        print(f"Failed to parse line: {line}")
                        print(f"Error: {e}")
    except FileNotFoundError:
        print("Error: reconstructed_edges.txt not found")
        return []
    
    return edges

def main():
    print("Analyzing edge reconstruction for duplicates...")
    print("=" * 70)
    
    edges = parse_edges_from_output()
    
    if not edges:
        print("No edges found!")
        return
    
    total_edges = len(edges)
    unique_edges = len(set(edges))
    duplicate_count = total_edges - unique_edges
    
    print(f"\nResults:")
    print(f"  Total edges in list: {total_edges}")
    print(f"  Unique edges: {unique_edges}")
    print(f"  Duplicates: {duplicate_count}")
    
    if duplicate_count > 0:
        print(f"\n⚠️  WARNING: {duplicate_count} duplicate edges found!")
        
        # Count occurrences
        edge_counts = Counter(edges)
        duplicates = [(edge, count) for edge, count in edge_counts.items() if count > 1]
        
        print(f"\nDuplicate edges ({len(duplicates)} unique duplicated pairs):")
        for edge, count in sorted(duplicates):
            print(f"  V{edge[0]:2d} -- V{edge[1]:2d}: appears {count} times")
    else:
        print(f"\n✓ No duplicates found! All {unique_edges} edges are unique.")
    
    print("=" * 70)

if __name__ == '__main__':
    main()
