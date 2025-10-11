"""
Check if V40--V24 and V44--V114 edges exist in view summary connectivity matrices.
"""
import numpy as np

# Load all summaries
unified = np.load('unified_summary.npy')
top_summary = np.load('top_view_summary.npy')
front_summary = np.load('front_view_summary.npy')
side_summary = np.load('side_view_summary.npy')

print("="*70)
print("CHECKING MISSING X-PARALLEL EDGES IN VIEW SUMMARIES")
print("="*70)

tolerance = 1e-5

def check_edge_in_view(v1_3d, v2_3d, view_summary, view_name, match_indices):
    """Check if edge exists in a view's connectivity matrix."""
    print(f"\n  {view_name} View (matching indices {match_indices}):")
    
    # Find vertices matching the projections
    v1_proj = [v1_3d[i] for i in match_indices]
    v2_proj = [v2_3d[i] for i in match_indices]
    
    print(f"    V1 projection: {v1_proj}")
    print(f"    V2 projection: {v2_proj}")
    
    v1_matches = []
    v2_matches = []
    
    for i in range(len(view_summary)):
        vertex_3d = view_summary[i, :3]
        vertex_proj = [vertex_3d[idx] for idx in match_indices]
        
        v1_dist = np.linalg.norm(np.array(vertex_proj) - np.array(v1_proj))
        v2_dist = np.linalg.norm(np.array(vertex_proj) - np.array(v2_proj))
        
        if v1_dist < tolerance:
            v1_matches.append(i)
        if v2_dist < tolerance:
            v2_matches.append(i)
    
    print(f"    V1 found at rows: {v1_matches[:5]}")
    print(f"    V2 found at rows: {v2_matches[:5]}")
    
    if not v1_matches or not v2_matches:
        print(f"    ✗ Vertices not found in {view_name} view")
        return False
    
    # Check connectivity matrix (starts at column 6)
    connectivity_start = 6
    found = False
    
    for i in v1_matches[:3]:
        for j in v2_matches[:3]:
            if i == j:
                continue
            
            # Check both directions
            if connectivity_start + j < view_summary.shape[1]:
                conn = view_summary[i, connectivity_start + j]
                if conn > 0:
                    print(f"    ✓ EDGE FOUND: row {i} → col {j}, "
                          f"connectivity = {conn}")
                    found = True
                    break
            
            if connectivity_start + i < view_summary.shape[1]:
                conn = view_summary[j, connectivity_start + i]
                if conn > 0 and not found:
                    print(f"    ✓ EDGE FOUND: row {j} → col {i}, "
                          f"connectivity = {conn}")
                    found = True
                    break
        if found:
            break
    
    if not found:
        print(f"    ✗ NO CONNECTIVITY in {view_name} view matrix")
    
    return found


# Check Edge 1: V40 -- V24
print("\n" + "="*70)
print("Edge 1: V40 -- V24 (X-PARALLEL)")
print("="*70)

v40 = unified[40, 1:4]
v24 = unified[24, 1:4]

print(f"V40: ({v40[0]:.6f}, {v40[1]:.6f}, {v40[2]:.6f})")
print(f"V24: ({v24[0]:.6f}, {v24[1]:.6f}, {v24[2]:.6f})")
print(f"Edge type: X-PARALLEL (same y, same z)")
print(f"Required views: Top (x,y) and Front (x,z)")

top_found = check_edge_in_view(v40, v24, top_summary, "Top", [0, 1])
front_found = check_edge_in_view(v40, v24, front_summary, "Front", [0, 2])

print(f"\n  SUMMARY for V40--V24:")
print(f"    Top view: {'FOUND ✓' if top_found else 'MISSING ✗'}")
print(f"    Front view: {'FOUND ✓' if front_found else 'MISSING ✗'}")
if top_found and front_found:
    print(f"    → Should be RECONSTRUCTED")
else:
    print(f"    → CANNOT be reconstructed (missing from view(s))")


# Check Edge 2: V44 -- V114
print("\n" + "="*70)
print("Edge 2: V44 -- V114 (X-PARALLEL)")
print("="*70)

v44 = unified[44, 1:4]
v114 = unified[114, 1:4]

print(f"V44: ({v44[0]:.6f}, {v44[1]:.6f}, {v44[2]:.6f})")
print(f"V114: ({v114[0]:.6f}, {v114[1]:.6f}, {v114[2]:.6f})")
print(f"Edge type: X-PARALLEL (same y, same z)")
print(f"Required views: Top (x,y) and Front (x,z)")

top_found = check_edge_in_view(v44, v114, top_summary, "Top", [0, 1])
front_found = check_edge_in_view(v44, v114, front_summary, "Front", [0, 2])

print(f"\n  SUMMARY for V44--V114:")
print(f"    Top view: {'FOUND ✓' if top_found else 'MISSING ✗'}")
print(f"    Front view: {'FOUND ✓' if front_found else 'MISSING ✗'}")
if top_found and front_found:
    print(f"    → Should be RECONSTRUCTED")
else:
    print(f"    → CANNOT be reconstructed (missing from view(s))")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("If edges are missing from the view connectivity matrices,")
print("they don't exist in the original solid's edge topology.")
print("This means they are NOT real edges - they are visual artifacts")
print("from looking at the projection, or they lie on face interiors.")
