"""
Check for collinear edges - multiple edges on the same line.
For X-parallel edges, they project to a single point in Side view.
"""
import numpy as np

unified = np.load('unified_summary.npy')
top_summary = np.load('top_view_summary.npy')
front_summary = np.load('front_view_summary.npy')

print("="*70)
print("COLLINEAR EDGE ANALYSIS")
print("="*70)

# Find all vertices on the line y=32.824, z=40.828 (X-parallel line)
tolerance = 1e-5
y_target = 32.823690
z_target = 40.828081

print(f"\nSearching for vertices on X-parallel line:")
print(f"  y = {y_target:.6f}")
print(f"  z = {z_target:.6f}")

vertices_on_line = []
for i in range(len(unified)):
    v = unified[i, 1:4]
    if abs(v[1] - y_target) < tolerance and abs(v[2] - z_target) < tolerance:
        vertices_on_line.append((i, v[0]))

# Sort by x coordinate
vertices_on_line.sort(key=lambda x: x[1])

print(f"\nFound {len(vertices_on_line)} vertices on this line:")
for idx, x_coord in vertices_on_line:
    print(f"  V{idx}: x = {x_coord:.6f}")

# Check which edges exist between these vertices
print(f"\nChecking edges between consecutive vertices:")
adj = unified[:, 10:]

for i in range(len(vertices_on_line) - 1):
    v1_idx, v1_x = vertices_on_line[i]
    v2_idx, v2_x = vertices_on_line[i + 1]
    
    edge_exists = adj[v1_idx, v2_idx] > 0 or adj[v2_idx, v1_idx] > 0
    
    print(f"  V{v1_idx} (x={v1_x:.3f}) → V{v2_idx} (x={v2_x:.3f}): "
          f"{'EXISTS ✓' if edge_exists else 'MISSING ✗'}")

# Check in Top view
print(f"\n{'='*70}")
print("TOP VIEW ANALYSIS")
print("="*70)
print("In Top view, all these vertices project to same (x,y).")
print("But Top view should see them as SEPARATE points (different x).")
print("\nChecking Top view connectivity:")

for i in range(len(vertices_on_line) - 1):
    v1_idx, v1_x = vertices_on_line[i]
    v2_idx, v2_x = vertices_on_line[i + 1]
    
    v1_3d = unified[v1_idx, 1:4]
    v2_3d = unified[v2_idx, 1:4]
    
    # Find in top view by x,y projection
    v1_matches = []
    v2_matches = []
    
    for j in range(len(top_summary)):
        tv = top_summary[j, :3]
        if (abs(tv[0] - v1_3d[0]) < tolerance and 
            abs(tv[1] - v1_3d[1]) < tolerance):
            v1_matches.append(j)
        if (abs(tv[0] - v2_3d[0]) < tolerance and 
            abs(tv[1] - v2_3d[1]) < tolerance):
            v2_matches.append(j)
    
    # Check connectivity in top view
    top_conn = False
    for j1 in v1_matches:
        for j2 in v2_matches:
            if 6 + j2 < top_summary.shape[1]:
                if top_summary[j1, 6 + j2] > 0:
                    top_conn = True
                    break
            if 6 + j1 < top_summary.shape[1]:
                if top_summary[j2, 6 + j1] > 0:
                    top_conn = True
                    break
        if top_conn:
            break
    
    print(f"  V{v1_idx}→V{v2_idx}: "
          f"Top view {'FOUND ✓' if top_conn else 'MISSING ✗'}")

# Check in Front view
print(f"\n{'='*70}")
print("FRONT VIEW ANALYSIS")
print("="*70)
print("In Front view, all these vertices project to same (x,z).")
print("They should appear as SEPARATE points (different x).")
print("\nChecking Front view connectivity:")

for i in range(len(vertices_on_line) - 1):
    v1_idx, v1_x = vertices_on_line[i]
    v2_idx, v2_x = vertices_on_line[i + 1]
    
    v1_3d = unified[v1_idx, 1:4]
    v2_3d = unified[v2_idx, 1:4]
    
    # Find in front view by x,z projection
    v1_matches = []
    v2_matches = []
    
    for j in range(len(front_summary)):
        fv = front_summary[j, :3]
        if (abs(fv[0] - v1_3d[0]) < tolerance and 
            abs(fv[2] - v1_3d[2]) < tolerance):
            v1_matches.append(j)
        if (abs(fv[0] - v2_3d[0]) < tolerance and 
            abs(fv[2] - v2_3d[2]) < tolerance):
            v2_matches.append(j)
    
    print(f"  V{v1_idx} (x={v1_x:.3f}): matches in front view = {v1_matches}")
    print(f"  V{v2_idx} (x={v2_x:.3f}): matches in front view = {v2_matches}")
    
    # Check connectivity in front view
    front_conn = False
    for j1 in v1_matches:
        for j2 in v2_matches:
            if 6 + j2 < front_summary.shape[1]:
                if front_summary[j1, 6 + j2] > 0:
                    front_conn = True
                    print(f"    → Connection found: [{j1}, 6+{j2}] = "
                          f"{front_summary[j1, 6 + j2]}")
                    break
            if 6 + j1 < front_summary.shape[1]:
                if front_summary[j2, 6 + j1] > 0:
                    front_conn = True
                    print(f"    → Connection found: [{j2}, 6+{j1}] = "
                          f"{front_summary[j2, 6 + j1]}")
                    break
        if front_conn:
            break
    
    print(f"  V{v1_idx}→V{v2_idx}: "
          f"Front view {'FOUND ✓' if front_conn else 'MISSING ✗'}")

print(f"\n{'='*70}")
print("CONCLUSION")
print("="*70)
print("If edges exist in Top view but NOT in Front view,")
print("the issue is in how Front view extracts/stores edges.")
print("Possible causes:")
print("1. Front view projection merges vertices with same (x,z)")
print("2. Front view only keeps ONE edge between projection points")
print("3. Hidden line removal is too aggressive in Front view")
