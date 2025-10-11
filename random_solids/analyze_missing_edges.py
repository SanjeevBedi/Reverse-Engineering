"""
Analyze why V40--V24 and V44--V114 edges were not detected.
These are X-PARALLEL edges visible in Top and Front views.
"""
import numpy as np

# Load unified summary
unified = np.load('unified_summary.npy')

print("="*70)
print("ANALYSIS: WHY X-PARALLEL EDGES V40--V24 AND V44--V114 ARE MISSING")
print("="*70)

# Edge 1: V40 -- V24
print("\n" + "="*70)
print("Edge 1: V40 -- V24")
print("="*70)
v40 = unified[40, 1:4]
v24 = unified[24, 1:4]
print(f"V40: ({v40[0]:.6f}, {v40[1]:.6f}, {v40[2]:.6f})")
print(f"V24: ({v24[0]:.6f}, {v24[1]:.6f}, {v24[2]:.6f})")
print(f"Edge type: X-PARALLEL (same y={v40[1]:.6f}, same z={v40[2]:.6f})")
print(f"Requires: Top View (x,y projection) AND Front View (x,z projection)")

# Check adjacency in unified summary
adj = unified[:, 10:]
print(f"\nIn unified summary adjacency: {adj[40, 24] > 0 or adj[24, 40] > 0}")

# Check nearby vertices at same y,z to understand the structure
print("\nOther vertices at y=32.824, z=40.828:")
tolerance = 1e-5
for i in range(len(unified)):
    vi = unified[i, 1:4]
    if abs(vi[1] - v40[1]) < tolerance and abs(vi[2] - v40[2]) < tolerance:
        print(f"  V{i}: x={vi[0]:.6f}")

# Check connections
print(f"\nV40 connects to:")
for i in range(len(unified)):
    if adj[40, i] > 0:
        vi = unified[i, 1:4]
        print(f"  V{i}: ({vi[0]:.6f}, {vi[1]:.6f}, {vi[2]:.6f})")

print(f"\nV24 connects to:")
for i in range(len(unified)):
    if adj[24, i] > 0:
        vi = unified[i, 1:4]
        print(f"  V{i}: ({vi[0]:.6f}, {vi[1]:.6f}, {vi[2]:.6f})")

# Edge 2: V44 -- V114
print("\n" + "="*70)
print("Edge 2: V44 -- V114")
print("="*70)
v44 = unified[44, 1:4]
v114 = unified[114, 1:4]
print(f"V44: ({v44[0]:.6f}, {v44[1]:.6f}, {v44[2]:.6f})")
print(f"V114: ({v114[0]:.6f}, {v114[1]:.6f}, {v114[2]:.6f})")
print(f"Edge type: X-PARALLEL (same y={v44[1]:.6f}, same z={v44[2]:.6f})")
print(f"Requires: Top View (x,y projection) AND Front View (x,z projection)")

print(f"\nIn unified summary adjacency: {adj[44, 114] > 0 or adj[114, 44] > 0}")

print(f"\nV44 connects to:")
for i in range(len(unified)):
    if adj[44, i] > 0:
        vi = unified[i, 1:4]
        print(f"  V{i}: ({vi[0]:.6f}, {vi[1]:.6f}, {vi[2]:.6f})")

print(f"\nV114 connects to:")
for i in range(len(unified)):
    if adj[114, i] > 0:
        vi = unified[i, 1:4]
        print(f"  V{i}: ({vi[0]:.6f}, {vi[1]:.6f}, {vi[2]:.6f})")

print("\n" + "="*70)
print("HYPOTHESIS")
print("="*70)
print("These X-PARALLEL edges should be detected in Phase 1 (Top View)")
print("or Phase 2 (Front View) of the edge reconstruction algorithm.")
print("\nPossible reasons they weren't detected:")
print("1. They don't exist in the view summary connectivity matrices")
print("2. The vertices don't match by projection in the view summaries")
print("3. The edge validation logic is too restrictive")
print("4. They were skipped because they failed a visibility check")
print("\nNext step: Check if these edges exist in the ORIGINAL solid")
print("by examining the OpenCASCADE edge topology.")
