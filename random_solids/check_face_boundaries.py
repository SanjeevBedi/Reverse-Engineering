"""
Check if outer_boundary in face_polygons contains all vertices including
those on collinear edges.
"""
import numpy as np

# We need to check what faces exist in the solid
# and whether their boundaries include all vertices

print("This analysis requires running V6_current.py with debug output")
print("to see the face boundaries being extracted.")
print("")
print("The key question: Does extract_wire_vertices_in_sequence() ")
print("return ALL vertices from the wire, including intermediate vertices")
print("on collinear edges (like V24, V40, V44, V114)?")
print("")
print("If YES: The problem is in how plot_four_views processes boundaries")
print("If NO: The problem is in extract_wire_vertices_in_sequence()")
