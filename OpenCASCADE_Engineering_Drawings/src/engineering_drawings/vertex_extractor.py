"""
Vertex extraction module for OpenCASCADE Engineering Drawings Generator.

This module handles the extraction of vertices from faces with proper orientation.
"""

import numpy as np
from typing import List, Optional

try:
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import (TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX,
                                 TopAbs_FORWARD, TopAbs_REVERSED)
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import topods
    
    # Try to import TopExp for enhanced vertex extraction
    try:
        from OCC.Core.TopExp import topexp
        from OCC.Core.TopoDS import TopoDS_Vertex
        TOPEXP_AVAILABLE = True
    except ImportError:
        TOPEXP_AVAILABLE = False
    
    OPENCASCADE_AVAILABLE = True
except ImportError as e:
    print(f"OpenCASCADE not available: {e}")
    OPENCASCADE_AVAILABLE = False
    TOPEXP_AVAILABLE = False


class VertexExtractor:
    """
    Extracts vertices from OpenCASCADE faces with proper orientation handling.
    
    This class provides methods to:
    - Extract vertices from faces in proper sequence
    - Handle edge orientation (FORWARD/REVERSED)
    - Maintain topological consistency
    - Remove duplicate vertices while preserving order
    """
    
    def __init__(self):
        """Initialize the vertex extractor."""
        if not OPENCASCADE_AVAILABLE:
            raise ImportError("OpenCASCADE is required for vertex extraction")
    
    def extract_face_vertices(self, face) -> List[List[float]]:
        """
        Extract vertices from a face using proper wire-based topology.
        
        Each face contains:
        - One outer wire (boundary loop)
        - Zero or more inner wires (holes/cutouts)
        
        Wire orientation determines edge traversal direction.
        
        Args:
            face: OpenCASCADE face object
            
        Returns:
            List of [x, y, z] vertex coordinates in proper order
        """
        vertices = []
        
        try:
            # Get face orientation for debugging
            face_orientation = face.Orientation()
            face_orientation_str = (
                "REVERSED" if face_orientation == TopAbs_REVERSED
                else "FORWARD"
            )
            print(f"        Face orientation: {face_orientation_str}")
            
            # Process all wires in the face
            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
            wire_count = 0
            
            while wire_explorer.More():
                wire = wire_explorer.Current()
                wire_count += 1
                
                # Get wire orientation - this is crucial for edge traversal
                wire_orientation = wire.Orientation()
                wire_orientation_str = (
                    "REVERSED" if wire_orientation == TopAbs_REVERSED
                    else "FORWARD"
                )
                
                print(f"        Wire {wire_count}: "
                      f"orientation = {wire_orientation_str}")
                
                # First wire is always the outer boundary
                if wire_count == 1:
                    print("        Processing outer boundary wire...")
                    vertices = self.extract_wire_vertices(wire, wire_count)
                    
                    if vertices:
                        print(f"        ✓ Extracted {len(vertices)} vertices "
                              f"from outer wire")
                        break  # Use only outer boundary for now
                    else:
                        print("        ✗ Failed to extract vertices "
                              "from outer wire")
                else:
                    # Inner wires are holes/cutouts - skip for basic impl
                    print(f"        Skipping inner wire {wire_count} "
                          f"(hole/cutout)")
                
                wire_explorer.Next()
            
            # If no vertices extracted, use fallback
            if not vertices:
                print("        Wire-based extraction failed, using fallback")
                vertices = self._extract_vertices_fallback(face)
                
        except Exception as e:
            print(f"      Error extracting vertices: {e}")
            vertices = []
        
        return vertices
    
    def extract_wire_vertices(self, wire, wire_id: int) -> List[List[float]]:
        """
        Extract vertices from a wire by traversing edges in sequence.
        
        This method handles proper edge orientation:
        - FORWARD edges: start → end vertex order
        - REVERSED edges: end → start vertex order
        
        Args:
            wire: OpenCASCADE wire object
            wire_id: Wire identifier for debugging
            
        Returns:
            List of [x, y, z] vertex coordinates in proper order
        """
        vertices = []
        
        try:
            print(f"          Traversing Wire {wire_id} edges...")
            
            # Method 1: Enhanced orientation method if TopExp available
            if TOPEXP_AVAILABLE:
                print("            ✓ TopExp available - " +
                      "using enhanced orientation method")
                vertices = self._extract_with_topexp(wire, wire_id)
            
            # Fallback: Basic edge traversal
            if not vertices:
                print("            Using fallback edge traversal method...")
                vertices = self._extract_basic_traversal(wire, wire_id)
                
        except Exception as e:
            print(f"          ✗ Error extracting vertices from "
                  f"wire {wire_id}: {e}")
            vertices = []
        
        return vertices
    
    def _extract_with_topexp(self, wire, wire_id: int) -> List[List[float]]:
        """Extract vertices using complete orientation hierarchy."""
        try:
            # Get wire orientation
            wire_orientation = wire.Orientation()
            is_wire_reversed = (wire_orientation == TopAbs_REVERSED)
            
            print(f"            Wire {wire_id} orientation: "
                  f"{'REVERSED' if is_wire_reversed else 'FORWARD'}")
            
            # Get edges in wire order
            edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
            edges = []
            
            while edge_explorer.More():
                edges.append(edge_explorer.Current())
                edge_explorer.Next()
            
            print(f"            Found {len(edges)} edges in wire {wire_id}")
            
            if not edges:
                return []
            
            # CRITICAL: Don't reverse edge order for wire orientation
            # The wire orientation affects the overall polygon orientation,
            # but edge connectivity should be preserved
            print("            Processing edges in natural wire order")
            
            # Build vertex sequence following edge connectivity
            vertex_sequence = []
            
            for edge_idx, edge in enumerate(edges):
                try:
                    # Extract vertices using topexp
                    if TOPEXP_AVAILABLE:
                        first_vertex = TopoDS_Vertex()
                        last_vertex = TopoDS_Vertex()
                        topexp.Vertices(edge, first_vertex, last_vertex, True)
                    else:
                        return []
                    
                    # Get vertex coordinates from edge
                    first_pnt = BRep_Tool.Pnt(first_vertex)
                    last_pnt = BRep_Tool.Pnt(last_vertex)
                    
                    start_vertex = [first_pnt.X(), first_pnt.Y(),
                                    first_pnt.Z()]
                    end_vertex = [last_pnt.X(), last_pnt.Y(), last_pnt.Z()]
                    
                    # Apply user's simplified orientation logic
                    if not is_wire_reversed:  # Wire FORWARD
                        # Forward wire: always select start vertex
                        selected_vertex = start_vertex
                        orientation_tag = "WF_START"
                    else:  # Wire REVERSED
                        # Reversed wire: always select end vertex
                        selected_vertex = end_vertex
                        orientation_tag = "WR_END"
                    
                    print(f"              Edge {edge_idx}: {orientation_tag} "
                          f"({start_vertex[0]:.1f},{start_vertex[1]:.1f},"
                          f"{start_vertex[2]:.1f}) → "
                          f"({end_vertex[0]:.1f},{end_vertex[1]:.1f},"
                          f"{end_vertex[2]:.1f}) "
                          f"SELECT ({selected_vertex[0]:.1f},"
                          f"{selected_vertex[1]:.1f},"
                          f"{selected_vertex[2]:.1f})")
                    
                    # Add selected vertex to sequence
                    vertex_sequence.append(selected_vertex)
                
                except Exception as e:
                    print(f"              Edge {edge_idx} "
                          f"extraction error: {e}")
                    return []
            
            # Remove consecutive duplicate vertices while preserving wire loop
            if len(vertex_sequence) > 1:
                cleaned_sequence = [vertex_sequence[0]]  # Keep first vertex
                
                for i in range(1, len(vertex_sequence)):
                    prev_vertex = np.array(cleaned_sequence[-1])
                    curr_vertex = np.array(vertex_sequence[i])
                    
                    # Only add if different from previous vertex
                    if np.linalg.norm(curr_vertex - prev_vertex) > 1e-6:
                        cleaned_sequence.append(vertex_sequence[i])
                
                vertex_sequence = cleaned_sequence
                print(f"            Removed consecutive duplicates: "
                      f"{len(vertex_sequence)} vertices remaining")
            
            # Duplicate first vertex at the end to close the wire
            if len(vertex_sequence) >= 1:
                first_vertex = vertex_sequence[0]
                vertex_sequence.append(first_vertex)
                print("            Added duplicate first vertex to close wire")
            
            # Note: Wire orientation is already handled in vertex selection
            # No additional reversal needed
            print(f"            ✓ Wire connectivity sequence: "
                  f"{len(vertex_sequence)} vertices")
            
            # Display final sequence
            if vertex_sequence:
                vertex_coords = " → ".join([
                    f"({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})"
                    for v in vertex_sequence
                ])
                print(f"            FINAL: {vertex_coords}")
            
            return vertex_sequence
            
        except Exception as e:
            print(f"            TopExp wire extraction error: {e}")
            return []
            
        except Exception as e:
            print(f"            ✗ Enhanced TopExp traversal failed: {e}")
            return []
    
    def _extract_basic_traversal(
            self, wire, wire_id: int) -> List[List[float]]:
        """Basic edge traversal fallback method."""
        edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
        vertex_list = []
        
        while edge_explorer.More():
            edge = edge_explorer.Current()
            
            # Get vertices from edge
            vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
            edge_vertices = []
            
            while vertex_explorer.More():
                vertex = topods.Vertex(vertex_explorer.Current())
                pnt = BRep_Tool.Pnt(vertex)
                v = [pnt.X(), pnt.Y(), pnt.Z()]
                edge_vertices.append(v)
                vertex_explorer.Next()
            
            vertex_list.extend(edge_vertices)
            edge_explorer.Next()
        
        # Remove duplicates while preserving order
        vertices = []
        seen = set()
        for v in vertex_list:
            v_tuple = tuple(np.round(v, 6))
            if v_tuple not in seen:
                vertices.append(v)
                seen.add(v_tuple)
        
        print(f"            ✓ Fallback method: {len(vertices)} vertices")
        
        # Print vertices for debugging
        if vertices:
            vertex_coords = " → ".join([f"({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})"
                                        for v in vertices])
            print(f"            VERTEX ORDER: {vertex_coords}")
        
        return vertices
    
    def _extract_vertices_fallback(self, face) -> List[List[float]]:
        """Extract vertices using face bounding box as fallback."""
        try:
            from OCC.Core.Bnd import Bnd_Box
            from OCC.Core.BRepBndLib import brepbndlib
            
            bbox = Bnd_Box()
            brepbndlib.Add(face, bbox)
            
            if not bbox.IsVoid():
                xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
                
                # Determine which plane the face lies in
                x_range = xmax - xmin
                y_range = ymax - ymin
                z_range = zmax - zmin
                
                tolerance = 1e-6
                
                if x_range < tolerance:  # X-normal face (YZ plane)
                    x = (xmin + xmax) / 2
                    return [[x, ymin, zmin], [x, ymax, zmin],
                            [x, ymax, zmax], [x, ymin, zmax]]
                elif y_range < tolerance:  # Y-normal face (XZ plane)
                    y = (ymin + ymax) / 2
                    return [[xmin, y, zmin], [xmax, y, zmin],
                            [xmax, y, zmax], [xmin, y, zmax]]
                elif z_range < tolerance:  # Z-normal face (XY plane)
                    z = (zmin + zmax) / 2
                    return [[xmin, ymin, z], [xmax, ymin, z],
                            [xmax, ymax, z], [xmin, ymax, z]]
            
        except Exception as e:
            print(f"      Error creating rectangular fallback: {e}")
        
        return []
