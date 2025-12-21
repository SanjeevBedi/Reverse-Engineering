"""
Face analysis module for OpenCASCADE Engineering Drawings Generator.

This module handles the analysis of face geometry and topology.
"""

import numpy as np
from typing import List, Dict, Any, Optional

try:
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import (TopAbs_FACE, TopAbs_SHELL, TopAbs_WIRE, 
                                 TopAbs_EDGE, TopAbs_VERTEX)
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, 
                                  GeomAbs_Sphere, GeomAbs_Cone)
    
    OPENCASCADE_AVAILABLE = True
except ImportError as e:
    print(f"OpenCASCADE not available: {e}")
    OPENCASCADE_AVAILABLE = False


class FaceAnalyzer:
    """
    Analyzes face geometry and topology from OpenCASCADE solids.
    
    This class provides methods to:
    - Extract faces from solids
    - Classify face types (planar, curved, complex)
    - Extract face normals and geometric properties
    - Validate face topology
    """
    
    def __init__(self):
        """Initialize the face analyzer."""
        if not OPENCASCADE_AVAILABLE:
            raise ImportError("OpenCASCADE is required for face analysis")
        
        self._face_count = 0
        self._shell_count = 0
        self._edge_count = 0
        self._vertex_count = 0
    
    def analyze_topology(self, solid: Any) -> Dict[str, int]:
        """
        Analyze the topology of a solid shape.
        
        Args:
            solid: OpenCASCADE solid to analyze
            
        Returns:
            dict: Dictionary containing topology counts
        """
        if solid is None:
            return {}
        
        print(f"\n" + "="*60)
        print("DETAILED SOLID GEOMETRY ANALYSIS")
        print("="*60)
        
        try:
            # Reset counters
            self._shell_count = 0
            self._face_count = 0
            self._edge_count = 0
            self._vertex_count = 0
            
            # Count shells
            shell_explorer = TopExp_Explorer(solid, TopAbs_SHELL)
            while shell_explorer.More():
                self._shell_count += 1
                shell_explorer.Next()
            
            # Count faces
            face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
            while face_explorer.More():
                self._face_count += 1
                face_explorer.Next()
            
            # Count edges
            edge_explorer = TopExp_Explorer(solid, TopAbs_EDGE)
            while edge_explorer.More():
                self._edge_count += 1
                edge_explorer.Next()
            
            # Count vertices
            vertex_explorer = TopExp_Explorer(solid, TopAbs_VERTEX)
            while vertex_explorer.More():
                self._vertex_count += 1
                vertex_explorer.Next()
            
            print(f"Topological Elements:")
            print(f"  • Shells: {self._shell_count}")
            print(f"  • Faces: {self._face_count}")
            print(f"  • Edges: {self._edge_count}")
            print(f"  • Vertices: {self._vertex_count}")
            
            # Analyze face types
            face_types = self._analyze_face_types(solid)
            
            print(f"\nFace Type Summary:")
            print(f"  • Planar faces: {face_types['planar']}")
            print(f"  • Curved faces: {face_types['curved']}")
            print(f"  • Complex faces: {face_types['complex']}")
            
            # Validate topology
            self._validate_topology()
            
            print(f"\n" + "="*60)
            print("GEOMETRY ANALYSIS COMPLETE")
            print("="*60)
            
            return {
                'shells': self._shell_count,
                'faces': self._face_count,
                'edges': self._edge_count,
                'vertices': self._vertex_count,
                'face_types': face_types
            }
            
        except Exception as e:
            print(f"✗ Solid analysis failed: {e}")
            return {}
    
    def _analyze_face_types(self, solid: Any) -> Dict[str, int]:
        """Analyze and classify face types in the solid."""
        face_types = {'planar': 0, 'curved': 0, 'complex': 0}
        
        face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
        face_num = 0
        
        print(f"\nFace Analysis:")
        while face_explorer.More():
            face_shape = face_explorer.Current()
            face_num += 1
            
            try:
                face = topods.Face(face_shape)
                surface = BRepAdaptor_Surface(face)
                
                # Classify surface type using GeomAbs constants
                surface_type = surface.GetType()
                
                # Convert enum to string for display
                type_name = str(surface_type).split('.')[-1] if hasattr(
                    surface_type, '__str__') else 'Unknown'
                
                print(f"  Face {face_num}: {type_name} (type={surface_type})")
                
                # Classify based on surface type
                if surface_type == GeomAbs_Plane:
                    face_types['planar'] += 1
                    print(f"    → Classified as PLANAR")
                elif surface_type in [GeomAbs_Cylinder, GeomAbs_Sphere, 
                                      GeomAbs_Cone]:
                    face_types['curved'] += 1
                    print(f"    → Classified as CURVED")
                else:
                    face_types['complex'] += 1
                    print(f"    → Classified as COMPLEX")
                    
            except Exception as e:
                print(f"  Face {face_num}: Analysis error - {e}")
                face_types['complex'] += 1
            
            face_explorer.Next()
        
        return face_types
    
    def _validate_topology(self):
        """Validate the topology for typical cuboid-based solids."""
        print(f"\nSolid Validation:")
        
        if self._shell_count == 1:
            print(f"  ✓ Single shell - solid is manifold")
        else:
            print(f"  ⚠️  Multiple shells ({self._shell_count}) - "
                  f"may indicate issues")
        
        # Expected face count for cuboid-based solids
        if 6 <= self._face_count <= 12:
            print(f"  ✓ Face count ({self._face_count}) is typical for "
                  f"cuboid-based solids")
        elif self._face_count < 6:
            print(f"  ⚠️  Low face count ({self._face_count}) - "
                  f"unexpected for cuboids")
        else:
            print(f"  ℹ️  High face count ({self._face_count}) - "
                  f"complex boolean result")
    
    def extract_faces_with_vertices(self, solid: Any, 
                                    vertex_extractor) -> List[Dict[str, Any]]:
        """
        Extract faces from solid with their vertex information.
        
        Args:
            solid: OpenCASCADE solid
            vertex_extractor: VertexExtractor instance
            
        Returns:
            List of face data dictionaries
        """
        if solid is None:
            return []
        
        faces = []
        
        print("  Traversing BRep topology: "
              "Solid -> Shells -> Faces -> Wires -> Edges -> Vertices")
        
        # Check shell count
        shell_explorer = TopExp_Explorer(solid, TopAbs_SHELL)
        shell_count = 0
        
        while shell_explorer.More():
            shell_count += 1
            shell_explorer.Next()
        
        print(f"  Found {shell_count} shells in solid")
        
        if shell_count > 2:
            print(f"  ✗ ABORTING: Found {shell_count} shells (expected ≤ 2)")
            print(f"    Complex multi-shell solids not supported")
            return []
        elif shell_count == 2:
            print(f"  ⚠️  WARNING: Found 2 shells - "
                  f"may indicate hollow solid")
        
        # Process shells
        shell_explorer = TopExp_Explorer(solid, TopAbs_SHELL)
        shell_num = 0
        face_count = 0
        
        while shell_explorer.More():
            shell = shell_explorer.Current()
            shell_num += 1
            print(f"  \nShell {shell_num}:")
            
            # Explore faces in shell
            face_explorer = TopExp_Explorer(shell, TopAbs_FACE)
            
            while face_explorer.More():
                face_shape = face_explorer.Current()
                face_count += 1
                
                try:
                    face = topods.Face(face_shape)
                    
                    # Extract vertices using vertex extractor
                    vertices = vertex_extractor.extract_face_vertices(face)
                    
                    # Get face normal
                    face_normal = self._get_face_normal(face)
                    
                    # Determine face type
                    face_type = self._get_face_type(face)
                    
                    if vertices and face_normal is not None:
                        face_data = {
                            'face_id': face_count,
                            'outer_boundary': vertices,
                            'normal': face_normal,
                            'face_type': face_type,
                            'vertex_count': len(vertices)
                        }
                        faces.append(face_data)
                        print(f"    Face {face_count}: ✓ {len(vertices)} "
                              f"vertices, {face_type} type")
                    else:
                        print(f"    Face {face_count}: ✗ Failed to extract "
                              f"geometry")
                
                except Exception as e:
                    print(f"    Face {face_count}: error processing - {e}")
                
                face_explorer.Next()
            
            shell_explorer.Next()
        
        print(f"  \n✓ Successfully extracted {len(faces)} faces from "
              f"{shell_count} shells")
        return faces
    
    def _get_face_normal(self, face: Any) -> Optional[np.ndarray]:
        """Get the face normal vector."""
        try:
            from OCC.Core.GeomLProp import GeomLProp_SLProps
            
            surface = BRepAdaptor_Surface(face)
            
            # Get parameter bounds
            u_min = surface.FirstUParameter()
            u_max = surface.LastUParameter()
            v_min = surface.FirstVParameter()
            v_max = surface.LastVParameter()
            
            # Use parameter center
            u_mid = (u_min + u_max) / 2.0
            v_mid = (v_min + v_max) / 2.0
            
            # Get surface properties
            surface_handle = surface.Surface()
            props = GeomLProp_SLProps(surface_handle, u_mid, v_mid, 1, 1e-6)
            
            if props.IsNormalDefined():
                normal_vec = props.Normal()
                
                # Apply face orientation
                if face.Orientation() == 1:  # TopAbs_REVERSED
                    normal_vec.Reverse()
                
                normal = np.array([normal_vec.X(), normal_vec.Y(), 
                                   normal_vec.Z()])
                return normal / np.linalg.norm(normal)
            
        except Exception as e:
            print(f"        Error getting face normal: {e}")
        
        return None
    
    def _get_face_type(self, face: Any) -> str:
        """Get the type of a face (planar, curved, complex)."""
        try:
            surface = BRepAdaptor_Surface(face)
            surface_type = surface.GetType()
            
            if surface_type == GeomAbs_Plane:
                return 'planar'
            elif surface_type in [GeomAbs_Cylinder, GeomAbs_Sphere, 
                                  GeomAbs_Cone]:
                return 'curved'
            else:
                return 'complex'
                
        except Exception:
            return 'unknown'
