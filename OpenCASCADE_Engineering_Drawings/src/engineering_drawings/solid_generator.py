"""
Solid generation module for OpenCASCADE Engineering Drawings Generator.

This module handles the creation of 3D solids using OpenCASCADE boolean operations.
"""

import random
import numpy as np
from typing import Any, Tuple, Optional

try:
    from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_FACE, TopAbs_EDGE
    
    OPENCASCADE_AVAILABLE = True
except ImportError as e:
    print(f"OpenCASCADE not available: {e}")
    OPENCASCADE_AVAILABLE = False


class SolidGenerator:
    """
    Handles creation of 3D solids using OpenCASCADE boolean operations.
    
    This class provides methods to:
    - Create basic geometric primitives (cuboids)
    - Apply transformations (translation, rotation)
    - Perform boolean operations (cut, fuse, common)
    - Validate solid integrity
    """
    
    def __init__(self, demo_mode=False):
        """Initialize the solid generator.
        
        Args:
            demo_mode: If True, use pre-computed demo data instead of OpenCASCADE
        """
        self.demo_mode = demo_mode
        if not demo_mode and not OPENCASCADE_AVAILABLE:
            raise ImportError("OpenCASCADE is required for solid generation. Use demo_mode=True for testing without OpenCASCADE.")
    
    def create_cuboid(
        self, 
        width: float, 
        height: float, 
        depth: float,
        origin: Tuple[float, float, float] = (0, 0, 0)
    ) -> Any:
        """
        Create a cuboid (box) with specified dimensions.
        
        Args:
            width: Width of the cuboid (X direction)
            height: Height of the cuboid (Y direction)
            depth: Depth of the cuboid (Z direction)
            origin: Origin point for the cuboid
            
        Returns:
            OpenCASCADE shape representing the cuboid
        """
        try:
            # Create the box at origin
            box_maker = BRepPrimAPI_MakeBox(width, height, depth)
            cuboid = box_maker.Shape()
            
            # Apply translation if origin is not (0,0,0)
            if origin != (0, 0, 0):
                transform = gp_Trsf()
                transform.SetTranslation(gp_Vec(origin[0], origin[1], origin[2]))
                cuboid.Move(TopLoc_Location(transform))
            
            print(f"✓ Created cuboid: {width}x{height}x{depth} at {origin}")
            return cuboid
            
        except Exception as e:
            print(f"✗ Failed to create cuboid: {e}")
            return None
    
    def create_boolean_solid(
        self,
        cuboid1_dims: Tuple[float, float, float] = (10, 20, 30),
        cuboid2_dims: Optional[Tuple[float, float, float]] = None,
        translation: Tuple[float, float, float] = (5, 10, 15),
        operation: str = "cut"
    ) -> Any:
        """
        Create a complex solid using boolean operations between two cuboids.
        
        Args:
            cuboid1_dims: Dimensions (width, height, depth) of first cuboid
            cuboid2_dims: Dimensions of second cuboid (auto-generated if None)
            translation: Translation vector for second cuboid
            operation: Boolean operation ("cut", "fuse", "common")
            
        Returns:
            OpenCASCADE shape representing the boolean result
        """
        print(f"Creating boolean solid with {operation} operation...")
        
        # Create first cuboid
        print(f"Creating first cuboid {cuboid1_dims}...")
        cuboid1 = self.create_cuboid(cuboid1_dims[0], cuboid1_dims[1], cuboid1_dims[2])
        
        if cuboid1 is None:
            return None
        
        # Generate second cuboid dimensions if not provided
        if cuboid2_dims is None:
            random.seed(42)  # For reproducible results
            width2 = random.uniform(8, 15)
            height2 = random.uniform(15, 25)
            depth2 = random.uniform(20, 35)
            cuboid2_dims = (width2, height2, depth2)
        
        print(f"Creating second cuboid {cuboid2_dims}...")
        
        # Create second cuboid at origin
        cuboid2 = self.create_cuboid(cuboid2_dims[0], cuboid2_dims[1], cuboid2_dims[2])
        
        if cuboid2 is None:
            return cuboid1
        
        # Apply translation to second cuboid
        transform = gp_Trsf()
        transform.SetTranslation(gp_Vec(translation[0], translation[1], translation[2]))
        cuboid2.Move(TopLoc_Location(transform))
        print(f"Applied translation {translation} to second cuboid")
        
        # Perform boolean operation
        try:
            if operation.lower() == "cut":
                bool_op = BRepAlgoAPI_Cut(cuboid1, cuboid2)
            elif operation.lower() == "fuse":
                bool_op = BRepAlgoAPI_Fuse(cuboid1, cuboid2)
            elif operation.lower() == "common":
                bool_op = BRepAlgoAPI_Common(cuboid1, cuboid2)
            else:
                print(f"✗ Unknown operation: {operation}. Using 'cut'")
                bool_op = BRepAlgoAPI_Cut(cuboid1, cuboid2)
            
            bool_op.Build()
            
            if bool_op.IsDone() and not bool_op.HasErrors():
                result_shape = bool_op.Shape()
                
                if self.validate_solid(result_shape):
                    print(f"✓ Boolean {operation} operation completed successfully")
                    return result_shape
                else:
                    print(f"✗ Boolean result failed validation, returning first cuboid")
                    return cuboid1
            else:
                print(f"✗ Boolean {operation} operation failed")
                if not bool_op.IsDone():
                    print("  Operation not completed (IsDone = False)")
                if bool_op.HasErrors():
                    print("  Operation has errors (HasErrors = True)")
                print("  Falling back to first cuboid")
                return cuboid1
                
        except Exception as e:
            print(f"✗ Boolean {operation} failed with exception: {e}")
            print("  Falling back to first cuboid")
            return cuboid1
    
    def validate_solid(self, shape: Any) -> bool:
        """
        Validate a solid shape to ensure it's properly formed.
        
        Args:
            shape: OpenCASCADE shape to validate
            
        Returns:
            bool: True if shape is valid, False otherwise
        """
        if shape is None:
            return False
        
        try:
            # Count geometric elements
            shell_count = 0
            face_count = 0
            edge_count = 0
            
            # Count shells - should be exactly 1 for a valid solid
            shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
            while shell_explorer.More():
                shell_count += 1
                shell_explorer.Next()
            
            # Count faces
            face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while face_explorer.More():
                face_count += 1
                face_explorer.Next()
            
            # Count edges
            edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            while edge_explorer.More():
                edge_count += 1
                edge_explorer.Next()
            
            print(f"  Shape validation:")
            print(f"    Shells: {shell_count}")
            print(f"    Faces: {face_count}")
            print(f"    Edges: {edge_count}")
            
            # For a valid solid, we should have exactly 1 shell
            if shell_count != 1:
                print(f"    ✗ Invalid shell count: {shell_count} (expected: 1)")
                return False
            
            # For cuboid-based solids, face count should be reasonable
            if face_count < 6 or face_count > 20:
                print(f"    ⚠️  Unusual face count: {face_count} (typical: 6-12)")
            
            print(f"    ✓ Shape validation passed")
            return True
            
        except Exception as e:
            print(f"    ✗ Shape validation failed: {e}")
            return False
    
    def get_solid_info(self, shape: Any) -> dict:
        """
        Get detailed information about a solid shape.
        
        Args:
            shape: OpenCASCADE shape to analyze
            
        Returns:
            dict: Dictionary containing shape information
        """
        if shape is None:
            return {}
        
        try:
            # Count topological elements
            shell_count = 0
            face_count = 0
            edge_count = 0
            vertex_count = 0
            
            shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
            while shell_explorer.More():
                shell_count += 1
                shell_explorer.Next()
            
            face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while face_explorer.More():
                face_count += 1
                face_explorer.Next()
            
            edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            while edge_explorer.More():
                edge_count += 1
                edge_explorer.Next()
            
            from OCC.Core.TopAbs import TopAbs_VERTEX
            vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
            while vertex_explorer.More():
                vertex_count += 1
                vertex_explorer.Next()
            
            return {
                'shells': shell_count,
                'faces': face_count,
                'edges': edge_count,
                'vertices': vertex_count,
                'is_valid': shell_count == 1 and face_count >= 6
            }
            
        except Exception as e:
            print(f"Error getting solid info: {e}")
            return {}
