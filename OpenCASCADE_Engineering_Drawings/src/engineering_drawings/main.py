"""
Main application entry point for OpenCASCADE Engineering Drawings Generator.

This module provides the main application class and entry point for generating
engineering drawings from 3D CAD models.
"""

import sys
from typing import List, Tuple, Dict, Any

# Import project modules
from .solid_generator import SolidGenerator
from .face_analyzer import FaceAnalyzer
from .vertex_extractor import VertexExtractor
from .projection_engine import ProjectionEngine
from .classification_algorithm import ClassificationAlgorithm
from .visualization import Visualizer


class EngineeringDrawingsGenerator:
    """
    Main class for generating engineering drawings from 3D CAD models.
    
    This class coordinates all the components needed to:
    1. Generate 3D solids with boolean operations
    2. Analyze face geometry and types
    3. Extract vertices with proper orientation
    4. Project faces to 2D polygons
    5. Classify polygons using historic algorithm
    6. Visualize results as engineering drawings
    """
    
    def __init__(self):
        """Initialize the engineering drawings generator with all components."""
        self.solid_generator = SolidGenerator()
        self.face_analyzer = FaceAnalyzer()
        self.vertex_extractor = VertexExtractor()
        self.projection_engine = ProjectionEngine()
        self.classification_algorithm = ClassificationAlgorithm()
        self.visualizer = Visualizer()
        
        self._current_solid = None
        self._current_faces = None
        self._current_arrays = None
    
    def create_complex_solid(
        self, 
        cuboid1_dims: Tuple[float, float, float] = (10, 20, 30),
        cuboid2_dims: Tuple[float, float, float] = None,
        translation: Tuple[float, float, float] = (5, 10, 15),
        operation: str = "cut"
    ) -> Any:
        """
        Create a complex 3D solid using boolean operations.
        
        Args:
            cuboid1_dims: Dimensions (width, height, depth) of first cuboid
            cuboid2_dims: Dimensions of second cuboid (auto-generated if None)
            translation: Translation vector for second cuboid
            operation: Boolean operation ("cut", "fuse", "common")
            
        Returns:
            OpenCASCADE solid shape
        """
        self._current_solid = self.solid_generator.create_boolean_solid(
            cuboid1_dims, cuboid2_dims, translation, operation
        )
        return self._current_solid
    
    def analyze_solid_geometry(self, solid: Any = None) -> List[Dict[str, Any]]:
        """
        Analyze solid geometry and extract face information.
        
        Args:
            solid: OpenCASCADE solid to analyze (uses current if None)
            
        Returns:
            List of face data dictionaries with geometry information
        """
        if solid is None:
            solid = self._current_solid
            
        if solid is None:
            raise ValueError("No solid available for analysis")
        
        # Analyze solid geometry
        self.face_analyzer.analyze_topology(solid)
        
        # Extract faces with vertex information
        self._current_faces = self.face_analyzer.extract_faces_with_vertices(
            solid, self.vertex_extractor
        )
        
        return self._current_faces
    
    def classify_faces(
        self, 
        faces: List[Dict[str, Any]] = None,
        projection_normal: List[float] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Classify faces using the historic polygon classification algorithm.
        
        Args:
            faces: Face data list (uses current if None)
            projection_normal: Projection direction [x, y, z] (interactive if None)
            
        Returns:
            Tuple of (array_B, array_C) containing visible and hidden polygons
        """
        if faces is None:
            faces = self._current_faces
            
        if faces is None:
            raise ValueError("No faces available for classification")
        
        # Get projection normal
        if projection_normal is None:
            projection_normal = self.projection_engine.get_projection_normal_interactive()
        else:
            projection_normal = self.projection_engine.normalize_vector(projection_normal)
        
        # Project faces to 2D polygons
        projected_polygons = self.projection_engine.project_faces_to_2d(
            faces, projection_normal
        )
        
        # Apply historic classification algorithm
        array_A, array_B, array_C = self.classification_algorithm.classify_polygons(
            projected_polygons, projection_normal
        )
        
        self._current_arrays = (array_A, array_B, array_C)
        return array_B, array_C
    
    def plot_engineering_drawings(
        self,
        array_B: List[Dict[str, Any]] = None,
        array_C: List[Dict[str, Any]] = None,
        projection_normal: List[float] = None,
        show_3d: bool = True
    ) -> None:
        """
        Generate and display engineering drawings.
        
        Args:
            array_B: Visible polygons (uses current if None)
            array_C: Hidden polygons (uses current if None)
            projection_normal: Projection direction for labeling
            show_3d: Whether to show 3D visualization
        """
        if array_B is None or array_C is None:
            if self._current_arrays is None:
                raise ValueError("No classified arrays available for plotting")
            _, array_B, array_C = self._current_arrays
        
        # Show 3D solid visualization first
        if show_3d and self._current_solid is not None:
            self.visualizer.visualize_3d_solid(self._current_solid)
        
        # Generate 2D engineering drawings
        self.visualizer.plot_arrays_visualization(
            array_A=[], 
            array_B=array_B, 
            array_C=array_C,
            unit_projection_normal=projection_normal or [0.577, 0.577, 0.577]
        )
    
    def generate_complete_drawings(
        self,
        cuboid1_dims: Tuple[float, float, float] = (10, 20, 30),
        cuboid2_dims: Tuple[float, float, float] = None,
        translation: Tuple[float, float, float] = (5, 10, 15),
        projection_normal: List[float] = None,
        operation: str = "cut"
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generate complete engineering drawings from start to finish.
        
        This is a convenience method that performs the entire workflow:
        1. Create complex solid
        2. Analyze geometry
        3. Classify faces
        4. Visualize results
        
        Args:
            cuboid1_dims: Dimensions of first cuboid
            cuboid2_dims: Dimensions of second cuboid (auto-generated if None)
            translation: Translation for second cuboid
            projection_normal: Projection direction (interactive if None)
            operation: Boolean operation type
            
        Returns:
            Tuple of (array_B, array_C) for further processing
        """
        print("=" * 60)
        print("OPENCASCADE ENGINEERING DRAWINGS GENERATOR")
        print("Complete Workflow Execution")
        print("=" * 60)
        
        # Step 1: Create solid
        print("\nStep 1: Creating complex 3D solid...")
        solid = self.create_complex_solid(
            cuboid1_dims, cuboid2_dims, translation, operation
        )
        
        # Step 2: Analyze geometry
        print("\nStep 2: Analyzing solid geometry...")
        faces = self.analyze_solid_geometry(solid)
        
        # Step 3: Classify faces
        print("\nStep 3: Classifying faces with historic algorithm...")
        array_B, array_C = self.classify_faces(faces, projection_normal)
        
        # Step 4: Visualize results
        print("\nStep 4: Generating engineering drawings...")
        self.plot_engineering_drawings(array_B, array_C, projection_normal)
        
        print(f"\n✓ Complete workflow finished successfully!")
        print(f"  → Generated {len(array_B)} visible polygons")
        print(f"  → Generated {len(array_C)} hidden/intersection polygons")
        
        return array_B, array_C


def main():
    """
    Main entry point for the engineering drawings application.
    
    This function can be called from the command line or imported and used
    in other applications.
    """
    try:
        # Create generator instance
        generator = EngineeringDrawingsGenerator()
        
        # Check for command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] in ["-h", "--help"]:
                print("OpenCASCADE Engineering Drawings Generator")
                print("Usage: engineering-drawings [options]")
                print("Options:")
                print("  -h, --help     Show this help message")
                print("  --interactive  Run in interactive mode")
                print("  --batch        Run with default parameters")
                return
            elif sys.argv[1] == "--batch":
                # Batch mode with default parameters
                generator.generate_complete_drawings()
                return
        
        # Interactive mode (default)
        print("Starting interactive engineering drawings generation...")
        
        # Get user preferences
        print("\nConfiguration options:")
        print("1. Use default cuboid dimensions (10x20x30 and auto-generated)")
        print("2. Specify custom dimensions")
        
        choice = input("Enter choice (1-2, default=1): ").strip()
        
        if choice == "2":
            try:
                print("\nEnter first cuboid dimensions:")
                w1 = float(input("Width: "))
                h1 = float(input("Height: "))
                d1 = float(input("Depth: "))
                cuboid1_dims = (w1, h1, d1)
                
                print("\nEnter second cuboid dimensions:")
                w2 = float(input("Width: "))
                h2 = float(input("Height: "))
                d2 = float(input("Depth: "))
                cuboid2_dims = (w2, h2, d2)
                
                print("\nEnter translation vector:")
                tx = float(input("X translation: "))
                ty = float(input("Y translation: "))
                tz = float(input("Z translation: "))
                translation = (tx, ty, tz)
                
            except ValueError:
                print("Invalid input, using default values")
                cuboid1_dims = (10, 20, 30)
                cuboid2_dims = None
                translation = (5, 10, 15)
        else:
            cuboid1_dims = (10, 20, 30)
            cuboid2_dims = None
            translation = (5, 10, 15)
        
        # Generate drawings
        generator.generate_complete_drawings(
            cuboid1_dims=cuboid1_dims,
            cuboid2_dims=cuboid2_dims,
            translation=translation
        )
        
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
