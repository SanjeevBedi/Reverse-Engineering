"""
Test suite for the OpenCASCADE Engineering Drawings Generator.

This module contains comprehensive tests for all components of the
engineering drawings generation system.
"""

import sys
import os
import pytest
import numpy as np

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from engineering_drawings.main import EngineeringDrawingsGenerator
from engineering_drawings.solid_generator import SolidGenerator
from engineering_drawings.face_analyzer import FaceAnalyzer


class TestSolidGenerator:
    """Test cases for the SolidGenerator class."""
    
    def test_create_cuboid(self):
        """Test basic cuboid creation."""
        generator = SolidGenerator()
        cuboid = generator.create_cuboid(10, 20, 30)
        assert cuboid is not None
    
    def test_create_boolean_solid(self):
        """Test boolean solid creation."""
        generator = SolidGenerator()
        solid = generator.create_boolean_solid()
        assert solid is not None


class TestFaceAnalyzer:
    """Test cases for the FaceAnalyzer class."""
    
    def test_face_analyzer_creation(self):
        """Test FaceAnalyzer instantiation."""
        analyzer = FaceAnalyzer()
        assert analyzer is not None
    
    def test_analyze_topology(self):
        """Test topology analysis."""
        generator = SolidGenerator()
        analyzer = FaceAnalyzer()
        
        # Create a simple cuboid
        cuboid = generator.create_cuboid(10, 20, 30)
        
        # Analyze topology
        analyzer.analyze_topology(cuboid)
        
        # Check that analysis completed
        assert hasattr(analyzer, '_face_count')
        assert analyzer._face_count > 0


class TestEngineeringDrawingsGenerator:
    """Test cases for the main EngineeringDrawingsGenerator class."""
    
    def test_generator_creation(self):
        """Test generator instantiation."""
        generator = EngineeringDrawingsGenerator()
        assert generator is not None
        assert generator.solid_generator is not None
        assert generator.face_analyzer is not None
    
    def test_create_complex_solid(self):
        """Test complex solid creation."""
        generator = EngineeringDrawingsGenerator()
        solid = generator.create_complex_solid()
        assert solid is not None
    
    def test_analyze_solid_geometry(self):
        """Test solid geometry analysis."""
        generator = EngineeringDrawingsGenerator()
        
        # Create a solid first
        solid = generator.create_complex_solid()
        
        # Analyze geometry
        faces = generator.analyze_solid_geometry(solid)
        
        # Check results
        assert isinstance(faces, list)
        assert len(faces) > 0
        
        # Check that we have planar faces
        planar_faces = [f for f in faces if f.get('face_type') == 'planar']
        assert len(planar_faces) > 0
    
    def test_complete_workflow(self):
        """Test the complete engineering drawings workflow."""
        generator = EngineeringDrawingsGenerator()
        
        # Run complete workflow with small dimensions for speed
        array_B, array_C = generator.generate_complete_drawings(
            cuboid1_dims=(5, 10, 15),
            cuboid2_dims=(2, 5, 8),
            translation=(2, 5, 8),
            projection_normal=[1, 1, 1]
        )
        
        # Check results
        assert isinstance(array_B, list)
        assert isinstance(array_C, list)
        
        # Should have some polygons
        total_polygons = len(array_B) + len(array_C)
        assert total_polygons > 0


class TestProjectionEngine:
    """Test cases for projection functionality."""
    
    def test_projection_normal_validation(self):
        """Test projection normal vector validation."""
        generator = EngineeringDrawingsGenerator()
        
        # Test various projection normals
        test_normals = [
            [1, 0, 0],    # X-axis
            [0, 1, 0],    # Y-axis
            [0, 0, 1],    # Z-axis
            [1, 1, 1],    # Isometric
            [1, 1, 0],    # 45-degree
        ]
        
        for normal in test_normals:
            # This should not raise an exception
            normalized = generator.projection_engine.normalize_vector(normal)
            assert len(normalized) == 3
            
            # Check that it's normalized (magnitude should be 1)
            magnitude = np.sqrt(sum(x**2 for x in normalized))
            assert abs(magnitude - 1.0) < 1e-10


def test_module_imports():
    """Test that all modules can be imported correctly."""
    # These imports should not raise exceptions
    from engineering_drawings.solid_generator import SolidGenerator
    from engineering_drawings.face_analyzer import FaceAnalyzer
    from engineering_drawings.main import EngineeringDrawingsGenerator
    
    # Basic instantiation should work
    generator = EngineeringDrawingsGenerator()
    assert generator is not None


def test_basic_workflow():
    """Test the basic workflow without visualization."""
    generator = EngineeringDrawingsGenerator()
    
    # Create solid
    solid = generator.create_complex_solid(
        cuboid1_dims=(5, 5, 5),
        cuboid2_dims=(2, 2, 2),
        translation=(1, 1, 1)
    )
    assert solid is not None
    
    # Analyze geometry
    faces = generator.analyze_solid_geometry(solid)
    assert len(faces) > 0
    
    # Classify faces
    array_B, array_C = generator.classify_faces(
        faces, projection_normal=[1, 1, 1]
    )
    assert isinstance(array_B, list)
    assert isinstance(array_C, list)


if __name__ == "__main__":
    # Run tests directly
    print("Running OpenCASCADE Engineering Drawings Tests...")
    
    # Simple test runner
    test_functions = [
        test_module_imports,
        test_basic_workflow,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed! 🎉")
    else:
        print("Some tests failed. Check the output above.")
        sys.exit(1)
