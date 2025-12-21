"""
Basic example demonstrating the OpenCASCADE Engineering Drawings Generator.

This example shows how to:
1. Create a simple complex solid using boolean operations
2. Analyze the solid's geometry
3. Generate engineering drawings with visible and hidden lines
"""

import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from engineering_drawings.main import EngineeringDrawingsGenerator


def basic_example():
    """
    Run a basic example with default parameters.
    """
    print("=" * 60)
    print("BASIC EXAMPLE: OpenCASCADE Engineering Drawings")
    print("=" * 60)
    
    # Create the generator
    generator = EngineeringDrawingsGenerator()
    
    # Generate complete drawings with default parameters
    # This creates a 10x20x30 cuboid with a cut operation
    array_B, array_C = generator.generate_complete_drawings()
    
    print(f"\n✓ Example completed successfully!")
    print(f"  → Generated {len(array_B)} visible polygons")
    print(f"  → Generated {len(array_C)} hidden/intersection polygons")
    
    return array_B, array_C


def custom_dimensions_example():
    """
    Run an example with custom dimensions.
    """
    print("=" * 60)
    print("CUSTOM DIMENSIONS EXAMPLE")
    print("=" * 60)
    
    # Create the generator
    generator = EngineeringDrawingsGenerator()
    
    # Define custom dimensions
    cuboid1_dims = (15, 25, 35)  # Larger first cuboid
    cuboid2_dims = (8, 12, 20)   # Custom second cuboid
    translation = (7, 12, 18)    # Custom translation
    projection_normal = [1, 1, 1]  # Isometric projection
    
    print(f"First cuboid dimensions: {cuboid1_dims}")
    print(f"Second cuboid dimensions: {cuboid2_dims}")
    print(f"Translation vector: {translation}")
    print(f"Projection direction: {projection_normal}")
    
    # Generate drawings with custom parameters
    array_B, array_C = generator.generate_complete_drawings(
        cuboid1_dims=cuboid1_dims,
        cuboid2_dims=cuboid2_dims,
        translation=translation,
        projection_normal=projection_normal,
        operation="cut"
    )
    
    print(f"\n✓ Custom example completed successfully!")
    return array_B, array_C


def step_by_step_example():
    """
    Run an example showing each step individually.
    """
    print("=" * 60)
    print("STEP-BY-STEP EXAMPLE")
    print("=" * 60)
    
    # Create the generator
    generator = EngineeringDrawingsGenerator()
    
    # Step 1: Create solid
    print("\nStep 1: Creating complex solid...")
    solid = generator.create_complex_solid(
        cuboid1_dims=(12, 18, 24),
        cuboid2_dims=(6, 9, 12),
        translation=(6, 9, 12),
        operation="cut"
    )
    print("✓ Complex solid created")
    
    # Step 2: Analyze geometry
    print("\nStep 2: Analyzing solid geometry...")
    faces = generator.analyze_solid_geometry(solid)
    print(f"✓ Found {len(faces)} faces")
    
    # Print face information
    planar_faces = [f for f in faces if f['face_type'] == 'planar']
    print(f"  → {len(planar_faces)} planar faces")
    
    # Step 3: Classify faces
    print("\nStep 3: Classifying faces...")
    projection_normal = [0.707, 0.707, 0]  # 45-degree angle
    array_B, array_C = generator.classify_faces(faces, projection_normal)
    print(f"✓ Classification complete")
    print(f"  → {len(array_B)} visible polygons")
    print(f"  → {len(array_C)} hidden polygons")
    
    # Step 4: Visualize
    print("\nStep 4: Generating visualizations...")
    generator.plot_engineering_drawings(array_B, array_C, projection_normal)
    print("✓ Engineering drawings generated")
    
    return array_B, array_C


def main():
    """
    Run all examples.
    """
    try:
        print("OpenCASCADE Engineering Drawings - Examples")
        print("Choose an example to run:")
        print("1. Basic example (default parameters)")
        print("2. Custom dimensions example")
        print("3. Step-by-step example")
        print("4. Run all examples")
        
        choice = input("\nEnter choice (1-4, default=1): ").strip()
        
        if choice == "2":
            custom_dimensions_example()
        elif choice == "3":
            step_by_step_example()
        elif choice == "4":
            print("\n" + "="*60)
            print("RUNNING ALL EXAMPLES")
            print("="*60)
            basic_example()
            custom_dimensions_example()
            step_by_step_example()
        else:
            basic_example()
            
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
