#!/usr/bin/env python3
"""
Analyze the final engineering model to see its hidden line properties
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Final_HLR_Engineering_Drawings import FinalHLREngineeringDrawings
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Extend.TopologyUtils import TopologyExplorer

def analyze_step_model(step_file):
    """Analyze a STEP model for hidden line properties"""
    print(f"Analyzing model: {step_file}")
    print("=" * 50)
    
    # Load the STEP file
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file)
    
    if status != IFSelect_RetDone:
        print("✗ Failed to load STEP file")
        return None
        
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    
    # Analyze the shape topology
    topo_exp = TopologyExplorer(shape)
    
    # Count topology elements
    shell_count = len(list(topo_exp.shells()))
    face_count = len(list(topo_exp.faces()))
    edge_count = len(list(topo_exp.edges()))
    vertex_count = len(list(topo_exp.vertices()))
    
    print(f"Shape contains:")
    print(f"  - {shell_count} shells")
    print(f"  - {face_count} faces")
    print(f"  - {edge_count} edges")
    print(f"  - {vertex_count} vertices")
    
    if shell_count == 1:
        print("✓ Single shell - good for HLR analysis")
    else:
        print("⚠ Multiple shells - may affect HLR")
        
    # Create HLR generator and analyze
    generator = FinalHLREngineeringDrawings()
    generator.integrated_shape = shape
    generator.main_shape = shape  # Set main_shape for compatibility
    
    # Extract edge information
    print("\nExtracting edges for HLR analysis...")
    generator.extract_edges_simple()
    
    # Show edge statistics
    total_edges = 0
    for view in ['front', 'top', 'side']:
        visible_count = len(generator.visible_edges[view])
        hidden_count = len(generator.hidden_edges[view])
        total_count = visible_count + hidden_count
        total_edges += total_count
        
        if total_count > 0:
            visible_pct = (visible_count / total_count) * 100
            hidden_pct = (hidden_count / total_count) * 100
            
            print(f"\n{view.upper()} VIEW:")
            print(f"  Visible edges: {visible_count} ({visible_pct:.1f}%)")
            print(f"  Hidden edges:  {hidden_count} ({hidden_pct:.1f}%)")
            print(f"  Total edges:   {total_count}")
        
    print(f"\nTotal edges processed: {total_edges}")
    
    # Test some sample points for visibility
    print(f"\nTesting visibility at key points:")
    
    # Get bounding box to test interior points
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import BRepBndLib_Add
    bbox = Bnd_Box()
    BRepBndLib_Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    
    test_points = [
        ((xmin + xmax)/2, (ymin + ymax)/2, (zmin + zmax)/2, "Center"),
        (xmin, ymin, zmin, "Min corner"),
        (xmax, ymax, zmax, "Max corner"),
        ((xmin + xmax)/4, (ymin + ymax)/4, (zmin + zmax)/4, "Quarter point"),
    ]
    
    for x, y, z, desc in test_points:
        print(f"\n{desc} ({x:.1f}, {y:.1f}, {z:.1f}):")
        for view in ['front', 'top', 'side']:
            visible = generator.geometric_visibility_test((x, y, z), view)
            print(f"  {view}: {'VISIBLE' if visible else 'HIDDEN'}")
    
    # Generate drawing
    print(f"\nGenerating drawing analysis...")
    generator.create_engineering_drawings_professional("model_analysis.png")
    print("Analysis drawing saved as 'model_analysis.png'")
    
    return generator

if __name__ == "__main__":
    step_file = "final_engineering_model.step"
    if len(sys.argv) > 1:
        step_file = sys.argv[1]
        
    if os.path.exists(step_file):
        analyze_step_model(step_file)
    else:
        print(f"STEP file not found: {step_file}")
