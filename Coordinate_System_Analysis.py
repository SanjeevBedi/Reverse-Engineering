#!/usr/bin/env python3
"""
Engineering Drawing Coordinate System Analysis
==============================================

Analyzes the coordinate system and projection logic for front and side views
to identify potential issues with the orthographic projections.
"""

import sys
import os
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid/')
from Random_Engineering_Drawings import RandomEngineeringDrawings
import numpy as np

def analyze_projection_coordinates():
    """Analyze the coordinate projection for different views"""
    print("=== PROJECTION COORDINATE ANALYSIS ===")
    
    generator = RandomEngineeringDrawings()
    
    # Create a test model to get actual dimensions
    generator.create_random_model()
    
    print(f"Model dimensions: {generator.base_length} x {generator.base_width} x {generator.base_height}")
    print(f"Generated features: {len(generator.features)}")
    
    # Print feature details
    for i, feature in enumerate(generator.features):
        print(f"Feature {i+1}: {feature}")
    
    print("\n=== VIEW COORDINATE MAPPINGS ===")
    
    # Test coordinate mappings for each view
    test_point_3d = (50, 30, 40)  # Sample 3D point
    
    print(f"3D Point: {test_point_3d}")
    
    # Front view: Y -> X (horizontal), Z -> Y (vertical)
    front_2d = (test_point_3d[1], test_point_3d[2])
    print(f"Front view (Y,Z): {front_2d}")
    
    # Top view: X -> X (horizontal), Y -> Y (vertical) 
    top_2d = (test_point_3d[0], test_point_3d[1])
    print(f"Top view (X,Y): {top_2d}")
    
    # Side view: X -> X (horizontal), Z -> Y (vertical)
    side_2d = (test_point_3d[0], test_point_3d[2])
    print(f"Side view (X,Z): {side_2d}")
    
    return generator

def test_edge_classification_consistency():
    """Test edge classification consistency across views"""
    print("\n=== EDGE CLASSIFICATION CONSISTENCY TEST ===")
    
    generator = analyze_projection_coordinates()
    
    # Extract edges and test visibility in different views
    generator.extract_edges_simple()
    
    # Count edges per view
    front_visible = len(generator.visible_edges['front'])
    front_hidden = len(generator.hidden_edges['front'])
    side_visible = len(generator.visible_edges['side'])
    side_hidden = len(generator.hidden_edges['side'])
    top_visible = len(generator.visible_edges['top'])
    top_hidden = len(generator.hidden_edges['top'])
    
    print(f"Front view: {front_visible} visible, {front_hidden} hidden")
    print(f"Side view: {side_visible} visible, {side_hidden} hidden") 
    print(f"Top view: {top_visible} visible, {top_hidden} hidden")
    
    # Check for suspicious patterns
    total_front = front_visible + front_hidden
    total_side = side_visible + side_hidden
    total_top = top_visible + top_hidden
    
    print(f"\nTotal edges per view: Front={total_front}, Side={total_side}, Top={total_top}")
    
    if total_front == 0:
        print("⚠️  WARNING: No edges found for front view!")
    if total_side == 0:
        print("⚠️  WARNING: No edges found for side view!")
    if total_top == 0:
        print("⚠️  WARNING: No edges found for top view!")
    
    # Check visibility ratios
    if total_front > 0:
        front_ratio = front_visible / total_front
        print(f"Front view visible ratio: {front_ratio:.2%}")
        if front_ratio < 0.1:
            print("⚠️  WARNING: Very few front view edges are visible!")
        elif front_ratio > 0.9:
            print("⚠️  WARNING: Too many front view edges are visible!")
    
    if total_side > 0:
        side_ratio = side_visible / total_side
        print(f"Side view visible ratio: {side_ratio:.2%}")
        if side_ratio < 0.1:
            print("⚠️  WARNING: Very few side view edges are visible!")
        elif side_ratio > 0.9:
            print("⚠️  WARNING: Too many side view edges are visible!")

def analyze_specific_problematic_edges():
    """Analyze specific edges that might be causing problems"""
    print("\n=== PROBLEMATIC EDGE ANALYSIS ===")
    
    generator = RandomEngineeringDrawings()
    generator.create_random_model()
    generator.extract_edges_simple()
    
    # Look for issues in edge visibility patterns
    print(f"Front view: {len(generator.visible_edges['front'])} visible, {len(generator.hidden_edges['front'])} hidden")
    print(f"Side view: {len(generator.visible_edges['side'])} visible, {len(generator.hidden_edges['side'])} hidden") 
    print(f"Top view: {len(generator.visible_edges['top'])} visible, {len(generator.hidden_edges['top'])} hidden")
    
    # Check for suspicious ratios
    views = ['front', 'side', 'top']
    for view in views:
        visible_count = len(generator.visible_edges[view])
        hidden_count = len(generator.hidden_edges[view])
        total = visible_count + hidden_count
        
        if total > 0:
            ratio = visible_count / total
            print(f"{view.title()} view visible ratio: {ratio:.2%}")
            
            if ratio < 0.2:
                print(f"⚠️  WARNING: Very few {view} view edges are visible! Possible over-hiding.")
            elif ratio > 0.8:
                print(f"⚠️  WARNING: Too many {view} view edges are visible! Possible under-hiding.")
        else:
            print(f"⚠️  WARNING: No edges found for {view} view!")
    
    return generator

def main():
    """Main analysis function"""
    print("Engineering Drawing Coordinate System Analysis")
    print("=" * 60)
    
    analyze_projection_coordinates()
    test_edge_classification_consistency()
    analyze_specific_problematic_edges()

if __name__ == "__main__":
    main()
