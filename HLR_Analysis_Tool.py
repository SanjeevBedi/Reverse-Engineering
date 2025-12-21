#!/usr/bin/env python3
"""
HLR Analysis Tool - Detailed analysis of hidden line removal accuracy
"""

import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

def analyze_hlr_output():
    """Analyze the HLR output for correctness issues."""
    print("HLR Analysis Tool")
    print("="*50)
    
    # Load the most recent drawing
    drawing_file = "random_engineering_drawings.png"
    
    if not os.path.exists(drawing_file):
        print(f"❌ Drawing file not found: {drawing_file}")
        return
    
    print(f"✓ Analyzing: {drawing_file}")
    
    # Load and analyze the image
    try:
        img = Image.open(drawing_file)
        img_array = np.array(img)
        
        print(f"Image dimensions: {img.size}")
        print(f"Image mode: {img.mode}")
        
        # Analyze the drawing content
        analyze_drawing_content(img_array)
        
    except Exception as e:
        print(f"❌ Error loading image: {e}")

def analyze_drawing_content(img_array):
    """Analyze the drawing content for HLR issues."""
    print("\nDrawing Content Analysis:")
    print("-" * 30)
    
    # Convert to grayscale for analysis
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    # Find line pixels (non-white areas)
    line_pixels = gray < 250  # Assuming white background
    line_count = np.sum(line_pixels)
    total_pixels = gray.size
    
    print(f"Line density: {line_count/total_pixels:.4f}")
    
    # Analyze line patterns
    analyze_line_patterns(img_array)

def analyze_line_patterns(img_array):
    """Analyze line patterns to identify solid vs dashed lines."""
    print("\nLine Pattern Analysis:")
    print("-" * 25)
    
    # This is a simplified analysis
    # In a real implementation, you'd need more sophisticated pattern recognition
    
    # Check for color variations (solid vs dashed lines)
    if len(img_array.shape) == 3:
        # Look for different colors/styles
        unique_colors = np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0)
        print(f"Unique colors found: {len(unique_colors)}")
        
        # Identify potential line colors
        non_white_colors = []
        for color in unique_colors:
            if not np.all(color > 240):  # Not white/near-white
                non_white_colors.append(color)
        
        print(f"Non-white colors (potential lines): {len(non_white_colors)}")
        for i, color in enumerate(non_white_colors):
            print(f"  Color {i+1}: RGB{tuple(color)}")

def identify_hlr_issues():
    """Identify specific HLR issues based on engineering drawing standards."""
    print("\nCommon HLR Issues to Check:")
    print("-" * 35)
    
    issues = [
        "1. Visible edges shown as dashed (should be solid)",
        "2. Hidden edges shown as solid (should be dashed)", 
        "3. Incorrect depth perception - front edges behind back edges",
        "4. Missing hidden edges that should be visible",
        "5. Extra hidden edges that should not be visible",
        "6. Inconsistent line weights or styles",
        "7. Overlapping or conflicting line representations"
    ]
    
    for issue in issues:
        print(f"  {issue}")

def suggest_hlr_improvements():
    """Suggest specific improvements for HLR algorithm."""
    print("\nSuggested HLR Improvements:")
    print("-" * 35)
    
    suggestions = [
        "1. Increase ray sampling resolution (current: 100 samples)",
        "2. Implement proper solid classification using BRepClass3d",
        "3. Add face normal analysis for front-facing vs back-facing",
        "4. Use Z-buffer depth testing for accurate visibility",
        "5. Implement proper edge-face relationship analysis",
        "6. Add geometric validation for edge visibility",
        "7. Use multiple ray directions for robust intersection testing"
    ]
    
    for suggestion in suggestions:
        print(f"  {suggestion}")

def debug_specific_geometry():
    """Debug the specific geometry in the simplified test model."""
    print("\nSimplified Test Model Debug:")
    print("-" * 35)
    
    print("Model composition:")
    print("  • Base cuboid: 94 x 61 x 57 mm")
    print("  • Cut feature: 8.0 x 17.0 x 67.0 mm at (21.0, 38.0, -5.0)")
    print("  • Boss feature: 33.0 x 27.0 x 10.0 mm at (2.0, 1.0, 57.0)")
    
    print("\nExpected visibility in each view:")
    print("Front view (looking along -Y):")
    print("  • Base cuboid front face should be visible")
    print("  • Cut creates visible opening on front face")
    print("  • Boss extends upward, front face visible")
    print("  • Back edges of cut should be hidden (dashed)")
    
    print("Top view (looking along -Z):")
    print("  • Base cuboid top face visible")
    print("  • Cut opening visible on top")
    print("  • Boss visible as rectangular protrusion")
    print("  • Bottom edges of features should be hidden")
    
    print("Side view (looking along -X):")
    print("  • Base cuboid side face visible")
    print("  • Cut visible as rectangular opening")
    print("  • Boss visible on top")
    print("  • Far-side edges should be hidden")

if __name__ == "__main__":
    analyze_hlr_output()
    print()
    identify_hlr_issues()
    print()
    suggest_hlr_improvements()
    print()
    debug_specific_geometry()
