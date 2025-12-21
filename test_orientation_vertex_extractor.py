#!/usr/bin/env python3
"""
Test the orientation-based vertex extractor with wire/edge logic.

This test validates that the new orientation logic correctly handles
mixed edge orientations found in problematic faces.
"""

import sys
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid')
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid/OpenCASCADE_Engineering_Drawings/src')

try:
    from engineering_drawings.vertex_extractor import VertexExtractor
    from V5_current import create_opencascade_solid, OPENCASCADE_AVAILABLE
    
    if OPENCASCADE_AVAILABLE:
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
except ImportError as e:
    print(f"Import error: {e}")
    OPENCASCADE_AVAILABLE = False


def test_orientation_vertex_extractor():
    """Test the orientation-based vertex extractor."""
    
    print("="*70)
    print("TESTING ORIENTATION-BASED VERTEX EXTRACTOR")
    print("="*70)
    
    if not OPENCASCADE_AVAILABLE:
        print("✗ OpenCASCADE not available - cannot run test")
        return
    
    # Create the test solid
    print("Creating OpenCASCADE solid...")
    solid = create_opencascade_solid()
    
    if solid is None:
        print("✗ Failed to create solid")
        return
    
    # Initialize the enhanced vertex extractor
    try:
        extractor = VertexExtractor()
        print("✓ VertexExtractor initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize VertexExtractor: {e}")
        return
    
    # Test the problematic faces (1, 2, 5)
    target_faces = [1, 2, 5]
    
    face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
    face_count = 0
    
    print(f"\nTesting faces {target_faces} with new orientation logic...")
    
    results = {}
    
    while face_explorer.More() and face_count < 6:
        face = face_explorer.Current()
        face_count += 1
        
        if face_count in target_faces:
            print(f"\n{'-'*60}")
            print(f"TESTING FACE {face_count}")
            print(f"{'-'*60}")
            
            try:
                vertices = extractor.extract_face_vertices(face)
                
                if vertices and len(vertices) >= 3:
                    print(f"✓ Extracted {len(vertices)} vertices")
                    
                    # Store results
                    results[face_count] = {
                        'success': True,
                        'vertex_count': len(vertices),
                        'vertices': vertices
                    }
                    
                    # Display vertex sequence
                    vertex_coords = " → ".join([
                        f"({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})" 
                        for v in vertices
                    ])
                    print(f"  Polygon: {vertex_coords}")
                    
                else:
                    print(f"✗ Failed to extract vertices")
                    results[face_count] = {'success': False}
                    
            except Exception as e:
                print(f"✗ Exception during extraction: {e}")
                results[face_count] = {'success': False, 'error': str(e)}
        
        face_explorer.Next()
    
    # Summary
    print(f"\n{'='*70}")
    print("ORIENTATION VERTEX EXTRACTOR TEST RESULTS")
    print("="*70)
    
    success_count = 0
    for face_id in target_faces:
        if face_id in results:
            if results[face_id].get('success'):
                vertex_count = results[face_id]['vertex_count']
                print(f"✓ Face {face_id}: {vertex_count} vertices extracted")
                success_count += 1
            else:
                error = results[face_id].get('error', 'Unknown error')
                print(f"✗ Face {face_id}: Failed - {error}")
        else:
            print(f"✗ Face {face_id}: Not tested")
    
    print(f"\nResults: {success_count}/{len(target_faces)} faces successful")
    
    if success_count == len(target_faces):
        print("🎉 ALL TARGET FACES EXTRACTED SUCCESSFULLY!")
        print("   New orientation logic appears to be working!")
    else:
        print("⚠️  Some faces still have issues")
    
    print("="*70)


if __name__ == "__main__":
    test_orientation_vertex_extractor()
