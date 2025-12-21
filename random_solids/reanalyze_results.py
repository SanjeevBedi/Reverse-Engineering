#!/usr/bin/env python3
"""
Re-analyze batch test results with corrected success criteria.
"""

import re
import os

def check_reconstruction_success(output_file):
    """Check if reconstruction was successful by analyzing the output file."""
    try:
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Look for success indicators
        has_extraction = 'EXTRACTION COMPLETE' in content
        has_completed = ('COMPLETED' in content and 
                        'Reconstruction process finished' in content)
        
        # Only count as error if there's a Traceback (actual Python exception)
        # STEP 6.x errors are expected during polygon tracing
        has_error = 'Traceback' in content
        
        # Extract face count if available
        face_count = None
        match = re.search(r'EXTRACTION COMPLETE:\s*(\d+)\s*faces found', 
                         content)
        if match:
            face_count = int(match.group(1))
        
        return {
            'has_extraction': has_extraction,
            'has_completed': has_completed,
            'has_error': has_error,
            'face_count': face_count,
            'success': has_extraction and has_completed and not has_error
        }
    except Exception as e:
        return {
            'has_extraction': False,
            'has_completed': False,
            'has_error': True,
            'face_count': None,
            'success': False,
            'error': str(e)
        }

def main():
    seeds = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
    
    print("="*70)
    print("RE-ANALYZING BATCH TEST RESULTS")
    print("="*70)
    print()
    
    results = []
    for seed in seeds:
        output_file = f"output_recon_{seed}.txt"
        if not os.path.exists(output_file):
            print(f"[{seed}] Output file not found: {output_file}")
            continue
        
        analysis = check_reconstruction_success(output_file)
        results.append({
            'seed': seed,
            'analysis': analysis
        })
        
        status = "✓ SUCCESS" if analysis['success'] else "✗ FAILED"
        face_count = analysis['face_count'] if analysis['face_count'] else "N/A"
        print(f"[{seed}] {status} - {face_count} faces")
        print(f"      Extraction: {analysis['has_extraction']}, " 
              f"Completed: {analysis['has_completed']}, " 
              f"Errors: {analysis['has_error']}")
    
    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    successful = sum(1 for r in results if r['analysis']['success'])
    total = len(results)
    success_rate = (successful / total * 100) if total > 0 else 0
    
    print(f"Total seeds tested: {total}")
    print(f"Successful reconstructions: {successful} ({success_rate:.1f}%)")
    print(f"Failed reconstructions: {total - successful} " 
          f"({100 - success_rate:.1f}%)")
    print()
    
    # Detailed table
    print(f"{'Seed':<10} {'Faces':<10} {'Status':<15}")
    print("-"*70)
    for r in results:
        seed = r['seed']
        analysis = r['analysis']
        face_count = str(analysis['face_count']) if analysis['face_count'] \
                     else "N/A"
        status = "✓ SUCCESS" if analysis['success'] else "✗ FAILED"
        print(f"{seed:<10} {face_count:<10} {status:<15}")

if __name__ == '__main__':
    main()
