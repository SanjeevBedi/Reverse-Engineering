#!/usr/bin/env python3
"""
Batch test script to build connectivity matrices and reconstruct solids for multiple seeds.
Saves individual outputs and generates a summary of success/failure rates.
"""

import subprocess
import sys
import os
import re
import argparse
import signal
from datetime import datetime

def run_command(cmd, output_file, timeout=900):
    """Run a command and save output to file with timeout (default 15 mins).
    
    Args:
        cmd: Command to run
        output_file: File to save output
        timeout: Timeout in seconds (default 900 = 15 minutes)
    
    Returns:
        tuple: (returncode, timed_out)
    """
    try:
        with open(output_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout
            )
        return result.returncode, False
    except subprocess.TimeoutExpired:
        with open(output_file, 'a') as f:
            f.write(f"\n\nTIMEOUT: Process exceeded {timeout}s limit\n")
        return -2, True
    except Exception as e:
        with open(output_file, 'a') as f:
            f.write(f"\n\nERROR: {str(e)}\n")
        return -1, False

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
        match = re.search(r'EXTRACTION COMPLETE:\s*(\d+)\s*faces found', content)
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
            'has_solid': False,
            'has_error': True,
            'face_count': None,
            'success': False,
            'error': str(e)
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Batch test solid reconstruction for multiple seeds'
    )
    parser.add_argument(
        '--start-seed', type=int, default=None,
        help='Start from this seed (default: start from beginning)'
    )
    parser.add_argument(
        '--timeout', type=int, default=900,
        help='Timeout in seconds for each operation (default: 900 = 15 min)'
    )
    args = parser.parse_args()
    
    # Configuration
    seeds = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91,
             2, 12, 22, 32, 42, 52, 62, 72, 82, 92,
             3, 13, 23, 33, 43, 53, 63, 73, 83, 93,
             4, 14, 24, 34, 44, 54, 64, 74, 84, 94,
             5, 15, 25, 35, 45, 55, 65, 75, 85, 95,
             7, 17, 27, 37, 47, 57, 67, 77, 87, 97,
             8, 18, 28, 38, 48, 58, 68, 78, 88, 98,
             9, 19, 29, 39, 49, 59, 69, 79, 89, 99,
             10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    python_exe = '/opt/anaconda3/envs/pyocc/bin/python'
    work_dir = '/Users/sbedi/Nextcloud/Python/Solid/random_solids'
    
    # Filter seeds if start_seed is specified
    if args.start_seed is not None:
        if args.start_seed in seeds:
            start_idx = seeds.index(args.start_seed)
            seeds = seeds[start_idx:]
            print(f"Starting from seed {args.start_seed}")
        else:
            print(f"Warning: Start seed {args.start_seed} not in list")
            print(f"Available seeds: {seeds}")
            return 1
    
    # Change to working directory
    os.chdir(work_dir)
    
    # Summary tracking
    results = []
    
    print("="*70)
    print("BATCH RECONSTRUCTION TEST")
    print("="*70)
    print(f"Testing seeds: {seeds}")
    print(f"Working directory: {work_dir}")
    print(f"Timeout per operation: {args.timeout}s ({args.timeout/60:.1f} min)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print()
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Processing Seed {seed}")
        print(f"{'='*70}")
        
        # # Step 1: Build connectivity matrix
        print(f"[{seed}] Step 1: Building solid and saving projections...")
        build_output = f"txtFiles/output_build_{seed}.txt"
        build_cmd = [python_exe, 'Build_Solid.py', '--seed',
                     str(seed), '--no-graphics']
        
        build_return, build_timeout = run_command(
            build_cmd, build_output, timeout=args.timeout)
        if build_timeout:
            print(f"[{seed}] Build TIMEOUT (>{args.timeout}s)")
        else:
            print(f"[{seed}] Build complete (exit code: {build_return})")
        
        # Step 2: Reconstruct solid from connectivity matrix
        print(f"[{seed}] Step 2: Reconstructing solid from connectivity...")
        recon_output = f"txtFiles/output_recon_{seed}.txt"
        recon_cmd = [python_exe, 'Reconstruct_Solid.py', '--seed',
                     str(seed), '--no-occ-viewer', '--no-graphics']
        
        recon_return, recon_timeout = run_command(
            recon_cmd, recon_output, timeout=args.timeout)
        if recon_timeout:
            print(f"[{seed}] Reconstruction TIMEOUT (>{args.timeout}s)")
        else:
            print(f"[{seed}] Reconstruction complete (exit: {recon_return})")
        
        # Step 3: Analyze results
        print(f"[{seed}] Step 3: Analyzing results...")
        analysis = check_reconstruction_success(recon_output)
        
        results.append({
            'seed': seed,
            'build_exit_code': build_return,
            'recon_exit_code': recon_return,
            'build_timeout': build_timeout,
            'recon_timeout': recon_timeout,
            'analysis': analysis,
            'build_output': build_output,
            'recon_output': recon_output
        })
        
        # Print immediate result
        status = "✓ SUCCESS" if analysis['success'] else "✗ FAILED"
        face_info = f" ({analysis['face_count']} faces)" if analysis['face_count'] else ""
        print(f"[{seed}] Result: {status}{face_info}")
    
    # Generate summary
    print(f"\n\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Summary statistics
    total = len(results)
    successful = sum(1 for r in results if r['analysis']['success'])
    failed = total - successful
    
    print(f"Total seeds tested: {total}")
    print(f"Successful reconstructions: {successful} ({100*successful/total:.1f}%)")
    print(f"Failed reconstructions: {failed} ({100*failed/total:.1f}%)")
    print()
    
    # Detailed results table
    print(f"{'Seed':<8} {'Build':<10} {'Recon':<10} {'Faces':<8} "
          f"{'Status':<15} {'Output'}")
    print("-" * 75)
    
    for r in results:
        seed = r['seed']
        if r['build_timeout']:
            build_code = "TIMEOUT"
        elif r['build_exit_code'] == 0:
            build_code = "OK"
        else:
            build_code = f"ERR({r['build_exit_code']})"
        
        if r['recon_timeout']:
            recon_code = "TIMEOUT"
        elif r['recon_exit_code'] == 0:
            recon_code = "OK"
        else:
            recon_code = f"ERR({r['recon_exit_code']})"
        
        faces = str(r['analysis']['face_count']) if \
                r['analysis']['face_count'] else "N/A"
        status = "✓ SUCCESS" if r['analysis']['success'] else "✗ FAILED"
        
        print(f"{seed:<8} {build_code:<10} {recon_code:<10} {faces:<8} "
              f"{status:<15} {r['recon_output']}")
    
    print()
    
    # Failed cases details
    if failed > 0:
        print(f"\n{'='*70}")
        print("FAILED CASES DETAILS")
        print(f"{'='*70}")
        
        for r in results:
            if not r['analysis']['success']:
                print(f"\nSeed {r['seed']}:")
                if r['build_timeout']:
                    print(f"  Build: TIMEOUT")
                if r['recon_timeout']:
                    print(f"  Reconstruction: TIMEOUT")
                print(f"  Build output: {r['build_output']}")
                print(f"  Recon output: {r['recon_output']}")
                if r['analysis'].get('error'):
                    print(f"  Error: {r['analysis']['error']}")
                if r['analysis']['has_error']:
                    print(f"  Contains errors in output")
    
    # Save summary to file
    summary_file = f"batch_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"BATCH RECONSTRUCTION TEST SUMMARY\n")
        f.write(f"{'='*70}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Seeds tested: {seeds}\n\n")
        f.write(f"Total: {total}\n")
        f.write(f"Successful: {successful} ({100*successful/total:.1f}%)\n")
        f.write(f"Failed: {failed} ({100*failed/total:.1f}%)\n\n")
        f.write(f"{'Seed':<8} {'Build':<10} {'Recon':<10} {'Faces':<8} "
                f"{'Status':<15} {'Output File'}\n")
        f.write("-" * 75 + "\n")
        
        for r in results:
            seed = r['seed']
            if r['build_timeout']:
                build_code = "TIMEOUT"
            elif r['build_exit_code'] == 0:
                build_code = "OK"
            else:
                build_code = f"ERR({r['build_exit_code']})"
            
            if r['recon_timeout']:
                recon_code = "TIMEOUT"
            elif r['recon_exit_code'] == 0:
                recon_code = "OK"
            else:
                recon_code = f"ERR({r['recon_exit_code']})"
            
            faces = str(r['analysis']['face_count']) if \
                    r['analysis']['face_count'] else "N/A"
            status = "SUCCESS" if r['analysis']['success'] else "FAILED"
            
            f.write(f"{seed:<8} {build_code:<10} {recon_code:<10} "
                    f"{faces:<8} {status:<15} {r['recon_output']}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"{'='*70}\n")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
