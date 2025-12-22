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
        # Capture output instead of redirecting to file handle
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout
        )
        # Write captured output to file
        with open(output_file, 'w') as f:
            f.write(result.stdout)
        return result.returncode, False
    except subprocess.TimeoutExpired as e:
        # Write any partial output captured before timeout
        with open(output_file, 'w') as f:
            if e.stdout:
                f.write(e.stdout.decode() if isinstance(e.stdout, bytes)
                        else e.stdout)
            f.write(f"\n\nTIMEOUT: Process exceeded {timeout}s limit\n")
        return -2, True
    except Exception as e:
        with open(output_file, 'w') as f:
            f.write(f"\n\nERROR: {str(e)}\n")
        return -1, False

def check_reconstruction_success(output_file):
    """Check if reconstruction was successful by analyzing the output file."""
    try:
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Look for completion marker
        has_completed = '[COMPLETED] Reconstruction process finished.' in content
        
        # Check for critical errors (Traceback)
        has_error = 'Traceback' in content
        
        # Check shell quality
        has_no_free_edges = 'Number of free edges: 0' in content
        has_positive_volume = 'Volume is positive' in content
        
        # Extract face counts
        reconstructed_faces = None
        matched_faces = None
        original_faces = None
        
        match = re.search(r'Total faces:\s*(\d+)', content)
        if match:
            reconstructed_faces = int(match.group(1))
        
        match = re.search(r'Matched faces:\s*(\d+)/(\d+)', content)
        if match:
            matched_faces = int(match.group(1))
            original_faces = int(match.group(2))
        
        # Extract volume
        volume = None
        match = re.search(r'Volume:\s*([\d.]+)\s*mm³', content)
        if match:
            volume = float(match.group(1))
        
        # Calculate success: completed, no errors, no free edges, has faces
        # Note: solid may be "invalid" by OCC standards but still reconstructed
        success = (has_completed and not has_error and 
                   has_no_free_edges and reconstructed_faces and 
                   reconstructed_faces > 0)
        
        return {
            'has_completed': has_completed,
            'has_no_free_edges': has_no_free_edges,
            'has_positive_volume': has_positive_volume,
            'has_error': has_error,
            'reconstructed_faces': reconstructed_faces,
            'original_faces': original_faces,
            'matched_faces': matched_faces,
            'volume': volume,
            'success': success
        }
    except Exception as e:
        return {
            'has_valid_solid': False,
            'has_no_free_edges': False,
            'has_solid_created': False,
            'has_error': True,
            'face_count': None,
            'volume': None,
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
    seeds = range(100, 130, 5)  # Seeds from 1000 to 1300 inclusive 
    # [201, 211, 221, 231, 241, 251, 261, 271, 281, 291,
    #             202, 212, 222, 232, 242, 252, 262, 272, 282, 292,
    #             203, 213, 223, 233, 243, 253, 263, 273, 283, 293,
    #             204, 214, 224, 234, 244, 254, 264, 274, 284, 294,
    #             205, 215, 225, 235, 245, 255, 265, 275, 285, 295,
    #             206, 216, 226, 236, 246, 256, 266, 276, 286, 296,
    #             207, 217, 227, 237, 247, 257, 267, 277, 287, 297,
    #             208, 218, 228, 238, 248, 258, 268, 278, 288, 298,
    #             209, 219, 229, 239, 249, 259, 269, 279, 289, 299,
    #             210, 220, 230, 240, 250, 260, 270, 280, 290, 300]         

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
                     str(seed), '--no-occ-viewer', '--no-graphics',
                     '--tolerance', '0.05']
        
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
        if analysis['matched_faces'] and analysis['original_faces']:
            face_info = (f" ({analysis['matched_faces']}/{analysis['original_faces']} "
                        f"matched, {analysis['reconstructed_faces']} total)")
        elif analysis['reconstructed_faces']:
            face_info = f" ({analysis['reconstructed_faces']} faces)"
        else:
            face_info = ""
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
        
        # Format face information
        if r['analysis']['matched_faces'] and r['analysis']['original_faces']:
            faces = f"{r['analysis']['matched_faces']}/{r['analysis']['original_faces']}"
        elif r['analysis']['reconstructed_faces']:
            faces = str(r['analysis']['reconstructed_faces'])
        else:
            faces = "N/A"
        
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
            
            # Format face information for file output
            if r['analysis']['matched_faces'] and r['analysis']['original_faces']:
                faces = (f"{r['analysis']['matched_faces']}/"
                        f"{r['analysis']['original_faces']}")
            elif r['analysis']['reconstructed_faces']:
                faces = str(r['analysis']['reconstructed_faces'])
            else:
                faces = "N/A"
            
            status = "SUCCESS" if r['analysis']['success'] else "FAILED"
            
            f.write(f"{seed:<8} {build_code:<10} {recon_code:<10} "
                    f"{faces:<8} {status:<15} {r['recon_output']}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"{'='*70}\n")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
