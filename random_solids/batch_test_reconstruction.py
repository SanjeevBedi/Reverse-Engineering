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
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

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
        
        # Check for valid solid created
        has_valid_solid = 'SUCCESS: Valid solid created!' in content
        
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
        
        # Extract volumes (original and reconstructed)
        original_volume = None
        reconstructed_volume = None
        volume_diff = None
        volume_diff_pct = None
        
        # Look for original volume
        match = re.search(r'Original volume:\s*([\d.]+)\s*mm³', content)
        if match:
            original_volume = float(match.group(1))
        
        # Look for reconstructed volume
        match = re.search(r'Reconstructed volume:\s*([\d.]+)\s*mm³', content)
        if match:
            reconstructed_volume = float(match.group(1))
        
        # Calculate volume difference if both are available
        if original_volume is not None and reconstructed_volume is not None:
            volume_diff = reconstructed_volume - original_volume
            volume_diff_pct = (volume_diff / original_volume) * 100 if original_volume != 0 else 0
        
        # Calculate success: completed, no errors, no free edges, has faces, valid solid
        success = (has_completed and not has_error and 
                   has_no_free_edges and has_valid_solid and 
                   reconstructed_faces and reconstructed_faces > 0)
        
        return {
            'has_completed': has_completed,
            'has_no_free_edges': has_no_free_edges,
            'has_positive_volume': has_positive_volume,
            'has_valid_solid': has_valid_solid,
            'has_error': has_error,
            'reconstructed_faces': reconstructed_faces,
            'original_faces': original_faces,
            'matched_faces': matched_faces,
            'original_volume': original_volume,
            'reconstructed_volume': reconstructed_volume,
            'volume_diff': volume_diff,
            'volume_diff_pct': volume_diff_pct,
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

def process_single_seed(seed, python_exe, work_dir, timeout):
    """Process a single seed (build and reconstruct)."""
    seed_start_time = time.time()
    
    # Change to working directory (each process gets its own copy)
    os.chdir(work_dir)
    
    print(f"[{seed}] Starting (PID: {os.getpid()})...")
    
    # Step 1: Build solid and save BREP
    build_output = f"txtFiles/output_build_{seed}.txt"
    build_cmd = [python_exe, 'Build_Solid.py', '--seed',
                 str(seed), '--no-graphics']
    
    build_return, build_timeout = run_command(
        build_cmd, build_output, timeout=timeout)
    
    # Step 2: Build enhanced connectivity matrix from BREP
    connectivity_output = f"txtFiles/output_connectivity_{seed}.txt"
    connectivity_cmd = [python_exe, 'build_solid_connectivity.py', '--seed', str(seed), '--no-graphics']
    
    connectivity_return, connectivity_timeout = run_command(
        connectivity_cmd, connectivity_output, timeout=timeout)
    
    # Move connectivity plot to Plots_From directory
    plot_file = f"solid_connectivity_validation_seed_{seed}.png"
    if os.path.exists(plot_file):
        plots_dir = "Plots_From"
        os.makedirs(plots_dir, exist_ok=True)
        import shutil
        shutil.move(plot_file, os.path.join(plots_dir, plot_file))
    
    # Step 3: Reconstruct solid from connectivity matrix
    recon_output = f"txtFiles/output_recon_{seed}.txt"
    recon_cmd = [python_exe, 'Reconstruct_Solid.py', '--seed',
                 str(seed), '--no-occ-viewer', '--no-graphics',
                 '--tolerance', '0.05']
    
    recon_return, recon_timeout = run_command(
        recon_cmd, recon_output, timeout=timeout)
    
    # Analyze results
    analysis = check_reconstruction_success(recon_output)
    
    result = {
        'seed': seed,
        'build_exit_code': build_return,
        'connectivity_exit_code': connectivity_return,
        'recon_exit_code': recon_return,
        'build_timeout': build_timeout,
        'connectivity_timeout': connectivity_timeout,
        'recon_timeout': recon_timeout,
        'analysis': analysis,
        'build_output': build_output,
        'connectivity_output': connectivity_output,
        'recon_output': recon_output,
        'elapsed_time': time.time() - seed_start_time
    }
    
    # Print result
    status = "✓ SUCCESS" if analysis['success'] else "✗ FAILED"
    if analysis['matched_faces'] and analysis['original_faces']:
        face_info = (f" ({analysis['matched_faces']}/{analysis['original_faces']} "
                    f"matched, {analysis['reconstructed_faces']} total)")
    elif analysis['reconstructed_faces']:
        face_info = f" ({analysis['reconstructed_faces']} faces)"
    else:
        face_info = ""
    
    print(f"[{seed}] Completed in {result['elapsed_time']:.1f}s - {status} [Seed {seed}]{face_info}")
    
    return result

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
    parser.add_argument(
        '--parallel', type=int, default=2,
        help='Number of seeds to process in parallel (default: 2)'
    )
    args = parser.parse_args()
    
    # Configuration
    seeds = range(0, 501, 5)  # Seeds from 1000 to 1300 inclusive 
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
    print(f"Testing seeds: {list(seeds)}")
    print(f"Working directory: {work_dir}")
    print(f"Parallel processes: {args.parallel}")
    print(f"Timeout per operation: {args.timeout}s ({args.timeout/60:.1f} min)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print()
    
    # Process seeds in parallel batches
    results = []
    
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all seeds
        future_to_seed = {
            executor.submit(process_single_seed, seed, python_exe, work_dir, args.timeout): seed
            for seed in seeds
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"[{seed}] Generated an exception: {exc}")
                # Create a failed result entry
                results.append({
                    'seed': seed,
                    'build_exit_code': -1,
                    'recon_exit_code': -1,
                    'build_timeout': False,
                    'recon_timeout': False,
                    'analysis': {
                        'success': False,
                        'has_error': True,
                        'error': str(exc)
                    },
                    'build_output': f"txtFiles/output_build_{seed}.txt",
                    'recon_output': f"txtFiles/output_recon_{seed}.txt",
                    'elapsed_time': 0
                })
    
    # Sort results by seed for consistent reporting
    results.sort(key=lambda x: x['seed'])
    
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
    print(f"{'Seed':<8} {'Build':<10} {'Connect':<10} {'Recon':<10} {'Faces':<8} "
          f"{'Valid':<7} {'Orig Vol':<10} {'Recon Vol':<10} {'Vol Diff %':<12} "
          f"{'Status':<15} {'Output'}")
    print("-" * 140)
    
    for r in results:
        seed = r['seed']
        if r['build_timeout']:
            build_code = "TIMEOUT"
        elif r['build_exit_code'] == 0:
            build_code = "OK"
        else:
            build_code = f"ERR({r['build_exit_code']})"
        
        if r.get('connectivity_timeout', False):
            conn_code = "TIMEOUT"
        elif r.get('connectivity_exit_code', 0) == 0:
            conn_code = "OK"
        else:
            conn_code = f"ERR({r.get('connectivity_exit_code', -1)})"
        
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
        
        # Format valid solid flag
        valid_solid = "✓" if r['analysis'].get('has_valid_solid', False) else "✗"
        
        # Format volumes
        if r['analysis'].get('original_volume') is not None:
            orig_vol = f"{r['analysis']['original_volume']:.2f}"
        else:
            orig_vol = "N/A"
        
        if r['analysis'].get('reconstructed_volume') is not None:
            recon_vol = f"{r['analysis']['reconstructed_volume']:.2f}"
        else:
            recon_vol = "N/A"
        
        # Format volume difference
        if r['analysis'].get('volume_diff_pct') is not None:
            vol_diff = f"{r['analysis']['volume_diff_pct']:+.2f}%"
        else:
            vol_diff = "N/A"
        
        status = "✓ SUCCESS" if r['analysis']['success'] else "✗ FAILED"
        
        print(f"{seed:<8} {build_code:<10} {conn_code:<10} {recon_code:<10} {faces:<8} "
              f"{valid_solid:<7} {orig_vol:<10} {recon_vol:<10} {vol_diff:<12} "
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
                if r.get('connectivity_timeout', False):
                    print(f"  Connectivity: TIMEOUT")
                if r['recon_timeout']:
                    print(f"  Reconstruction: TIMEOUT")
                print(f"  Build output: {r['build_output']}")
                print(f"  Connectivity output: {r.get('connectivity_output', 'N/A')}")
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
        f.write(f"{'Seed':<8} {'Build':<10} {'Connect':<10} {'Recon':<10} {'Faces':<8} "
                f"{'Valid':<7} {'Orig Vol':<10} {'Recon Vol':<10} {'Vol Diff %':<12} "
                f"{'Status':<15} {'Output File'}\n")
        f.write("-" * 140 + "\n")
        
        for r in results:
            seed = r['seed']
            if r['build_timeout']:
                build_code = "TIMEOUT"
            elif r['build_exit_code'] == 0:
                build_code = "OK"
            else:
                build_code = f"ERR({r['build_exit_code']})"
            
            if r.get('connectivity_timeout', False):
                conn_code = "TIMEOUT"
            elif r.get('connectivity_exit_code', 0) == 0:
                conn_code = "OK"
            else:
                conn_code = f"ERR({r.get('connectivity_exit_code', -1)})"
            
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
            
            # Format valid solid flag
            valid_solid = "Y" if r['analysis'].get('has_valid_solid', False) else "N"
            
            # Format volumes
            if r['analysis'].get('original_volume') is not None:
                orig_vol = f"{r['analysis']['original_volume']:.2f}"
            else:
                orig_vol = "N/A"
            
            if r['analysis'].get('reconstructed_volume') is not None:
                recon_vol = f"{r['analysis']['reconstructed_volume']:.2f}"
            else:
                recon_vol = "N/A"
            
            # Format volume difference
            if r['analysis'].get('volume_diff_pct') is not None:
                vol_diff = f"{r['analysis']['volume_diff_pct']:+.2f}%"
            else:
                vol_diff = "N/A"
            
            status = "SUCCESS" if r['analysis']['success'] else "FAILED"
            
            f.write(f"{seed:<8} {build_code:<10} {recon_code:<10} "
                    f"{faces:<8} {valid_solid:<7} {orig_vol:<10} {recon_vol:<10} {vol_diff:<12} "
                    f"{status:<15} {r['recon_output']}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"{'='*70}\n")
    
    # Filter successful solids with matching face counts
    successful_seeds = []
    for r in results:
        if (r['analysis']['success'] and 
            r['analysis'].get('matched_faces') and 
            r['analysis'].get('original_faces') and
            r['analysis']['matched_faces'] == r['analysis']['original_faces']):
            successful_seeds.append(r['seed'])
    
    # Prepare training data if we have successful seeds
    if successful_seeds:
        print(f"Preparing training data for {len(successful_seeds)} successful seeds...")
        print(f"Seeds: {successful_seeds}\n")
        
        # Create seed list as comma-separated string
        seeds_str = ','.join(map(str, successful_seeds))
        
        # Parameters for training data preparation
        tolerance = 0.1
        augment = 5
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"NN/training_data_{timestamp}.npz"
        
        # New prepare_training_data.py CLI interface
        # Note: Use pyocc environment for data preparation (OCC dependencies)
        prep_cmd = (f"conda run -n pyocc python NN/prepare_training_data.py "
                   f"--seeds {seeds_str} "
                   f"--tolerance {tolerance} "
                   f"--augment {augment} "
                   f"--input-dir Output "
                   f"--output {output_file}")
        
        print(f"Running: {prep_cmd}")
        print(f"  Tolerance: {tolerance} mm")
        print(f"  Augmentation: {augment}x per seed")
        print(f"  Total samples: {len(successful_seeds) * augment}")
        print("="*70)
        
        try:
            result = subprocess.run(
                prep_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout (increased for larger datasets)
            )
            
            if result.returncode == 0:
                print(f"✓ Training data prepared successfully!")
                print(f"Output: {output_file}")
                if result.stdout:
                    print(result.stdout)
                
                # Now train the model using the tf environment
                print(f"\n{'='*70}")
                print("TRAINING MODEL")
                print(f"{'='*70}\n")
                
                model_output = output_file.replace('.npz', '.h5')
                train_log = output_file.replace('.npz', '_training.txt')
                
                # Use tf environment for training (TensorFlow dependencies)
                train_cmd = (f"conda run -n tf python NN/train_model.py "
                           f"--data {output_file} "
                           f"--epochs 100 "
                           f"--batch-size 8 "
                           f"--model {model_output}")
                
                print(f"Running: {train_cmd}")
                print(f"  Data: {output_file}")
                print(f"  Model output: {model_output}")
                print(f"  Training log: {train_log}")
                print("="*70)
                
                try:
                    with open(train_log, 'w') as log_file:
                        train_result = subprocess.run(
                            train_cmd,
                            shell=True,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            text=True,
                            timeout=3600  # 60 minute timeout for training
                        )
                    
                    if train_result.returncode == 0:
                        print(f"✓ Model training completed successfully!")
                        print(f"Model saved: {model_output}")
                        print(f"Training log: {train_log}")
                    else:
                        print(f"✗ Model training failed (exit code: {train_result.returncode})")
                        print(f"Check training log: {train_log}")
                        
                except subprocess.TimeoutExpired:
                    print("✗ Model training timed out after 60 minutes")
                except Exception as e:
                    print(f"✗ Error running model training: {e}")
                
            else:
                print(f"✗ Training data preparation failed (exit code: {result.returncode})")
                if result.stderr:
                    print(f"Error output:\n{result.stderr}")
                if result.stdout:
                    print(f"Output:\n{result.stdout}")
                    
        except subprocess.TimeoutExpired:
            print("✗ Training data preparation timed out after 30 minutes")
        except Exception as e:
            print(f"✗ Error running training data preparation: {e}")
        
        print(f"{'='*70}\n")
    else:
        print("No successful seeds with matching face counts found.")
        print("Skipping training data preparation.\n")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
