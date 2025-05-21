#!/usr/bin/env python3
"""
Script to download multiple Beaker experiments from a text file.
Each line in the text file should contain a Beaker experiment ID.
Uses multiprocessing for parallel downloads.
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple
import sys
from multiprocessing import Pool, cpu_count
from functools import partial

from utils.beaker_integration import download_experiment_results, check_beaker_available

def read_experiment_ids(file_path: str) -> List[str]:
    """Read experiment IDs from a text file."""
    with open(file_path, 'r') as f:
        # Strip whitespace and filter out empty lines
        return [line.strip() for line in f if line.strip()]

def download_single_experiment(experiment_id: str, output_dir: str) -> Tuple[bool, str, str]:
    """Download a single Beaker experiment."""
    success, message, experiment_path = download_experiment_results(experiment_id, output_dir)
    return success, message, experiment_path

def download_experiments(experiment_ids: List[str], output_dir: str, num_processes: int = None) -> None:
    """Download multiple Beaker experiments to the specified output directory using multiprocessing."""
    # Check if Beaker is available
    beaker_available, message = check_beaker_available()
    if not beaker_available:
        print(f"Error: {message}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)  # Leave one CPU free
    
    total_experiments = len(experiment_ids)
    print(f"Starting download of {total_experiments} experiments to {output_dir}")
    
    successful = 0
    failed = 0
    
    if num_processes == 1:
        # Use single-threaded implementation for debugging
        print("Using single-threaded implementation")
        for i, experiment_id in enumerate(experiment_ids, 1):
            print(f"\nProcessing experiment {i}/{total_experiments}: {experiment_id}")
            success, message, experiment_path = download_experiment_results(experiment_id, output_dir)
            
            if success:
                print(f"✓ Success: {message}")
                print(f"  Downloaded to: {experiment_path}")
                successful += 1
            else:
                print(f"✗ Failed: {message}")
                failed += 1
    else:
        # Use multiprocessing for parallel downloads
        print(f"Using {num_processes} processes for parallel downloads")
        download_func = partial(download_single_experiment, output_dir=output_dir)
        
        with Pool(processes=num_processes) as pool:
            for i, (success, message, experiment_path) in enumerate(
                pool.imap_unordered(download_func, experiment_ids), 1
            ):
                print(f"\nProcessing experiment {i}/{total_experiments}")
                if success:
                    print(f"✓ Success: {message}")
                    print(f"  Downloaded to: {experiment_path}")
                    successful += 1
                else:
                    print(f"✗ Failed: {message}")
                    failed += 1
    
    # Print summary
    print("\nDownload Summary:")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

def main():
    parser = argparse.ArgumentParser(
        description="Download multiple Beaker experiments from a text file containing experiment IDs"
    )
    parser.add_argument(
        "experiment_ids_file",
        type=str,
        help="Path to text file containing Beaker experiment IDs (one per line)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs",
        help="Directory where experiment results will be saved (default: outputs)"
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=None,
        help="Number of processes to use for parallel downloads (default: CPU count - 1)"
    )
    
    args = parser.parse_args()
    
    # Read experiment IDs from file
    try:
        experiment_ids = read_experiment_ids(args.experiment_ids_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.experiment_ids_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading experiment IDs file: {str(e)}")
        sys.exit(1)
    
    if not experiment_ids:
        print("Error: No experiment IDs found in the input file")
        sys.exit(1)
    
    # Download the experiments
    download_experiments(experiment_ids, args.output_dir, args.processes)

if __name__ == "__main__":
    main() 