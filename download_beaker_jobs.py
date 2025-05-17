    #!/usr/bin/env python3
"""
Script to download multiple Beaker jobs from a text file.
Each line in the text file should contain a Beaker job ID.
"""
import argparse
import os
from pathlib import Path
from typing import List
import sys

from utils.beaker_integration import download_job_results, check_beaker_available

def read_job_ids(file_path: str) -> List[str]:
    """Read job IDs from a text file."""
    with open(file_path, 'r') as f:
        # Strip whitespace and filter out empty lines
        return [line.strip() for line in f if line.strip()]

def download_jobs(job_ids: List[str], output_dir: str) -> None:
    """Download multiple Beaker jobs to the specified output directory."""
    # Check if Beaker is available
    beaker_available, message = check_beaker_available()
    if not beaker_available:
        print(f"Error: {message}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Download each job
    total_jobs = len(job_ids)
    successful = 0
    failed = 0
    
    print(f"Starting download of {total_jobs} jobs to {output_dir}")
    
    for i, job_id in enumerate(job_ids, 1):
        print(f"\nProcessing job {i}/{total_jobs}: {job_id}")
        success, message, job_path = download_job_results(job_id, output_dir)
        
        if success:
            print(f"✓ Success: {message}")
            print(f"  Downloaded to: {job_path}")
            successful += 1
        else:
            print(f"✗ Failed: {message}")
            failed += 1
    
    # Print summary
    print("\nDownload Summary:")
    print(f"Total jobs: {total_jobs}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

def main():
    parser = argparse.ArgumentParser(
        description="Download multiple Beaker jobs from a text file containing job IDs"
    )
    parser.add_argument(
        "job_ids_file",
        type=str,
        help="Path to text file containing Beaker job IDs (one per line)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs",
        help="Directory where job results will be saved (default: outputs)"
    )
    
    args = parser.parse_args()
    
    # Read job IDs from file
    try:
        job_ids = read_job_ids(args.job_ids_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.job_ids_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading job IDs file: {str(e)}")
        sys.exit(1)
    
    if not job_ids:
        print("Error: No job IDs found in the input file")
        sys.exit(1)
    
    # Download the jobs
    download_jobs(job_ids, args.output_dir)

if __name__ == "__main__":
    main() 