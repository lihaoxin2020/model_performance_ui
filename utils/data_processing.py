"""
Data processing utilities for integrating external results into the model performance dashboard
"""
import os
import json
import shutil
from pathlib import Path
import uuid

def import_beaker_job_results(beaker_job_path, output_dir, model_name=None):
    """
    Import downloaded Beaker job results into the model_performance_ui outputs directory
    
    Args:
        beaker_job_path (str): Path to the downloaded Beaker job results
        output_dir (str): Output directory where model results are stored
        model_name (str, optional): Name to use for the model. If None, tries to extract from metrics.json
        
    Returns:
        tuple: (success, message, imported_directory)
    """
    # Convert paths to Path objects
    beaker_path = Path(beaker_job_path)
    output_path = Path(output_dir)
    
    # Check if the beaker job directory exists
    if not beaker_path.exists() or not beaker_path.is_dir():
        return False, f"Beaker job directory {beaker_job_path} does not exist", None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Look for metrics.json
    metrics_file = beaker_path / "metrics.json"
    if not metrics_file.exists():
        return False, f"No metrics.json found in {beaker_job_path}", None
    
    # Extract dataset name from prediction files
    prediction_files = list(beaker_path.glob("task-*-*predictions.jsonl"))
    if not prediction_files:
        return False, f"No prediction files found in {beaker_job_path}", None
    
    # Try to extract dataset name from prediction file
    dataset_name = "unknown"
    for file in prediction_files:
        file_name = file.name
        parts = file_name.split('-')
        if len(parts) >= 3:
            dataset_name = parts[2].split(':')[0]  # Handle potential `:` in filename
            break
    
    # Determine model name
    if not model_name:
        # Try to extract from metrics.json
        try:
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
                # Try to get model name from metrics
                if "config" in metrics_data and "model" in metrics_data["config"]:
                    model_name = metrics_data["config"]["model"]
                else:
                    # Generate a unique model name based on the job ID
                    job_id = beaker_path.name.replace("beaker_job_", "").replace("_", "")
                    model_name = f"beaker_model_{job_id[:8]}"
        except Exception:
            # If there's an error, generate a fallback name
            model_name = f"beaker_model_{uuid.uuid4().hex[:8]}"
    
    # Create destination directory name
    dest_dir_name = f"lmeval-{model_name}-on-{dataset_name}"
    dest_path = output_path / dest_dir_name
    
    # Check if directory already exists
    if dest_path.exists():
        # Add a unique suffix to prevent overwriting
        unique_suffix = uuid.uuid4().hex[:6]
        dest_dir_name = f"{dest_dir_name}-{unique_suffix}"
        dest_path = output_path / dest_dir_name
    
    # Create destination directory
    os.makedirs(dest_path, exist_ok=True)
    
    # Copy all files from Beaker job to destination
    for file in beaker_path.glob("*"):
        if file.is_file():
            shutil.copy2(file, dest_path)
    
    return True, f"Successfully imported Beaker job results as {dest_dir_name}", str(dest_path) 