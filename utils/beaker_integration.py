"""
Beaker integration utilities for downloading model evaluation results.
"""
import os
import tempfile
from pathlib import Path
import shutil
from typing import Optional, List

# Import beaker client
try:
    from beaker import Beaker
    from beaker.exceptions import JobNotFound, ExperimentNotFound
    BEAKER_AVAILABLE = True
except ImportError:
    BEAKER_AVAILABLE = False

def check_beaker_available():
    """Check if beaker client is available and configured"""
    if not BEAKER_AVAILABLE:
        return False, "Beaker Python client not installed. Install with 'pip install beaker-py'."
    
    try:
        # Try to initialize client from environment
        Beaker.from_env()
        return True, "Beaker client initialized successfully."
    except Exception as e:
        return False, f"Beaker client initialization failed: {str(e)}"

def download_experiment_results(experiment_id: str, output_dir: Optional[str] = None) -> tuple[bool, str, Optional[str]]:
    """
    Download all output files from a Beaker experiment by its ID.
    
    Args:
        experiment_id (str): The Beaker experiment ID
        output_dir (str, optional): Directory where files will be saved.
                                   If None, uses the input directory from Streamlit session state.
    
    Returns:
        tuple: (success, message, path_to_results)
    """
    if not BEAKER_AVAILABLE:
        return False, "Beaker Python client not installed.", None
    
    try:
        # Initialize Beaker client
        beaker = Beaker.from_env()
        
        # Create output directory if it doesn't exist
        if output_dir is None:
            # Use input directory from Streamlit session state
            import streamlit as st
            if 'input_dir' not in st.session_state:
                return False, "No input directory set in Streamlit session state.", None
            output_dir = st.session_state.input_dir
        
        output_path = Path(output_dir)
        os.makedirs(output_path, exist_ok=True)
        
        # Get experiment information
        try:
            experiment = beaker.experiment.get(experiment_id)
        except ExperimentNotFound as e:
            return False, f"Experiment with ID {experiment_id} not found.", None
        
        # Find the result dataset using a different approach
        # First, try to use the experiment's task ID to check for associated results
        result_dataset_id = None
        
        # Check experiment status for completion
        for job in experiment.jobs:
            # if hasattr(job, 'status'):
            if not job.status.finalized:
                return False, f"Experiment {experiment_id} is not yet finalized. Wait for the experiment to complete.", None
        
        # Use the beaker HTTP API to get experiment execution details
        try:
            # Check if experiment has a result in the execution
            result_dataset_id = beaker.experiment.results(experiment_id).id
            
        except Exception:
            # If HTTP request fails, try alternative methods
            pass
        
        # If we still don't have a result dataset ID, try using a workload
        if not result_dataset_id:
            try:
                # Some experiments store results in a dataset with the same name as the experiment ID
                result_datasets = list(beaker.dataset.list(name_or_description=experiment_id, limit=1))
                if result_datasets:
                    result_dataset_id = result_datasets[0].id
            except Exception:
                # If this fails, continue
                pass
        
        # If we still can't find a result dataset, return an error
        if not result_dataset_id:
            return False, f"No result dataset found for experiment {experiment_id}. The experiment may not have completed successfully or doesn't have results.", None
            
        # Download the dataset files
        try:
            # Get the dataset object
            dataset = beaker.dataset.get(result_dataset_id)
            
            # Fetch the files from the result dataset
            dir_name = experiment.name
            assert dir_name is not None, "No experiment name found in experiment environment variables"
            output_experiment_path = output_path / dir_name
            beaker.dataset.fetch(dataset, target=output_experiment_path)
            
            return True, f"Successfully downloaded all results for experiment {experiment_id}", str(output_experiment_path)
        
        except FileExistsError as e:
            return True, f"Local directory already exists: {str(e)}", str(output_experiment_path)
        except Exception as e:
            return False, f"Error downloading dataset files: {str(e)}", None
    
    except Exception as e:
        return False, f"Error downloading experiment outputs: {str(e)}", None

def process_downloaded_job(job_path: str) -> Optional[dict]:
    """
    Process a downloaded job to prepare it for the model_performance_ui
    
    Args:
        job_path (str): Path to the downloaded job results
        
    Returns:
        dict: Information about the job, including path and metadata
    """
    path = Path(job_path)
    
    # Check if directory exists
    if not path.exists() or not path.is_dir():
        return None
    
    # Look for metrics.json
    metrics_file = path / "metrics.json"
    has_metrics = metrics_file.exists()
    
    # Look for prediction files
    prediction_files = list(path.glob("task-*-*predictions.jsonl"))
    has_predictions = len(prediction_files) > 0
    
    # If it has metrics and predictions, consider it valid
    if has_metrics and has_predictions:
        # Get dataset name from prediction file
        dataset_name = "unknown"
        if prediction_files:
            # Try to extract dataset name from the prediction file name
            # Format is typically task-000-dataset-predictions.jsonl
            file_name = prediction_files[0].name
            parts = file_name.split('-')
            if len(parts) >= 3:
                dataset_name = parts[2]
        
        return {
            "path": str(path),
            "has_metrics": has_metrics,
            "has_predictions": has_predictions,
            "dataset": dataset_name,
            "job_id": path.name.replace("beaker_job_", "").replace("_", ""),
            "prediction_files": [str(f) for f in prediction_files]
        }
    
    return None 