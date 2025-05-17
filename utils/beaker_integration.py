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

def download_job_results(job_id: str, output_dir: Optional[str] = None) -> tuple[bool, str, Optional[str]]:
    """
    Download all output files from a Beaker job by its ID.
    
    Args:
        job_id (str): The Beaker job ID
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
        
        # Get job information
        job = beaker.job.get(job_id)
        
        # Check if job exists
        if not job:
            return False, f"Job with ID {job_id} not found.", None
        
        # Find the result dataset using a different approach
        # First, try to use the job's task ID to check for associated results
        result_dataset_id = None
        
        # Check job status for completion
        if hasattr(job, 'status'):
            if not job.status.finalized:
                return False, f"Job {job_id} is not yet finalized. Wait for the job to complete.", None
        
        # Use the beaker HTTP API to get job execution details
        try:
            # Make a direct HTTP request to get job details with execution info
            # job_details = beaker.http_request(f"jobs/{job_id}").json()
            
            # Check if job has a result in the execution
            result_dataset_id = beaker.job.results(job_id).id
            
        except Exception:
            # If HTTP request fails, try alternative methods
            pass
        
        # If we still don't have a result dataset ID, try using a workload
        # if not result_dataset_id and hasattr(job, 'workload_id'):
        #     try:
        #         # Get workload details
        #         workload = beaker.workload.get(job.workload_id)
                
        #         # Check if workload has jobs with results
        #         for workload_job in beaker.workload.jobs(workload):
        #             if workload_job.id == job_id:
        #                 # Try to get result from REST API
        #                 job_details = beaker.http_request(f"jobs/{job_id}").json()
        #                 if 'execution' in job_details and 'result' in job_details['execution']:
        #                     if 'beaker' in job_details['execution']['result']:
        #                         result_dataset_id = job_details['execution']['result']['beaker']
        #                         break
        #     except Exception:
        #         # If workload methods fail, continue
        #         pass
        
        # As a last resort, try to fetch the dataset using a direct naming convention
        if not result_dataset_id:
            try:
                # Some jobs store results in a dataset with the same name as the job ID
                result_datasets = list(beaker.dataset.list(name_or_description=job_id, limit=1))
                if result_datasets:
                    result_dataset_id = result_datasets[0].id
            except Exception:
                # If this fails, continue
                pass
        
        # If we still can't find a result dataset, return an error
        if not result_dataset_id:
            return False, f"No result dataset found for job {job_id}. The job may not have completed successfully or doesn't have results.", None
            
        # Download the dataset files
        try:
            # Get the dataset object
            job_details = job.to_json()
            dataset = beaker.dataset.get(result_dataset_id)
            
            # Fetch the files from the result dataset
            dir_name = None

            env_vars = job_details['execution']['spec']['envVars']

            for env_var in env_vars:
                if env_var['name'] == 'BEAKER_EXPERIMENT_NAME':
                    dir_name = env_var['value']
                    break
            assert dir_name is not None, "No experiment name found in job environment variables"
            output_job_path = output_path / dir_name
            # from IPython import embed
            # embed()
            beaker.dataset.fetch(dataset, target=output_job_path)
            
            return True, f"Successfully downloaded all results for job {job_id}", str(output_job_path)
        
        except FileExistsError as e:
            return True, f"Local directory already exists: {str(e)}", str(output_job_path)
        except Exception as e:
            return False, f"Error downloading dataset files: {str(e)}", None
    
    except Exception as e:
        return False, f"Error downloading job outputs: {str(e)}", None

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