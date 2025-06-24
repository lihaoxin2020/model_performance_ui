"""
Beaker integration utilities for downloading model evaluation results.
"""
import os
import tempfile
from pathlib import Path
import shutil
from typing import Optional, List, Dict, Tuple
from datetime import datetime

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

def list_workspace_experiments_cached(workspace_name: Optional[str] = None, force_refresh: bool = False) -> Tuple[bool, str, List[Dict], bool]:
    """
    List experiments from a Beaker workspace with intelligent caching.
    Only fetches new experiments since the last load.
    
    Args:
        workspace_name (str, optional): The workspace name. If None, uses default workspace.
        force_refresh (bool): Force refresh all experiments from scratch.
    
    Returns:
        tuple: (success, message, experiments_list, has_new_experiments)
    """
    if not BEAKER_AVAILABLE:
        return False, "Beaker Python client not installed.", [], False
    
    try:
        # Initialize Beaker client
        beaker = Beaker.from_env()
        
        # Get workspace info
        if workspace_name:
            try:
                workspace = beaker.workspace.get(workspace_name)
            except Exception as e:
                return False, f"Workspace '{workspace_name}' not found: {str(e)}", [], False
        else:
            workspace = beaker.workspace.get()
        
        workspace_id = workspace.id
        
        # Import streamlit for session state access
        import streamlit as st
        
        # Cache key for this workspace
        cache_key = f"experiments_cache_{workspace_id}"
        timestamp_key = f"experiments_timestamp_{workspace_id}"
        
        # Get cached experiments and last update timestamp
        cached_experiments = st.session_state.get(cache_key, [])
        last_timestamp = st.session_state.get(timestamp_key, None)
        
        if force_refresh:
            # Clear cache if force refresh
            cached_experiments = []
            last_timestamp = None
        
        # If we have cached experiments, find the newest one's creation time
        latest_cached_time = None
        if cached_experiments:
            try:
                # Parse timestamps and find the latest
                latest_cached_time = max(
                    datetime.fromisoformat(exp.get("created_iso", "1970-01-01T00:00:00"))
                    for exp in cached_experiments
                    if exp.get("created_iso")
                )
            except:
                latest_cached_time = None
        
        # Fetch new experiments (we'll fetch a larger batch to ensure we catch all new ones)
        fetch_limit = 200 if not cached_experiments else 100
        experiments = beaker.workspace.experiments(workspace, limit=fetch_limit)
        
        # Convert to list and process
        all_fetched_experiments = []
        new_experiments = []
        current_time = datetime.now()
        
        for exp in experiments:
            # Check if experiment is finalized
            is_finalized = all(job.status.finalized for job in exp.jobs) if exp.jobs else False
            
            # Get creation time
            created_str = exp.created.strftime("%Y-%m-%d %H:%M:%S") if exp.created else "Unknown"
            created_iso = exp.created.isoformat() if exp.created else None
            
            # Check if experiment has results
            has_results = False
            try:
                result_dataset = beaker.experiment.results(exp.id)
                has_results = result_dataset is not None
            except:
                has_results = False
            
            experiment_data = {
                "id": exp.id,
                "name": exp.name or exp.id,
                "created": created_str,
                "created_iso": created_iso,
                "finalized": is_finalized,
                "has_results": has_results,
                "author": exp.author.name if exp.author else "Unknown",
                "description": exp.description or ""
            }
            
            all_fetched_experiments.append(experiment_data)
            
            # Check if this is a new experiment
            if not any(cached_exp["id"] == exp.id for cached_exp in cached_experiments):
                new_experiments.append(experiment_data)
        
        # Merge cached experiments with new experiments
        # Create a dict for fast lookup and deduplication
        all_experiments_dict = {exp["id"]: exp for exp in cached_experiments}
        
        # Add new experiments
        for exp in new_experiments:
            all_experiments_dict[exp["id"]] = exp
        
        # Convert back to list and sort by creation time (newest first)
        merged_experiments = list(all_experiments_dict.values())
        merged_experiments.sort(
            key=lambda x: x.get("created_iso", "1970-01-01T00:00:00"),
            reverse=True
        )
        
        # Update cache
        st.session_state[cache_key] = merged_experiments
        st.session_state[timestamp_key] = current_time.isoformat()
        
        # Prepare response message
        total_count = len(merged_experiments)
        new_count = len(new_experiments)
        
        if force_refresh:
            message = f"Refreshed all {total_count} experiments in workspace '{workspace.full_name}'"
        elif new_count > 0:
            message = f"Found {new_count} new experiments. Total: {total_count} experiments in workspace '{workspace.full_name}'"
        else:
            message = f"No new experiments found. Total: {total_count} cached experiments in workspace '{workspace.full_name}'"
        
        return True, message, merged_experiments, new_count > 0
    
    except Exception as e:
        return False, f"Error listing workspace experiments: {str(e)}", [], False

def get_paginated_experiments(workspace_name: Optional[str] = None, page: int = 0, page_size: int = 50, force_refresh: bool = False) -> Tuple[bool, str, List[Dict], bool, int]:
    """
    Get paginated experiments using cached data.
    
    Args:
        workspace_name (str, optional): The workspace name. If None, uses default workspace.
        page (int): Page number (0-based).
        page_size (int): Number of experiments per page.
        force_refresh (bool): Force refresh all experiments from scratch.
    
    Returns:
        tuple: (success, message, page_experiments, has_more, total_count)
    """
    # Get all cached experiments
    success, message, all_experiments, has_new = list_workspace_experiments_cached(workspace_name, force_refresh)
    
    if not success:
        return False, message, [], False, 0
    
    # Calculate pagination
    total_count = len(all_experiments)
    start_idx = page * page_size
    end_idx = start_idx + page_size
    
    # Get page experiments
    page_experiments = all_experiments[start_idx:end_idx]
    has_more = end_idx < total_count
    
    # Update message with pagination info
    page_message = f"Page {page + 1} ({len(page_experiments)} experiments)"
    if has_new:
        page_message += " - New experiments detected!"
    
    return True, page_message, page_experiments, has_more, total_count

def clear_experiments_cache(workspace_name: Optional[str] = None):
    """
    Clear cached experiments for a workspace.
    
    Args:
        workspace_name (str, optional): The workspace name. If None, clears default workspace cache.
    """
    try:
        import streamlit as st
        
        if workspace_name:
            # We need to get the workspace ID, but this requires a Beaker call
            # For simplicity, we'll clear all experiment caches
            keys_to_remove = [key for key in st.session_state.keys() if key.startswith("experiments_")]
        else:
            # Clear all experiment caches
            keys_to_remove = [key for key in st.session_state.keys() if key.startswith("experiments_")]
        
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
                
        return True
    except Exception:
        return False

def list_workspace_experiments(workspace_name: Optional[str] = None, limit: int = 50, offset: int = 0) -> Tuple[bool, str, List[Dict]]:
    """
    List experiments from a Beaker workspace with pagination support.
    
    Args:
        workspace_name (str, optional): The workspace name. If None, uses default workspace.
        limit (int): Maximum number of experiments to return per page.
        offset (int): Number of experiments to skip (for pagination).
    
    Returns:
        tuple: (success, message, experiments_list)
    """
    if not BEAKER_AVAILABLE:
        return False, "Beaker Python client not installed.", []
    
    try:
        # Initialize Beaker client
        beaker = Beaker.from_env()
        
        # Get workspace info
        if workspace_name:
            try:
                workspace = beaker.workspace.get(workspace_name)
            except Exception as e:
                return False, f"Workspace '{workspace_name}' not found: {str(e)}", []
        else:
            workspace = beaker.workspace.get()
        
        # List experiments in the workspace with pagination
        # Note: We request more than needed to implement offset ourselves since
        # the Beaker API might not support offset directly
        total_to_fetch = offset + limit
        experiments = beaker.workspace.experiments(workspace, limit=min(total_to_fetch, 200))
        
        # Convert to list to support slicing
        all_experiments = list(experiments)
        
        # Apply offset and limit
        paginated_experiments = all_experiments[offset:offset + limit]
        
        # Convert to list of dictionaries with relevant info
        experiment_list = []
        for exp in paginated_experiments:
            # Check if experiment is finalized
            is_finalized = all(job.status.finalized for job in exp.jobs) if exp.jobs else False
            
            # Get creation time
            created = exp.created.strftime("%Y-%m-%d %H:%M:%S") if exp.created else "Unknown"
            
            # Check if experiment has results
            has_results = False
            try:
                result_dataset = beaker.experiment.results(exp.id)
                has_results = result_dataset is not None
            except:
                has_results = False
            
            experiment_list.append({
                "id": exp.id,
                "name": exp.name or exp.id,
                "created": created,
                "finalized": is_finalized,
                "has_results": has_results,
                "author": exp.author.name if exp.author else "Unknown",
                "description": exp.description or ""
            })
        
        # Check if there are more experiments available
        has_more = len(all_experiments) > offset + limit
        
        return True, f"Found {len(experiment_list)} experiments (page {offset//limit + 1}){' - more available' if has_more else ''} in workspace '{workspace.full_name}'", experiment_list
    
    except Exception as e:
        return False, f"Error listing workspace experiments: {str(e)}", []

def get_workspace_info() -> Tuple[bool, str, Optional[str]]:
    """
    Get current workspace information.
    
    Returns:
        tuple: (success, message, workspace_name)
    """
    if not BEAKER_AVAILABLE:
        return False, "Beaker Python client not installed.", None
    
    try:
        beaker = Beaker.from_env()
        workspace = beaker.workspace.get()
        return True, f"Current workspace: {workspace.full_name}", workspace.full_name
    except Exception as e:
        return False, f"Error getting workspace info: {str(e)}", None

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
            if output_experiment_path.exists():
                return True, f"Local directory already exists at {output_experiment_path}", str(output_experiment_path)
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