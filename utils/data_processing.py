"""
Data processing utilities for integrating external results into the model performance dashboard
"""
import os
import json
import pickle
import shutil
import hashlib
from pathlib import Path
import uuid
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import streamlit as st
from datetime import datetime

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

def get_cache_key(data_identifier: str, selected_models: list, selected_datasets: list) -> str:
    """Generate a cache key based on data identifier and selections"""
    key_data = {
        "identifier": data_identifier,
        "models": sorted(selected_models),
        "datasets": sorted(selected_datasets)
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def get_cache_dir() -> Path:
    """Get the cache directory for storing processed data"""
    if 'input_dir' in st.session_state:
        cache_dir = Path(st.session_state.input_dir) / ".cache"
    else:
        cache_dir = Path(".cache")
    
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

def save_to_cache(cache_key: str, data: Any, cache_type: str = "general") -> bool:
    """
    Save data to cache
    
    Args:
        cache_key (str): Unique identifier for the cached data
        data (Any): Data to cache
        cache_type (str): Type of cache (general, domain, predictions, etc.)
    
    Returns:
        bool: True if saved successfully
    """
    try:
        cache_dir = get_cache_dir() / cache_type
        cache_dir.mkdir(exist_ok=True)
        
        cache_file = cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump({
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "cache_key": cache_key
            }, f)
        
        return True
    except Exception as e:
        st.warning(f"Failed to save to cache: {str(e)}")
        return False

def load_from_cache(cache_key: str, cache_type: str = "general", max_age_hours: int = 24) -> Optional[Any]:
    """
    Load data from cache
    
    Args:
        cache_key (str): Unique identifier for the cached data
        cache_type (str): Type of cache (general, domain, predictions, etc.)
        max_age_hours (int): Maximum age of cache in hours
    
    Returns:
        Any: Cached data if available and not expired, None otherwise
    """
    try:
        cache_dir = get_cache_dir() / cache_type
        cache_file = cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Check if cache is not too old
        cache_time = datetime.fromisoformat(cached_data["timestamp"])
        current_time = datetime.now()
        age_hours = (current_time - cache_time).total_seconds() / 3600
        
        if age_hours > max_age_hours:
            # Remove expired cache
            cache_file.unlink()
            return None
        
        return cached_data["data"]
    except Exception as e:
        # If there's any error loading cache, just return None
        return None

def clear_cache(cache_type: Optional[str] = None) -> bool:
    """
    Clear cache files
    
    Args:
        cache_type (str, optional): Specific cache type to clear. If None, clears all cache.
    
    Returns:
        bool: True if cleared successfully
    """
    try:
        cache_dir = get_cache_dir()
        
        if cache_type:
            cache_subdir = cache_dir / cache_type
            if cache_subdir.exists():
                shutil.rmtree(cache_subdir)
        else:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
        
        return True
    except Exception as e:
        st.warning(f"Failed to clear cache: {str(e)}")
        return False

def get_cache_info() -> Dict[str, Any]:
    """Get information about cache usage"""
    cache_dir = get_cache_dir()
    
    if not cache_dir.exists():
        return {"total_size": 0, "file_count": 0, "cache_types": []}
    
    total_size = 0
    file_count = 0
    cache_types = []
    
    for cache_type_dir in cache_dir.iterdir():
        if cache_type_dir.is_dir():
            cache_types.append(cache_type_dir.name)
            for cache_file in cache_type_dir.glob("*.pkl"):
                total_size += cache_file.stat().st_size
                file_count += 1
    
    return {
        "total_size": total_size,
        "file_count": file_count,
        "cache_types": cache_types,
        "size_mb": total_size / (1024 * 1024)
    } 