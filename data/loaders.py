"""
Data loading utilities for model performance analysis
"""
import os
import json
import glob
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer
from .task_handlers import BaseTaskHandler
from utils.data_processing import get_cache_key, save_to_cache, load_from_cache
import streamlit as st

# Define datasets and their high-level categorization
DATASET_INFO = {
    "gpqa": {"type": "knowledge", "full_name": "GPQA"},
    "olympiadbench": {"type": "problem_solving", "full_name": "OlympiadBench"},
    "scibench": {"type": "knowledge", "full_name": "SciBench"},
    "scieval": {"type": "knowledge", "full_name": "SciEval"},
    "sciknoweval": {"type": "knowledge", "full_name": "SciKnowEval"},
    "mmlu_pro": {"type": "knowledge", "full_name": "MMLU-Pro"}
}

def extract_model_name(directory):
    """Extract model name from directory path"""
    # Get just the directory name without the path
    dir_name = os.path.basename(directory)
    
    # Extract model name from directory
    model_match = re.match(r'lmeval-([^-]+(?:-[^-]+)*)-on-', dir_name)
    if model_match:
        return model_match.group(1)
    return dir_name

def extract_dataset_name(directory):
    """Extract dataset name from directory path"""
    # Get just the directory name without the path
    dir_name = os.path.basename(directory)
    
    # Check specifically for mmlu_pro with its variant
    if "on-mmlu_pro" in dir_name:
        return "mmlu_pro"
    
    # Check for other datasets
    # Match anything between "on-" and the next dash OR end of string
    dataset_match = re.search(r'on-([^-]+)', dir_name)
    if dataset_match:
        return dataset_match.group(1)
    
    return "unknown"

def load_metrics(directory):
    """Load metrics from a model's directory"""
    metrics_file = os.path.join(directory, "metrics.json")
    
    if not os.path.exists(metrics_file):
        return None
    
    with open(metrics_file, 'r') as f:
        return json.load(f)

def compute_token_length(prediction):
    """Compute token length for a prediction if not present"""
    if "num_tokens" in prediction:
        return prediction["num_tokens"]
    
    # Get the output text
    output_text = prediction.get("continuation", "")
    if not output_text:
        return 0
    
    # Initialize tokenizer (using GPT-2 tokenizer as default)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Tokenize and get length
    tokens = tokenizer.encode(output_text)
    return len(tokens)

def load_predictions(directory):
    """Load predictions from a model's directory"""
    # For MMLU Pro, files are structured like task-000-mmlu_pro_math:cot-predictions.jsonl
    # For other datasets, they follow task-000-dataset-predictions.jsonl
    predictions_files = glob.glob(os.path.join(directory, "task-*-*predictions.jsonl"))
    
    if not predictions_files:
        return []
    
    predictions = []
    for pred_file in predictions_files:
        with open(pred_file, 'r') as f:
            for idx, line in enumerate(f):
                try:
                    pred = json.loads(line)
                    # Add metadata about source file if not present
                    if "metadata" not in pred:
                        pred["metadata"] = {}
                    pred["metadata"]["filename"] = os.path.basename(pred_file)
                    pred["metadata"]["line_number"] = idx
                    pred["metadata"]["directory"] = directory
                    
                    # Add token length if not present
                    if "num_tokens" not in pred:
                        # Check if "model_output" field exists and contains continuation
                        if "model_output" in pred and isinstance(pred["model_output"], list):
                            num_tokens = []
                            for output in pred["model_output"]:
                                num_tokens.append(compute_token_length(output))
                            pred["num_tokens"] = np.mean(num_tokens) if num_tokens else 0
                        else:
                            # Fall back to direct computation
                            pred["num_tokens"] = compute_token_length(pred)
                    predictions.append(pred)
                except json.JSONDecodeError:
                    continue
    
    return predictions

def get_dataset_domains(dataset_name, predictions=None, directory=None):
    """Load domains from dataset if available using the appropriate task handler"""
    # Get the appropriate task handler for this dataset
    handler = BaseTaskHandler.get_handler_for_dataset(dataset_name)
    
    # Get domain information using the handler
    return handler.get_domains(predictions=predictions, directory=directory)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_model_data(input_dir="outputs"):
    """Load all model data from the input directory with caching"""
    
    # Create cache key based on input directory and file modification times
    cache_key = get_cache_key(f"model_data_{input_dir}", [], [])
    
    # Try to load from cache first
    cached_data = load_from_cache(cache_key, "model_data", max_age_hours=1)
    if cached_data is not None:
        return cached_data["performance_df"], cached_data["all_models"], cached_data["model_datasets"]
    
    # Find all model directories
    directories = glob.glob(os.path.join(input_dir, "lmeval-*-on-*"))
    
    # Prepare data structures
    all_models = []
    model_datasets = defaultdict(list)
    performance_data = []
    
    # Process each directory
    for directory in directories:
        model_name = extract_model_name(directory)
        dataset_name = extract_dataset_name(directory)
        
        if model_name not in all_models:
            all_models.append(model_name)
        
        model_datasets[model_name].append(dataset_name)
        
        # Load metrics
        metrics_data = load_metrics(directory)
        if not metrics_data:
            continue
        
        # Extract primary score
        primary_score = 0
        extra_metrics = {}
        if "metrics" in metrics_data and len(metrics_data["metrics"]) > 0:
            primary_score = metrics_data["metrics"][0].get("primary_score", 0)
            extra_metrics = metrics_data["metrics"][0].get("extra_metrics", {})
        
        # Load predictions and compute average token length
        predictions = load_predictions(directory)
        avg_tokens = np.mean([p.get("num_tokens", 0) for p in predictions]) if predictions else 0
        
        # Add to performance data
        performance_data.append({
            "model": model_name,
            "dataset": dataset_name,
            "display_dataset": DATASET_INFO.get(dataset_name, {}).get("full_name", dataset_name),
            "accuracy": primary_score,
            "extra_metrics": extra_metrics,
            "directory": directory,
            "avg_tokens": avg_tokens
        })
    
    # Convert to dataframe
    if performance_data:
        performance_df = pd.DataFrame(performance_data)
        
        # Cache the results
        cache_data = {
            "performance_df": performance_df,
            "all_models": all_models,
            "model_datasets": dict(model_datasets)
        }
        save_to_cache(cache_key, cache_data, "model_data")
        
        return performance_df, all_models, model_datasets
    
    return pd.DataFrame(), [], {}

def load_domain_data(selected_models, selected_datasets, performance_df):
    """Load domain data for selected models and datasets with caching"""
    
    # Create cache key based on selected models and datasets
    cache_key = get_cache_key("domain_data", selected_models, selected_datasets)
    
    # Try to load from cache first
    cached_data = load_from_cache(cache_key, "domain_data", max_age_hours=2)
    if cached_data is not None:
        return cached_data["domain_df"], cached_data["subdomain_df"]
    
    domain_data = defaultdict(list)
    subdomain_data = defaultdict(list)
    
    # Filter performance data
    filtered_df = performance_df[
        (performance_df['model'].isin(selected_models)) & 
        (performance_df['dataset'].isin(selected_datasets))
    ]
    
    # Process each model/dataset combination
    for _, row in filtered_df.iterrows():
        model = row['model']
        dataset = row['dataset']
        directory = row['directory']
        
        # Load predictions with caching
        predictions = load_predictions_cached(directory, model, dataset)
        if not predictions:
            continue
        
        # Get task handler for this dataset
        handler = BaseTaskHandler.get_handler_for_dataset(dataset)
        
        # Process predictions with the task handler if needed
        predictions = handler.process_predictions(predictions)
        
        # Get domains for this dataset
        domains_info = handler.get_domains(predictions=predictions, directory=directory)
        if not domains_info:
            continue
        
        # Process high-level domains
        process_domain_performance(predictions, domains_info["high_level"], model, dataset, domain_data, handler)
        
        # Process subdomains
        process_domain_performance(predictions, domains_info["subdomain"], model, dataset, subdomain_data, handler)
    
    # Convert to dataframes
    domain_df = pd.DataFrame([
        {
            "model": item["model"],
            "dataset": item["dataset"],
            "domain": item["domain"],
            "accuracy": item["accuracy"],
            "correct": item["correct"],
            "total": item["total"]
        }
        for items in domain_data.values() for item in items
    ])
    
    subdomain_df = pd.DataFrame([
        {
            "model": item["model"],
            "dataset": item["dataset"],
            "subdomain": item["domain"],
            "accuracy": item["accuracy"],
            "correct": item["correct"],
            "total": item["total"]
        }
        for items in subdomain_data.values() for item in items
    ])
    
    # Cache the results
    cache_data = {
        "domain_df": domain_df,
        "subdomain_df": subdomain_df
    }
    save_to_cache(cache_key, cache_data, "domain_data")
    
    return domain_df, subdomain_df

def load_predictions_cached(directory, model, dataset):
    """Load predictions with caching support"""
    
    # Create cache key for this specific predictions file
    cache_key = get_cache_key(f"predictions_{directory}", [model], [dataset])
    
    # Try to load from cache first
    cached_predictions = load_from_cache(cache_key, "predictions", max_age_hours=12)
    if cached_predictions is not None:
        return cached_predictions
    
    # Load predictions normally
    predictions = load_predictions(directory)
    
    # Cache the predictions if they exist
    if predictions:
        save_to_cache(cache_key, predictions, "predictions")
    
    return predictions

def process_domain_performance(predictions, domains, model, dataset, domain_data, handler):
    """Process performance for each domain"""
    # Calculate per-domain performance
    domain_results = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for pred in predictions:
        native_id = pred.get("native_id")
        
        # Skip predictions with no native_id or native_id not in domains
        if native_id is None or native_id not in domains:
            continue
            
        domain = domains[native_id]
        # Use the handler to check correctness
        is_correct = handler.check_correctness(pred)
        
        if is_correct:
            domain_results[domain]["correct"] += 1
        domain_results[domain]["total"] += 1
    
    # Convert results to list
    for domain, counts in domain_results.items():
        if counts["total"] >= 5:  # Only consider domains with at least 5 samples
            domain_acc = counts["correct"] / counts["total"]
            domain_data[domain].append({
                "model": model,
                "dataset": dataset,
                "domain": domain,
                "accuracy": domain_acc,
                "correct": counts["correct"],
                "total": counts["total"]
            }) 