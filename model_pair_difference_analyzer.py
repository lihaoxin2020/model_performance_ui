#!/usr/bin/env python3
"""
Model Pair Difference Analyzer

This script compares correctness differences between pairs of models and measures 
the alignment between different difference sets.

Example usage:
- Extract instances where o3-mini-high is correct and o3-mini-low is incorrect
- Extract instances where o4-mini-high is correct and o4-mini-low is incorrect  
- Compare how similar these difference sets are

Usage:
    python model_pair_difference_analyzer.py /path/to/lmeval-api \
        --pair1 o3-mini-high o3-mini-low \
        --pair2 o4-mini-high o4-mini-low \
        --dataset ArpanSarkar/ReasoningIntensiveStrict
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import sys

# Import components from performance_analyzer
from performance_analyzer import ModelPerformanceAnalyzer

class ModelPairDifferenceAnalyzer:
    def __init__(self, lmeval_dir: str, dataset_name: str = None):
        """
        Initialize the analyzer with the lmeval directory.
        
        Args:
            lmeval_dir (str): Path to lmeval-api directory containing model directories
            dataset_name (str): Name of the dataset to use for reference
        """
        self.lmeval_dir = Path(lmeval_dir)
        self.dataset_name = dataset_name
        
        # Initialize the performance analyzer to get task mapping
        self.analyzer = ModelPerformanceAnalyzer(str(self.lmeval_dir), dataset_name)
        print(f"Initialized with dataset: {dataset_name}")
        
    def _find_model_directory(self, model_name: str) -> Optional[Path]:
        """Find the directory for a given model name."""
        model_dir = self.lmeval_dir / model_name
        if model_dir.exists() and model_dir.is_dir():
            return model_dir
        
        print(f"Warning: Model directory not found: {model_dir}")
        return None
    
    def _find_prediction_files(self, model_dir: Path) -> Dict[str, Path]:
        """Find all prediction files for a model and group by task."""
        prediction_files = {}
        
        # Look for prediction files in subdirectories
        for subdir in model_dir.glob('lmeval-*'):
            if subdir.is_dir():
                for pred_file in subdir.glob('*predictions.jsonl'):
                    # Extract task name from file path
                    task_name = self.analyzer._extract_task_from_file_path(pred_file)
                    prediction_files[task_name] = pred_file
                    
        return prediction_files
    
    def _load_predictions_for_model(self, model_name: str) -> Dict[str, pd.DataFrame]:
        """Load all prediction files for a model."""
        model_dir = self._find_model_directory(model_name)
        if not model_dir:
            return {}
            
        prediction_files = self._find_prediction_files(model_dir)
        predictions = {}
        
        for task_name, file_path in prediction_files.items():
            try:
                df = self.analyzer._load_predictions(file_path)
                # Filter to matching instances in reference dataset
                filtered_df = self.analyzer._filter_matching_instances(df, task_name)
                if not filtered_df.empty:
                    predictions[task_name] = filtered_df
                    print(f"Loaded {len(filtered_df)} instances for {model_name}/{task_name}")
                else:
                    print(f"No matching instances for {model_name}/{task_name}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return predictions
    
    def _get_correctness_for_prediction(self, row) -> bool:
        """Extract correctness from a prediction row."""
        # Try different correctness metrics in order of preference
        if 'exact_match_flex' in row and pd.notna(row['exact_match_flex']):
            return bool(row['exact_match_flex'])
        elif 'exact_match_simple' in row and pd.notna(row['exact_match_simple']):
            return bool(row['exact_match_simple'])
        elif 'correctness' in row and pd.notna(row['correctness']):
            return bool(row['correctness'])
        elif 'metrics' in row and isinstance(row['metrics'], dict):
            # Check metrics field
            for metric_name in ['exact_match_flex', 'exact_match_simple', 'correctness']:
                if metric_name in row['metrics']:
                    return bool(row['metrics'][metric_name])
        
        # Fallback to prediction/label comparison
        if 'prediction' in row and 'label' in row:
            return row['prediction'] == row['label']
            
        return False
    
    def _create_instance_key(self, row) -> Tuple[str, str]:
        """Create a unique key for an instance (task, doc_id)."""
        task = row.get('task', '')
        doc_id = str(row.get('doc_id', ''))
        return (task, doc_id)
    
    def extract_difference_set(self, model_high: str, model_low: str) -> Set[Tuple[str, str]]:
        """
        Extract instances where model_high is correct and model_low is incorrect.
        
        Args:
            model_high (str): Name of the high-performance model
            model_low (str): Name of the low-performance model
            
        Returns:
            Set[Tuple[str, str]]: Set of (task, doc_id) tuples where high is correct and low is incorrect
        """
        print(f"\nExtracting difference set for {model_high} vs {model_low}")
        
        # Load predictions for both models
        high_predictions = self._load_predictions_for_model(model_high)
        low_predictions = self._load_predictions_for_model(model_low)
        
        if not high_predictions or not low_predictions:
            print(f"Could not load predictions for both models")
            return set()
        
        # Find common tasks
        common_tasks = set(high_predictions.keys()) & set(low_predictions.keys())
        print(f"Common tasks: {common_tasks}")
        
        difference_set = set()
        
        for task in common_tasks:
            high_df = high_predictions[task]
            low_df = low_predictions[task]
            
            # Create mappings from instance key to correctness
            high_correctness = {}
            low_correctness = {}
            
            for _, row in high_df.iterrows():
                key = self._create_instance_key(row)
                high_correctness[key] = self._get_correctness_for_prediction(row)
                
            for _, row in low_df.iterrows():
                key = self._create_instance_key(row)
                low_correctness[key] = self._get_correctness_for_prediction(row)
            
            # Find instances where high is correct and low is incorrect
            common_instances = set(high_correctness.keys()) & set(low_correctness.keys())
            
            task_differences = 0
            for instance_key in common_instances:
                if high_correctness[instance_key] and not low_correctness[instance_key]:
                    difference_set.add(instance_key)
                    task_differences += 1
            
            print(f"Task {task}: {task_differences}/{len(common_instances)} instances where {model_high} correct and {model_low} incorrect")
        
        print(f"Total difference set size: {len(difference_set)} instances")
        return difference_set
    
    def calculate_alignment_metrics(self, set1: Set[Tuple[str, str]], set2: Set[Tuple[str, str]]) -> Dict[str, float]:
        """
        Calculate alignment metrics between two difference sets.
        
        Args:
            set1: First difference set
            set2: Second difference set
            
        Returns:
            Dict with alignment metrics
        """
        intersection = set1 & set2
        union = set1 | set2
        
        # Basic set metrics
        jaccard = len(intersection) / len(union) if union else 0.0
        overlap_coeff = len(intersection) / min(len(set1), len(set2)) if set1 and set2 else 0.0
        
        # Precision/recall treating set1 as "true" and set2 as "predicted"
        precision = len(intersection) / len(set2) if set2 else 0.0
        recall = len(intersection) / len(set1) if set1 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'intersection_size': len(intersection),
            'set1_size': len(set1),
            'set2_size': len(set2),
            'union_size': len(union),
            'jaccard_similarity': jaccard,
            'overlap_coefficient': overlap_coeff,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _extract_benchmark_name(self, task_name: str) -> str:
        """Extract benchmark name from task name."""
        # Remove common suffixes and extract base benchmark name
        task_name = task_name.replace(':cot', '').replace('_0shot', '').replace('_few_shot', '')
        
        # Handle specific patterns
        if task_name.startswith('lab_bench_'):
            return 'lab_bench'
        elif task_name.startswith('mmlu_pro'):
            return 'mmlu_pro'
        elif task_name.startswith('scibench'):
            return 'scibench'
        elif task_name.startswith('gpqa'):
            return 'gpqa'
        elif task_name.startswith('supergpqa'):
            return 'supergpqa'
        elif task_name.startswith('scieval'):
            return 'scieval'
        elif task_name.startswith('sciknoweval'):
            return 'sciknoweval'
        elif task_name.startswith('sciriff'):
            return 'sciriff'
        elif task_name.startswith('ugphysics'):
            return 'ugphysics'
        elif task_name.startswith('olympiadbench'):
            return 'olympiadbench'
        else:
            # For other cases, try to extract base name before underscore
            base_name = task_name.split('_')[0] if '_' in task_name else task_name
            return base_name

    def _should_exclude_task(self, task_name: str) -> bool:
        """Check if task should be excluded from analysis."""
        excluded_tasks = [
            'lab_bench_DbQA:cot',
            'lab_bench_LitQA2:cot', 
            'lab_bench_SuppQA:cot'
        ]
        return task_name in excluded_tasks

    def analyze_by_benchmark(self, set1: Set[Tuple[str, str]], set2: Set[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
        """Analyze alignment broken down by benchmark, excluding specified tasks."""
        # Group by benchmark, excluding certain tasks
        set1_by_benchmark = defaultdict(set)
        set2_by_benchmark = defaultdict(set)
        
        for task, doc_id in set1:
            if not self._should_exclude_task(task):
                benchmark = self._extract_benchmark_name(task)
                set1_by_benchmark[benchmark].add((task, doc_id))
            
        for task, doc_id in set2:
            if not self._should_exclude_task(task):
                benchmark = self._extract_benchmark_name(task)
                set2_by_benchmark[benchmark].add((task, doc_id))
        
        benchmark_metrics = {}
        all_benchmarks = set(set1_by_benchmark.keys()) | set(set2_by_benchmark.keys())
        
        for benchmark in all_benchmarks:
            benchmark_set1 = set1_by_benchmark.get(benchmark, set())
            benchmark_set2 = set2_by_benchmark.get(benchmark, set())
            
            benchmark_metrics[benchmark] = self.calculate_alignment_metrics(benchmark_set1, benchmark_set2)
        
        return benchmark_metrics

    def analyze_by_task(self, set1: Set[Tuple[str, str]], set2: Set[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
        """Analyze alignment broken down by task (kept for backward compatibility)."""
        # Group by task, excluding certain tasks
        set1_by_task = defaultdict(set)
        set2_by_task = defaultdict(set)
        
        for task, doc_id in set1:
            if not self._should_exclude_task(task):
                set1_by_task[task].add(doc_id)
            
        for task, doc_id in set2:
            if not self._should_exclude_task(task):
                set2_by_task[task].add(doc_id)
        
        task_metrics = {}
        all_tasks = set(set1_by_task.keys()) | set(set2_by_task.keys())
        
        for task in all_tasks:
            task_set1 = set1_by_task.get(task, set())
            task_set2 = set2_by_task.get(task, set())
            
            # Convert back to full tuples for consistency
            task_full_set1 = {(task, doc_id) for doc_id in task_set1}
            task_full_set2 = {(task, doc_id) for doc_id in task_set2}
            
            task_metrics[task] = self.calculate_alignment_metrics(task_full_set1, task_full_set2)
        
        return task_metrics
    
    def calculate_agreement_metrics(self, pair1: Tuple[str, str], pair2: Tuple[str, str]) -> Dict:
        """
        Calculate agreement metrics treating pair1 as ground truth and pair2 as predictions.
        For each instance, determine if there's a difference (high correct, low incorrect) and 
        measure how well pair2 agrees with pair1's patterns.
        
        Args:
            pair1: (high_model, low_model) tuple for ground truth pair
            pair2: (high_model, low_model) tuple for prediction pair
            
        Returns:
            Dict containing agreement metrics
        """
        model1_high, model1_low = pair1
        model2_high, model2_low = pair2
        
        print(f"\nCalculating agreement metrics (treating {model1_high} vs {model1_low} as ground truth)")
        
        # Load predictions for all models
        high1_predictions = self._load_predictions_for_model(model1_high)
        low1_predictions = self._load_predictions_for_model(model1_low)
        high2_predictions = self._load_predictions_for_model(model2_high)
        low2_predictions = self._load_predictions_for_model(model2_low)
        
        if not all([high1_predictions, low1_predictions, high2_predictions, low2_predictions]):
            print("Could not load predictions for all models")
            return {}
        
        # Find common tasks across all models
        all_tasks = (set(high1_predictions.keys()) & set(low1_predictions.keys()) & 
                    set(high2_predictions.keys()) & set(low2_predictions.keys()))
        print(f"Common tasks across all models: {all_tasks}")
        
        # For each instance, determine if there's a difference for each pair
        pair1_labels = []  # True if high1 correct and low1 incorrect, False otherwise
        pair2_predictions = []  # True if high2 correct and low2 incorrect, False otherwise
        instance_keys = []
        
        for task in all_tasks:
            # Skip excluded tasks
            if self._should_exclude_task(task):
                continue
            # Get correctness mappings for each model
            high1_correctness = {}
            low1_correctness = {}
            high2_correctness = {}
            low2_correctness = {}
            
            for _, row in high1_predictions[task].iterrows():
                key = self._create_instance_key(row)
                high1_correctness[key] = self._get_correctness_for_prediction(row)
                
            for _, row in low1_predictions[task].iterrows():
                key = self._create_instance_key(row)
                low1_correctness[key] = self._get_correctness_for_prediction(row)
                
            for _, row in high2_predictions[task].iterrows():
                key = self._create_instance_key(row)
                high2_correctness[key] = self._get_correctness_for_prediction(row)
                
            for _, row in low2_predictions[task].iterrows():
                key = self._create_instance_key(row)
                low2_correctness[key] = self._get_correctness_for_prediction(row)
            
            # Find instances common to all four models
            common_instances = (set(high1_correctness.keys()) & set(low1_correctness.keys()) & 
                              set(high2_correctness.keys()) & set(low2_correctness.keys()))
            
            for instance_key in common_instances:
                # Pair1 label: True if high1 correct and low1 incorrect
                pair1_diff = high1_correctness[instance_key] and not low1_correctness[instance_key]
                
                # Pair2 prediction: True if high2 correct and low2 incorrect  
                pair2_diff = high2_correctness[instance_key] and not low2_correctness[instance_key]
                
                pair1_labels.append(pair1_diff)
                pair2_predictions.append(pair2_diff)
                instance_keys.append(instance_key)
        
        if not pair1_labels:
            print("No common instances found")
            return {}
        
        # Convert to numpy arrays for easier calculation
        y_true = np.array(pair1_labels)
        y_pred = np.array(pair2_predictions)
        
        # Calculate classification metrics
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Confusion matrix with proper handling for edge cases
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape == (1, 1):
                # Only one class present
                if len(np.unique(y_true)) == 1 and np.unique(y_true)[0] == 0:
                    # All negative cases
                    tn = cm[0, 0]
                    fp = fn = tp = 0
                else:
                    # All positive cases
                    tp = cm[0, 0]
                    tn = fp = fn = 0
            else:
                tn, fp, fn, tp = cm.ravel()
        except ImportError:
            print("Warning: sklearn not available, using manual calculations")
            # Manual calculations
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            total = len(y_true)
            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate agreement breakdown
        total_instances = len(y_true)
        pair1_positives = np.sum(y_true)  # instances where pair1 shows difference
        pair1_negatives = total_instances - pair1_positives
        pair2_positives = np.sum(y_pred)  # instances where pair2 shows difference
        
        return {
            'total_instances': total_instances,
            'pair1_differences': int(pair1_positives),
            'pair1_non_differences': int(pair1_negatives),
            'pair2_differences': int(pair2_positives),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'agreement_rate': accuracy,  # Same as accuracy but clearer name
            'pair1_difficulty_rate': pair1_positives / total_instances,
            'pair2_difficulty_rate': pair2_positives / total_instances
        }

    def calculate_benchmark_agreement_metrics(self, pair1: Tuple[str, str], pair2: Tuple[str, str]) -> Dict[str, Dict]:
        """
        Calculate agreement metrics by benchmark, treating pair1 as ground truth.
        
        Args:
            pair1: (high_model, low_model) tuple for ground truth pair
            pair2: (high_model, low_model) tuple for prediction pair
            
        Returns:
            Dict mapping benchmark names to agreement metrics
        """
        model1_high, model1_low = pair1
        model2_high, model2_low = pair2
        
        print(f"\nCalculating benchmark-level agreement metrics")
        
        # Load predictions for all models
        high1_predictions = self._load_predictions_for_model(model1_high)
        low1_predictions = self._load_predictions_for_model(model1_low)
        high2_predictions = self._load_predictions_for_model(model2_high)
        low2_predictions = self._load_predictions_for_model(model2_low)
        
        if not all([high1_predictions, low1_predictions, high2_predictions, low2_predictions]):
            print("Could not load predictions for all models")
            return {}
        
        # Find common tasks across all models
        all_tasks = (set(high1_predictions.keys()) & set(low1_predictions.keys()) & 
                    set(high2_predictions.keys()) & set(low2_predictions.keys()))
        
        # Group instances by benchmark
        benchmark_data = defaultdict(lambda: {
            'pair1_labels': [],
            'pair2_predictions': [],
            'instance_keys': []
        })
        
        for task in all_tasks:
            # Skip excluded tasks
            if self._should_exclude_task(task):
                continue
                
            benchmark = self._extract_benchmark_name(task)
            
            # Get correctness mappings for each model
            high1_correctness = {}
            low1_correctness = {}
            high2_correctness = {}
            low2_correctness = {}
            
            for _, row in high1_predictions[task].iterrows():
                key = self._create_instance_key(row)
                high1_correctness[key] = self._get_correctness_for_prediction(row)
                
            for _, row in low1_predictions[task].iterrows():
                key = self._create_instance_key(row)
                low1_correctness[key] = self._get_correctness_for_prediction(row)
                
            for _, row in high2_predictions[task].iterrows():
                key = self._create_instance_key(row)
                high2_correctness[key] = self._get_correctness_for_prediction(row)
                
            for _, row in low2_predictions[task].iterrows():
                key = self._create_instance_key(row)
                low2_correctness[key] = self._get_correctness_for_prediction(row)
            
            # Find instances common to all four models
            common_instances = (set(high1_correctness.keys()) & set(low1_correctness.keys()) & 
                              set(high2_correctness.keys()) & set(low2_correctness.keys()))
            
            for instance_key in common_instances:
                # Pair1 label: True if high1 correct and low1 incorrect
                pair1_diff = high1_correctness[instance_key] and not low1_correctness[instance_key]
                
                # Pair2 prediction: True if high2 correct and low2 incorrect  
                pair2_diff = high2_correctness[instance_key] and not low2_correctness[instance_key]
                
                benchmark_data[benchmark]['pair1_labels'].append(pair1_diff)
                benchmark_data[benchmark]['pair2_predictions'].append(pair2_diff)
                benchmark_data[benchmark]['instance_keys'].append(instance_key)
        
        # Calculate metrics for each benchmark
        benchmark_metrics = {}
        for benchmark, data in benchmark_data.items():
            if not data['pair1_labels']:
                continue
                
            # Convert to numpy arrays
            y_true = np.array(data['pair1_labels'])
            y_pred = np.array(data['pair2_predictions'])
            
            # Calculate classification metrics
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # Confusion matrix with proper handling for edge cases
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                if cm.shape == (1, 1):
                    # Only one class present
                    if len(np.unique(y_true)) == 1 and np.unique(y_true)[0] == 0:
                        # All negative cases
                        tn = cm[0, 0]
                        fp = fn = tp = 0
                    else:
                        # All positive cases
                        tp = cm[0, 0]
                        tn = fp = fn = 0
                else:
                    tn, fp, fn, tp = cm.ravel()
                    
            except ImportError:
                # Manual calculations
                tp = np.sum((y_true == 1) & (y_pred == 1))
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                
                total = len(y_true)
                accuracy = (tp + tn) / total if total > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Calculate agreement breakdown
            total_instances = len(y_true)
            pair1_positives = np.sum(y_true)
            pair2_positives = np.sum(y_pred)
            
            benchmark_metrics[benchmark] = {
                'total_instances': total_instances,
                'pair1_differences': int(pair1_positives),
                'pair1_non_differences': int(total_instances - pair1_positives),
                'pair2_differences': int(pair2_positives),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'confusion_matrix': {
                    'true_negative': int(tn),
                    'false_positive': int(fp),
                    'false_negative': int(fn),
                    'true_positive': int(tp)
                },
                'pair1_difficulty_rate': pair1_positives / total_instances,
                'pair2_difficulty_rate': pair2_positives / total_instances
            }
        
        return benchmark_metrics
    
    def compare_model_pairs(self, pair1: Tuple[str, str], pair2: Tuple[str, str]) -> Dict:
        """
        Compare two model pairs and analyze their difference set alignment.
        
        Args:
            pair1: (high_model, low_model) tuple for first pair
            pair2: (high_model, low_model) tuple for second pair
            
        Returns:
            Dict containing comparison results
        """
        model1_high, model1_low = pair1
        model2_high, model2_low = pair2
        
        print(f"Comparing model pairs:")
        print(f"  Pair 1: {model1_high} vs {model1_low}")
        print(f"  Pair 2: {model2_high} vs {model2_low}")
        
        # Extract difference sets (original analysis)
        diff_set1 = self.extract_difference_set(model1_high, model1_low)
        diff_set2 = self.extract_difference_set(model2_high, model2_low)
        
        # Calculate alignment metrics (original analysis)
        alignment_metrics = {}
        benchmark_metrics = {}
        if diff_set1 and diff_set2:
            alignment_metrics = self.calculate_alignment_metrics(diff_set1, diff_set2)
            benchmark_metrics = self.analyze_by_benchmark(diff_set1, diff_set2)
        
        # Calculate agreement metrics (new analysis treating pair1 as ground truth)
        agreement_metrics = self.calculate_agreement_metrics(pair1, pair2)
        benchmark_agreement_metrics = self.calculate_benchmark_agreement_metrics(pair1, pair2)
        
        results = {
            'pair1': {'high': model1_high, 'low': model1_low},
            'pair2': {'high': model2_high, 'low': model2_low},
            'difference_set_analysis': {
                'set1_size': len(diff_set1),
                'set2_size': len(diff_set2),
                'alignment_metrics': alignment_metrics,
                'benchmark_breakdown': benchmark_metrics
            },
            'agreement_analysis': agreement_metrics,
            'benchmark_agreement_analysis': benchmark_agreement_metrics,
            'dataset': self.dataset_name
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print formatted results."""
        if not results:
            print("No results to display")
            return
            
        print("\n" + "="*80)
        print("MODEL PAIR COMPARISON ANALYSIS")
        print("="*80)
        
        pair1 = results['pair1']
        pair2 = results['pair2']
        
        print(f"\nDataset: {results['dataset']}")
        print(f"Pair 1: {pair1['high']} vs {pair1['low']}")
        print(f"Pair 2: {pair2['high']} vs {pair2['low']}")
        
        # Print agreement analysis (treating pair1 as ground truth)
        agreement = results.get('agreement_analysis', {})
        if agreement:
            print(f"\n" + "="*50)
            print("AGREEMENT ANALYSIS (Pair 1 as Ground Truth)")
            print("="*50)
            print(f"  Total common instances: {agreement['total_instances']}")
            print(f"  Pair 1 difficulty rate: {agreement['pair1_difficulty_rate']:.3f} ({agreement['pair1_differences']}/{agreement['total_instances']})")
            print(f"  Pair 2 difficulty rate: {agreement['pair2_difficulty_rate']:.3f} ({agreement['pair2_differences']}/{agreement['total_instances']})")
            print(f"\n  Agreement Metrics:")
            print(f"    Accuracy: {agreement['accuracy']:.3f}")
            print(f"    Precision: {agreement['precision']:.3f}")
            print(f"    Recall: {agreement['recall']:.3f}")
            print(f"    F1 score: {agreement['f1_score']:.3f}")
            print(f"    Specificity: {agreement['specificity']:.3f}")
            
            # Print confusion matrix
            cm = agreement['confusion_matrix']
            print(f"\n  Confusion Matrix:")
            print(f"    True Positives:  {cm['true_positive']:4d} (both pairs show difference)")
            print(f"    False Positives: {cm['false_positive']:4d} (only pair 2 shows difference)")
            print(f"    False Negatives: {cm['false_negative']:4d} (only pair 1 shows difference)")
            print(f"    True Negatives:  {cm['true_negative']:4d} (neither pair shows difference)")
        
        # Print difference set analysis
        diff_analysis = results.get('difference_set_analysis', {})
        metrics = diff_analysis.get('alignment_metrics', {})
        if metrics:
            print(f"\n" + "="*50)
            print("DIFFERENCE SET OVERLAP ANALYSIS")
            print("="*50)
            print(f"  Difference set 1 size: {diff_analysis['set1_size']}")
            print(f"  Difference set 2 size: {diff_analysis['set2_size']}")
            print(f"  Intersection size: {metrics['intersection_size']}")
            print(f"  Union size: {metrics['union_size']}")
            print(f"  Jaccard similarity: {metrics['jaccard_similarity']:.3f}")
            print(f"  Overlap coefficient: {metrics['overlap_coefficient']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1 score: {metrics['f1_score']:.3f}")
            
            print(f"\nBenchmark-level Breakdown:")
            benchmark_breakdown = diff_analysis.get('benchmark_breakdown', {})
            for benchmark, benchmark_metrics in sorted(benchmark_breakdown.items()):
                if benchmark_metrics['set1_size'] > 0 or benchmark_metrics['set2_size'] > 0:
                    print(f"  {benchmark}:")
                    print(f"    Set1: {benchmark_metrics['set1_size']}, Set2: {benchmark_metrics['set2_size']}, Overlap: {benchmark_metrics['intersection_size']}")
                    print(f"    Jaccard: {benchmark_metrics['jaccard_similarity']:.3f}, F1: {benchmark_metrics['f1_score']:.3f}")
        
        # Print benchmark-level agreement analysis
        benchmark_agreement = results.get('benchmark_agreement_analysis', {})
        if benchmark_agreement:
            print(f"\n" + "="*50)
            print("BENCHMARK-LEVEL AGREEMENT ANALYSIS")
            print("="*50)
            for benchmark in sorted(benchmark_agreement.keys()):
                metrics = benchmark_agreement[benchmark]
                print(f"\n  {benchmark}:")
                print(f"    Total instances: {metrics['total_instances']}")
                print(f"    Pair 1 difficulty rate: {metrics['pair1_difficulty_rate']:.3f} ({metrics['pair1_differences']}/{metrics['total_instances']})")
                print(f"    Pair 2 difficulty rate: {metrics['pair2_difficulty_rate']:.3f} ({metrics['pair2_differences']}/{metrics['total_instances']})")
                print(f"    Accuracy: {metrics['accuracy']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze correctness differences between model pairs')
    parser.add_argument('lmeval_dir', help='Path to lmeval-api directory containing model directories')
    parser.add_argument('--pair1', nargs=2, required=True, metavar=('HIGH', 'LOW'),
                        help='First model pair: high-performance model and low-performance model')
    parser.add_argument('--pair2', nargs=2, required=True, metavar=('HIGH', 'LOW'),
                        help='Second model pair: high-performance model and low-performance model')
    parser.add_argument('--dataset', default=None,
                        help='HuggingFace dataset to use for reference (default: ArpanSarkar/ReasoningIntensiveStrict)')
    parser.add_argument('--output', help='Output file to save results (JSON format)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ModelPairDifferenceAnalyzer(args.lmeval_dir, args.dataset)
    
    # Compare model pairs
    results = analyzer.compare_model_pairs(
        pair1=tuple(args.pair1),
        pair2=tuple(args.pair2)
    )
    
    # Print results
    analyzer.print_results(results)
    
    # Save results if requested
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main() 