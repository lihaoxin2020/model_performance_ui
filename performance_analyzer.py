import os
import json
import pandas as pd
import numpy as np
import re
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class ModelPerformanceAnalyzer:
    def __init__(self, predictions_dir: str, dataset_name: str = "ArpanSarkar/ReasoningIntensiveStrict"):
        """
        Initialize the analyzer with the directory containing prediction files.
        
        Args:
            predictions_dir (str): Path to directory containing model prediction directories
            dataset_name (str): Name of the dataset to load from Hugging Face (default: ArpanSarkar/ReasoningIntensiveStrict)
        """
        self.predictions_dir = Path(predictions_dir)
        self.dataset_name = dataset_name
        print(f"Loading the dataset: {dataset_name}...")
        self.reference_dataset = load_dataset(dataset_name)
        self.math_reference_dataset = load_dataset("ArpanSarkar/merged_bench_annotated")
        print(f"Dataset loaded with {len(self.reference_dataset['train'])} instances")
        self.task_doc_mapping = self._create_task_doc_mapping()
        self.math_mapping = self._create_math_mapping()
        print(f"Found {len(self.task_doc_mapping)} unique (taskname, doc_id) pairs in the reference dataset")
        
        # Group by task for display purposes
        tasks_summary = {}
        for (task_name, doc_id), exists in self.task_doc_mapping.items():
            if exists:
                if task_name in tasks_summary:
                    tasks_summary[task_name] += 1
                else:
                    tasks_summary[task_name] = 1
        
        print(f"Grouped into {len(tasks_summary)} unique tasks")
        for task, count in list(tasks_summary.items())[:5]:
            print(f"Sample task: {task} with {count} document IDs")
        
        # Log math distribution in the dataset
        total_instances = len(self.reference_dataset['train'])
        
        # Count math instances only for items that exist in the main reference dataset
        math_instances = 0
        for (task_name, doc_id), exists in self.task_doc_mapping.items():
            # if "math" in task_name:
                # from IPython import embed; embed()
            if exists and self.math_mapping.get((task_name, doc_id), False):
                math_instances += 1
        
        non_math_instances = total_instances - math_instances
        
        has_requires_math_field = 'requires_math' in self.math_reference_dataset['train'].features
        math_method = "requires_math field" if has_requires_math_field else "heuristic detection"
        print(f"Math distribution in dataset (using {math_method}): {math_instances} require math ({math_instances/total_instances:.1%}), {non_math_instances} don't require math ({non_math_instances/total_instances:.1%})")
        
    def _create_task_doc_mapping(self) -> Dict[Tuple[str, str], bool]:
        """
        Create a mapping of (taskname, doc_id) to existence for quick lookup.
        """
        mapping = {}
        for item in self.reference_dataset['train']:
            task_name = item['taskname']
            doc_id = str(item['doc_id'])
            mapping[(task_name, doc_id)] = True
        return mapping
    
    def _create_math_mapping(self) -> Dict[Tuple[str, str], bool]:
        """
        Create a mapping of (taskname, doc_id) to requires_math field for quick lookup.
        If requires_math field is not available, use heuristic approach based on task names and content.
        
        Returns:
            Dict[Tuple[str, str], bool]: Mapping from (taskname, doc_id) to requires_math boolean
        """
        mapping = {}
        has_requires_math_field = 'requires_math' in self.math_reference_dataset['train'].features
        
        # Math-related task name patterns
        math_task_patterns = [
            'math', 'algebra', 'geometry', 'calculus', 'arithmetic', 'trigonometry',
            'statistics', 'probability', 'number_theory', 'combinatorics',
            'mmlu_pro_math', 'hendrycks_math', 'minerva_math', 'deepmind_math',
            'scibench', 'sat_math', 'aime', 'amc'
        ]
        
        # Math-related keywords in questions
        math_keywords = [
            'calculate', 'compute', 'solve', 'equation', 'formula', 'theorem',
            'derivative', 'integral', 'matrix', 'vector', 'polynomial',
            'logarithm', 'exponential', 'coefficient', 'probability',
            'factorial', 'permutation', 'combination', 'geometric',
            'algebraic', 'trigonometric', 'cosine', 'sine', 'tangent',
            'sum', 'product', 'quotient', 'remainder', 'prime', 'factor',
            'square root', 'cube root', 'absolute value', 'inequality',
            '$', '\\(', '\\)', '\\[', '\\]', '$$', 'frac{', '\\sqrt{'
        ]
        
        for item in self.math_reference_dataset['train']:
            task_name = item['taskname']
            doc_id = item['doc_id']
            
            if has_requires_math_field:
                # Use the actual field if available
                requires_math = item.get('requires_math', "NO") == "YES"
            else:
                raise ValueError(f"requires_math field not found in dataset {self.dataset_name}")
            mapping[(task_name, str(doc_id))] = requires_math
        return mapping
    
    def _load_predictions(self, file_path: Path) -> pd.DataFrame:
        """
        Load predictions from a JSON or JSONL file.
        
        Args:
            file_path (Path): Path to the prediction file
            
        Returns:
            pd.DataFrame: DataFrame containing predictions
        """
        if str(file_path).endswith('.jsonl'):
            # Handle JSONL files
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            # Handle JSON files
            with open(file_path, 'r') as f:
                data = json.load(f)
                
        return pd.DataFrame(data)
    
    def _filter_matching_instances(self, predictions_df: pd.DataFrame, task_name: str) -> pd.DataFrame:
        """
        Filter predictions to only include instances that match the reference dataset.
        
        Args:
            predictions_df (pd.DataFrame): DataFrame containing predictions
            task_name (str): Task name extracted from the file path
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        # Adjust column names if needed
        if 'taskname' in predictions_df.columns and 'task' not in predictions_df.columns:
            predictions_df['task'] = predictions_df['taskname']
        elif 'task_name' in predictions_df.columns and 'task' not in predictions_df.columns:
            predictions_df['task'] = predictions_df['task_name']
        
        # If no task column, add it using the extracted task name
        if 'task' not in predictions_df.columns:
            predictions_df['task'] = task_name
            
        # Create a mask for matching instances
        def is_match(row):
            # Get task name from row, preferring existing task field
            row_task = row.get('task', task_name)
            doc_id = row.get('doc_id', '')
            
            # Check if this (taskname, doc_id) combination exists in reference dataset
            return self.task_doc_mapping.get((row_task, str(doc_id)), False)
            
        matching_mask = predictions_df.apply(is_match, axis=1)
        return predictions_df[matching_mask]
    
    def _filter_matching_instances_math_only(self, predictions_df: pd.DataFrame, task_name: str) -> pd.DataFrame:
        """
        Filter predictions for math-only mode - use math_reference_dataset and exclude sciriff/sciknow tasks.
        
        Args:
            predictions_df (pd.DataFrame): DataFrame containing predictions
            task_name (str): Task name extracted from the file path
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        # Adjust column names if needed
        if 'taskname' in predictions_df.columns and 'task' not in predictions_df.columns:
            predictions_df['task'] = predictions_df['taskname']
        elif 'task_name' in predictions_df.columns and 'task' not in predictions_df.columns:
            predictions_df['task'] = predictions_df['task_name']
        
        # If no task column, add it using the extracted task name
        if 'task' not in predictions_df.columns:
            predictions_df['task'] = task_name
            
        # Create a mask for matching instances in math_reference_dataset
        def is_math_match(row):
            # Get task name from row, preferring existing task field
            row_task = row.get('task', task_name)
            doc_id = str(row.get('doc_id', ''))
            
            # Filter out sciriff and sciknow tasks
            if 'sciriff' in row_task.lower() or 'sciknow' in row_task.lower():
                return False
            
            # Check if this (taskname, doc_id) combination exists in math_reference_dataset
            return (row_task, doc_id) in self.math_mapping
            
        matching_mask = predictions_df.apply(is_math_match, axis=1)
        return predictions_df[matching_mask]
    
    def _parse_directory_name(self, dir_name: str) -> Tuple[str, str]:
        """
        Parse the directory name to extract model name and benchmark name.
        
        Args:
            dir_name (str): Directory name in format lmeval-[model_name]-on-[benchmark_name]-[hash]
            
        Returns:
            Tuple[str, str]: (model_name, benchmark_name)
        """
        pattern = r'lmeval-(.+)-on-(.+)-[a-f0-9]{10}'
        match = re.match(pattern, dir_name)
        
        if match:
            model_name = match.group(1)
            benchmark_name = match.group(2)
            return model_name, benchmark_name
        
        # Fallback if pattern doesn't match
        parts = dir_name.split('-')
        if len(parts) >= 4 and parts[0] == 'lmeval' and parts[-2] == 'on':
            model_name = '-'.join(parts[1:-2])
            benchmark_name = parts[-1]
        else:
            model_name = dir_name
            benchmark_name = "unknown"
            
        return model_name, benchmark_name
    
    def _extract_task_from_file_path(self, file_path: Path) -> str:
        """
        Extract task name from the prediction file path.
        
        Args:
            file_path (Path): Path to the prediction file
            
        Returns:
            str: Extracted task name
        """
        # First look for task pattern in the filename
        task_match = re.search(r'task-\d+-(.+)-predictions', file_path.stem)
        if task_match:
            return task_match.group(1)
        
        # Also check for task in the parent directory path
        for parent in file_path.parts:
            task_match = re.search(r'task-\d+-(.+)-predictions', parent)
            if task_match:
                return task_match.group(1)
                
        # If no match, try to extract benchmark name from directory
        for parent in file_path.parts:
            on_match = re.search(r'-on-(.+)-[a-f0-9]{10}', parent)
            if on_match:
                return on_match.group(1)
        
        # Default to the file stem if no pattern matches
        return file_path.stem
    
    def _find_prediction_files(self, subdirectory: Path) -> List[Path]:
        """
        Find all prediction files in a subdirectory.
        
        Args:
            subdirectory (Path): Path to subdirectory
            
        Returns:
            List[Path]: List of paths to prediction files
        """
        prediction_files = []
        # Look for files with "predictions" in the name
        for file_path in subdirectory.glob('**/*predictions*.json*'):
            prediction_files.append(file_path)
        
        # If no prediction files found, look for any JSON/JSONL files
        if not prediction_files:
            for file_path in subdirectory.glob('**/*.json*'):
                # Skip metrics.json as it's not a prediction file
                if file_path.name != 'metrics.json':
                    prediction_files.append(file_path)
                    
        return prediction_files
    
    def _analyze_math_performance(self, task_df: pd.DataFrame, accuracy_series: pd.Series) -> Dict:
        """
        Analyze performance specifically for math vs non-math questions.
        
        Args:
            task_df (pd.DataFrame): DataFrame containing predictions for a specific task
            accuracy_series (pd.Series): Series indicating correctness for each instance
            
        Returns:
            Dict: Dictionary containing math performance statistics
        """
        math_stats = {
            'total_instances': len(task_df),
            'math_instances': 0,
            'non_math_instances': 0,
            'correct_math': 0,
            'correct_non_math': 0,
            'math_accuracy': 0.0,
            'non_math_accuracy': 0.0,
            'math_distribution': 0.0
        }
        
        # Initialize counters
        math_correct = 0
        math_total = 0
        non_math_correct = 0
        non_math_total = 0
        
        # Check each instance for math requirement and correctness
        for idx, row in task_df.iterrows():
            task_name = row.get('task', '')
            doc_id = row.get('doc_id', '')
            
            # Get whether this instance requires math
            requires_math = self.math_mapping.get((task_name, str(doc_id)), False)
            
            # Get whether this instance was answered correctly
            is_correct = accuracy_series.loc[idx] if idx in accuracy_series.index else False
            
            # from IPython import embed; embed()
            if requires_math:
                math_total += 1
                if is_correct:
                    math_correct += 1
            else:
                non_math_total += 1
                if is_correct:
                    non_math_correct += 1
        
        # Update statistics
        math_stats.update({
            'math_instances': math_total,
            'non_math_instances': non_math_total,
            'correct_math': math_correct,
            'correct_non_math': non_math_correct,
            'math_accuracy': math_correct / math_total if math_total > 0 else 0.0,
            'non_math_accuracy': non_math_correct / non_math_total if non_math_total > 0 else 0.0,
            'math_distribution': math_total / len(task_df) if len(task_df) > 0 else 0.0
        })
        
        return math_stats

    def analyze_performance(self, math_only: bool = False) -> Dict:
        """
        Analyze performance across all prediction files in the directory structure.
        
        Args:
            math_only (bool): If True, use math-only filtering (math_reference_dataset, exclude sciriff/sciknow)
        
        Returns:
            Dict: Dictionary containing performance metrics by model, benchmark, and task
        """
        results = {}
        
        # Process each subdirectory
        for subdir in tqdm(list(self.predictions_dir.glob('*')), desc="Processing model directories"):
            if not subdir.is_dir():
                continue
                
            model_name, benchmark_name = self._parse_directory_name(subdir.name)
            
            # Find prediction files in this subdirectory
            prediction_files = self._find_prediction_files(subdir)
            
            for pred_file in prediction_files:
                try:
                    # Extract task name from file path
                    task_name = self._extract_task_from_file_path(pred_file)
                    
                    predictions_df = self._load_predictions(pred_file)
                    if math_only:
                        filtered_df = self._filter_matching_instances_math_only(predictions_df, task_name)
                    else:
                        filtered_df = self._filter_matching_instances(predictions_df, task_name)
                    
                    if filtered_df.empty:
                        print(f"No matching instances found in {pred_file}")
                        continue
                    
                    # Ensure we have necessary columns
                    if 'prediction' not in filtered_df.columns:
                        for col in ['pred', 'output', 'generated', 'prediction_text', 'completion']:
                            if col in filtered_df.columns:
                                filtered_df['prediction'] = filtered_df[col]
                                break
                    
                    if 'label' not in filtered_df.columns:
                        for col in ['target', 'ground_truth', 'answer', 'label_text', 'correct_answer']:
                            if col in filtered_df.columns:
                                filtered_df['label'] = filtered_df[col]
                                break
                    
                    # Calculate metrics by task
                    for task in filtered_df['task'].unique():
                        task_df = filtered_df[filtered_df['task'] == task]
                        
                        # Calculate accuracy - using existing metrics if available
                        accuracy_series = None
                        if 'exact_match_flex' in task_df.columns:
                            # Use pre-calculated exact match metric
                            accuracy = task_df['exact_match_flex'].mean()
                            accuracy_series = task_df['exact_match_flex']
                        elif 'exact_match_simple' in task_df.columns:
                            # Use pre-calculated exact match simple metric
                            accuracy = task_df['exact_match_simple'].mean()
                            accuracy_series = task_df['exact_match_simple']
                        elif 'correctness' in task_df.columns:
                            # Use pre-calculated correctness metric
                            accuracy = task_df['correctness'].mean()
                            accuracy_series = task_df['correctness']
                        elif 'metrics' in task_df.columns:
                            # Look for metrics within a metrics field (common in some datasets)
                            def extract_metric(row):
                                if isinstance(row.get('metrics'), dict):
                                    # Check multiple possible metric fields in order of preference
                                    for metric_name in ['exact_match_flex', 'exact_match_simple', 'correctness']:
                                        if metric_name in row['metrics']:
                                            return row['metrics'][metric_name]
                                return None
                            
                            metrics = task_df.apply(extract_metric, axis=1)
                            if not metrics.isna().all():
                                accuracy = metrics.mean()
                                accuracy_series = metrics
                            elif 'prediction' in task_df.columns and 'label' in task_df.columns:
                                # Fall back to prediction/label comparison if metrics not available
                                accuracy_series = (task_df['prediction'] == task_df['label'])
                                accuracy = accuracy_series.mean()
                            else:
                                print(f"Skipping task {task}: No accuracy metrics available")
                                continue
                        elif 'prediction' in task_df.columns and 'label' in task_df.columns:
                            # Fall back to prediction/label comparison if metrics not available
                            accuracy_series = (task_df['prediction'] == task_df['label'])
                            accuracy = accuracy_series.mean()
                        else:
                            print(f"Skipping task {task}: No accuracy metrics available")
                            continue
                        
                        # Analyze math performance
                        math_stats = self._analyze_math_performance(task_df, accuracy_series)
                        
                        # Log math performance for this task
                        print(f"Task {task}: {math_stats['correct_math']}/{math_stats['math_instances']} math correct ({math_stats['math_accuracy']:.1%}), "
                              f"{math_stats['correct_non_math']}/{math_stats['non_math_instances']} non-math correct ({math_stats['non_math_accuracy']:.1%}), "
                              f"math distribution: {math_stats['math_distribution']:.1%}")
                        
                        # Initialize nested dictionaries if needed
                        if model_name not in results:
                            results[model_name] = {}
                        if benchmark_name not in results[model_name]:
                            results[model_name][benchmark_name] = {}
                        
                        results[model_name][benchmark_name][task] = {
                            'accuracy': accuracy,
                            'num_instances': len(task_df),
                            'math_stats': math_stats
                        }
                except Exception as e:
                    print(f"Error processing {pred_file}: {e}")
        
        return results
    
    def generate_math_summary_report(self, results: Dict, output_file: Optional[str] = None):
        """
        Generate a comprehensive math performance summary report.
        
        Args:
            results (Dict): Results from analyze_performance()
            output_file (str, optional): Path to save the math summary report
        """
        if output_file is None:
            dataset_short_name = self.dataset_name.split('/')[-1]
            output_file = f"math-analysis-{dataset_short_name}-report.tsv"
        
        # Aggregate math statistics
        all_math_rows = []
        model_aggregates = {}
        
        for model_name, benchmarks in results.items():
            model_aggregates[model_name] = {
                'total_instances': 0,
                'math_instances': 0,
                'non_math_instances': 0,
                'correct_math': 0,
                'correct_non_math': 0
            }
            
            for benchmark_name, tasks in benchmarks.items():
                for task, metrics in tasks.items():
                    if 'math_stats' in metrics:
                        math_stats = metrics['math_stats']
                        
                        # Add task-level math row
                        all_math_rows.append({
                            'model': model_name,
                            'benchmark': benchmark_name,
                            'task': task,
                            'total_instances': math_stats['total_instances'],
                            'math_instances': math_stats['math_instances'],
                            'non_math_instances': math_stats['non_math_instances'],
                            'correct_math': math_stats['correct_math'],
                            'correct_non_math': math_stats['correct_non_math'],
                            'math_accuracy': math_stats['math_accuracy'],
                            'non_math_accuracy': math_stats['non_math_accuracy'],
                            'math_distribution': math_stats['math_distribution']
                        })
                        
                        # Add to model aggregates
                        model_aggregates[model_name]['total_instances'] += math_stats['total_instances']
                        model_aggregates[model_name]['math_instances'] += math_stats['math_instances']
                        model_aggregates[model_name]['non_math_instances'] += math_stats['non_math_instances']
                        model_aggregates[model_name]['correct_math'] += math_stats['correct_math']
                        model_aggregates[model_name]['correct_non_math'] += math_stats['correct_non_math']
        
        # Add model aggregate rows
        for model_name, agg in model_aggregates.items():
            if agg['total_instances'] > 0:
                all_math_rows.append({
                    'model': model_name,
                    'benchmark': 'ALL_BENCHMARKS',
                    'task': 'MODEL_AGGREGATE',
                    'total_instances': agg['total_instances'],
                    'math_instances': agg['math_instances'],
                    'non_math_instances': agg['non_math_instances'],
                    'correct_math': agg['correct_math'],
                    'correct_non_math': agg['correct_non_math'],
                    'math_accuracy': agg['correct_math'] / agg['math_instances'] if agg['math_instances'] > 0 else 0.0,
                    'non_math_accuracy': agg['correct_non_math'] / agg['non_math_instances'] if agg['non_math_instances'] > 0 else 0.0,
                    'math_distribution': agg['math_instances'] / agg['total_instances'] if agg['total_instances'] > 0 else 0.0
                })
        
        if all_math_rows:
            # Create DataFrame and save to TSV
            math_df = pd.DataFrame(all_math_rows)
            math_df.to_csv(output_file, sep='\t', index=False)
            print(f"Math analysis report saved to {output_file} (TSV format)")
            
            # Print summary statistics
            print("\n" + "="*80)
            print("MATH PERFORMANCE ANALYSIS SUMMARY")
            print("="*80)
            
            # Overall statistics
            aggregate_rows = math_df[math_df['task'] == 'MODEL_AGGREGATE']
            if not aggregate_rows.empty:
                print("\nOverall Math Performance by Model:")
                for _, row in aggregate_rows.iterrows():
                    print(f"{row['model']}:")
                    print(f"  Total instances: {row['total_instances']}")
                    print(f"  Math instances: {row['math_instances']} ({row['math_distribution']:.1%})")
                    print(f"  Non-math instances: {row['non_math_instances']} ({(1-row['math_distribution']):.1%})")
                    print(f"  Math accuracy: {row['math_accuracy']:.1%} ({row['correct_math']}/{row['math_instances']})")
                    print(f"  Non-math accuracy: {row['non_math_accuracy']:.1%} ({row['correct_non_math']}/{row['non_math_instances']})")
                    print()
            
            # Task-level breakdown for tasks with both math and non-math questions
            task_rows = math_df[~math_df['task'].isin(['MODEL_AGGREGATE'])]
            mixed_tasks = task_rows[(task_rows['math_instances'] > 0) & (task_rows['non_math_instances'] > 0)]
            if not mixed_tasks.empty:
                print("Tasks with both math and non-math questions:")
                for _, row in mixed_tasks.iterrows():
                    print(f"{row['task']} ({row['model']}):")
                    print(f"  Math: {row['math_accuracy']:.1%} ({row['correct_math']}/{row['math_instances']})")
                    print(f"  Non-math: {row['non_math_accuracy']:.1%} ({row['correct_non_math']}/{row['non_math_instances']})")
                    print(f"  Distribution: {row['math_distribution']:.1%} math, {(1-row['math_distribution']):.1%} non-math")
                    print()
        else:
            print("No math statistics available to generate report.")

    def generate_summary_report(self, results: Dict, output_file: Optional[str] = None):
        """
        Generate a summary report of the performance analysis.
        
        Args:
            results (Dict): Results from analyze_performance()
            output_file (str, optional): Path to save the summary report. If None, defaults to [model_name]-[dataset_name]-results.tsv
        """
        summary_rows = []
        
        # Get unique model names (to use in default filename if needed)
        model_names = list(results.keys())
        
        # Extract the dataset name for the filename (use last part after /)
        dataset_short_name = self.dataset_name.split('/')[-1]
        
        # Set default output file name based on model name if not provided
        if output_file is None:
            if len(model_names) == 1:
                # If only one model, use its name and dataset name
                output_file = f"{model_names[0]}-{dataset_short_name}-results.tsv"
            else:
                # If multiple models, use combined name with dataset
                output_file = f"combined-{dataset_short_name}-results.tsv"
        
        # Collect all rows for the summary report
        for model_name, benchmarks in results.items():
            model_total_instances = 0
            model_weighted_accuracy = 0
            
            for benchmark_name, tasks in benchmarks.items():
                benchmark_total_instances = 0
                benchmark_weighted_accuracy = 0
                
                for task, metrics in tasks.items():
                    accuracy = metrics['accuracy']
                    num_instances = metrics['num_instances']
                    
                    # Add to model and benchmark totals
                    model_total_instances += num_instances
                    model_weighted_accuracy += accuracy * num_instances
                    benchmark_total_instances += num_instances
                    benchmark_weighted_accuracy += accuracy * num_instances
                    
                    # Add task-level row
                    summary_rows.append({
                        'model': model_name,
                        'benchmark': benchmark_name,
                        'task': task,
                        'accuracy': accuracy,
                        'num_instances': num_instances
                    })
                
                # Calculate benchmark average (if there were instances)
                if benchmark_total_instances > 0:
                    # Add benchmark average row
                    summary_rows.append({
                        'model': model_name,
                        'benchmark': benchmark_name,
                        'task': 'BENCHMARK_AVERAGE',
                        'accuracy': benchmark_weighted_accuracy / benchmark_total_instances,
                        'num_instances': benchmark_total_instances
                    })
            
            # Calculate model average (if there were instances)
            if model_total_instances > 0:
                # Add model average row
                summary_rows.append({
                    'model': model_name,
                    'benchmark': 'ALL_BENCHMARKS',
                    'task': 'MODEL_AVERAGE',
                    'accuracy': model_weighted_accuracy / model_total_instances,
                    'num_instances': model_total_instances
                })
        
        if not summary_rows:
            print("No matching instances found in any of the prediction files.")
            print(f"Creating empty summary report at {output_file}")
            # Create an empty DataFrame with the expected columns
            summary_df = pd.DataFrame(columns=['model', 'benchmark', 'task', 'accuracy', 'num_instances'])
            summary_df.to_csv(output_file, sep='\t', index=False)
            return
            
        # Create DataFrame and save to TSV
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_file, sep='\t', index=False)
        print(f"Summary report saved to {output_file} (TSV format)")
        
        # Print overall statistics by model and benchmark
        print("\nOverall Statistics by Model:")
        model_stats = summary_df[~summary_df['task'].isin(['BENCHMARK_AVERAGE', 'MODEL_AVERAGE'])].groupby('model').agg({
            'accuracy': lambda x: np.average(x, weights=summary_df.loc[x.index, 'num_instances']),
            'num_instances': 'sum'
        }).round(4)
        print(model_stats)
        
        print("\nOverall Statistics by Model and Benchmark:")
        benchmark_stats = summary_df[~summary_df['task'].isin(['BENCHMARK_AVERAGE', 'MODEL_AVERAGE'])].groupby(['model', 'benchmark']).agg({
            'accuracy': lambda x: np.average(x, weights=summary_df.loc[x.index, 'num_instances']),
            'num_instances': 'sum'
        }).round(4)
        print(benchmark_stats)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze model performance across benchmarks')
    parser.add_argument('predictions_dir', help='Directory containing model prediction directories')
    parser.add_argument('--output', help='Output file path (defaults to [model_name]-[dataset_name]-results.tsv)')
    parser.add_argument('--dataset', default="ArpanSarkar/ReasoningIntensiveLoose", 
                        help='HuggingFace dataset to use for matching (default: ArpanSarkar/ReasoningIntensiveLoose)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--list-dataset-tasks', action='store_true', 
                        help='List all tasks in the specified dataset and exit')
    parser.add_argument('--math-analysis', action='store_true', default=True,
                        help='Generate math performance analysis (enabled by default)')
    parser.add_argument('--no-math-analysis', action='store_true',
                        help='Disable math performance analysis')
    parser.add_argument('--math-only', action='store_true',
                        help='Generate only math analysis report (skip regular performance report)')
    args = parser.parse_args()
    
    # First initialize to load the dataset
    analyzer = ModelPerformanceAnalyzer(args.predictions_dir, dataset_name=args.dataset)
    
    # If requested, list all tasks in the dataset and exit
    if args.list_dataset_tasks:
        print(f"\nTasks available in {args.dataset} dataset:")
        
        # Group by task for display purposes
        tasks_summary = {}
        for (task_name, doc_id), exists in analyzer.task_doc_mapping.items():
            if exists:
                if task_name in tasks_summary:
                    tasks_summary[task_name] += 1
                else:
                    tasks_summary[task_name] = 1
        
        for task in sorted(tasks_summary.keys()):
            doc_count = tasks_summary[task]
            print(f"  - {task} ({doc_count} documents)")
        return
        
    # Otherwise continue with analysis
    results = analyzer.analyze_performance(args.math_only)
    
    # Determine what reports to generate based on arguments
    generate_math = args.math_analysis and not args.no_math_analysis
    
    if args.math_only:
        # Generate only math analysis
        analyzer.generate_math_summary_report(results)
    else:
        # Generate regular summary report
        analyzer.generate_summary_report(results, args.output)
        
        # Generate math analysis if requested
        if generate_math:
            analyzer.generate_math_summary_report(results)

if __name__ == "__main__":
    main() 