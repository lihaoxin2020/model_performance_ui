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
        print(f"Dataset loaded with {len(self.reference_dataset['train'])} instances")
        self.task_doc_mapping = self._create_task_doc_mapping()
        print(f"Found {len(self.task_doc_mapping)} unique tasks in the reference dataset")
        for task, ids in list(self.task_doc_mapping.items())[:5]:
            print(f"Sample task: {task} with {len(ids)} document IDs")
        
    def _create_task_doc_mapping(self) -> Dict[str, List[str]]:
        """
        Create a mapping of taskname to list of doc_ids for quick lookup.
        """
        mapping = {}
        for item in self.reference_dataset['train']:
            task_name = item['taskname']
            doc_id = item['doc_id']
            if task_name in mapping:
                mapping[task_name].append(doc_id)
            else:
                mapping[task_name] = [doc_id]
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
            
            if row_task in self.task_doc_mapping:
                return doc_id in self.task_doc_mapping[row_task]
            return False
            
        matching_mask = predictions_df.apply(is_match, axis=1)
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
    
    def analyze_performance(self) -> Dict:
        """
        Analyze performance across all prediction files in the directory structure.
        
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
                        if 'exact_match_flex' in task_df.columns:
                            # Use pre-calculated exact match metric
                            accuracy = task_df['exact_match_flex'].mean()
                        elif 'exact_match_simple' in task_df.columns:
                            # Use pre-calculated exact match simple metric
                            accuracy = task_df['exact_match_simple'].mean()
                        elif 'correctness' in task_df.columns:
                            # Use pre-calculated correctness metric
                            accuracy = task_df['correctness'].mean()
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
                            elif 'prediction' in task_df.columns and 'label' in task_df.columns:
                                # Fall back to prediction/label comparison if metrics not available
                                accuracy = (task_df['prediction'] == task_df['label']).mean()
                            else:
                                print(f"Skipping task {task}: No accuracy metrics available")
                                continue
                        elif 'prediction' in task_df.columns and 'label' in task_df.columns:
                            # Fall back to prediction/label comparison if metrics not available
                            accuracy = (task_df['prediction'] == task_df['label']).mean()
                        else:
                            print(f"Skipping task {task}: No accuracy metrics available")
                            continue
                        
                        # Initialize nested dictionaries if needed
                        if model_name not in results:
                            results[model_name] = {}
                        if benchmark_name not in results[model_name]:
                            results[model_name][benchmark_name] = {}
                        
                        results[model_name][benchmark_name][task] = {
                            'accuracy': accuracy,
                            'num_instances': len(task_df)
                        }
                except Exception as e:
                    print(f"Error processing {pred_file}: {e}")
        
        return results
    
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
    parser.add_argument('--dataset', default="ArpanSarkar/ReasoningIntensiveStrict", 
                        help='HuggingFace dataset to use for matching (default: ArpanSarkar/ReasoningIntensiveStrict)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--list-dataset-tasks', action='store_true', 
                        help='List all tasks in the specified dataset and exit')
    args = parser.parse_args()
    
    # First initialize to load the dataset
    analyzer = ModelPerformanceAnalyzer(args.predictions_dir, dataset_name=args.dataset)
    
    # If requested, list all tasks in the dataset and exit
    if args.list_dataset_tasks:
        print(f"\nTasks available in {args.dataset} dataset:")
        tasks = sorted(analyzer.task_doc_mapping.keys())
        for task in tasks:
            doc_count = len(analyzer.task_doc_mapping[task])
            print(f"  - {task} ({doc_count} documents)")
        return
        
    # Otherwise continue with analysis
    results = analyzer.analyze_performance()
    analyzer.generate_summary_report(results, args.output)

if __name__ == "__main__":
    main() 