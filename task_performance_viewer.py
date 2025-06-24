#!/usr/bin/env python3
"""
Task Performance Viewer

This script displays performance metrics for each task/subtask from lmeval results.
It reads metrics-all.jsonl files from benchmark subdirectories and presents them
in a clear, organized format.

The script supports two modes:
1. Single model mode: Analyze one model's performance across tasks
2. Comparison mode: Compare multiple models side-by-side

Usage:
    # Single model mode
    python task_performance_viewer.py <model_directory> [options]
    
    # Comparison mode
    python task_performance_viewer.py --comparison-dir <directory_with_models> --comparison-mode [options]
    
Example:
    # Single model analysis
    python task_performance_viewer.py lmeval/Qwen3-32B
    python task_performance_viewer.py lmeval/deepeek-r1 --benchmark mmlu_pro
    python task_performance_viewer.py lmeval/Qwen3-32B --format table --sort-by primary_score
    
    # Multi-model comparison
    python task_performance_viewer.py --comparison-dir lmeval_results/ --comparison-mode
    python task_performance_viewer.py --comparison-dir models/ --benchmark mmlu --comparison-mode
    python task_performance_viewer.py --comparison-dir models/ --comparison-mode --show-individual-tasks
"""

import argparse
import json
import os
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tabulate import tabulate


class TaskPerformanceViewer:
    """Viewer for task-level performance metrics from lmeval results."""
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the viewer with a model directory.
        
        Args:
            model_dir (str, optional): Path to the model directory containing lmeval results
        """
        if model_dir:
            self.model_dir = Path(model_dir)
            if not self.model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            self.model_name = self._extract_model_name()
        else:
            self.model_dir = None
            self.model_name = None
        
    def _extract_model_name(self) -> str:
        """Extract model name from the directory path."""
        # If the directory name follows the pattern lmeval/model_name, extract model_name
        if self.model_dir.parent.name == "lmeval":
            return self.model_dir.name
        return self.model_dir.name
    
    def _parse_directory_name(self, dir_name: str) -> Tuple[str, str]:
        """
        Parse lmeval directory name to extract model and benchmark info.
        
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
            benchmark_name = benchmark_name.replace('::', '-').replace(':', '-')
            return model_name, benchmark_name
        
        # Fallback pattern for different naming conventions
        if '-on-' in dir_name:
            parts = dir_name.split('-on-')
            if len(parts) >= 2:
                model_part = parts[0].replace('lmeval-', '')
                benchmark_part = parts[1].split('-')[0]  # Remove hash
                benchmark_part = benchmark_part.replace('::', '-').replace(':', '-')
                return model_part, benchmark_part
        
        return dir_name, "unknown"
    
    def _load_metrics_file(self, metrics_path: Path) -> List[Dict]:
        """
        Load and parse a metrics-all.jsonl file.
        
        Args:
            metrics_path (Path): Path to the metrics-all.jsonl file
            
        Returns:
            List[Dict]: List of metric dictionaries
        """
        metrics = []
        try:
            with open(metrics_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            metric_data = json.loads(line)
                            metrics.append(metric_data)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse line {line_num} in {metrics_path}: {e}")
                            continue
        except Exception as e:
            print(f"Error reading {metrics_path}: {e}")
            return []
        
        return metrics
    
    def _extract_task_metrics(self, metric_data: Dict) -> Dict:
        """
        Extract relevant metrics from a task metric entry.
        
        Args:
            metric_data (Dict): Raw metric data from JSONL
            
        Returns:
            Dict: Cleaned metrics with key information
        """
        task_name = metric_data.get('task_name', 'unknown')
        
        task_core = metric_data.get('task_config', {}).get('task_core', 'unknown')
        num_instances = metric_data.get('num_instances', 0)
        
        # Extract main metrics
        metrics = metric_data.get('metrics', {})
        
        # Extract primary score and other important metrics
        result = {
            'task_name': task_name,
            'task_core': task_core,
            'num_instances': num_instances,
            'primary_score': metrics.get('primary_score', None),
            'exact_match': metrics.get('exact_match_macro', metrics.get('exact_match', None)),
            'exact_match_simple': metrics.get('exact_match_simple_macro', metrics.get('exact_match_simple', None)),
        }
        
        # Add extra metrics if available
        extra_metrics = metrics.get('extra_metrics', {})
        if extra_metrics:
            result.update({
                'max_tokens_reached': extra_metrics.get('max_tokens_reached_macro', None),
                'avg_tokens': extra_metrics.get('num_tokens_macro', None),
                'total_price': extra_metrics.get('total_price', None),
                'answer_format_correct': extra_metrics.get('answer_format_correct_macro', None)
            })
        
        # Extract special sciriff metrics for custom aggregation
        sciriff_metrics = {}
        if 'sciriff' in task_name.lower():
            # Extract specific metrics based on task type
            if 'evidence_inference' in task_name:
                sciriff_metrics['f1_overlap'] = metrics.get('f1_overlap', None)
            elif 'qasper_abstractive_qa' in task_name:
                sciriff_metrics['llm_score'] = metrics.get('llm_score', None)
                sciriff_metrics['f1_evidence_all'] = metrics.get('f1_evidence_all', None)
            elif 'scifact_entailment' in task_name:
                sciriff_metrics['f1_label'] = metrics.get('f1_label', None)
                sciriff_metrics['f1_evidence_token'] = metrics.get('f1_evidence_token', None)
        
        # Add sciriff metrics to result
        result.update(sciriff_metrics)
        
        # Calculate processing time per instance if available
        processing_time = metric_data.get('processing_time', None)
        if processing_time and num_instances > 0:
            result['time_per_instance'] = processing_time / num_instances
        
        return result
    
    def discover_benchmarks(self, benchmark_filter: Optional[str] = None) -> List[Tuple[str, Path]]:
        """
        Discover all benchmark directories in the model directory.
        
        Args:
            benchmark_filter (str, optional): Filter to only include benchmarks matching this string
            
        Returns:
            List[Tuple[str, Path]]: List of (benchmark_name, path) tuples
        """
        if not self.model_dir:
            raise ValueError("Model directory not set")
            
        benchmarks = []
        
        for subdir in self.model_dir.iterdir():
            if not subdir.is_dir():
                continue
                
            # Look for metrics-all.jsonl file
            metrics_file = subdir / "metrics-all.jsonl"
            if not metrics_file.exists():
                continue
            
            # Extract benchmark name from directory
            _, benchmark_name = self._parse_directory_name(subdir.name)
            
            # Apply filter if specified
            if benchmark_filter and benchmark_filter.lower() not in benchmark_name.lower():
                continue
                
            benchmarks.append((benchmark_name, subdir))
        
        return sorted(benchmarks)
    
    def discover_models(self, comparison_dir: str, benchmark_filter: Optional[str] = None) -> Dict[str, List[Tuple[str, Path]]]:
        """
        Discover all models and their benchmarks in a comparison directory.
        
        Args:
            comparison_dir (str): Path to directory containing model subdirectories
            benchmark_filter (str, optional): Filter to only include benchmarks matching this string
            
        Returns:
            Dict[str, List[Tuple[str, Path]]]: Dictionary mapping model names to lists of (benchmark_name, path) tuples
        """
        comparison_path = Path(comparison_dir)
        if not comparison_path.exists():
            raise FileNotFoundError(f"Comparison directory not found: {comparison_dir}")
        
        models = {}
        
        for model_subdir in comparison_path.iterdir():
            if not model_subdir.is_dir():
                continue
            
            model_name = model_subdir.name
            
            # Temporarily set model directory to discover benchmarks
            original_model_dir = self.model_dir
            original_model_name = self.model_name
            
            self.model_dir = model_subdir
            self.model_name = model_name
            
            try:
                benchmarks = self.discover_benchmarks(benchmark_filter)
                if benchmarks:  # Only include models that have benchmark data
                    models[model_name] = benchmarks
            except Exception as e:
                print(f"Warning: Could not process model {model_name}: {e}")
            finally:
                # Restore original values
                self.model_dir = original_model_dir
                self.model_name = original_model_name
        
        return models
    
    def load_all_metrics(self, benchmark_filter: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Load metrics from all discovered benchmarks.
        If multiple benchmark directories have the same name, their metrics will be averaged.
        
        Args:
            benchmark_filter (str, optional): Filter to only include benchmarks matching this string
            
        Returns:
            Dict[str, List[Dict]]: Dictionary mapping benchmark names to lists of task metrics
        """
        # First, collect all metrics grouped by benchmark name
        all_metrics_by_name = {}
        benchmarks = self.discover_benchmarks(benchmark_filter)
        
        print(f"Found {len(benchmarks)} benchmark(s) for model {self.model_name}")
        
        for benchmark_name, benchmark_path in benchmarks:
            metrics_file = benchmark_path / "metrics-all.jsonl"
            print(f"Loading metrics from {benchmark_name}...")
            
            raw_metrics = self._load_metrics_file(metrics_file)
            
            # Process metrics - first entry is usually aggregate, rest are individual tasks
            processed_metrics = []
            for i, metric_data in enumerate(raw_metrics):
                processed = self._extract_task_metrics(metric_data)
                processed['is_aggregate'] = (i == 0)  # First entry is typically aggregate
                processed['benchmark'] = benchmark_name
                processed['benchmark_path'] = str(benchmark_path)
                processed_metrics.append(processed)
            
            # Group by benchmark name (in case there are multiple dirs with same name)
            if benchmark_name not in all_metrics_by_name:
                all_metrics_by_name[benchmark_name] = []
            all_metrics_by_name[benchmark_name].append(processed_metrics)
            
            print(f"  Loaded {len(processed_metrics)} metric entries ({len(processed_metrics)-1} tasks + 1 aggregate)")
        
        # Average metrics for benchmarks with the same name
        final_metrics = {}
        for benchmark_name, metrics_lists in all_metrics_by_name.items():
            if len(metrics_lists) == 1:
                # Only one run for this benchmark, use as-is
                final_metrics[benchmark_name] = metrics_lists[0]
            else:
                # Multiple runs for this benchmark, average them
                print(f"  Averaging {len(metrics_lists)} runs for benchmark {benchmark_name}")
                averaged_metrics = self._average_benchmark_metrics(metrics_lists, benchmark_name)
                final_metrics[benchmark_name] = averaged_metrics
        
        # Apply custom aggregation for sciriff, supergpqa, and lab_bench
        final_metrics = self._create_custom_aggregates(final_metrics)
        
        return final_metrics
    
    def _average_benchmark_metrics(self, metrics_lists: List[List[Dict]], benchmark_name: str) -> List[Dict]:
        """
        Average metrics from multiple runs of the same benchmark.
        
        Args:
            metrics_lists: List of metrics lists from different runs
            benchmark_name: Name of the benchmark
            
        Returns:
            List[Dict]: Averaged metrics list
        """
        if not metrics_lists:
            return []
        
        # Separate aggregates and individual tasks
        all_aggregates = []
        all_tasks_by_name = {}
        
        for metrics_list in metrics_lists:
            for metric in metrics_list:
                if metric.get('is_aggregate', False):
                    all_aggregates.append(metric)
                else:
                    task_name = metric['task_name']
                    if task_name not in all_tasks_by_name:
                        all_tasks_by_name[task_name] = []
                    all_tasks_by_name[task_name].append(metric)
        
        # Average aggregate metrics
        averaged_metrics = []
        if all_aggregates:
            averaged_aggregate = self._average_metrics_entries(all_aggregates, benchmark_name, is_aggregate=True)
            averaged_metrics.append(averaged_aggregate)
        
        # Average individual task metrics
        for task_name, task_metrics in all_tasks_by_name.items():
            averaged_task = self._average_metrics_entries(task_metrics, benchmark_name, is_aggregate=False)
            averaged_metrics.append(averaged_task)
        
        return averaged_metrics
    
    def _average_metrics_entries(self, metrics_entries: List[Dict], benchmark_name: str, is_aggregate: bool = False) -> Dict:
        """
        Average a list of metrics entries (either aggregates or individual tasks).
        If there are at least 5 entries, also compute standard deviation as error range.
        
        Args:
            metrics_entries: List of metrics dictionaries to average
            benchmark_name: Name of the benchmark
            is_aggregate: Whether these are aggregate metrics
            
        Returns:
            Dict: Averaged metrics entry with std_dev if applicable
        """
        if not metrics_entries:
            return {}
        
        # Use first entry as template
        template = metrics_entries[0].copy()
        
        # Numeric fields to average
        numeric_fields = [
            'primary_score', 'exact_match', 'exact_match_simple', 
            'max_tokens_reached', 'avg_tokens', 'total_price', 
            'answer_format_correct', 'time_per_instance',
            'f1_overlap', 'llm_score', 'f1_evidence_all', 'f1_label', 'f1_evidence_token'
        ]
        
        # Collect values for each field
        field_values = {}
        field_counts = {}
        total_instances = 0
        
        for entry in metrics_entries:
            # Sum instances
            total_instances += entry.get('num_instances', 0)
            
            # Collect values for each numeric field
            for field in numeric_fields:
                if entry.get(field) is not None:
                    if field not in field_values:
                        field_values[field] = []
                        field_counts[field] = 0
                    field_values[field].append(entry[field])
                    field_counts[field] += 1
        
        # Calculate averages and standard deviations
        compute_std = len(metrics_entries) >= 5
        
        for field in numeric_fields:
            if field in field_values and field_counts[field] > 0:
                values = field_values[field]
                template[field] = sum(values) / len(values)
                
                # Compute standard deviation if we have at least 5 results
                if compute_std and len(values) >= 2:  # Need at least 2 values for std
                    try:
                        std_dev = statistics.stdev(values)
                        template[f'{field}_std'] = std_dev
                    except statistics.StatisticsError:
                        template[f'{field}_std'] = None
                else:
                    template[f'{field}_std'] = None
            else:
                template[field] = None
                template[f'{field}_std'] = None
        
        # Update metadata
        template['num_instances'] = total_instances
        template['benchmark'] = benchmark_name
        template['benchmark_path'] = f"averaged_from_{len(metrics_entries)}_runs"
        template['is_aggregate'] = is_aggregate
        template['num_runs'] = len(metrics_entries)
        
        # Update task name for aggregates to indicate averaging
        if is_aggregate:
            original_task_name = template.get('task_name', 'unknown')
            std_indicator = " (±std)" if compute_std else ""
            template['task_name'] = f"{original_task_name} (avg of {len(metrics_entries)} runs{std_indicator})"
        
        return template
    
    def load_multi_model_metrics(self, comparison_dir: str, benchmark_filter: Optional[str] = None) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Load metrics from all models in a comparison directory.
        
        Args:
            comparison_dir (str): Path to directory containing model subdirectories
            benchmark_filter (str, optional): Filter to only include benchmarks matching this string
            
        Returns:
            Dict[str, Dict[str, List[Dict]]]: Dictionary mapping model names to their metrics data
        """
        models_data = {}
        discovered_models = self.discover_models(comparison_dir, benchmark_filter)
        
        print(f"Found {len(discovered_models)} models for comparison")
        
        for model_name, benchmarks in discovered_models.items():
            print(f"\nProcessing model: {model_name}")
            
            # Temporarily set model directory to load metrics
            original_model_dir = self.model_dir
            original_model_name = self.model_name
            
            self.model_dir = Path(comparison_dir) / model_name
            self.model_name = model_name
            
            try:
                model_metrics = self.load_all_metrics(benchmark_filter)
                models_data[model_name] = model_metrics
            except Exception as e:
                print(f"Error loading metrics for model {model_name}: {e}")
            finally:
                # Restore original values
                self.model_dir = original_model_dir
                self.model_name = original_model_name
        
        return models_data
    
    def create_comparison_table(self, models_data: Dict[str, Dict[str, List[Dict]]], 
                              metric_type: str = "primary_score",
                              show_aggregates_only: bool = True) -> pd.DataFrame:
        """
        Create a comparison table with models as rows and tasks as columns.
        
        Args:
            models_data (Dict): Multi-model metrics data from load_multi_model_metrics()
            metric_type (str): Which metric to compare ("primary_score", "exact_match")
            show_aggregates_only (bool): Whether to show only aggregate scores or individual tasks
            
        Returns:
            pd.DataFrame: Comparison table
        """
        # Collect all unique benchmarks/tasks across all models
        all_benchmarks = set()
        all_tasks = set()
        
        for model_name, model_metrics in models_data.items():
            for benchmark_name, metrics_list in model_metrics.items():
                all_benchmarks.add(benchmark_name)
                for metric in metrics_list:
                    if show_aggregates_only and not metric.get('is_aggregate', False):
                        continue
                    if not show_aggregates_only and metric.get('is_aggregate', False):
                        continue
                    task_key = f"{benchmark_name}" if metric.get('is_aggregate', False) else f"{benchmark_name}_{metric['task_name']}"
                    all_tasks.add(task_key)
        
        # Create comparison matrix
        comparison_data = {}
        
        for model_name, model_metrics in models_data.items():
            model_row = {}
            
            for task_key in sorted(all_tasks):
                # if task_key == 'lab_bench':
                #     from IPython import embed; embed()
                score = None
                
                # Parse task key to get benchmark and task name
                # if '_' in task_key and not any(task_key.startswith(b + '_') for b in ['lab_bench', 'super_gpqa']):
                #     benchmark_name = task_key.split('_')[0]
                #     task_name = '_'.join(task_key.split('_')[1:])
                # else:
                benchmark_name = task_key
                task_name = None
                
                if benchmark_name in model_metrics:
                    metrics_list = model_metrics[benchmark_name]
                    
                    # Find the right metric entry
                    for metric in metrics_list:
                        if show_aggregates_only and metric.get('is_aggregate', False):
                            score = metric.get(metric_type)
                            break
                        elif not show_aggregates_only and not metric.get('is_aggregate', False) and metric['task_name'] == task_name:
                            score = metric.get(metric_type)
                            break
                
                model_row[task_key] = score
            
            comparison_data[model_name] = model_row
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(comparison_data, orient='index')
        
        # Sort columns by benchmark name for better organization
        benchmark_order = sorted(all_benchmarks)
        ordered_columns = []
        for benchmark in benchmark_order:
            benchmark_columns = [col for col in df.columns if col.startswith(benchmark)]
            ordered_columns.extend(sorted(benchmark_columns))
        
        df = df[ordered_columns]
        
        return df
    
    def display_comparison_table(self, models_data: Dict[str, Dict[str, List[Dict]]], 
                               metric_type: str = "primary_score",
                               format_type: str = "table",
                               show_aggregates_only: bool = True,
                               precision: int = 4):
        """
        Display a comparison table of models vs tasks.
        
        Args:
            models_data (Dict): Multi-model metrics data
            metric_type (str): Which metric to compare
            format_type (str): Output format
            show_aggregates_only (bool): Whether to show only aggregate scores
            precision (int): Number of decimal places to show
        """
        df = self.create_comparison_table(models_data, metric_type, show_aggregates_only)
        
        if df.empty:
            print("No data available for comparison.")
            return
        
        print(f"\n{'='*120}")
        print(f"MODEL COMPARISON - {metric_type.upper()}")
        if show_aggregates_only:
            print("(Showing aggregate scores only)")
        else:
            print("(Showing individual task scores)")
        print(f"{'='*120}")
        
        # Format the DataFrame for display
        df_display = df.copy()
        for col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.{precision}f}" if pd.notna(x) else "N/A"
            )
        
        if format_type == "table":
            print(tabulate(df_display, headers=df_display.columns, tablefmt="grid", stralign="center"))
        else:
            print(df_display.to_string())
        
        # Calculate and display averages (excluding knowledge tasks)
        print(f"\n{'='*60}")
        print("AVERAGES (excluding knowledge benchmarks)")
        print(f"{'='*60}")
        
        # Filter out knowledge columns and calculate averages
        non_knowledge_cols = [col for col in df.columns if 'knowledge' not in col.lower()]
        if non_knowledge_cols:
            avg_scores = {}
            for model in df.index:
                model_scores = [df.loc[model, col] for col in non_knowledge_cols if pd.notna(df.loc[model, col])]
                if model_scores:
                    avg_scores[model] = sum(model_scores) / len(model_scores)
            
            # Sort models by average score
            sorted_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            
            for i, (model, avg_score) in enumerate(sorted_models, 1):
                print(f"{i:2d}. {model:.<30} {avg_score:.{precision}f}")
    
    def display_summary(self, all_metrics: Dict[str, List[Dict]], format_type: str = "simple"):
        """
        Display a summary of all benchmarks and their aggregate scores.
        
        Args:
            all_metrics (Dict): Metrics data from load_all_metrics()
            format_type (str): Output format ("simple", "table", "detailed")
        """
        print(f"\n{'='*80}")
        print(f"PERFORMANCE SUMMARY - {self.model_name}")
        print(f"{'='*80}")
        
        summary_data = []
        total_instances = 0
        total_weighted_score = 0
        valid_scores = []
        
        for benchmark_name, metrics_list in all_metrics.items():
            # Find aggregate metrics (usually first entry)
            aggregate = next((m for m in metrics_list if m.get('is_aggregate', False)), None)
            
            if aggregate:
                primary_score = aggregate['primary_score']
                exact_match = aggregate['exact_match']
                num_instances = aggregate['num_instances']
                
                # Format primary score with std if available
                primary_score_str = "N/A"
                if primary_score is not None:
                    primary_score_std = aggregate.get('primary_score_std')
                    if primary_score_std is not None:
                        primary_score_str = f"{primary_score:.4f} ± {primary_score_std:.4f}"
                    else:
                        primary_score_str = f"{primary_score:.4f}"
                
                # Format exact match with std if available
                exact_match_str = "N/A"
                if exact_match is not None:
                    exact_match_std = aggregate.get('exact_match_std')
                    if exact_match_std is not None:
                        exact_match_str = f"{exact_match:.4f} ± {exact_match_std:.4f}"
                    else:
                        exact_match_str = f"{exact_match:.4f}"
                
                summary_data.append({
                    'Benchmark': benchmark_name,
                    'Primary Score': primary_score_str,
                    'Exact Match': exact_match_str,
                    'Tasks': len(metrics_list) - 1,  # Subtract 1 for aggregate
                    'Total Instances': num_instances,
                    'Runs': aggregate.get('num_runs', 1)
                })
                
                # Collect data for average calculation - exclude benchmarks containing "knowledge"
                if primary_score is not None and "knowledge" not in benchmark_name.lower():
                    total_instances += num_instances
                    total_weighted_score += primary_score * num_instances
                    valid_scores.append(primary_score)
        
        # Calculate averages (excluding knowledge tasks)
        if valid_scores:
            # Weighted average (by number of instances)
            weighted_avg = total_weighted_score / total_instances if total_instances > 0 else 0
            # Simple average (unweighted)
            simple_avg = sum(valid_scores) / len(valid_scores)
            
            # Count tasks excluding knowledge benchmarks
            total_tasks = sum(item['Tasks'] for item in summary_data 
                            if isinstance(item['Tasks'], int) and "knowledge" not in item['Benchmark'].lower())
            
            # Add average row
            summary_data.append({
                'Benchmark': '─' * 20 + ' AVERAGE (excl. knowledge) ' + '─' * 20,
                'Primary Score': f"{simple_avg:.4f}",
                'Exact Match': "N/A",
                'Tasks': f"{total_tasks}",
                'Total Instances': total_instances
            })
        
        if summary_data:
            if format_type == "table":
                print(tabulate(summary_data, headers="keys", tablefmt="grid"))
            else:
                for item in summary_data:
                    if '─' in item['Benchmark']:  # Average row
                        print(f"\n{item['Benchmark']}")
                        print(f"{'Average Score (excl. knowledge)':.<30} {item['Primary Score']:>8} ({item['Tasks']} total tasks, {item['Total Instances']} total instances)")
                    else:
                        print(f"{item['Benchmark']:.<30} {item['Primary Score']:>8} ({item['Tasks']} tasks, {item['Total Instances']} instances)")
        else:
            print("No aggregate metrics found.")
    
    def display_detailed_tasks(self, all_metrics: Dict[str, List[Dict]], 
                             sort_by: str = "task_name", 
                             format_type: str = "table",
                             show_aggregate: bool = False):
        """
        Display detailed task-level metrics.
        
        Args:
            all_metrics (Dict): Metrics data from load_all_metrics()
            sort_by (str): Column to sort by ("task_name", "primary_score", "exact_match", "num_instances")
            format_type (str): Output format ("simple", "table", "detailed")
            show_aggregate (bool): Whether to include aggregate metrics in the detailed view
        """
        print(f"\n{'='*120}")
        print(f"DETAILED TASK PERFORMANCE - {self.model_name}")
        print(f"{'='*120}")
        
        # Collect all task data
        all_tasks = []
        for benchmark_name, metrics_list in all_metrics.items():
            for metric in metrics_list:
                # Skip aggregate unless explicitly requested
                if metric.get('is_aggregate', False) and not show_aggregate:
                    continue
                all_tasks.append(metric)
        
        if not all_tasks:
            print("No task metrics found.")
            return
        
        # Sort tasks
        reverse_sort = sort_by in ['primary_score', 'exact_match', 'num_instances']
        try:
            all_tasks.sort(key=lambda x: x.get(sort_by, 0) or 0, reverse=reverse_sort)
        except (TypeError, KeyError):
            print(f"Warning: Could not sort by {sort_by}, using default order")
        
        if format_type == "table":
            # Prepare data for tabular display
            table_data = []
            for task in all_tasks:
                # Format primary score with std if available
                primary_score_str = "N/A"
                if task['primary_score'] is not None:
                    primary_score_std = task.get('primary_score_std')
                    if primary_score_std is not None:
                        primary_score_str = f"{task['primary_score']:.4f} ± {primary_score_std:.4f}"
                    else:
                        primary_score_str = f"{task['primary_score']:.4f}"
                
                # Format exact match with std if available
                exact_match_str = "N/A"
                if task['exact_match'] is not None:
                    exact_match_std = task.get('exact_match_std')
                    if exact_match_std is not None:
                        exact_match_str = f"{task['exact_match']:.4f} ± {exact_match_std:.4f}"
                    else:
                        exact_match_str = f"{task['exact_match']:.4f}"
                else:
                    print("  Exact Match: N/A")
                
                row = {
                    'Benchmark': task['benchmark'],
                    'Task': task['task_name'],
                    'Primary Score': primary_score_str,
                    'Exact Match': exact_match_str,
                    'Instances': task['num_instances'],
                }
                
                # Add optional metrics if available
                if task.get('avg_tokens') is not None:
                    avg_tokens_std = task.get('avg_tokens_std')
                    if avg_tokens_std is not None:
                        row['Avg Tokens'] = f"{task['avg_tokens']:.1f} ± {avg_tokens_std:.1f}"
                    else:
                        row['Avg Tokens'] = f"{task['avg_tokens']:.1f}"
                if task.get('time_per_instance') is not None:
                    time_std = task.get('time_per_instance_std')
                    if time_std is not None:
                        row['Time/Instance'] = f"{task['time_per_instance']:.2f} ± {time_std:.2f}s"
                    else:
                        row['Time/Instance'] = f"{task['time_per_instance']:.2f}s"
                
                # Add runs information if available
                if task.get('num_runs') and task.get('num_runs') > 1:
                    row['Runs'] = task['num_runs']
                
                table_data.append(row)
            
            print(tabulate(table_data, headers="keys", tablefmt="grid"))
            
        elif format_type == "detailed":
            # Detailed format with extra information
            for task in all_tasks:
                print(f"\nTask: {task['task_name']}")
                print(f"  Benchmark: {task['benchmark']}")
                print(f"  Task Core: {task['task_core']}")
                print(f"  Instances: {task['num_instances']}")
                
                # Show runs information if available
                if task.get('num_runs') and task.get('num_runs') > 1:
                    print(f"  Runs: {task['num_runs']}")
                
                # Primary score with std if available
                if task['primary_score'] is not None:
                    primary_score_std = task.get('primary_score_std')
                    if primary_score_std is not None:
                        print(f"  Primary Score: {task['primary_score']:.4f} ± {primary_score_std:.4f}")
                    else:
                        print(f"  Primary Score: {task['primary_score']:.4f}")
                else:
                    print("  Primary Score: N/A")
                
                # Exact match with std if available
                if task['exact_match'] is not None:
                    exact_match_std = task.get('exact_match_std')
                    if exact_match_std is not None:
                        print(f"  Exact Match: {task['exact_match']:.4f} ± {exact_match_std:.4f}")
                    else:
                        print(f"  Exact Match: {task['exact_match']:.4f}")
                
                if task.get('exact_match_simple') is not None:
                    exact_match_simple_std = task.get('exact_match_simple_std')
                    if exact_match_simple_std is not None:
                        print(f"  Exact Match (Simple): {task['exact_match_simple']:.4f} ± {exact_match_simple_std:.4f}")
                    else:
                        print(f"  Exact Match (Simple): {task['exact_match_simple']:.4f}")
                
                if task.get('avg_tokens') is not None:
                    avg_tokens_std = task.get('avg_tokens_std')
                    if avg_tokens_std is not None:
                        print(f"  Average Tokens: {task['avg_tokens']:.1f} ± {avg_tokens_std:.1f}")
                    else:
                        print(f"  Average Tokens: {task['avg_tokens']:.1f}")
                
                if task.get('max_tokens_reached') is not None:
                    max_tokens_std = task.get('max_tokens_reached_std')
                    if max_tokens_std is not None:
                        print(f"  Max Tokens Reached: {task['max_tokens_reached']:.1%} ± {max_tokens_std:.1%}")
                    else:
                        print(f"  Max Tokens Reached: {task['max_tokens_reached']:.1%}")
                
                if task.get('answer_format_correct') is not None:
                    answer_format_std = task.get('answer_format_correct_std')
                    if answer_format_std is not None:
                        print(f"  Answer Format Correct: {task['answer_format_correct']:.1%} ± {answer_format_std:.1%}")
                    else:
                        print(f"  Answer Format Correct: {task['answer_format_correct']:.1%}")
                
                if task.get('time_per_instance') is not None:
                    time_std = task.get('time_per_instance_std')
                    if time_std is not None:
                        print(f"  Time per Instance: {task['time_per_instance']:.2f} ± {time_std:.2f}s")
                    else:
                        print(f"  Time per Instance: {task['time_per_instance']:.2f}s")
                
                if task.get('total_price') is not None:
                    price_std = task.get('total_price_std')
                    if price_std is not None:
                        print(f"  Total Cost: ${task['total_price']:.4f} ± ${price_std:.4f}")
                    else:
                        print(f"  Total Cost: ${task['total_price']:.4f}")
                        
        else:  # simple format
            for task in all_tasks:
                # Format score with std if available
                if task['primary_score'] is not None:
                    primary_score_std = task.get('primary_score_std')
                    if primary_score_std is not None:
                        score_str = f"{task['primary_score']:.4f} ± {primary_score_std:.4f}"
                    else:
                        score_str = f"{task['primary_score']:.4f}"
                else:
                    score_str = "N/A"
                
                # Add runs info if available
                runs_info = ""
                if task.get('num_runs') and task.get('num_runs') > 1:
                    runs_info = f" [{task['num_runs']} runs]"
                
                print(f"{task['benchmark']:.<20} {task['task_name']:.<40} {score_str:>15} ({task['num_instances']} instances{runs_info})")
    
    def export_to_tsv(self, all_metrics: Dict[str, List[Dict]], output_file: str):
        """
        Export all metrics to a TSV file.
        
        Args:
            all_metrics (Dict): Metrics data from load_all_metrics()
            output_file (str): Path to output TSV file
        """
        # Collect all task data
        all_tasks = []
        for benchmark_name, metrics_list in all_metrics.items():
            for metric in metrics_list:
                all_tasks.append(metric)
        
        if not all_tasks:
            print("No data to export.")
            return
        
        # Convert to DataFrame and save as TSV
        df = pd.DataFrame(all_tasks)
        df.to_csv(output_file, index=False, sep='\t')
        print(f"Exported {len(all_tasks)} records to {output_file}")
    
    def export_summary_to_tsv(self, all_metrics: Dict[str, List[Dict]], output_file: str):
        """
        Export summary table (benchmark aggregates) to a TSV file.
        
        Args:
            all_metrics (Dict): Metrics data from load_all_metrics()
            output_file (str): Path to output TSV file
        """
        summary_data = []
        total_instances = 0
        total_weighted_score = 0
        valid_scores = []
        
        for benchmark_name, metrics_list in all_metrics.items():
            # Find aggregate metrics (usually first entry)
            aggregate = next((m for m in metrics_list if m.get('is_aggregate', False)), None)
            
            if aggregate:
                primary_score = aggregate['primary_score']
                exact_match = aggregate['exact_match']
                num_instances = aggregate['num_instances']
                
                # Format primary score with std if available
                primary_score_str = "N/A"
                if primary_score is not None:
                    primary_score_std = aggregate.get('primary_score_std')
                    if primary_score_std is not None:
                        primary_score_str = f"{primary_score:.4f} ± {primary_score_std:.4f}"
                    else:
                        primary_score_str = f"{primary_score:.4f}"
                
                # Format exact match with std if available
                exact_match_str = "N/A"
                if exact_match is not None:
                    exact_match_std = aggregate.get('exact_match_std')
                    if exact_match_std is not None:
                        exact_match_str = f"{exact_match:.4f} ± {exact_match_std:.4f}"
                    else:
                        exact_match_str = f"{exact_match:.4f}"
                
                summary_data.append({
                    'Benchmark': benchmark_name,
                    'Primary_Score': primary_score_str,
                    'Exact_Match': exact_match_str,
                    'Tasks': len(metrics_list) - 1,  # Subtract 1 for aggregate
                    'Total_Instances': num_instances,
                    'Time_Per_Instance': aggregate.get('time_per_instance'),
                    'Total_Cost': aggregate.get('total_price')
                })
                
                # Collect data for average calculation - exclude benchmarks containing "knowledge"
                if primary_score is not None and "knowledge" not in benchmark_name.lower():
                    total_instances += num_instances
                    total_weighted_score += primary_score * num_instances
                    valid_scores.append(primary_score)
        
        # Calculate averages (excluding knowledge tasks)
        if valid_scores:
            # Weighted average (by number of instances)
            weighted_avg = total_weighted_score / total_instances if total_instances > 0 else 0
            # Simple average (unweighted)
            simple_avg = sum(valid_scores) / len(valid_scores)
            
            # Count tasks excluding knowledge benchmarks
            total_tasks = sum(item['Tasks'] for item in summary_data 
                            if isinstance(item['Tasks'], int) and "knowledge" not in item['Benchmark'].lower())
            
            # Add average row
            summary_data.append({
                'Benchmark': 'AVERAGE_EXCL_KNOWLEDGE',
                'Primary_Score': simple_avg,
                'Exact_Match': None,
                'Tasks': total_tasks,
                'Total_Instances': total_instances,
                'Time_Per_Instance': None,
                'Total_Cost': None
            })
            
            # Add weighted average row
            summary_data.append({
                'Benchmark': 'WEIGHTED_AVERAGE_EXCL_KNOWLEDGE',
                'Primary_Score': weighted_avg,
                'Exact_Match': None,
                'Tasks': total_tasks,
                'Total_Instances': total_instances,
                'Time_Per_Instance': None,
                'Total_Cost': None
            })
        
        if summary_data:
            # Convert to DataFrame and save as TSV
            df = pd.DataFrame(summary_data)
            df.to_csv(output_file, index=False, sep='\t')
            print(f"Exported summary table with {len(summary_data)} benchmarks to {output_file}")
        else:
            print("No summary data to export.")
    
    def export_comparison_summary_to_tsv(self, models_data: Dict[str, Dict[str, List[Dict]]], 
                                       output_file: str, 
                                       metric_type: str = "primary_score"):
        """
        Export comparison summary (model averages) to a TSV file.
        
        Args:
            models_data (Dict): Multi-model metrics data
            output_file (str): Path to output TSV file
            metric_type (str): Which metric to use for comparison
        """
        summary_data = []
        
        for model_name, model_metrics in models_data.items():
            model_scores = []
            total_tasks = 0
            total_instances = 0
            total_cost = 0
            total_time = 0
            
            # Collect benchmark-level aggregates for this model
            for benchmark_name, metrics_list in model_metrics.items():
                # Skip knowledge benchmarks for average calculation
                if "knowledge" in benchmark_name.lower():
                    continue
                    
                # Find aggregate metrics
                aggregate = next((m for m in metrics_list if m.get('is_aggregate', False)), None)
                if aggregate and aggregate.get(metric_type) is not None:
                    model_scores.append(aggregate[metric_type])
                    total_tasks += len(metrics_list) - 1  # Subtract 1 for aggregate
                    total_instances += aggregate.get('num_instances', 0)
                    
                    if aggregate.get('total_price'):
                        total_cost += aggregate['total_price']
                    if aggregate.get('time_per_instance') and aggregate.get('num_instances'):
                        total_time += aggregate['time_per_instance'] * aggregate['num_instances']
            
            # Calculate average score for this model
            avg_score = sum(model_scores) / len(model_scores) if model_scores else None
            avg_time_per_instance = total_time / total_instances if total_instances > 0 else None
            
            summary_data.append({
                'Model': model_name,
                f'Average_{metric_type.title()}': avg_score,
                'Benchmarks': len(model_scores),
                'Total_Tasks': total_tasks,
                'Total_Instances': total_instances,
                'Total_Cost': total_cost if total_cost > 0 else None,
                'Avg_Time_Per_Instance': avg_time_per_instance
            })
        
        # Sort by average score (descending)
        if summary_data:
            summary_data.sort(key=lambda x: x[f'Average_{metric_type.title()}'] or 0, reverse=True)
            
            # Convert to DataFrame and save as TSV
            df = pd.DataFrame(summary_data)
            df.to_csv(output_file, index=False, sep='\t')
            print(f"Exported comparison summary with {len(summary_data)} models to {output_file}")
        else:
            print("No comparison summary data to export.")
    
    def _create_custom_aggregates(self, all_metrics: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Create custom aggregates for sciriff, supergpqa, and lab_bench benchmarks.
        
        Args:
            all_metrics (Dict): Original metrics data
            
        Returns:
            Dict: Modified metrics data with custom aggregates
        """
        modified_metrics = {}
        
        # Collect sciriff, supergpqa, and lab_bench data for aggregation
        sciriff_data = []
        supergpqa_data = []
        lab_bench_data = []
        
        for benchmark_name, metrics_list in all_metrics.items():
            if 'sciriff' in benchmark_name.lower():
                sciriff_data.extend(metrics_list)
            elif 'supergpqa' in benchmark_name.lower():
                supergpqa_data.extend(metrics_list)
            elif 'lab_bench' in benchmark_name.lower():
                lab_bench_data.extend(metrics_list)
            else:
                # Keep other benchmarks as-is
                modified_metrics[benchmark_name] = metrics_list
        
        # Create sciriff aggregate if we have sciriff data
        if sciriff_data:
            sciriff_aggregate = self._create_sciriff_aggregate(sciriff_data)
            if sciriff_aggregate:
                modified_metrics['sciriff'] = sciriff_aggregate
        
        # Create supergpqa aggregate if we have supergpqa data
        if supergpqa_data:
            supergpqa_aggregate = self._create_supergpqa_aggregate(supergpqa_data)
            if supergpqa_aggregate:
                modified_metrics['supergpqa'] = supergpqa_aggregate
        
        # Create lab_bench aggregate if we have lab_bench data
        if lab_bench_data:
            lab_bench_aggregate = self._create_lab_bench_aggregate(lab_bench_data)
            if lab_bench_aggregate:
                modified_metrics['lab_bench'] = lab_bench_aggregate
        
        return modified_metrics
    
    def _create_sciriff_aggregate(self, sciriff_data: List[Dict]) -> List[Dict]:
        """
        Create a custom sciriff aggregate from the 5 specific metrics.
        
        Returns:
            List[Dict]: List containing the sciriff aggregate and individual tasks
        """
        # Find the individual task metrics (not aggregates)
        tasks = [item for item in sciriff_data if not item.get('is_aggregate', False)]
        
        # Extract the 5 specific metrics
        sciriff_metrics = {}
        total_instances = 0
        total_time = 0
        total_cost = 0
        
        for task in tasks:
            total_instances += task.get('num_instances', 0)
            
            # Add time and cost if available
            if task.get('time_per_instance') and task.get('num_instances'):
                total_time += task['time_per_instance'] * task['num_instances']
            if task.get('total_price'):
                total_cost += task['total_price']
            
            # Collect the 5 specific metrics
            for metric_name in ['f1_overlap', 'llm_score', 'f1_evidence_all', 'f1_label', 'f1_evidence_token']:
                if task.get(metric_name) is not None:
                    sciriff_metrics[metric_name] = task[metric_name]
        
        # Calculate average of the 5 metrics
        metric_values = [v for v in sciriff_metrics.values() if v is not None]
        if len(metric_values) >= 5:  # Only create aggregate if we have all 5 metrics
            avg_score = sum(metric_values) / len(metric_values)
            
            # Create aggregate entry
            aggregate = {
                'task_name': 'sciriff (5-metric average)',
                'task_core': 'sciriff_aggregate',
                'num_instances': total_instances,
                'primary_score': avg_score,
                'exact_match': None,
                'exact_match_simple': None,
                'is_aggregate': True,
                'benchmark': 'sciriff',
                'benchmark_path': 'aggregated',
                'time_per_instance': total_time / total_instances if total_instances > 0 else None,
                'total_price': total_cost if total_cost > 0 else None,
            }
            
            # Add the individual sciriff metrics to the aggregate for reference
            aggregate.update(sciriff_metrics)
            
            return [aggregate] + tasks
        else:
            # If we don't have all 5 metrics, return tasks as-is
            print(f"Warning: Could not create sciriff aggregate, found only {len(metric_values)}/5 required metrics")
            return tasks
    
    def _create_supergpqa_aggregate(self, supergpqa_data: List[Dict]) -> List[Dict]:
        """
        Create a supergpqa aggregate by averaging Engineering and Science scores.
        
        Returns:
            List[Dict]: List containing the supergpqa aggregate and individual tasks
        """
        # Find aggregate entries (not individual tasks)
        aggregates = [item for item in supergpqa_data if item.get('is_aggregate', False)]
        tasks = [item for item in supergpqa_data if not item.get('is_aggregate', False)]
        
        if len(aggregates) >= 2:  # We expect Engineering and Science aggregates
            # Calculate average primary score
            primary_scores = [agg['primary_score'] for agg in aggregates if agg['primary_score'] is not None]
            exact_matches = [agg['exact_match'] for agg in aggregates if agg['exact_match'] is not None]
            
            total_instances = sum(agg.get('num_instances', 0) for agg in aggregates)
            total_time = 0
            total_cost = 0
            
            # Calculate weighted averages for time and cost
            for agg in aggregates:
                if agg.get('time_per_instance') and agg.get('num_instances'):
                    total_time += agg['time_per_instance'] * agg['num_instances']
                if agg.get('total_price'):
                    total_cost += agg['total_price']
            
            if primary_scores:
                avg_primary = sum(primary_scores) / len(primary_scores)
                avg_exact_match = sum(exact_matches) / len(exact_matches) if exact_matches else None
                
                # Create aggregate entry
                aggregate = {
                    'task_name': 'supergpqa (Engineering + Science average)',
                    'task_core': 'supergpqa_aggregate', 
                    'num_instances': total_instances,
                    'primary_score': avg_primary,
                    'exact_match': avg_exact_match,
                    'exact_match_simple': avg_exact_match,
                    'is_aggregate': True,
                    'benchmark': 'supergpqa',
                    'benchmark_path': 'aggregated',
                    'time_per_instance': total_time / total_instances if total_instances > 0 else None,
                    'total_price': total_cost if total_cost > 0 else None,
                }
                
                return [aggregate] + tasks
        
        # If we can't create aggregate, return original data
        return supergpqa_data
    
    def _create_lab_bench_aggregate(self, lab_bench_data: List[Dict]) -> List[Dict]:
        """
        Create a custom lab_bench aggregate that only considers CloningScenarios, ProtocolQA, and SeqQA.
        
        Returns:
            List[Dict]: List containing the lab_bench aggregate and individual tasks
        """
        # Find the individual task metrics (not aggregates)
        tasks = [item for item in lab_bench_data if not item.get('is_aggregate', False)]
        
        # Filter tasks to only include CloningScenarios, ProtocolQA, and SeqQA
        filtered_tasks = [task for task in tasks if any(x in task.get('task_name', '') for x in ['CloningScenarios', 'ProtocolQA', 'SeqQA'])]
        
        if not filtered_tasks:
            print("Warning: No CloningScenarios, ProtocolQA, or SeqQA tasks found in lab_bench data.")
            return lab_bench_data
        
        # Calculate aggregate metrics
        total_instances = sum(task.get('num_instances', 0) for task in filtered_tasks)
        total_time = 0
        total_cost = 0
        primary_scores = []
        exact_matches = []
        
        for task in filtered_tasks:
            if task.get('primary_score') is not None:
                primary_scores.append(task['primary_score'])
            if task.get('exact_match') is not None:
                exact_matches.append(task['exact_match'])
            if task.get('time_per_instance') and task.get('num_instances'):
                total_time += task['time_per_instance'] * task['num_instances']
            if task.get('total_price'):
                total_cost += task['total_price']
        
        # Calculate average primary score and exact match
        avg_primary = sum(primary_scores) / len(primary_scores) if primary_scores else None
        avg_exact_match = sum(exact_matches) / len(exact_matches) if exact_matches else None
        
        # Create aggregate entry
        aggregate = {
            'task_name': 'lab_bench (CloningScenarios + ProtocolQA + SeqQA average)',
            'task_core': 'lab_bench_aggregate',
            'num_instances': total_instances,
            'primary_score': avg_primary,
            'exact_match': avg_exact_match,
            'exact_match_simple': avg_exact_match,
            'is_aggregate': True,
            'benchmark': 'lab_bench',
            'benchmark_path': 'aggregated',
            'time_per_instance': total_time / total_instances if total_instances > 0 else None,
            'total_price': total_cost if total_cost > 0 else None,
        }
        
        return [aggregate] + tasks


def main():
    parser = argparse.ArgumentParser(
        description="View task-level performance metrics from lmeval results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s lmeval/Qwen3-32B
  %(prog)s lmeval/deepeek-r1 --benchmark mmlu_pro
  %(prog)s lmeval/Qwen3-32B --format table --sort-by primary_score
  %(prog)s lmeval/deepeek-r1 --detailed --export results.tsv
  %(prog)s --comparison-dir lmeval_results/ --comparison-mode
  %(prog)s --comparison-dir models/ --benchmark mmlu --comparison-mode
  %(prog)s lmeval/Qwen3-32B --export-summary summary.tsv
  %(prog)s --comparison-dir models/ --comparison-mode --export-comparison-summary model_rankings.tsv
        """
    )
    
    parser.add_argument('model_dir', nargs='?', help='Path to model directory containing lmeval results')
    parser.add_argument('--benchmark', help='Filter to specific benchmark (substring match)')
    parser.add_argument('--format', choices=['simple', 'table', 'detailed'], default='table',
                        help='Output format (default: table)')
    parser.add_argument('--sort-by', choices=['task_name', 'primary_score', 'exact_match', 'num_instances'],
                        default='task_name', help='Sort tasks by this column (default: task_name)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed task view')
    parser.add_argument('--show-aggregate', action='store_true', 
                        help='Include aggregate metrics in detailed view')
    parser.add_argument('--export', help='Export results to TSV file')
    parser.add_argument('--summary-only', action='store_true', 
                        help='Show only benchmark summary, not individual tasks')
    parser.add_argument('--comparison-dir', help='Path to directory containing model subdirectories for comparison')
    parser.add_argument('--comparison-mode', action='store_true', help='Show multi-model comparison')
    parser.add_argument('--metric-type', choices=['primary_score', 'exact_match'], default='primary_score',
                        help='Metric to use for comparison (default: primary_score)')
    parser.add_argument('--show-individual-tasks', action='store_true',
                        help='In comparison mode, show individual tasks instead of aggregates')
    parser.add_argument('--export-summary', help='Export summary table to TSV file')
    parser.add_argument('--export-comparison-summary', help='Export comparison summary (model averages) to TSV file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.comparison_mode and not args.comparison_dir:
        parser.error("--comparison-mode requires --comparison-dir")
    
    if not args.comparison_mode and not args.model_dir:
        parser.error("model_dir is required unless using --comparison-mode with --comparison-dir")
    
    try:
        viewer = TaskPerformanceViewer(args.model_dir)
        
        if args.comparison_mode:
            # Multi-model comparison mode
            models_data = viewer.load_multi_model_metrics(args.comparison_dir, args.benchmark)
            
            if not models_data:
                print(f"No models with metrics found in {args.comparison_dir}")
                return 1
            
            # Display comparison table
            viewer.display_comparison_table(
                models_data, 
                metric_type=args.metric_type,
                format_type=args.format, 
                show_aggregates_only=not args.show_individual_tasks
            )
            
            # Export comparison if requested
            if args.export:
                df = viewer.create_comparison_table(
                    models_data, 
                    metric_type=args.metric_type,
                    show_aggregates_only=not args.show_individual_tasks
                )
                df.to_csv(args.export, sep='\t')
                print(f"Exported comparison table to {args.export}")
            
            # Export comparison summary if requested
            if args.export_comparison_summary:
                viewer.export_comparison_summary_to_tsv(
                    models_data, 
                    args.export_comparison_summary, 
                    args.metric_type
                )
        
        else:
            # Single model mode (original functionality)
            all_metrics = viewer.load_all_metrics(args.benchmark)
            
            if not all_metrics:
                print(f"No metrics found in {args.model_dir}")
                return 1
            
            # Display summary
            viewer.display_summary(all_metrics, args.format)
            
            # Display detailed tasks unless summary-only
            if not args.summary_only:
                format_type = 'detailed' if args.detailed else args.format
                viewer.display_detailed_tasks(all_metrics, args.sort_by, format_type, args.show_aggregate)
            
            # Export if requested
            if args.export:
                viewer.export_to_tsv(all_metrics, args.export)
            
            # Export summary if requested
            if args.export_summary:
                viewer.export_summary_to_tsv(all_metrics, args.export_summary)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 