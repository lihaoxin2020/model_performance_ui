import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import re

# Assuming performance_analyzer.py is in the same directory or PYTHONPATH is configured
try:
    from .performance_analyzer import ModelPerformanceAnalyzer
except ImportError:
    from performance_analyzer import ModelPerformanceAnalyzer

# Special aggregated task names
LAB_BENCH_KNOWLEDGE = "lab_bench_knowledge"
LAB_BENCH_GENERAL = "lab_bench"
MMLU_KNOWLEDGE = "mmlu_knowledge" 
MMLU_GENERAL = "mmlu"
GPQA_KNOWLEDGE = "gpqa_knowledge"
GPQA_GENERAL = "gpqa"

# Regex patterns for subtasks
LAB_BENCH_KNOWLEDGE_PATTERN = r"^lab_bench_.*:cot_knowledge$"
LAB_BENCH_GENERAL_PATTERN = r"^lab_bench_.*:cot$"
MMLU_KNOWLEDGE_PATTERN = r"^mmlu_pro_.*:cot_knowledge$"
MMLU_GENERAL_PATTERN = r"^mmlu_pro_.*:cot$"
GPQA_KNOWLEDGE_PATTERN = r"^gpqa_knowledge$"
GPQA_GENERAL_PATTERN = r"^gpqa$"


def aggregate_subtasks_union(task_dfs: Dict[str, List[pd.DataFrame]], task_name: str) -> Optional[pd.DataFrame]:
    """
    Aggregates subtasks based on the requested task name, taking union across all runs.
    
    Args:
        task_dfs: Dictionary of task DataFrames (lists for multiple runs)
        task_name: The requested task name (e.g., "lab_bench_knowledge")
    
    Returns:
        Aggregated DataFrame (union across all runs) or None if no matching subtasks found
    """
    if task_name == LAB_BENCH_KNOWLEDGE:
        pattern = LAB_BENCH_KNOWLEDGE_PATTERN
    elif task_name == LAB_BENCH_GENERAL:
        pattern = LAB_BENCH_GENERAL_PATTERN
    elif task_name == MMLU_KNOWLEDGE:
        pattern = MMLU_KNOWLEDGE_PATTERN
    elif task_name == MMLU_GENERAL:
        pattern = MMLU_GENERAL_PATTERN
    elif task_name == GPQA_KNOWLEDGE:
        pattern = GPQA_KNOWLEDGE_PATTERN
    elif task_name == GPQA_GENERAL:
        pattern = GPQA_GENERAL_PATTERN
    else:
        return None
    
    # Find matching subtasks and collect all runs
    all_runs_data = []
    matching_subtask_names = []
    
    for actual_task_name, df_list in task_dfs.items():
        if re.match(pattern, actual_task_name):
            matching_subtask_names.append(actual_task_name)
            for run_idx, df in enumerate(df_list):
                # Create unique doc_ids by prefixing with task name
                df_copy = df.copy()
                df_copy['doc_id'] = actual_task_name + "_" + df_copy['doc_id'].astype(str)
                all_runs_data.append(df_copy)
    
    if not all_runs_data:
        return None
    # Combine all runs from all matching subtasks (union)
    aggregated_df = pd.concat(all_runs_data, ignore_index=True)
    # First group by doc_id and check if any is_correct is True
    correct_by_id = aggregated_df.groupby('doc_id')['is_correct'].any().reset_index()
    # Then merge back with original df after deduping to preserve other columns
    deduped_df = aggregated_df.drop_duplicates(subset=['doc_id']).reset_index(drop=True)
    # Update is_correct based on the grouped results
    aggregated_df = deduped_df.merge(correct_by_id, on='doc_id', suffixes=('_orig', '')).drop('is_correct_orig', axis=1)
    
    total_runs = sum(len(df_list) for task, df_list in task_dfs.items() if re.match(pattern, task))
    print(f"Aggregated {len(matching_subtask_names)} subtasks for '{task_name}' (union across {total_runs} total runs): {len(aggregated_df)} unique instances")
    
    return aggregated_df


def aggregate_subtasks_multi_run(task_dfs: Dict[str, List[pd.DataFrame]], task_name: str) -> Optional[List[pd.DataFrame]]:
    """
    Aggregates subtasks based on the requested task name, preserving run structure for evaluation.
    
    Args:
        task_dfs: Dictionary of task DataFrames (lists for multiple runs)
        task_name: The requested task name (e.g., "lab_bench_knowledge")
    
    Returns:
        List of aggregated DataFrames (one per run) or None if no matching subtasks found
    """
    if task_name == LAB_BENCH_KNOWLEDGE:
        pattern = LAB_BENCH_KNOWLEDGE_PATTERN
    elif task_name == LAB_BENCH_GENERAL:
        pattern = LAB_BENCH_GENERAL_PATTERN
    elif task_name == MMLU_KNOWLEDGE:
        pattern = MMLU_KNOWLEDGE_PATTERN
    elif task_name == MMLU_GENERAL:
        pattern = MMLU_GENERAL_PATTERN
    elif task_name == GPQA_KNOWLEDGE:
        pattern = GPQA_KNOWLEDGE_PATTERN
    elif task_name == GPQA_GENERAL:
        pattern = GPQA_GENERAL_PATTERN
    else:
        return None
    
    # Find matching subtasks
    matching_subtasks = {}
    max_runs = 0
    
    for actual_task_name, df_list in task_dfs.items():
        if re.match(pattern, actual_task_name):
            matching_subtasks[actual_task_name] = df_list
            max_runs = max(max_runs, len(df_list))
    
    if not matching_subtasks:
        return None
    
    # Create aggregated DataFrames for each run
    aggregated_runs = []
    for run_idx in range(max_runs):
        run_data = []
        for task_name_key, df_list in matching_subtasks.items():
            if run_idx < len(df_list):
                df_copy = df_list[run_idx].copy()
                df_copy['doc_id'] = task_name_key + "_" + df_copy['doc_id'].astype(str)
                run_data.append(df_copy)
        
        if run_data:
            aggregated_run_df = pd.concat(run_data, ignore_index=True)
            aggregated_run_df = aggregated_run_df.drop_duplicates(subset=['doc_id']).reset_index(drop=True)
            aggregated_runs.append(aggregated_run_df)
    
    print(f"Aggregated {len(matching_subtasks)} subtasks for '{task_name}' preserving {len(aggregated_runs)} runs: {[len(df) for df in aggregated_runs]} instances per run")
    
    return aggregated_runs


def get_task_dataframes(analyzer: ModelPerformanceAnalyzer, directory_path: Path) -> Dict[str, List[pd.DataFrame]]:
    """
    Processes prediction files in a directory to extract instance-level correctness.
    
    Args:
        analyzer: An initialized ModelPerformanceAnalyzer instance.
        directory_path: Path to the directory containing model prediction subdirectories.

    Returns:
        A dictionary mapping task names to lists of DataFrames with 'doc_id' and 'is_correct' columns.
        Multiple DataFrames per task represent different runs/files.
    """
    task_dfs: Dict[str, List[pd.DataFrame]] = {}
    
    for subdir in directory_path.glob('*'):
        if not subdir.is_dir():
            continue
        
        prediction_files = analyzer._find_prediction_files(subdir)

        for pred_file in prediction_files:
            try:
                task_name_from_file_path = analyzer._extract_task_from_file_path(pred_file)
                predictions_df = analyzer._load_predictions(pred_file)
                
                if predictions_df.empty or 'doc_id' not in predictions_df.columns:
                    if 'doc_id' not in predictions_df.columns:
                        print(f"Warning: File {pred_file.name} - Missing 'doc_id' column, skipping.")
                    continue

                # Ensure doc_id is string for consistent processing
                predictions_df['doc_id'] = predictions_df['doc_id'].astype(str)
                filtered_df = predictions_df

                if 'task' not in filtered_df.columns:
                    if 'taskname' in filtered_df.columns:
                        filtered_df['task'] = filtered_df['taskname']
                    elif 'task_name' in filtered_df.columns:
                        filtered_df['task'] = filtered_df['task_name']
                    else:
                        filtered_df['task'] = task_name_from_file_path
                
                if 'prediction' not in filtered_df.columns:
                    for col_name in ['pred', 'output', 'generated', 'prediction_text', 'completion']:
                        if col_name in filtered_df.columns:
                            filtered_df['prediction'] = filtered_df[col_name]
                            break
                
                if 'label' not in filtered_df.columns:
                    for col_name in ['target', 'ground_truth', 'answer', 'label_text', 'correct_answer']:
                        if col_name in filtered_df.columns:
                            filtered_df['label'] = filtered_df[col_name]
                            break

                for task_in_df in filtered_df['task'].unique():
                    task_df = filtered_df[filtered_df['task'] == task_in_df].copy()
                    accuracy_series = None
                    
                    if 'exact_match_flex' in task_df.columns:
                        accuracy_series = task_df['exact_match_flex'].astype(bool)
                    elif 'exact_match_simple' in task_df.columns:
                        accuracy_series = task_df['exact_match_simple'].astype(bool)
                    elif 'correctness' in task_df.columns:
                        accuracy_series = task_df['correctness'].astype(bool)
                    elif 'metrics' in task_df.columns:
                        def get_metric_from_dict(metric_dict_val):
                            if isinstance(metric_dict_val, dict):
                                for metric_key in ['exact_match_flex', 'exact_match_simple', 'correctness']:
                                    if metric_key in metric_dict_val:
                                        return metric_dict_val[metric_key]
                            return None
                        
                        metrics_series = pd.Series(task_df['metrics'])
                        extracted_metrics = metrics_series.apply(get_metric_from_dict)
                        if not bool(extracted_metrics.isna().all()):
                            accuracy_series = extracted_metrics.fillna(0).astype(float).astype(bool)
                        elif 'prediction' in task_df.columns and 'label' in task_df.columns:
                            accuracy_series = (task_df['prediction'].astype(str) == task_df['label'].astype(str))
                        else:
                            print(f"Warning: Task {task_in_df} in {pred_file.name} - 'metrics' column unusable and no prediction/label.")
                            continue
                    elif 'prediction' in task_df.columns and 'label' in task_df.columns:
                        accuracy_series = (task_df['prediction'].astype(str) == task_df['label'].astype(str))
                    else:
                        print(f"Warning: Task {task_in_df} in {pred_file.name} - No way to determine correctness.")
                        continue
                    
                    if accuracy_series is not None:
                        task_df['is_correct'] = accuracy_series
                        
                        # Keep only the columns we need and remove duplicates
                        current_df_slice = pd.DataFrame(task_df[['doc_id', 'is_correct']]).drop_duplicates(subset=['doc_id'])
                        
                        if task_in_df in task_dfs:
                            # Add as new run
                            task_dfs[task_in_df].append(current_df_slice.reset_index(drop=True))
                        else:
                            # Create new task entry
                            task_dfs[task_in_df] = [current_df_slice.reset_index(drop=True)]

            except Exception as e:
                print(f"Error processing file {pred_file}: {e}")
                import traceback
                traceback.print_exc()
                
    return task_dfs


def get_task_df_union(task_dfs: Dict[str, List[pd.DataFrame]], task_name: str, dir_name: str) -> pd.DataFrame:
    """
    Gets a task DataFrame with union across all runs, handling aggregation if needed.
    Used for dir_A and dir_B.
    
    Args:
        task_dfs: Dictionary of task DataFrames (lists for multiple runs)
        task_name: The requested task name
        dir_name: Directory name for error messages
    
    Returns:
        The requested DataFrame (union across all runs)
    
    Raises:
        SystemExit if task not found
    """    
    # Try to aggregate subtasks (union across all runs)
    aggregated_df = aggregate_subtasks_union(task_dfs, task_name)
    if aggregated_df is not None:
        return aggregated_df
    
    # Task not found
    print(f"\nError: Task '{task_name}' not found in {dir_name}. Available tasks: {list(task_dfs.keys())}")
    print(f"Note: For aggregated tasks, use: {LAB_BENCH_KNOWLEDGE}, {LAB_BENCH_GENERAL}, {MMLU_KNOWLEDGE}, {MMLU_GENERAL}, {GPQA_KNOWLEDGE}, {GPQA_GENERAL}")
    exit(1)


def get_task_df_multi_run(task_dfs: Dict[str, List[pd.DataFrame]], task_name: str, dir_name: str) -> List[pd.DataFrame]:
    """
    Gets task DataFrames preserving run structure, handling aggregation if needed.
    Used for dir_C evaluation.
    
    Args:
        task_dfs: Dictionary of task DataFrames (lists for multiple runs)
        task_name: The requested task name
        dir_name: Directory name for error messages
    
    Returns:
        List of DataFrames (one per run)
    
    Raises:
        SystemExit if task not found
    """
    # Try to find the task directly first
    # if task_name in task_dfs:
    #     return task_dfs[task_name]
    
    # Try to aggregate subtasks (preserving run structure)
    aggregated_runs = aggregate_subtasks_multi_run(task_dfs, task_name)
    if aggregated_runs is not None:
        return aggregated_runs
    
    # Task not found
    print(f"\nError: Task '{task_name}' not found in {dir_name}. Available tasks: {list(task_dfs.keys())}")
    print(f"Note: For aggregated tasks, use: {LAB_BENCH_KNOWLEDGE}, {LAB_BENCH_GENERAL}, {MMLU_KNOWLEDGE}, {MMLU_GENERAL}, {GPQA_KNOWLEDGE}, {GPQA_GENERAL}")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description="Compare model performance on specific data slices.")
    parser.add_argument("dir_A", type=str, help="Directory for model A predictions (should be correct)")
    parser.add_argument("dir_B", type=str, help="Directory for model B predictions (should be correct, used for exclusion)")
    parser.add_argument("dir_C", type=str, help="Directory for model C predictions (to be evaluated on target instances)")
    parser.add_argument("--task_A", required=True, help="Task name in dir_A that should be correct")
    parser.add_argument("--task_B", required=True, help="Task name in dir_B that should be correct (for exclusion)")
    parser.add_argument("--task_C", required=True, help="Task name in dir_C to evaluate on target instances")
    parser.add_argument("--dataset", default=None, 
                        help="HuggingFace dataset name used by ModelPerformanceAnalyzer")
    
    args = parser.parse_args()

    dir_A_path = Path(args.dir_A)
    dir_B_path = Path(args.dir_B)
    dir_C_path = Path(args.dir_C)

    print(f"Initializing ModelPerformanceAnalyzer with dataset: {args.dataset}")
    analyzer = ModelPerformanceAnalyzer(predictions_dir=str(dir_A_path), dataset_name=args.dataset)
    
    print(f"\nProcessing dir_A predictions from: {args.dir_A}")
    data_A = get_task_dataframes(analyzer, dir_A_path)
    
    print(f"\nProcessing dir_B predictions from: {args.dir_B}")
    data_B = get_task_dataframes(analyzer, dir_B_path)
    
    print(f"\nProcessing dir_C predictions from: {args.dir_C}")
    data_C = get_task_dataframes(analyzer, dir_C_path)

    # Get task DataFrames - union for A/B, preserve runs for C
    df_A = get_task_df_union(data_A, args.task_A, args.dir_A)
    df_B = get_task_df_union(data_B, args.task_B, args.dir_B)
    df_C_runs = get_task_df_multi_run(data_C, args.task_C, args.dir_C)

    print(f"\nIdentifying target instances:")
    print(f"  - dir_A task '{args.task_A}': finding correct instances (union across runs)")
    print(f"  - dir_B task '{args.task_B}': finding correct instances for exclusion (union across runs)")
    print(f"  - Target = (dir_A correct) AND NOT (dir_B correct)")

    # Get instances where task_A is correct
    correct_A_instances = set(df_A[df_A['is_correct'] == True]['doc_id'].astype(str))
    print(f"  - Task '{args.task_A}' correct instances: {len(correct_A_instances)}")

    # Get instances where task_B is correct (to exclude)
    correct_B_instances = set(df_B[df_B['is_correct'] == True]['doc_id'].astype(str))
    print(f"  - Task '{args.task_B}' correct instances: {len(correct_B_instances)}")

    # Target instances = A correct AND B incorrect
    target_instances = correct_A_instances - correct_B_instances
    print(f"  - Target instances (A correct AND B incorrect): {len(target_instances)}")

    if not target_instances:
        print(f"No target instances found. All instances where '{args.task_A}' is correct are also correct in '{args.task_B}'.")
        return

    print(f"\nEvaluating task '{args.task_C}' from dir_C on {len(target_instances)} target instances across {len(df_C_runs)} runs...")

    # Evaluate across all runs
    run_results = []
    total_correct = 0
    total_evaluated = 0

    for run_idx, df_C_run in enumerate(df_C_runs):
        # Filter to target instances
        eval_C_on_target = df_C_run[df_C_run['doc_id'].astype(str).isin(list(target_instances))]
        
        if eval_C_on_target.empty:
            print(f"  Run {run_idx + 1}: No predictions for target instances")
            continue
        
        num_evaluable = len(eval_C_on_target)
        correct_on_target = eval_C_on_target['is_correct'].sum()
        accuracy_on_target = correct_on_target / num_evaluable if num_evaluable > 0 else 0.0
        
        run_results.append({
            'run': run_idx + 1,
            'correct': correct_on_target,
            'total': num_evaluable,
            'accuracy': accuracy_on_target
        })
        
        total_correct += correct_on_target
        total_evaluated += num_evaluable
        
        print(f"  Run {run_idx + 1}: {correct_on_target}/{num_evaluable} = {accuracy_on_target:.4f}")

    if not run_results:
        print(f"No runs had predictions for target instances.")
        return

    # Calculate statistics across runs
    accuracies = [r['accuracy'] for r in run_results]
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0
    overall_accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0.0

    print(f"\n--- Results ---")
    print(f"Task A (correct filter): '{args.task_A}' from {args.dir_A}")
    print(f"Task B (exclusion filter): '{args.task_B}' from {args.dir_B}")
    print(f"Task C (evaluation): '{args.task_C}' from {args.dir_C}")
    print(f"")
    print(f"Target instances: {len(target_instances)}")
    print(f"Number of runs evaluated: {len(run_results)}")
    print(f"")
    print(f"Per-run results:")
    for result in run_results:
        print(f"  Run {result['run']}: {result['correct']}/{result['total']} = {result['accuracy']:.4f}")
    print(f"")
    print(f"Summary statistics:")
    print(f"  Average accuracy across runs: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  Overall accuracy (pooled): {overall_accuracy:.4f} ({total_correct}/{total_evaluated})")
    
    if total_evaluated < len(target_instances) * len(df_C_runs):
        missing_total = len(target_instances) * len(df_C_runs) - total_evaluated
        print(f"  Note: {missing_total} target instance evaluations missing across all runs")


if __name__ == "__main__":
    main()
