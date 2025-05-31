import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Assuming performance_analyzer.py is in the same directory or PYTHONPATH is configured
# For direct execution, ensure this import works (e.g., run from parent dir as module)
try:
    from .performance_analyzer import ModelPerformanceAnalyzer
except ImportError:
    from performance_analyzer import ModelPerformanceAnalyzer

def get_task_dataframes(analyzer: ModelPerformanceAnalyzer, directory_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Processes prediction files in a directory to extract instance-level correctness.

    Args:
        analyzer: An initialized ModelPerformanceAnalyzer instance (used for its methods and dataset mappings).
        directory_path: Path to the directory containing model prediction subdirectories.

    Returns:
        A dictionary mapping task names to DataFrames. Each DataFrame contains
        'doc_id' and 'is_correct' columns for instances of that task.
    """
    all_task_dfs: Dict[str, pd.DataFrame] = {}
    
    # predictions_dir is an attribute of the analyzer, but we want to process a specific path
    # for this function, as it might be called for dir_a and dir_b independently.
    # However, the analyzer itself is initialized with a predictions_dir.
    # Let's use the directory_path passed to the function for globbing subdirs.

    for subdir in directory_path.glob('*'):
        if not subdir.is_dir():
            continue
        
        # _parse_directory_name is a method of the analyzer instance
        # model_name, benchmark_name = analyzer._parse_directory_name(subdir.name) # Not strictly needed for this func

        # _find_prediction_files is a method of the analyzer instance
        prediction_files = analyzer._find_prediction_files(subdir)

        for pred_file in prediction_files:
            try:
                # _extract_task_from_file_path is a method of the analyzer instance
                task_name_from_file_path = analyzer._extract_task_from_file_path(pred_file)
                
                # _load_predictions is a method of the analyzer instance
                predictions_df = analyzer._load_predictions(pred_file)
                if predictions_df.empty:
                    continue

                # _filter_matching_instances uses analyzer.task_doc_mapping
                # It also adds 'task' column if not present using task_name_from_file_path
                # filtered_df = analyzer._filter_matching_instances(predictions_df, task_name_from_file_path)
                filtered_df = predictions_df # Use all predictions without filtering against the reference dataset
                
                # Ensure 'task' column is present, as _filter_matching_instances would have handled this.
                # It might have been added by _load_predictions if 'taskname' or 'task_name' existed,
                # or it might be added by _filter_matching_instances.
                # Replicating part of _filter_matching_instances logic here if 'task' is still missing.
                if 'task' not in filtered_df.columns:
                    if 'taskname' in filtered_df.columns:
                        filtered_df['task'] = filtered_df['taskname']
                    elif 'task_name' in filtered_df.columns:
                        filtered_df['task'] = filtered_df['task_name']
                    else:
                        # If still no 'task' column, add it using the extracted task name from file path
                        filtered_df['task'] = task_name_from_file_path
                
                if filtered_df.empty:
                    # print(f"No matching instances in {pred_file.name} for task {task_name_from_file_path}")
                    continue

                # Ensure 'prediction' column exists by checking common names
                if 'prediction' not in filtered_df.columns:
                    for col_name in ['pred', 'output', 'generated', 'prediction_text', 'completion']:
                        if col_name in filtered_df.columns:
                            filtered_df['prediction'] = filtered_df[col_name]
                            break
                
                # Ensure 'label' column exists by checking common names
                if 'label' not in filtered_df.columns:
                    for col_name in ['target', 'ground_truth', 'answer', 'label_text', 'correct_answer']:
                        if col_name in filtered_df.columns:
                            filtered_df['label'] = filtered_df[col_name]
                            break

                for task_in_df in filtered_df['task'].unique():
                    task_df = filtered_df[filtered_df['task'] == task_in_df].copy()
                    accuracy_series = None
                    
                    # Determine correctness (is_correct column)
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
                        
                        extracted_metrics_series = task_df['metrics'].apply(get_metric_from_dict)
                        if not extracted_metrics_series.isna().all():
                            accuracy_series = extracted_metrics_series.fillna(0).astype(float).astype(bool)
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
                        task_df['doc_id'] = task_df['doc_id'].astype(str)
                        
                        # We need unique (task, doc_id) pairs.
                        # current_df_slice might have duplicates if a doc_id appears multiple times
                        # in the same task_df (e.g. from different parts of a file).
                        # We should ensure doc_ids are unique within a task.
                        # The filtering via task_doc_mapping should ideally handle this.
                        # Let's select relevant columns and drop duplicates on (task, doc_id).
                        
                        current_df_slice = task_df[['task', 'doc_id', 'is_correct']].drop_duplicates(subset=['doc_id'])
                        # The key for all_task_dfs is task_in_df.
                        # So current_df_slice should only contain doc_id and is_correct for that task.
                        
                        if task_in_df not in all_task_dfs:
                            all_task_dfs[task_in_df] = current_df_slice[['doc_id', 'is_correct']]
                        else:
                            # Concatenate and keep unique doc_ids, preferring first occurrence if any overlap.
                            # This handles if different files contribute to the same task_in_df.
                            all_task_dfs[task_in_df] = pd.concat([all_task_dfs[task_in_df], current_df_slice[['doc_id', 'is_correct']]]).drop_duplicates(subset=['doc_id']).reset_index(drop=True)
            
            except Exception as e:
                print(f"Error processing file {pred_file}: {e}")
                import traceback
                traceback.print_exc()
                
    return all_task_dfs

def main():
    parser = argparse.ArgumentParser(description="Compare model performance on specific data slices.")
    parser.add_argument("dir_a", type=str, help="Directory for Model A predictions.")
    parser.add_argument("dir_b", type=str, help="Directory for Model B predictions.")
    parser.add_argument("--dataset", default="ArpanSarkar/ReasoningIntensiveStrict", 
                        help="HuggingFace dataset name used for instance matching (default: ArpanSarkar/ReasoningIntensiveStrict). This dataset is loaded by ModelPerformanceAnalyzer to filter instances.")
    parser.add_argument("--task_A_correct", default="gpqa_knowledge", 
                        help="Task name where Model A (from dir_a) is expected to be correct.")
    parser.add_argument("--task_A_incorrect", default="gpqa", 
                        help="Task name where Model A (from dir_a) is expected to be incorrect.")
    parser.add_argument("--task_B_eval", default="gpqa", 
                        help="Task name on which Model B (from dir_b) will be evaluated using the identified slice.")
    
    args = parser.parse_args()

    dir_a_path = Path(args.dir_a)
    dir_b_path = Path(args.dir_b)

    # Initialize one analyzer instance. It's used for its helper methods and dataset mappings.
    # The specific predictions_dir used at initialization doesn't strictly matter here
    # as get_task_dataframes will operate on dir_a_path and dir_b_path.
    # However, it needs *a* valid directory to load the reference dataset against.
    # We can use dir_a_path for initialization.
    print(f"Initializing ModelPerformanceAnalyzer with dataset: {args.dataset}")
    analyzer = ModelPerformanceAnalyzer(predictions_dir=str(dir_a_path), dataset_name=args.dataset)
    
    print(f"\nProcessing Model A predictions from: {args.dir_a}")
    data_A = get_task_dataframes(analyzer, dir_a_path)
    
    print(f"\nProcessing Model B predictions from: {args.dir_b}")
    # We can reuse the same analyzer instance as its state (dataset mappings) is what we need.
    data_B = get_task_dataframes(analyzer, dir_b_path)

    df_A_correct_task = data_A.get(args.task_A_correct)
    df_A_incorrect_task = data_A.get(args.task_A_incorrect)
    df_B_eval_task = data_B.get(args.task_B_eval)

    if df_A_correct_task is None:
        print(f"\nError: Task '{args.task_A_correct}' not found or empty in Model A's data ({args.dir_a}). Available tasks: {list(data_A.keys())}")
        return
    if df_A_incorrect_task is None:
        print(f"\nError: Task '{args.task_A_incorrect}' not found or empty in Model A's data ({args.dir_a}). Available tasks: {list(data_A.keys())}")
        return
    if df_B_eval_task is None:
        print(f"\nError: Task '{args.task_B_eval}' not found or empty in Model B's data ({args.dir_b}). Available tasks: {list(data_B.keys())}")
        return

    print(f"\nIdentifying target instances based on Model A performance:")
    print(f"  - Correct on task: '{args.task_A_correct}'")
    print(f"  - Incorrect on task: '{args.task_A_incorrect}'")

    # Merge Model A's task results on 'doc_id'
    # Suffixes are important if 'is_correct' is the column name from get_task_dataframes
    merged_A = pd.merge(
        df_A_correct_task.rename(columns={'is_correct': 'is_correct_task_A_correct'}), 
        df_A_incorrect_task.rename(columns={'is_correct': 'is_correct_task_A_incorrect'}), 
        on='doc_id', 
        how='inner' # Only consider doc_ids present in both tasks for Model A
    )

    if merged_A.empty:
        print(f"No common doc_ids found between task '{args.task_A_correct}' and '{args.task_A_incorrect}' for Model A.")
        return

    # Filter for the specified conditions
    target_instances_A_df = merged_A[
        (merged_A['is_correct_task_A_correct'] == True) & 
        (merged_A['is_correct_task_A_incorrect'] == False)
    ]
    
    target_doc_ids_df = target_instances_A_df[['doc_id']].drop_duplicates()

    if target_doc_ids_df.empty:
        print(f"No instances found where Model A is correct on '{args.task_A_correct}' AND incorrect on '{args.task_A_incorrect}'.")
        return
    
    num_target_instances = len(target_doc_ids_df)
    print(f"Found {num_target_instances} target instances based on Model A's performance.")

    print(f"\nEvaluating Model B on task '{args.task_B_eval}' for these {num_target_instances} target instances...")

    # Filter Model B's evaluation task data for these target doc_ids
    eval_B_on_target_slice = pd.merge(
        df_B_eval_task, 
        target_doc_ids_df, 
        on='doc_id', 
        how='inner' # Only evaluate on target doc_ids that Model B also has predictions for
    )

    if eval_B_on_target_slice.empty:
        print(f"Model B (task '{args.task_B_eval}') has no predictions for any of the {num_target_instances} target instances.")
        print(f"Target doc_ids sample: {target_doc_ids_df['doc_id'].head().tolist()}")
        print(f"Model B doc_ids sample for task '{args.task_B_eval}': {df_B_eval_task['doc_id'].head().tolist()}")
        return

    num_evaluable_B = len(eval_B_on_target_slice)
    correct_B_on_target = eval_B_on_target_slice['is_correct'].sum()
    accuracy_B_on_target = correct_B_on_target / num_evaluable_B if num_evaluable_B > 0 else 0.0
    
    print(f"\n--- Performance of Model B on the Slice ---")
    print(f"Task for Model B: '{args.task_B_eval}'")
    print(f"Number of target instances from Model A criteria: {num_target_instances}")
    print(f"Number of these instances Model B had predictions for: {num_evaluable_B}")
    if num_evaluable_B < num_target_instances:
        print(f"  (Note: Model B did not cover all {num_target_instances} target instances for task '{args.task_B_eval}')")
    
    print(f"Model B - Correct predictions on slice: {correct_B_on_target}")
    print(f"Model B - Total predictions on slice: {num_evaluable_B}")
    print(f"Model B - Accuracy on slice: {accuracy_B_on_target:.4f} ({correct_B_on_target}/{num_evaluable_B})")

if __name__ == "__main__":
    main() 