import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import re # Added for MMLU task pattern matching

# Assuming performance_analyzer.py is in the same directory or PYTHONPATH is configured
# For direct execution, ensure this import works (e.g., run from parent dir as module)
try:
    from .performance_analyzer import ModelPerformanceAnalyzer
except ImportError:
    from performance_analyzer import ModelPerformanceAnalyzer

# User-friendly names for aggregated MMLU tasks (used in command-line arguments)
USER_MMLU_KNOWLEDGE = "mmlu_knowledge"
USER_MMLU_GENERAL = "mmlu" # Represents mmlu_pro_*:cot tasks

# Internal keys for aggregated DataFrames
AGG_MMLU_KNOWLEDGE_TASK_KEY = "MMLU_PRO_COT_KNOWLEDGE_AGGREGATED"
AGG_MMLU_GENERAL_TASK_KEY = "MMLU_PRO_COT_AGGREGATED"

# Regex patterns for MMLU subtasks
MMLU_KNOWLEDGE_PATTERN = r"^mmlu_pro_.*:cot_knowledge$"
MMLU_GENERAL_PATTERN = r"^mmlu_pro_.*:cot$" # This will match :cot, and we'll ensure it's not also :cot_knowledge if needed, but distinct endings handle this.


def get_task_dataframes(analyzer: ModelPerformanceAnalyzer, directory_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Processes prediction files in a directory to extract instance-level correctness.
    Aggregates MMLU subtasks if present.

    Args:
        analyzer: An initialized ModelPerformanceAnalyzer instance.
        directory_path: Path to the directory containing model prediction subdirectories.

    Returns:
        A dictionary mapping task names (including aggregated MMLU tasks) to DataFrames.
        Each DataFrame contains 'doc_id' and 'is_correct' columns.
    """
    individual_task_dfs: Dict[str, pd.DataFrame] = {}
    
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
                predictions_df['original_doc_id'] = predictions_df['doc_id'].astype(str)
                # The 'doc_id' column will be overwritten for MMLU tasks later for unique ID during aggregation
                # For non-MMLU, 'doc_id' will remain same as 'original_doc_id' for now

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
                        # 'original_doc_id' is already set from predictions_df
                        # 'doc_id' will be the same as 'original_doc_id' for non-MMLU tasks
                        # For MMLU tasks, 'doc_id' will be task-prefixed later if needed for aggregation step
                        task_df['doc_id'] = task_df['original_doc_id'] # Initialize 'doc_id' to 'original_doc_id'

                        current_df_slice = task_df[['doc_id', 'original_doc_id', 'is_correct']].drop_duplicates(subset=['original_doc_id'])
                        
                        if task_in_df not in individual_task_dfs:
                            individual_task_dfs[task_in_df] = current_df_slice
                        else:
                            individual_task_dfs[task_in_df] = pd.concat([individual_task_dfs[task_in_df], current_df_slice]).drop_duplicates(subset=['original_doc_id']).reset_index(drop=True)

            except Exception as e:
                print(f"Error processing file {pred_file}: {e}")
                import traceback
                traceback.print_exc()

    # Aggregate MMLU tasks
    final_aggregated_dfs: Dict[str, pd.DataFrame] = {}
    mmlu_knowledge_subtask_dfs_list: List[pd.DataFrame] = []
    mmlu_general_subtask_dfs_list: List[pd.DataFrame] = []

    for task_name, df_content in individual_task_dfs.items():
        # df_content already has 'doc_id' (which is original_doc_id at this stage) and 'original_doc_id'
        if re.match(MMLU_KNOWLEDGE_PATTERN, task_name):
            df_copy = df_content.copy()
            # Overwrite 'doc_id' with the unique prefixed ID for aggregation internal key
            df_copy['doc_id'] = task_name + "_" + df_copy['original_doc_id'].astype(str)
            mmlu_knowledge_subtask_dfs_list.append(df_copy[['doc_id', 'original_doc_id', 'is_correct']])
        elif re.match(MMLU_GENERAL_PATTERN, task_name): 
            df_copy = df_content.copy()
            df_copy['doc_id'] = task_name + "_" + df_copy['original_doc_id'].astype(str)
            mmlu_general_subtask_dfs_list.append(df_copy[['doc_id', 'original_doc_id', 'is_correct']])
        else:
            # For non-MMLU tasks, 'doc_id' is already the original_doc_id.
            # Ensure it has all three columns for consistency if other parts rely on it.
            final_aggregated_dfs[task_name] = df_content[['doc_id', 'original_doc_id', 'is_correct']]

    if mmlu_knowledge_subtask_dfs_list:
        # Concatenate using the unique 'doc_id' (task_name + original_doc_id)
        # The 'original_doc_id' column is preserved.
        agg_df = pd.concat(mmlu_knowledge_subtask_dfs_list).drop_duplicates(subset=['doc_id']).reset_index(drop=True)
        final_aggregated_dfs[AGG_MMLU_KNOWLEDGE_TASK_KEY] = agg_df
        print(f"Aggregated {len(mmlu_knowledge_subtask_dfs_list)} MMLU knowledge subtasks into '{AGG_MMLU_KNOWLEDGE_TASK_KEY}' with {len(agg_df)} unique doc_ids.")


    if mmlu_general_subtask_dfs_list:
        agg_df = pd.concat(mmlu_general_subtask_dfs_list).drop_duplicates(subset=['doc_id']).reset_index(drop=True)
        final_aggregated_dfs[AGG_MMLU_GENERAL_TASK_KEY] = agg_df
        print(f"Aggregated {len(mmlu_general_subtask_dfs_list)} MMLU general subtasks into '{AGG_MMLU_GENERAL_TASK_KEY}' with {len(agg_df)} unique doc_ids.")
                
    return final_aggregated_dfs

def main():
    parser = argparse.ArgumentParser(description="Compare model performance on specific data slices, with MMLU aggregation.")
    parser.add_argument("dir_a", type=str, help="Directory for Model A predictions.")
    parser.add_argument("dir_b", type=str, help="Directory for Model B predictions.")
    parser.add_argument("--dataset", default="ArpanSarkar/ReasoningIntensiveStrict", 
                        help="HuggingFace dataset name used by ModelPerformanceAnalyzer (default: ArpanSarkar/ReasoningIntensiveStrict). Note: this script uses all instances from prediction files, not filtering by this dataset.")
    parser.add_argument("--task_A_correct", default="gpqa_knowledge", 
                        help=f"Task name where Model A is correct. Use '{USER_MMLU_KNOWLEDGE}' or '{USER_MMLU_GENERAL}' for aggregated MMLU tasks.")
    parser.add_argument("--task_A_incorrect", default="gpqa", 
                        help=f"Task name where Model A is incorrect. Use '{USER_MMLU_KNOWLEDGE}' or '{USER_MMLU_GENERAL}' for aggregated MMLU tasks.")
    parser.add_argument("--task_B_eval", default="gpqa", 
                        help=f"Task name for Model B evaluation. Use '{USER_MMLU_KNOWLEDGE}' or '{USER_MMLU_GENERAL}' for aggregated MMLU tasks.")
    
    args = parser.parse_args()

    dir_a_path = Path(args.dir_a)
    dir_b_path = Path(args.dir_b)

    print(f"Initializing ModelPerformanceAnalyzer with dataset: {args.dataset} (Note: for this script, analyzer is primarily used for file processing utilities, not dataset filtering).")
    # Initialize analyzer once; its dataset_name is for its internal _filter_matching_instances, which we are bypassing.
    # However, _create_task_doc_mapping uses it, so it must be valid if that part of analyzer were used.
    analyzer = ModelPerformanceAnalyzer(predictions_dir=str(dir_a_path), dataset_name=args.dataset) 
    
    print(f"\nProcessing Model A predictions from: {args.dir_a}")
    data_A = get_task_dataframes(analyzer, dir_a_path)
    
    print(f"\nProcessing Model B predictions from: {args.dir_b}")
    data_B = get_task_dataframes(analyzer, dir_b_path)

    def map_user_task_to_internal_key(user_task_name: str) -> str:
        if user_task_name == USER_MMLU_KNOWLEDGE:
            return AGG_MMLU_KNOWLEDGE_TASK_KEY
        if user_task_name == USER_MMLU_GENERAL:
            return AGG_MMLU_GENERAL_TASK_KEY
        return user_task_name

    task_A_correct_key = map_user_task_to_internal_key(args.task_A_correct)
    task_A_incorrect_key = map_user_task_to_internal_key(args.task_A_incorrect)
    task_B_eval_key = map_user_task_to_internal_key(args.task_B_eval)

    df_A_correct_task = data_A.get(task_A_correct_key)
    df_A_incorrect_task = data_A.get(task_A_incorrect_key)
    df_B_eval_task = data_B.get(task_B_eval_key)

    # Update print statements to show the key being used (which might be an aggregated key)
    if df_A_correct_task is None:
        print(f"\nError: Task '{task_A_correct_key}' (mapped from '{args.task_A_correct}') not found or empty in Model A's data ({args.dir_a}). Available tasks: {list(data_A.keys())}")
        return
    if df_A_incorrect_task is None:
        print(f"\nError: Task '{task_A_incorrect_key}' (mapped from '{args.task_A_incorrect}') not found or empty in Model A's data ({args.dir_a}). Available tasks: {list(data_A.keys())}")
        return
    if df_B_eval_task is None:
        print(f"\nError: Task '{task_B_eval_key}' (mapped from '{args.task_B_eval}') not found or empty in Model B's data ({args.dir_b}). Available tasks: {list(data_B.keys())}")
        return

    print(f"\nIdentifying target instances based on Model A performance:")
    print(f"  - Correct on task: '{task_A_correct_key}' (from input '{args.task_A_correct}')")
    print(f"  - Incorrect on task: '{task_A_incorrect_key}' (from input '{args.task_A_incorrect}')")

    # Ensure 'original_doc_id' exists for merging
    if 'original_doc_id' not in df_A_correct_task.columns or 'original_doc_id' not in df_A_incorrect_task.columns:
        print("\nError: 'original_doc_id' column missing in one of the dataframes for Model A. Cannot merge.")
        # Print available columns for debugging
        if 'original_doc_id' not in df_A_correct_task.columns:
            print(f"Columns in df_A_correct_task ({task_A_correct_key}): {df_A_correct_task.columns.tolist()}")
        if 'original_doc_id' not in df_A_incorrect_task.columns:
            print(f"Columns in df_A_incorrect_task ({task_A_incorrect_key}): {df_A_incorrect_task.columns.tolist()}")
        return

    merged_A = pd.merge(
        df_A_correct_task.rename(columns={'is_correct': 'is_correct_task_A_correct'}), 
        df_A_incorrect_task.rename(columns={'is_correct': 'is_correct_task_A_incorrect'}), 
        on='original_doc_id', 
        how='inner'
    )

    if merged_A.empty:
        print(f"No common original_doc_ids found between task '{task_A_correct_key}' and '{task_A_incorrect_key}' for Model A based on 'original_doc_id' merging.")
        return

    target_instances_A_df = merged_A[
        (merged_A['is_correct_task_A_correct'] == True) & 
        (merged_A['is_correct_task_A_incorrect'] == False)
    ]
    
    # We need the 'original_doc_id' to filter Model B's data
    # Filter to keep only rows where doc IDs before colon match
    # target_instances_A_df['doc_id_before_colon'] = target_instances_A_df['original_doc_id'].str.split(':').str[0]
    target_instances_A_df = target_instances_A_df[target_instances_A_df['doc_id_x'].str.split(':').str[0] == target_instances_A_df['doc_id_y'].str.split(':').str[0]]
    target_original_doc_ids_df = target_instances_A_df[['doc_id_y']].drop_duplicates()

    if target_original_doc_ids_df.empty:
        print(f"No instances found where Model A is correct on '{task_A_correct_key}' AND incorrect on '{task_A_incorrect_key}'.")
        return
    
    num_target_instances = len(target_original_doc_ids_df)
    print(f"Found {num_target_instances} target instances based on Model A's performance (using original_doc_id).")

    print(f"\nEvaluating Model B on task '{task_B_eval_key}' (from input '{args.task_B_eval}') for these {num_target_instances} target instances...")

    if 'original_doc_id' not in df_B_eval_task.columns:
        print(f"\nError: 'original_doc_id' column missing in dataframe for Model B task '{task_B_eval_key}'. Cannot filter.")
        print(f"Columns in df_B_eval_task ({task_B_eval_key}): {df_B_eval_task.columns.tolist()}")
        return
    # from IPython import embed; embed()
    eval_B_on_target_slice = pd.merge(
        df_B_eval_task, 
        target_original_doc_ids_df.rename(columns={'doc_id_y': 'doc_id'}),
        on='doc_id',
        how='inner'
    )
    from IPython import embed; embed()

    if eval_B_on_target_slice.empty:
        print(f"Model B (task '{task_B_eval_key}') has no predictions for any of the {num_target_instances} target instances based on 'doc_id'.")
        print(f"Target doc_ids sample: {target_original_doc_ids_df['doc_id'].head().tolist()}")
        if 'doc_id' in df_B_eval_task:
            print(f"Model B doc_ids sample for task '{task_B_eval_key}': {df_B_eval_task['doc_id'].head().tolist()}")
        return

    num_evaluable_B = len(eval_B_on_target_slice)
    correct_B_on_target = eval_B_on_target_slice['is_correct'].sum()
    accuracy_B_on_target = correct_B_on_target / num_evaluable_B if num_evaluable_B > 0 else 0.0
    
    print(f"\n--- Performance of Model B on the Slice ---")
    print(f"Task for Model B: '{task_B_eval_key}' (from input '{args.task_B_eval}')")
    print(f"Number of target instances from Model A criteria: {num_target_instances}")
    print(f"Number of these instances Model B had predictions for (matched on original_doc_id): {num_evaluable_B}")
    if num_evaluable_B < num_target_instances:
        print(f"  (Note: Model B did not cover all {num_target_instances} target instances for task '{task_B_eval_key}')")
    
    print(f"Model B - Correct predictions on slice: {correct_B_on_target}")
    print(f"Model B - Total predictions on slice: {num_evaluable_B}")
    print(f"Model B - Accuracy on slice: {accuracy_B_on_target:.4f} ({correct_B_on_target}/{num_evaluable_B})")

if __name__ == "__main__":
    main()
