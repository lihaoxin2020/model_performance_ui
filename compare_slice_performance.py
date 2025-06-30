import argparse
import pandas as pd
import numpy as np
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

# User-friendly names for aggregated LAB_BENCH tasks (used in command-line arguments)
USER_LAB_BENCH_KNOWLEDGE = "lab_bench_knowledge"
USER_LAB_BENCH_GENERAL = "lab_bench" # Represents lab_bench_*:cot tasks

# Internal keys for aggregated DataFrames
AGG_MMLU_KNOWLEDGE_TASK_KEY = "MMLU_PRO_COT_KNOWLEDGE_AGGREGATED"
AGG_MMLU_GENERAL_TASK_KEY = "MMLU_PRO_COT_AGGREGATED"

# Internal keys for aggregated LAB_BENCH DataFrames
AGG_LAB_BENCH_KNOWLEDGE_TASK_KEY = "LAB_BENCH_COT_KNOWLEDGE_AGGREGATED"
AGG_LAB_BENCH_GENERAL_TASK_KEY = "LAB_BENCH_COT_AGGREGATED"

# Regex patterns for MMLU subtasks
MMLU_KNOWLEDGE_PATTERN = r"^mmlu_pro_.*:cot_knowledge$"
MMLU_GENERAL_PATTERN = r"^mmlu_pro_.*:cot$" # This will match :cot, and we'll ensure it's not also :cot_knowledge if needed, but distinct endings handle this.

# Regex patterns for LAB_BENCH subtasks
LAB_BENCH_KNOWLEDGE_PATTERN = r"^lab_bench_.*:cot_knowledge$"
LAB_BENCH_GENERAL_PATTERN = r"^lab_bench_.*:cot$" # This will match :cot, and we'll ensure it's not also :cot_knowledge if needed, but distinct endpoints handle this.


def get_task_dataframes(analyzer: ModelPerformanceAnalyzer, directory_path: Path) -> Dict[str, List[pd.DataFrame]]:
    """
    Processes prediction files in a directory to extract instance-level correctness.
    Aggregates MMLU and LAB_BENCH subtasks if present.

    Args:
        analyzer: An initialized ModelPerformanceAnalyzer instance.
        directory_path: Path to the directory containing model prediction subdirectories.

    Returns:
        A dictionary mapping task names (including aggregated MMLU and LAB_BENCH tasks) to lists of DataFrames.
        Each DataFrame contains 'doc_id' and 'is_correct' columns.
        Multiple DataFrames per task represent different runs/files.
    """
    individual_task_dfs: Dict[str, List[pd.DataFrame]] = {}
    
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
                            individual_task_dfs[task_in_df] = [current_df_slice]
                        else:
                            individual_task_dfs[task_in_df].append(current_df_slice)

            except Exception as e:
                print(f"Error processing file {pred_file}: {e}")
                import traceback
                traceback.print_exc()

    # Aggregate MMLU and LAB_BENCH tasks while preserving run structure
    final_aggregated_dfs: Dict[str, List[pd.DataFrame]] = {}
    
    # Track MMLU subtasks by run index
    mmlu_knowledge_by_run: Dict[int, List[pd.DataFrame]] = {}
    mmlu_general_by_run: Dict[int, List[pd.DataFrame]] = {}
    
    # Track LAB_BENCH subtasks by run index
    lab_bench_knowledge_by_run: Dict[int, List[pd.DataFrame]] = {}
    lab_bench_general_by_run: Dict[int, List[pd.DataFrame]] = {}
    
    max_runs = 0

    # First, determine the maximum number of runs across all tasks
    for task_name, df_list in individual_task_dfs.items():
        if (re.match(MMLU_KNOWLEDGE_PATTERN, task_name) or re.match(MMLU_GENERAL_PATTERN, task_name) or
            re.match(LAB_BENCH_KNOWLEDGE_PATTERN, task_name) or re.match(LAB_BENCH_GENERAL_PATTERN, task_name)):
            max_runs = max(max_runs, len(df_list))

    # Collect MMLU and LAB_BENCH subtasks by run index
    for task_name, df_list in individual_task_dfs.items():
        if re.match(MMLU_KNOWLEDGE_PATTERN, task_name):
            for run_idx, df_content in enumerate(df_list):
                if run_idx not in mmlu_knowledge_by_run:
                    mmlu_knowledge_by_run[run_idx] = []
                df_copy = df_content.copy()
                # Overwrite 'doc_id' with the unique prefixed ID for aggregation internal key
                df_copy['doc_id'] = task_name + "_" + df_copy['original_doc_id'].astype(str)
                mmlu_knowledge_by_run[run_idx].append(df_copy[['doc_id', 'original_doc_id', 'is_correct']])
        elif re.match(MMLU_GENERAL_PATTERN, task_name): 
            for run_idx, df_content in enumerate(df_list):
                if run_idx not in mmlu_general_by_run:
                    mmlu_general_by_run[run_idx] = []
                df_copy = df_content.copy()
                df_copy['doc_id'] = task_name + "_" + df_copy['original_doc_id'].astype(str)
                mmlu_general_by_run[run_idx].append(df_copy[['doc_id', 'original_doc_id', 'is_correct']])
        elif re.match(LAB_BENCH_KNOWLEDGE_PATTERN, task_name):
            for run_idx, df_content in enumerate(df_list):
                if run_idx not in lab_bench_knowledge_by_run:
                    lab_bench_knowledge_by_run[run_idx] = []
                df_copy = df_content.copy()
                # Overwrite 'doc_id' with the unique prefixed ID for aggregation internal key
                df_copy['doc_id'] = task_name + "_" + df_copy['original_doc_id'].astype(str)
                lab_bench_knowledge_by_run[run_idx].append(df_copy[['doc_id', 'original_doc_id', 'is_correct']])
        elif re.match(LAB_BENCH_GENERAL_PATTERN, task_name): 
            for run_idx, df_content in enumerate(df_list):
                if run_idx not in lab_bench_general_by_run:
                    lab_bench_general_by_run[run_idx] = []
                df_copy = df_content.copy()
                df_copy['doc_id'] = task_name + "_" + df_copy['original_doc_id'].astype(str)
                lab_bench_general_by_run[run_idx].append(df_copy[['doc_id', 'original_doc_id', 'is_correct']])
        else:
            # For non-MMLU/LAB_BENCH tasks, keep the list structure for multiple runs
            final_aggregated_dfs[task_name] = [df[['doc_id', 'original_doc_id', 'is_correct']] for df in df_list]

    # Create aggregated dataframes for each run
    if mmlu_knowledge_by_run:
        mmlu_knowledge_aggregated_runs = []
        for run_idx in sorted(mmlu_knowledge_by_run.keys()):
            run_dfs = mmlu_knowledge_by_run[run_idx]
            if run_dfs:
                agg_df = pd.concat(run_dfs).drop_duplicates(subset=['doc_id']).reset_index(drop=True)
                mmlu_knowledge_aggregated_runs.append(agg_df)
        
        final_aggregated_dfs[AGG_MMLU_KNOWLEDGE_TASK_KEY] = mmlu_knowledge_aggregated_runs
        total_subtasks = sum(len(mmlu_knowledge_by_run[run_idx]) for run_idx in mmlu_knowledge_by_run)
        print(f"Aggregated {total_subtasks} MMLU knowledge subtasks across {len(mmlu_knowledge_aggregated_runs)} runs into '{AGG_MMLU_KNOWLEDGE_TASK_KEY}'.")

    if mmlu_general_by_run:
        mmlu_general_aggregated_runs = []
        for run_idx in sorted(mmlu_general_by_run.keys()):
            run_dfs = mmlu_general_by_run[run_idx]
            if run_dfs:
                agg_df = pd.concat(run_dfs).drop_duplicates(subset=['doc_id']).reset_index(drop=True)
                mmlu_general_aggregated_runs.append(agg_df)
        
        final_aggregated_dfs[AGG_MMLU_GENERAL_TASK_KEY] = mmlu_general_aggregated_runs
        total_subtasks = sum(len(mmlu_general_by_run[run_idx]) for run_idx in mmlu_general_by_run)
        print(f"Aggregated {total_subtasks} MMLU general subtasks across {len(mmlu_general_aggregated_runs)} runs into '{AGG_MMLU_GENERAL_TASK_KEY}'.")

    # Create aggregated dataframes for LAB_BENCH tasks
    if lab_bench_knowledge_by_run:
        lab_bench_knowledge_aggregated_runs = []
        for run_idx in sorted(lab_bench_knowledge_by_run.keys()):
            run_dfs = lab_bench_knowledge_by_run[run_idx]
            if run_dfs:
                agg_df = pd.concat(run_dfs).drop_duplicates(subset=['doc_id']).reset_index(drop=True)
                lab_bench_knowledge_aggregated_runs.append(agg_df)
        
        final_aggregated_dfs[AGG_LAB_BENCH_KNOWLEDGE_TASK_KEY] = lab_bench_knowledge_aggregated_runs
        total_subtasks = sum(len(lab_bench_knowledge_by_run[run_idx]) for run_idx in lab_bench_knowledge_by_run)
        print(f"Aggregated {total_subtasks} LAB_BENCH knowledge subtasks across {len(lab_bench_knowledge_aggregated_runs)} runs into '{AGG_LAB_BENCH_KNOWLEDGE_TASK_KEY}'.")

    if lab_bench_general_by_run:
        lab_bench_general_aggregated_runs = []
        for run_idx in sorted(lab_bench_general_by_run.keys()):
            run_dfs = lab_bench_general_by_run[run_idx]
            if run_dfs:
                agg_df = pd.concat(run_dfs).drop_duplicates(subset=['doc_id']).reset_index(drop=True)
                lab_bench_general_aggregated_runs.append(agg_df)
        
        final_aggregated_dfs[AGG_LAB_BENCH_GENERAL_TASK_KEY] = lab_bench_general_aggregated_runs
        total_subtasks = sum(len(lab_bench_general_by_run[run_idx]) for run_idx in lab_bench_general_by_run)
        print(f"Aggregated {total_subtasks} LAB_BENCH general subtasks across {len(lab_bench_general_aggregated_runs)} runs into '{AGG_LAB_BENCH_GENERAL_TASK_KEY}'.")
                
    return final_aggregated_dfs

def main():
    parser = argparse.ArgumentParser(description="Compare model performance on specific data slices, with MMLU and LAB_BENCH aggregation.")
    parser.add_argument("dir_b", type=str, help="Directory for Model B predictions.")
    parser.add_argument("--dataset", default="ArpanSarkar/ReasoningIntensiveStrict", 
                        help="HuggingFace dataset name used by ModelPerformanceAnalyzer (default: ArpanSarkar/ReasoningIntensiveStrict). Note: this script uses all instances from prediction files, not filtering by this dataset.")
    parser.add_argument("--dir_A_correct", type=str, required=True,
                        help="Directory for Model A predictions on correct tasks.")
    parser.add_argument("--dir_A_incorrect", type=str, required=True,
                        help="Directory for Model A predictions on incorrect task.")
    parser.add_argument("--dir_correct", type=str, required=False,
                        help="Directory for exclusion filtering - instances where this model is correct will be excluded from target set.")
    parser.add_argument("--task_A_correct", nargs='+', default=["gpqa_knowledge"], 
                        help=f"Task name(s) where Model A should be correct (multiple values allowed). Use '{USER_MMLU_KNOWLEDGE}' or '{USER_MMLU_GENERAL}' for aggregated MMLU tasks, '{USER_LAB_BENCH_KNOWLEDGE}' or '{USER_LAB_BENCH_GENERAL}' for aggregated LAB_BENCH tasks.")
    parser.add_argument("--task_A_incorrect", default="gpqa", 
                        help=f"Task name where Model A should be incorrect. Use '{USER_MMLU_KNOWLEDGE}' or '{USER_MMLU_GENERAL}' for aggregated MMLU tasks, '{USER_LAB_BENCH_KNOWLEDGE}' or '{USER_LAB_BENCH_GENERAL}' for aggregated LAB_BENCH tasks.")
    parser.add_argument("--task_correct", default=None,
                        help=f"Task name for exclusion filtering. Instances where this task is correct will be excluded from target set. Use '{USER_MMLU_KNOWLEDGE}' or '{USER_MMLU_GENERAL}' for aggregated MMLU tasks, '{USER_LAB_BENCH_KNOWLEDGE}' or '{USER_LAB_BENCH_GENERAL}' for aggregated LAB_BENCH tasks.")
    parser.add_argument("--task_B_eval", default="gpqa", 
                        help=f"Task name for Model B evaluation. Use '{USER_MMLU_KNOWLEDGE}' or '{USER_MMLU_GENERAL}' for aggregated MMLU tasks, '{USER_LAB_BENCH_KNOWLEDGE}' or '{USER_LAB_BENCH_GENERAL}' for aggregated LAB_BENCH tasks.")
    parser.add_argument("--mode", choices=["union", "intersection", "first_run"], default="union",
                        help="Mode for combining results: 'union' takes union of all intersections (default), 'intersection' takes intersection of all intersections, 'first_run' uses only first run intersection.")
    
    args = parser.parse_args()

    dir_A_correct_path = Path(args.dir_A_correct)
    dir_A_incorrect_path = Path(args.dir_A_incorrect)
    dir_b_path = Path(args.dir_b)
    
    # Handle optional exclusion directory
    dir_correct_path = None
    if args.dir_correct:
        dir_correct_path = Path(args.dir_correct)

    print(f"Initializing ModelPerformanceAnalyzer with dataset: {args.dataset} (Note: for this script, analyzer is primarily used for file processing utilities, not dataset filtering).")
    # Initialize analyzer once; its dataset_name is for its internal _filter_matching_instances, which we are bypassing.
    # However, _create_task_doc_mapping uses it, so it must be valid if that part of analyzer were used.
    analyzer = ModelPerformanceAnalyzer(predictions_dir=str(dir_A_correct_path), dataset_name=args.dataset) 
    
    print(f"\nProcessing Model A correct task predictions from: {args.dir_A_correct}")
    data_A_correct = get_task_dataframes(analyzer, dir_A_correct_path)
    
    print(f"\nProcessing Model A incorrect task predictions from: {args.dir_A_incorrect}")
    data_A_incorrect = get_task_dataframes(analyzer, dir_A_incorrect_path)
    
    # Load exclusion data if provided
    data_correct = None
    if dir_correct_path and args.task_correct:
        print(f"\nProcessing exclusion task predictions from: {args.dir_correct}")
        data_correct = get_task_dataframes(analyzer, dir_correct_path)
    
    print(f"\nProcessing Model B predictions from: {args.dir_b}")
    data_B = get_task_dataframes(analyzer, dir_b_path)

    def map_user_task_to_internal_key(user_task_name: str) -> str:
        if user_task_name == USER_MMLU_KNOWLEDGE:
            return AGG_MMLU_KNOWLEDGE_TASK_KEY
        if user_task_name == USER_MMLU_GENERAL:
            return AGG_MMLU_GENERAL_TASK_KEY
        if user_task_name == USER_LAB_BENCH_KNOWLEDGE:
            return AGG_LAB_BENCH_KNOWLEDGE_TASK_KEY
        if user_task_name == USER_LAB_BENCH_GENERAL:
            return AGG_LAB_BENCH_GENERAL_TASK_KEY
        return user_task_name

    # Map all task_A_correct entries
    task_A_correct_keys = [map_user_task_to_internal_key(task) for task in args.task_A_correct]
    task_A_incorrect_key = map_user_task_to_internal_key(args.task_A_incorrect)
    task_B_eval_key = map_user_task_to_internal_key(args.task_B_eval)
    task_correct_key = map_user_task_to_internal_key(args.task_correct) if args.task_correct else None

    # Collect dataframes for all correct tasks
    df_A_correct_tasks = {}
    for i, task_key in enumerate(task_A_correct_keys):
        df_list = data_A_correct.get(task_key)
        if df_list is None:
            print(f"\nError: Task '{task_key}' (mapped from '{args.task_A_correct[i]}') not found or empty in Model A's correct data ({args.dir_A_correct}). Available tasks: {list(data_A_correct.keys())}")
            return
        df_A_correct_tasks[task_key] = df_list

    df_A_incorrect_task = data_A_incorrect.get(task_A_incorrect_key)
    df_B_eval_task = data_B.get(task_B_eval_key)

    if df_A_incorrect_task is None:
        print(f"\nError: Task '{task_A_incorrect_key}' (mapped from '{args.task_A_incorrect}') not found or empty in Model A's incorrect data ({args.dir_A_incorrect}). Available tasks: {list(data_A_incorrect.keys())}")
        return
    if df_B_eval_task is None:
        print(f"\nError: Task '{task_B_eval_key}' (mapped from '{args.task_B_eval}') not found or empty in Model B's data ({args.dir_b}). Available tasks: {list(data_B.keys())}")
        return

    # Handle exclusion task validation
    df_correct_task = None
    if task_correct_key and data_correct:
        df_correct_task = data_correct.get(task_correct_key)
        if df_correct_task is None:
            print(f"\nError: Task '{task_correct_key}' (mapped from '{args.task_correct}') not found or empty in exclusion data ({args.dir_correct}). Available tasks: {list(data_correct.keys())}")
            return

    print(f"\nIdentifying target instances based on Model A performance:")
    print(f"  - Correct on any of tasks: {[f'{key} (from {orig})' for key, orig in zip(task_A_correct_keys, args.task_A_correct)]}")
    print(f"  - Incorrect on task: '{task_A_incorrect_key}' (from input '{args.task_A_incorrect}')")
    if df_correct_task is not None:
        print(f"  - Exclusion: removing instances correct on task '{task_correct_key}' (from input '{args.task_correct}')")

    # Ensure 'original_doc_id' exists for all dataframes
    for task_key, df_list in df_A_correct_tasks.items():
        for df in df_list:
            if 'original_doc_id' not in df.columns:
                print(f"\nError: 'original_doc_id' column missing in dataframe for task '{task_key}'. Cannot proceed.")
                print(f"Columns in df for {task_key}: {df.columns.tolist()}")
                return
    
    if 'original_doc_id' not in df_A_incorrect_task[0].columns:
        print(f"\nError: 'original_doc_id' column missing in dataframe for task '{task_A_incorrect_key}'. Cannot proceed.")
        print(f"Columns in df_A_incorrect_task ({task_A_incorrect_key}): {df_A_incorrect_task[0].columns.tolist()}")
        return
    
    # Validate exclusion task columns if provided
    if df_correct_task is not None:
        if 'original_doc_id' not in df_correct_task[0].columns:
            print(f"\nError: 'original_doc_id' column missing in dataframe for exclusion task '{task_correct_key}'. Cannot proceed.")
            print(f"Columns in df_correct_task ({task_correct_key}): {df_correct_task[0].columns.tolist()}")
            return

    # Get instances where the incorrect task is incorrect (from all runs)
    incorrect_instances_all_runs = set()
    for df in df_A_incorrect_task:
        incorrect_instances = set(df[df['is_correct'] == False]['original_doc_id'].astype(str))
        incorrect_instances_all_runs.update(incorrect_instances)
    print(f"  - Task '{task_A_incorrect_key}': {len(incorrect_instances_all_runs)} incorrect instances across all runs")

    if args.mode == "union":
        print(f"  - Using union mode: intersecting with all runs of correct tasks")
        # For each correct task and run, find intersection with incorrect instances
        # Then take union of all intersections
        target_instances_union = set()
        
        for task_key, df_list in df_A_correct_tasks.items():
            task_intersections = set()
            for i, df in enumerate(df_list):
                correct_instances = set(df[df['is_correct'] == True]['original_doc_id'].astype(str))
                intersection = correct_instances.intersection(incorrect_instances_all_runs)
                task_intersections.update(intersection)
                print(f"  - Task '{task_key}' run {i+1}: {len(correct_instances)} correct instances, {len(intersection)} intersecting with incorrect task")
            
            target_instances_union.update(task_intersections)
            print(f"  - Task '{task_key}' total intersections: {len(task_intersections)} instances")
        
        print(f"  - Union of all intersections: {len(target_instances_union)} instances")
    
    elif args.mode == "intersection":
        print(f"  - Using intersection mode: finding instances consistent across all runs of correct tasks")
        # For each correct task, find intersection of all its runs with incorrect instances
        # Then take intersection of all these intersections
        target_instances_union = None
        
        for task_key, df_list in df_A_correct_tasks.items():
            task_intersection_of_intersections = None
            all_task_intersections = []
            
            for i, df in enumerate(df_list):
                correct_instances = set(df[df['is_correct'] == True]['original_doc_id'].astype(str))
                intersection = correct_instances.intersection(incorrect_instances_all_runs)
                all_task_intersections.append(intersection)
                print(f"  - Task '{task_key}' run {i+1}: {len(correct_instances)} correct instances, {len(intersection)} intersecting with incorrect task")
            
            # Take intersection of all runs for this task
            if all_task_intersections:
                task_intersection_of_intersections = all_task_intersections[0]
                for intersection_set in all_task_intersections[1:]:
                    task_intersection_of_intersections = task_intersection_of_intersections.intersection(intersection_set)
                
                print(f"  - Task '{task_key}' intersection across all runs: {len(task_intersection_of_intersections)} instances")
                
                # Union with other tasks (or initialize if first task)
                if target_instances_union is None:
                    target_instances_union = task_intersection_of_intersections
                else:
                    target_instances_union = target_instances_union.union(task_intersection_of_intersections)
        
        if target_instances_union is None:
            target_instances_union = set()
        
        print(f"  - Final intersection result: {len(target_instances_union)} instances")
    
    else:  # first_run mode
        print(f"  - Using first-run mode: intersecting only with first run of correct tasks")
        # Only use intersection with first run of each correct task
        target_instances_union = set()
        
        for task_key, df_list in df_A_correct_tasks.items():
            if df_list:  # Ensure there's at least one run
                df_first_run = df_list[0]
                correct_instances = set(df_first_run[df_first_run['is_correct'] == True]['original_doc_id'].astype(str))
                intersection = correct_instances.intersection(incorrect_instances_all_runs)
                target_instances_union.update(intersection)
                print(f"  - Task '{task_key}' first run: {len(correct_instances)} correct instances, {len(intersection)} intersecting with incorrect task")
        
        print(f"  - Total intersections (first run only): {len(target_instances_union)} instances")

    # Apply exclusion filtering if specified
    if df_correct_task is not None:
        print(f"\n  - Applying exclusion filtering:")
        initial_target_count = len(target_instances_union)
        
        # Get instances where the exclusion task is correct (from all runs)
        exclusion_correct_instances_all_runs = set()
        for df in df_correct_task:
            correct_instances = set(df[df['is_correct'] == True]['original_doc_id'].astype(str))
            exclusion_correct_instances_all_runs.update(correct_instances)
        
        print(f"  - Task '{task_correct_key}': {len(exclusion_correct_instances_all_runs)} correct instances to exclude across all runs")
        
        # Remove instances where exclusion task is correct
        target_instances_union = target_instances_union - exclusion_correct_instances_all_runs
        excluded_count = initial_target_count - len(target_instances_union)
        
        print(f"  - Excluded {excluded_count} instances where '{task_correct_key}' was correct")
        print(f"  - Final target set after exclusion: {len(target_instances_union)} instances")

    if not target_instances_union:
        print(f"No target instances found after filtering criteria.")
        return

    print(f"\nEvaluating Model B on task '{task_B_eval_key}' (from input '{args.task_B_eval}') for these {len(target_instances_union)} target instances...")

    if 'original_doc_id' not in df_B_eval_task[0].columns:
        print(f"\nError: 'original_doc_id' column missing in dataframe for Model B task '{task_B_eval_key}'. Cannot filter.")
        print(f"Columns in df_B_eval_task ({task_B_eval_key}): {df_B_eval_task[0].columns.tolist()}")
        return

    # Filter Model B's predictions to only include target instances and evaluate across all runs
    all_run_results = []
    total_correct_across_runs = 0
    total_evaluated_across_runs = 0
    
    for i, df_b_run in enumerate(df_B_eval_task):
        eval_B_on_target_slice = df_b_run[df_b_run['original_doc_id'].astype(str).isin(target_instances_union)]
        
        if eval_B_on_target_slice.empty:
            print(f"Model B run {i+1} (task '{task_B_eval_key}') has no predictions for target instances.")
            continue
            
        num_evaluable_B = len(eval_B_on_target_slice)
        correct_B_on_target = eval_B_on_target_slice['is_correct'].sum()
        accuracy_B_on_target = correct_B_on_target / num_evaluable_B if num_evaluable_B > 0 else 0.0
        
        all_run_results.append({
            'run': i+1,
            'correct': correct_B_on_target,
            'total': num_evaluable_B,
            'accuracy': accuracy_B_on_target
        })
        
        total_correct_across_runs += correct_B_on_target
        total_evaluated_across_runs += num_evaluable_B
        
        print(f"  Run {i+1}: {correct_B_on_target}/{num_evaluable_B} = {accuracy_B_on_target:.4f}")

    if not all_run_results:
        print(f"Model B (task '{task_B_eval_key}') has no predictions for any of the {len(target_instances_union)} target instances across all runs.")
        print(f"Target original_doc_ids sample: {list(target_instances_union)[:10]}")
        if 'original_doc_id' in df_B_eval_task[0]:
            print(f"Model B original_doc_ids sample for task '{task_B_eval_key}': {df_B_eval_task[0]['original_doc_id'].head().tolist()}")
        return

    # Calculate average accuracy and standard deviation across runs
    accuracies = [r['accuracy'] for r in all_run_results if r['accuracy'] < 1.0]
    avg_accuracy_across_runs = np.mean(accuracies)
    std_accuracy_across_runs = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0
    overall_accuracy = total_correct_across_runs / total_evaluated_across_runs if total_evaluated_across_runs > 0 else 0.0
    
    print(f"\n--- Performance of Model B on the Slice ---")
    print(f"Task for Model B: '{task_B_eval_key}' (from input '{args.task_B_eval}')")
    mode_descriptions = {
        "union": "union of intersections",
        "intersection": "intersection of intersections", 
        "first_run": "first-run intersection"
    }
    mode_description = mode_descriptions[args.mode]
    exclusion_info = f" with exclusion filtering" if df_correct_task is not None else ""
    print(f"Number of target instances from {mode_description} criteria{exclusion_info}: {len(target_instances_union)}")
    print(f"Number of runs evaluated: {len(all_run_results)}")
    
    print(f"\nPer-run results:")
    for result in all_run_results:
        print(f"  Run {result['run']}: {result['correct']}/{result['total']} = {result['accuracy']:.4f}")
    
    print(f"\nAggregate results:")
    print(f"  Average accuracy across runs: {avg_accuracy_across_runs:.4f} ± {std_accuracy_across_runs:.4f}")
    print(f"  Overall accuracy (total correct/total evaluated): {overall_accuracy:.4f} ({total_correct_across_runs}/{total_evaluated_across_runs})")
    
    if total_evaluated_across_runs < len(target_instances_union):
        print(f"  (Note: Model B did not cover all {len(target_instances_union)} target instances for task '{task_B_eval_key}' across all runs)")

if __name__ == "__main__":
    main()
