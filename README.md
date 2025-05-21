# Model Performance Analyzer

This tool analyzes model performance across different benchmarks by comparing predictions against reference datasets from Hugging Face. By default, it uses the ReasoningIntensiveStrict dataset, but can be configured to use any HuggingFace dataset.

## Requirements

Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure

The program works with nested directory structures where model predictions are stored in files with paths like:
```
predictions_dir/
  lmeval-model1-on-benchmark1-1234567890/
    task-000-subtask1-predictions.jsonl
    task-001-subtask2-predictions.jsonl
    metrics.json
  lmeval-model2-on-benchmark1-1234567890/
    task-000-subtask1-predictions.jsonl
    ...
  model_dir/
    lmeval-model1-on-benchmark2-1234567890/
      task-000-subtask3-predictions.jsonl
      ...
```

The tool handles various directory structures and naming patterns:
- Model directories with pattern: `lmeval-[model_name]-on-[benchmark_name]-[hash]`
- Prediction files with task names embedded in the filename: `task-000-[task_name]-predictions.jsonl`
- Deeply nested prediction files in subdirectories

## Usage

Basic usage:
```bash
python performance_analyzer.py /path/to/predictions/directory
```

This will automatically generate a TSV file named `[model_name]-[dataset_name]-results.tsv`. If multiple models are analyzed, the file will be named `combined-[dataset_name]-results.tsv`.

### Using a custom dataset

By default, the program uses the ReasoningIntensiveStrict dataset, but you can specify any HuggingFace dataset:

```bash
python performance_analyzer.py /path/to/predictions/directory --dataset another-dataset/custom-dataset-name
```

### Other options

You can also specify a custom output file:
```bash
python performance_analyzer.py /path/to/predictions/directory --output custom_name.tsv
```

To list all tasks in the dataset:
```bash
python performance_analyzer.py /path/to/predictions/directory --list-dataset-tasks
```

## Command Line Options

```
positional arguments:
  predictions_dir       Directory containing model prediction directories

options:
  -h, --help            Show this help message and exit
  --output OUTPUT       Output file path (defaults to [model_name]-[dataset_name]-results.tsv)
  --dataset DATASET     HuggingFace dataset to use for matching (default: ArpanSarkar/ReasoningIntensiveStrict)
  --verbose             Enable verbose output
  --list-dataset-tasks  List all tasks in the specified dataset and exit
```

## Task Matching

The program uses several strategies to match instances from prediction files with the reference dataset:

1. **Task Name Matching**:
   - Matches using lowercase task names for case-insensitivity
   - Extracts simplified task names by removing benchmark prefixes (e.g., "scieval_physics" → "physics")
   - Tries multiple task name variants from both the prediction file and directory structure

2. **Document ID Matching**:
   - Looks for document IDs in various fields ('doc_id', 'id', 'example_id', 'instance_id')
   - Converts all IDs to strings for consistent matching

3. **Task Extraction from File Paths**:
   - Extracts task names from file paths using regex patterns
   - Checks both filenames and parent directories for task names
   - Falls back to benchmark names if specific task names aren't found

## Performance Calculation

The program uses the following approaches to calculate performance metrics, in order of preference:

1. **Pre-calculated metrics** - Uses these fields if available:
   - `exact_match_flex` - Direct accuracy metric
   - `exact_match_simple` - Alternative exact match metric 
   - `correctness` - Binary correctness indicator (0 or 1)
   - Metrics within a nested `metrics` dictionary containing any of the above fields

2. **Direct comparison** - Falls back to comparing these fields:
   - `prediction` vs `label`
   - Other variations of prediction/label fields

This approach allows the program to work with a variety of evaluation output formats.

## Data Format

The program is flexible with JSON/JSONL data formats and field names:
- For task names: `task`, `taskname`, or `task_name`
- For predictions: `prediction`, `pred`, `output`, `generated`, `prediction_text`, or `completion`
- For ground truth: `label`, `target`, `ground_truth`, `answer`, `label_text`, or `correct_answer`
- For metrics: `exact_match_flex`, `exact_match_simple`, `correctness`, or nested `metrics` dictionary

## Output

The program generates a TSV (Tab-Separated Values) file containing:
- Model name (extracted from directory name)
- Benchmark name (extracted from directory name)
- Task name (extracted from file path or data)
- Accuracy
- Number of instances

The TSV file also includes summary rows:
- `BENCHMARK_AVERAGE` rows showing weighted average accuracy for each benchmark
- `MODEL_AVERAGE` rows showing weighted average accuracy across all benchmarks for each model

Output file naming:
- If analyzing one model: `[model_name]-[dataset_name]-results.tsv`
- If analyzing multiple models: `combined-[dataset_name]-results.tsv`
- Can be overridden with the `--output` parameter

The program also prints overall statistics by model and by model-benchmark combination.

## Troubleshooting

If no matching instances are found:
1. Use the `--list-dataset-tasks` flag to see what tasks are available in the dataset
2. Check if your prediction files have task names that match those in the dataset
3. Verify that your prediction files contain document IDs that match the dataset
4. Use the `--verbose` flag for additional debugging information

## Example

If you have a directory structure like:
```
eval_results/
  qwen-insturct-synthetic_1-sft-sciriff-grpo/
    lmeval-qwen-insturct-synthetic_1-sft-sciriff-grpo-on-gpqa-3a4ccb006b/
      task-000-gpqa-predictions.jsonl
  gpt4-results/
    lmeval-gpt-4-on-gsm8k-0987654321/
      task-000-gsm8k-predictions.jsonl
```

Run with default dataset:
```bash
python performance_analyzer.py eval_results
```

This will generate output files:
- `qwen-insturct-synthetic_1-sft-sciriff-grpo-ReasoningIntensiveStrict-results.tsv`
- `gpt-4-ReasoningIntensiveStrict-results.tsv`

Or with a custom dataset:
```bash
python performance_analyzer.py eval_results --dataset allenai/rainbow
```

This will generate output files:
- `qwen-insturct-synthetic_1-sft-sciriff-grpo-rainbow-results.tsv`
- `gpt-4-rainbow-results.tsv`

These files will contain:
1. Performance metrics for each task within each benchmark
2. Benchmark averages for each model
3. Overall model averages across all benchmarks 