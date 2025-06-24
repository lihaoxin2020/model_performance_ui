# Task Performance Viewer

A script to display performance metrics for each task/subtask from lmeval evaluation results. It reads `metrics-all.jsonl` files from benchmark subdirectories and presents them in a clear, organized format.

## Features

- **Benchmark Discovery**: Automatically finds all benchmark directories within a model directory
- **Flexible Filtering**: Filter results by benchmark name
- **Multiple Display Formats**: Choose from simple, table, or detailed output formats
- **Sorting Options**: Sort tasks by name, primary score, exact match, or number of instances
- **Summary Views**: Show just benchmark summaries or detailed task breakdowns
- **CSV Export**: Export results to CSV for further analysis
- **Rich Metrics**: Display primary scores, exact match rates, processing time, token usage, and costs
- **Custom Aggregation**: Special handling for sciriff and supergpqa benchmarks with custom scoring

## Custom Benchmark Aggregation

### Sciriff Benchmark
The script automatically creates a custom aggregate for sciriff benchmarks by averaging 5 specific metrics:
- `f1_overlap` from sciriff_evidence_inference
- `llm_score` and `f1_evidence_all` from sciriff_qasper_abstractive_qa  
- `f1_label` and `f1_evidence_token` from sciriff_scifact_entailment

The primary score for sciriff is calculated as: `(f1_overlap + llm_score + f1_evidence_all + f1_label + f1_evidence_token) / 5`

### Supergpqa Benchmark
When both supergpqa_Engineering-cot and supergpqa_Science-cot are present, they are automatically combined into a single "supergpqa" benchmark with the primary score calculated as the average of the two component benchmarks.

## Requirements

Install the required dependencies:

```bash
pip install pandas tabulate
```

## Usage

### Basic Usage

```bash
# View all benchmarks for a model
python task_performance_viewer.py lmeval/Qwen3-32B

# Filter to a specific benchmark
python task_performance_viewer.py lmeval/deepeek-r1 --benchmark mmlu_pro

# Show only benchmark summaries
python task_performance_viewer.py lmeval/Qwen3-32B --summary-only
```

### Display Formats

```bash
# Table format (default)
python task_performance_viewer.py lmeval/model_name --format table

# Simple text format
python task_performance_viewer.py lmeval/model_name --format simple

# Detailed format with all metrics
python task_performance_viewer.py lmeval/model_name --format detailed
python task_performance_viewer.py lmeval/model_name --detailed  # shorthand
```

### Sorting and Filtering

```bash
# Sort by primary score (highest first)
python task_performance_viewer.py lmeval/model_name --sort-by primary_score

# Sort by number of instances
python task_performance_viewer.py lmeval/model_name --sort-by num_instances

# Filter to specific benchmarks
python task_performance_viewer.py lmeval/model_name --benchmark mmlu
python task_performance_viewer.py lmeval/model_name --benchmark science
```

### Export Options

```bash
# Export all data to CSV
python task_performance_viewer.py lmeval/model_name --export results.csv

# Export with filtering
python task_performance_viewer.py lmeval/model_name --benchmark mmlu --export mmlu_results.csv
```

## Example Output

### Summary View
```
================================================================================
PERFORMANCE SUMMARY - deepeek-r1
================================================================================
+----------------------------+-----------------+---------------+---------+-------------------+
| Benchmark                  |   Primary Score | Exact Match   |   Tasks |   Total Instances |
+============================+=================+===============+=========+===================+
| mmlu_pro-0shot_cot-scillm  |          0.8686 | 0.8686        |       7 |              1400 |
| sciriff                    |          0.5221 | N/A           |       3 |               541 |
| supergpqa                  |          0.5706 | 0.0027        |       2 |              6167 |
+----------------------------+-----------------+---------------+---------+-------------------+
| ─────────── AVERAGE ────────|          0.6330 | N/A           |      47 |             16368 |
+----------------------------+-----------------+---------------+---------+-------------------+
```

**Note**: The summary view automatically includes an average score row at the bottom, showing the simple (unweighted) average of all benchmark primary scores, along with total task and instance counts. **Benchmarks containing "knowledge" in their name are excluded from the average calculation** but still displayed in the table.

### Example with Knowledge Task Exclusion
```
+---------------------------------------------------------------------+-----------------+
| Benchmark                                                           |   Primary Score |
+=====================================================================+=================+
| gpqa                                                                |          0.3839 |
| gpqa_knowledge                                                      |          0.5402 |  # excluded
| mmlu_pro-0shot_cot-scillm                                           |          0.5686 |
| mmlu_pro-cot_knowledge-scillm                                       |          0.6814 |  # excluded
| olympiadbench-scillm                                                |          0.1824 |
| ──────────────────── AVERAGE (excl. knowledge) ──────────────────── |          0.3528 |
+---------------------------------------------------------------------+-----------------+
```
In this example, `gpqa_knowledge` (0.5402) and `mmlu_pro-cot_knowledge-scillm` (0.6814) are displayed but excluded from the average calculation, resulting in a lower average score of 0.3528 instead of 0.3958 if knowledge tasks were included.

### Detailed Task View with Custom Aggregates
```
Task: sciriff (5-metric average)
  Benchmark: sciriff
  Task Core: sciriff_aggregate
  Instances: 541
  Primary Score: 0.5221
  Exact Match: N/A
  Time per Instance: 4.10s

Task: supergpqa (Engineering + Science average)
  Benchmark: supergpqa
  Task Core: supergpqa_aggregate
  Instances: 6167
  Primary Score: 0.5706
  Exact Match: 0.0027
  Time per Instance: 13.42s
```

## Directory Structure

The script expects the following directory structure:

```
lmeval/
├── model_name/
│   ├── lmeval-model-on-benchmark1-hash/
│   │   ├── metrics-all.jsonl        # Main metrics file
│   │   ├── metrics.json
│   │   └── task-*-predictions.jsonl
│   ├── lmeval-model-on-benchmark2-hash/
│   │   ├── metrics-all.jsonl
│   │   └── ...
│   └── ...
```

## Metrics Extracted

- **Primary Score**: The main performance metric for each task (with custom calculation for sciriff)
- **Exact Match**: Exact match accuracy (when available)
- **Task Information**: Task name, core task, number of instances
- **Performance Metrics**: Processing time per instance, token usage
- **Cost Information**: Total evaluation cost (when available)
- **Format Compliance**: Answer format correctness rates
- **Special Metrics**: Individual sciriff metrics (f1_overlap, llm_score, etc.) for detailed analysis

## Options Reference

| Option | Description |
|--------|-------------|
| `model_dir` | Path to model directory containing lmeval results |
| `--benchmark` | Filter to specific benchmark (substring match) |
| `--format` | Output format: `simple`, `table`, `detailed` |
| `--sort-by` | Sort column: `task_name`, `primary_score`, `exact_match`, `num_instances` |
| `--detailed` | Show detailed task view with all metrics |
| `--show-aggregate` | Include aggregate metrics in detailed view |
| `--export` | Export results to CSV file |
| `--summary-only` | Show only benchmark summary, not individual tasks |

## Integration with Performance Analyzer

This script complements the existing `performance_analyzer.py` by:

- Focusing on individual task performance rather than aggregate analysis
- Reading directly from `metrics-all.jsonl` files for the most accurate metrics
- Providing flexible display options for different use cases
- Supporting rapid exploration of model performance across multiple benchmarks
- **Custom aggregation for sciriff and supergpqa** for standardized benchmark reporting

Use `performance_analyzer.py` for detailed cross-model comparisons and mathematical analysis, and use `task_performance_viewer.py` for quick task-level performance inspection with standardized benchmark aggregation. 