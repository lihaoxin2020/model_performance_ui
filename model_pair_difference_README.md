# Model Pair Difference Analyzer

This script compares correctness differences between pairs of models and measures the alignment between different difference sets.

## Overview

The analyzer extracts instances where one model in a pair answers correctly while the other answers incorrectly, then compares how similar these "difference sets" are between different model pairs.

For example:
- Extract instances where `o3-mini-high` is correct and `o3-mini-low` is incorrect
- Extract instances where `o4-mini-high` is correct and `o4-mini-low` is incorrect  
- Compare how similar these two difference sets are

## Features

- **Difference Set Extraction**: Identifies instances where high-performance model succeeds but low-performance model fails
- **Alignment Metrics**: Calculates Jaccard similarity, overlap coefficient, precision, recall, and F1 score
- **Task-level Breakdown**: Provides per-task analysis of difference set alignment
- **Multiple Correctness Metrics**: Supports exact_match_flex, exact_match_simple, correctness, and prediction/label comparison

## Installation & Dependencies

The script uses components from `performance_analyzer.py` and requires:
- pandas
- numpy
- datasets (for HuggingFace dataset loading)
- tqdm

## Usage

### Command Line Interface

```bash
# Basic usage comparing o3-mini and o4-mini pairs
python model_pair_difference_analyzer.py /home/haoxinl/lmeval-api \
    --pair1 o3-mini-high o3-mini-low \
    --pair2 o4-mini-high o4-mini-low

# With custom dataset and output file
python model_pair_difference_analyzer.py /path/to/lmeval-api \
    --pair1 model1-high model1-low \
    --pair2 model2-high model2-low \
    --dataset ArpanSarkar/ReasoningIntensiveLoose_with_SuperGPQA \
    --output comparison_results.json
```

### Arguments

- `lmeval_dir`: Path to lmeval-api directory containing model directories
- `--pair1`: First model pair (high-performance model, low-performance model)
- `--pair2`: Second model pair (high-performance model, low-performance model)
- `--dataset`: HuggingFace dataset for reference (default: ArpanSarkar/ReasoningIntensiveStrict)
- `--output`: Optional output file to save results in JSON format
- `--verbose`: Enable verbose output

### Test Script

Run the test script to see the analyzer in action:

```bash
python test_pair_analyzer.py
```

## Example Output

```
MODEL PAIR COMPARISON ANALYSIS
================================================================================

Dataset: ArpanSarkar/ReasoningIntensiveStrict
Pair 1: o3-mini-high vs o3-mini-low
Pair 2: o4-mini-high vs o4-mini-low

==================================================
AGREEMENT ANALYSIS (Pair 1 as Ground Truth)
==================================================
  Total common instances: 2847
  Pair 1 difficulty rate: 0.312 (888/2847)
  Pair 2 difficulty rate: 0.298 (849/2847)

  Agreement Metrics:
    Accuracy: 0.756
    Precision: 0.623
    Recall: 0.578
    F1 score: 0.600
    Specificity: 0.834

  Confusion Matrix:
    True Positives:   513 (both pairs show difference)
    False Positives:  336 (only pair 2 shows difference)
    False Negatives:  375 (only pair 1 shows difference)
    True Negatives:  1623 (neither pair shows difference)

==================================================
DIFFERENCE SET OVERLAP ANALYSIS
==================================================
  Difference set 1 size: 888
  Difference set 2 size: 849
  Intersection size: 513
  Union size: 1224
  Jaccard similarity: 0.419
  Overlap coefficient: 0.604
  Precision: 0.604
  Recall: 0.578
  F1 score: 0.591
```

## Metrics Explained

### Agreement Analysis (treating Pair 1 as ground truth):
- **Accuracy**: How often pair 2 agrees with pair 1 about whether an instance shows a difficulty difference
- **Precision**: When pair 2 says there's a difference, how often is pair 1 correct?
- **Recall**: When pair 1 says there's a difference, how often does pair 2 agree?
- **Specificity**: When pair 1 says there's no difference, how often does pair 2 agree?
- **F1 Score**: Harmonic mean of precision and recall

### Confusion Matrix:
- **True Positives**: Both pairs show difference (high correct, low incorrect)
- **False Positives**: Only pair 2 shows difference 
- **False Negatives**: Only pair 1 shows difference
- **True Negatives**: Neither pair shows difference

### Difference Set Overlap Analysis:
- **Jaccard Similarity**: |intersection| / |union| - measures overall similarity
- **Overlap Coefficient**: |intersection| / min(|set1|, |set2|) - measures how much the smaller set overlaps
- **Precision**: |intersection| / |set2| - treating set2 as predictions
- **Recall**: |intersection| / |set1| - treating set1 as ground truth
- **F1 Score**: Harmonic mean of precision and recall

## Directory Structure Expected

The script expects the following directory structure:

```
/home/haoxinl/lmeval-api/
├── o3-mini-high/
│   ├── lmeval-o3-mini-on-gpqa-{hash}/
│   │   ├── task-000-gpqa-predictions.jsonl
│   │   └── ...
│   └── lmeval-o3-mini-on-scibench-{hash}/
│       ├── task-000-scibench-predictions.jsonl
│       └── ...
├── o3-mini-low/
│   └── ...
├── o4-mini-high/
│   └── ...
└── o4-mini-low/
    └── ...
```

## How It Works

1. **Load Predictions**: Loads prediction files for all four models
2. **Filter Instances**: Uses reference dataset to filter to matching instances
3. **Extract Correctness**: Determines correctness using available metrics
4. **Build Difference Sets**: For each pair, finds instances where high model is correct and low model is incorrect
5. **Calculate Alignment**: Compares the two difference sets using multiple similarity metrics
6. **Task Breakdown**: Provides per-task analysis of the alignment

## Use Cases

- **Model Comparison**: Understanding which instances cause systematic differences between model pairs
- **Error Analysis**: Identifying common failure patterns across different model families
- **Benchmark Analysis**: Analyzing whether different model pairs struggle with similar types of questions
- **Model Development**: Understanding whether improvements in one model family generalize to others 