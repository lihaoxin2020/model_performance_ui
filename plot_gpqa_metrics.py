#!/usr/bin/env python3
import os
import json
import re
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

def parse_directory_name(dir_name):
    """Parse model information from directory name following the pattern in performance_analyzer.py"""
    pattern = r'lmeval-(.+)-on-(.+)-[a-f0-9]{10}'
    match = re.match(pattern, dir_name)
    
    if match:
        model_name = match.group(1)
        benchmark_name = match.group(2)
        return model_name, benchmark_name
    
    # Fallback if pattern doesn't match
    parts = dir_name.split('-')
    if len(parts) >= 4 and parts[0] == 'lmeval' and 'on' in parts:
        on_index = parts.index('on')
        model_name = '-'.join(parts[1:on_index])
        benchmark_name = '-'.join(parts[on_index+1:-1]) if len(parts) > on_index+2 else parts[on_index+1]
    else:
        model_name = dir_name
        benchmark_name = "unknown"
        
    return model_name, benchmark_name

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot maj_at_k metrics from GPQA evaluations')
    parser.add_argument('--exclude-k', type=int, nargs='+', help='K values to exclude from the plot', default=[])
    args = parser.parse_args()
    
    # K values to exclude
    exclude_k_values = args.exclude_k
    print(f"Excluding k values: {exclude_k_values}")
    
    # Path to output directories
    base_dir = "/home/jovyan/workspace/lmeval/qwen-gpqa_diamond-majAtK"
    
    # Get all directories
    directories = [d for d in glob.glob(f"{base_dir}/*") if os.path.isdir(d)]
    
    print(f"Found {len(directories)} directories to process")
    
    # Dictionary to store metrics by model and maj_at_k value
    metrics_data = {}
    
    # Collect metrics from each directory
    for dir_path in directories:
        dir_name = os.path.basename(dir_path)
        model_name, benchmark_name = parse_directory_name(dir_name)
        
        metrics_file = os.path.join(dir_path, "metrics.json")
        if not os.path.exists(metrics_file):
            print(f"Warning: metrics.json not found in {dir_path}")
            continue
        
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                
            # Extract maj_at_k_exact_match_flex metrics
            flex_metrics = {}
            for metric in metrics.get('metrics', []):
                for k, v in metric.items():
                    if k.startswith('maj_at_') and k.endswith('_exact_match_flex'):
                        # Extract k value from the metric name
                        k_value = int(k.split('_')[2])
                        # Skip excluded k values
                        if k_value not in exclude_k_values:
                            flex_metrics[k_value] = v
            
            if flex_metrics:
                metrics_data[model_name] = flex_metrics
                print(f"Extracted metrics for {model_name}")
            else:
                print(f"No maj_at_k_exact_match_flex metrics found in {dir_path}")
        
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error processing {dir_path}: {e}")
    
    if not metrics_data:
        print("No metrics data found. Exiting.")
        return
    
    # Plot the metrics
    plt.figure(figsize=(12, 8))
    
    # Create x-axis for the different k values
    k_values = sorted(set(k for model_metrics in metrics_data.values() for k in model_metrics.keys()))
    
    # Colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_data)))
    
    # Plot each model's metrics
    for i, (model_name, model_metrics) in enumerate(metrics_data.items()):
        # Sort by k value for plotting
        x = sorted(model_metrics.keys())
        y = [model_metrics[k] for k in x]
        
        # Create a shorter display name for the legend
        display_name = model_name.replace('qwen-instruct-', 'qwen-')
        display_name = display_name.replace('synthetic_1_qwen_math_only-sft', 'math-sft')
        display_name = display_name.replace('synthetic_1-sft-sciriff-r1-distill-filtered-sft', 'synth1-sciriff-r1')
        display_name = display_name.replace('insturct-synthetic_1-sft-sciriff-grpo', 'synth1-sciriff-grpo')
        display_name = display_name.replace('insturct-synthetic_1-sft', 'synth1-sft')
        display_name = display_name.replace('Qwen2.5-7B-Instruct', 'Qwen2.5-7B')
        
        plt.plot(x, y, 'o-', label=display_name, color=colors[i], linewidth=2, markersize=8)
    
    # Set labels and title
    plt.xlabel('Majority at K', fontsize=14)
    plt.ylabel('Exact Match Flex Score', fontsize=14)
    title = 'GPQA Performance: maj_at_k_exact_match_flex by Model'
    if exclude_k_values:
        title += f' (Excluded K: {", ".join(map(str, exclude_k_values))})'
    plt.title(title, fontsize=16)
    
    # Set x-ticks to only the k values we have
    plt.xticks(k_values)
    
    # Set y-axis limits with a bit of padding
    all_values = [v for model_metrics in metrics_data.values() for v in model_metrics.values()]
    min_value = max(0, min(all_values) - 0.05)
    max_value = min(1, max(all_values) + 0.05)
    plt.ylim(min_value, max_value)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Add value labels on the points
    for model_name, model_metrics in metrics_data.items():
        for k, v in model_metrics.items():
            plt.text(k, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    
    # Tight layout
    plt.tight_layout()
    
    # Generate filename with excluded k values
    suffix = ""
    if exclude_k_values:
        suffix = f"_exclude_{'-'.join(map(str, sorted(exclude_k_values)))}"
    
    # Save the figure
    output_path = os.path.join("/home/jovyan/workspace/model_performance_ui", f"gpqa_maj_at_k_performance{suffix}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Save data as CSV for further analysis
    df = pd.DataFrame({model: {f"maj_at_{k}": v for k, v in metrics.items()} 
                      for model, metrics in metrics_data.items()})
    df = df.transpose()
    csv_path = os.path.join("/home/jovyan/workspace/model_performance_ui", f"gpqa_maj_at_k_performance{suffix}.csv")
    df.to_csv(csv_path)
    print(f"Data saved to {csv_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main() 