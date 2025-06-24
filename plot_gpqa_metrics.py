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

# Set matplotlib style for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'serif'],
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'axes.linewidth': 1,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2,
    'lines.markersize': 6
})

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
    parser.add_argument('--base-dir', type=str, default="/home/haoxinl/lmeval/maj_at_k-gpqa_pro", 
                       help='Base directory containing evaluation results (default: /home/haoxinl/lmeval/maj_at_k-gpqa_pro)')
    args = parser.parse_args()
    
    # K values to exclude
    exclude_k_values = args.exclude_k
    print(f"Excluding k values: {exclude_k_values}")
    
    # Path to output directories
    base_dir = args.base_dir
    print(f"Using base directory: {base_dir}")
    
    # Verify base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Base directory '{base_dir}' does not exist.")
        return
    
    # Get all directories
    directories = [d for d in glob.glob(f"{base_dir}/*") if os.path.isdir(d)]
    if directories[0].endswith('jsonl'):
        directories = [base_dir]
    
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
                # Create a combined key with both model and task name
                combined_key = f"{model_name} ({benchmark_name})"
                metrics_data[combined_key] = flex_metrics
                print(f"Extracted metrics for {model_name} on {benchmark_name}")
            else:
                print(f"No maj_at_k_exact_match_flex metrics found in {dir_path}")
        
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error processing {dir_path}: {e}")
    
    if not metrics_data:
        print("No metrics data found. Exiting.")
        return
    
    # Create figure with academic proportions
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Create x-axis for the different k values
    k_values = sorted(set(k for model_metrics in metrics_data.values() for k in model_metrics.keys()))
    
    # Professional color palette suitable for academic papers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Check if we have exactly 2 models for dual y-axis
    model_items = list(metrics_data.items())
    if len(model_items) == 2:
        # Dual y-axis plot
        ax2 = ax.twinx()  # Create second y-axis
        axes = [ax, ax2]
        
        # Calculate data ranges for both models first
        model_data = []
        for combined_name, model_metrics in model_items:
            x = sorted(model_metrics.keys())
            y = [model_metrics[k] * 100 for k in x]  # Convert to percentage
            model_data.append((combined_name, x, y))
        
        # Determine common unit scale based on the larger data range
        ranges = []
        for _, _, y in model_data:
            data_range = max(y) - min(y)
            ranges.append(data_range)
        
        # Use the larger range to determine appropriate tick spacing
        max_range = max(ranges)
        if max_range <= 2:
            tick_spacing = 0.5
        elif max_range <= 5:
            tick_spacing = 1
        elif max_range <= 10:
            tick_spacing = 2
        elif max_range <= 20:
            tick_spacing = 5
        else:
            tick_spacing = 10
        
        for i, (combined_name, x, y) in enumerate(model_data):
            # Create a shorter display name for the legend
            display_name = combined_name.replace('qwen-instruct-', 'qwen-')
            display_name = display_name.replace('synthetic_1_qwen_math_only-sft', 'math-sft')
            display_name = display_name.replace('synthetic_1-sft-sciriff-r1-distill-filtered-sft', 'synth1-sciriff-r1')
            display_name = display_name.replace('insturct-synthetic_1-sft-sciriff-grpo', 'synth1-sciriff-grpo')
            display_name = display_name.replace('insturct-synthetic_1-sft', 'synth1-sft')
            display_name = display_name.replace('Qwen2.5-7B-Instruct', 'Qwen2.5-7B')
            
            # Plot on respective axis
            current_ax = axes[i]
            color = colors[i % len(colors)]
            
            current_ax.plot(x, y, 'o-', label=display_name, color=color, 
                          linewidth=2, markersize=6, markerfacecolor='white', 
                          markeredgewidth=1.5, markeredgecolor=color)
            
            # Calculate axis range with same unit scale
            data_min = min(y)
            data_max = max(y)
            data_center = (data_min + data_max) / 2
            
            # Create range that's centered on the data but uses common tick spacing
            # Ensure we have at least 4-6 ticks visible
            desired_ticks = 5
            half_range = (desired_ticks - 1) * tick_spacing / 2
            
            min_value = max(0, data_center - half_range)
            max_value = min(100, data_center + half_range)
            
            # Adjust if we hit the boundaries
            if min_value == 0:
                max_value = min(100, desired_ticks * tick_spacing)
            elif max_value == 100:
                min_value = max(0, 100 - desired_ticks * tick_spacing)
            
            current_ax.set_ylim(min_value, max_value)
            
            # Set consistent tick spacing for both axes
            ticks = np.arange(
                np.ceil(min_value / tick_spacing) * tick_spacing,
                np.floor(max_value / tick_spacing) * tick_spacing + tick_spacing,
                tick_spacing
            )
            current_ax.set_yticks(ticks)
            
            # Color-code the y-axis labels and ticks to match the line
            current_ax.tick_params(axis='y', labelcolor=color)
            if i == 0:
                current_ax.set_ylabel('Accuracy (%)', fontweight='bold', color=color)
            else:
                current_ax.set_ylabel('Accuracy (%)', fontweight='bold', color=color)
        
        # Set common x-axis properties
        ax.set_xlabel('# of Iterations', fontweight='bold')
        ax.set_title('GPQA Performance (Dual Scale)', fontweight='bold', pad=20)
        ax.set_xlim(0, max(k_values) * 1.05)
        
        # Add grid only to primary axis
        ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Customize spines
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color('#333333')
        for spine in ax2.spines.values():
            spine.set_linewidth(1)
            spine.set_color('#333333')
        
        # Create combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax.legend(lines1 + lines2, labels1 + labels2, 
                          frameon=True, fancybox=False, shadow=False, 
                          framealpha=0.9, edgecolor='#333333')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(1)
        
    else:
        # Original single y-axis plot for multiple models
        for i, (combined_name, model_metrics) in enumerate(metrics_data.items()):
            # Sort by k value for plotting
            x = sorted(model_metrics.keys())
            y = [model_metrics[k] * 100 for k in x]  # Convert to percentage
            
            # Create a shorter display name for the legend
            display_name = combined_name.replace('qwen-instruct-', 'qwen-')
            display_name = display_name.replace('synthetic_1_qwen_math_only-sft', 'math-sft')
            display_name = display_name.replace('synthetic_1-sft-sciriff-r1-distill-filtered-sft', 'synth1-sciriff-r1')
            display_name = display_name.replace('insturct-synthetic_1-sft-sciriff-grpo', 'synth1-sciriff-grpo')
            display_name = display_name.replace('insturct-synthetic_1-sft', 'synth1-sft')
            display_name = display_name.replace('Qwen2.5-7B-Instruct', 'Qwen2.5-7B')
            
            # Plot with professional styling
            ax.plot(x, y, 'o-', label=display_name, color=colors[i % len(colors)], 
                   linewidth=2, markersize=6, markerfacecolor='white', 
                   markeredgewidth=1.5, markeredgecolor=colors[i % len(colors)])
        
        # Set labels and formatting
        ax.set_xlabel('# of Iterations', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title('GPQA Performance', fontweight='bold', pad=20)
        
        # Set axis limits and ticks
        ax.set_xlim(0, max(k_values) * 1.05)
        
        # Set y-axis limits with tighter bounds to emphasize trends
        all_values = [v * 100 for model_metrics in metrics_data.values() for v in model_metrics.values()]
        data_min = min(all_values)
        data_max = max(all_values)
        data_range = data_max - data_min
        
        # Use tighter padding to focus on the data range
        padding = max(1, data_range * 0.05)  # 5% of data range or minimum 1%
        min_value = max(0, data_min - padding)
        max_value = min(100, data_max + padding)
        
        # If the range is very small, ensure we have at least some visible range
        if max_value - min_value < 2:
            center = (max_value + min_value) / 2
            min_value = max(0, center - 1)
            max_value = min(100, center + 1)
        
        ax.set_ylim(min_value, max_value)
        
        # Customize grid
        ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Customize spines
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color('#333333')
        
        # Add legend with professional styling
        legend = ax.legend(frameon=True, fancybox=False, shadow=False, 
                          framealpha=0.9, edgecolor='#333333')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Generate filename with excluded k values
    suffix = ""
    if exclude_k_values:
        suffix = f"_exclude_{'-'.join(map(str, sorted(exclude_k_values)))}"
    
    # Save the figure with high quality for academic papers
    output_path = os.path.join("..", f"gpqa_maj_at_k_performance{suffix}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    
    # Also save as PDF for vector graphics (better for academic papers)
    pdf_path = os.path.join("..", f"gpqa_maj_at_k_performance{suffix}.pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='pdf')
    
    print(f"Plot saved to {output_path}")
    print(f"PDF version saved to {pdf_path}")
    
    # Save data as CSV for further analysis
    df = pd.DataFrame({model: {f"maj_at_{k}": v for k, v in metrics.items()} 
                      for model, metrics in metrics_data.items()})
    df = df.transpose()
    csv_path = os.path.join("..", f"gpqa_maj_at_k_performance{suffix}.csv")
    df.to_csv(csv_path)
    print(f"Data saved to {csv_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main() 