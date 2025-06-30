#!/usr/bin/env python3
"""
Domain Performance Analyzer

This script extends the TaskPerformanceViewer to analyze model performance
by scientific domains (Physics, Chemistry, Biology, etc.) based on 
subtask categorization.

Usage:
    python domain_performance_analyzer.py <model_directory> [options]
    python domain_performance_analyzer.py --comparison-dir <directory> --comparison-mode [options]
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tabulate import tabulate

# Import the base TaskPerformanceViewer
sys.path.append(str(Path(__file__).parent))
from task_performance_viewer import TaskPerformanceViewer


class DomainPerformanceAnalyzer(TaskPerformanceViewer):
    """Analyzer for domain-wise performance metrics."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Define domain mappings based on the provided tables
        self.domain_mappings = {
            'Physics': {
                'GPQA': ['Physics'],
                'MMLU-Pro': ['physics'],
                'SciBench': ['fund', 'thermo', 'class'],
                'OlympiadBench-COMP': ['physics_en'],
                'SciKnowEval.L5': ['physics_problem_solving'],
                'SciEval': ['physics_knowledge_application', 'physics_scientific_calculation'],
                'UGPhysics': [
                    'Electrodynamics', 'Thermodynamics', 'GeometricalOptics', 'Relativity',
                    'ClassicalElectromagnetism', 'ClassicalMechanics', 'WaveOptics',
                    'QuantumMechanics', 'TheoreticalMechanics', 'AtomicPhysics',
                    'SemiconductorPhysics', 'Solid-StatePhysics', 'StatisticalMechanics'
                ],
                'SuperGPQA': ['Physics']
            },
            'Chemistry': {
                'GPQA': ['Chemistry'],
                'MMLU-Pro': ['chemistry'],
                'SciBench': ['quan', 'chemc', 'atkins', 'matter'],
                'SciKnowEval.L5': ['chemical_procedure_generation', 'chemical_reagent_generation'],
                'SciEval': ['chemistry_knowledge_application', 'chemistry_scientific_calculation'],
                'SuperGPQA': ['Chemistry']
            },
            'Computer Science': {
                'MMLU-Pro': ['computer science'],
                'SciRIFF-ReasEval': ['Qasper'],
                'SuperGPQA': ['Computer Science and Technology']
            },
            'Math': {
                'MMLU-Pro': ['math'],
                'SciBench': ['calc', 'stat', 'diff'],
                'OlympiadBench-COMP': ['maths_en'],
                'SuperGPQA': ['Mathematics']
            },
            'Biology': {
                'GPQA': ['Biology'],
                'MMLU-Pro': ['biology'],
                'LabBench': ['CloningScenarios', 'ProtocolQA', 'SeqQA'],
                'SciKnowEval.L5': ['biological_procedure_generation', 'biological_reagent_generation'],
                'SciEval': ['biology_knowledge_application', 'biology_scientific_calculation'],
                'SuperGPQA': ['Biology']
            },
            'Medicine': {
                'MMLU-Pro': ['health'],
                'SciRIFF-ReasEval': ['SciFact', 'Evidence Inference']
            },
            'Material Science': {
                'SciKnowEval.L5': [
                    'crystal_structure_and_composition',
                    'specified_band_gap_material_generation',
                    'property_and_usage_analysis'
                ],
                'SuperGPQA': ['Materials Science and Engineering']
            },
            'Engineering': {
                'MMLU-Pro': ['engineering'],
                'SuperGPQA': [
                    'Control Science and Engineering',
                    'Information and Communication Engineering',
                    'Electrical Engineering',
                    'Chemical Engineering and Technology',
                    'Power Engineering and Engineering Thermophysics',
                    'Electronic Science and Technology',
                    'Hydraulic Engineering',
                    'Mechanics',
                    'Mechanical Engineering',
                    'Civil Engineering',
                    'Optical Engineering',
                    'Nuclear Science and Technology',
                    'Instrument Science and Technology',
                    'Systems Science'
                ]
            }
        }
    
    def classify_task_by_domain(self, task_name: str, benchmark_name: str) -> Optional[str]:
        """
        Classify a task into a domain based on task name and benchmark.
        
        Args:
            task_name (str): Name of the task
            benchmark_name (str): Name of the benchmark
            
        Returns:
            Optional[str]: Domain name if classified, None otherwise
        """
        # Normalize names for matching
        task_lower = task_name.lower()
        benchmark_lower = benchmark_name.lower()
        
        # Check each domain's mappings
        for domain, benchmark_mappings in self.domain_mappings.items():
            for bench_pattern, subtasks in benchmark_mappings.items():
                # Check if benchmark name matches
                if bench_pattern.lower() in benchmark_lower:
                    # Check if task matches any subtask
                    for subtask in subtasks:
                        if subtask.lower() in task_lower:
                            return domain
                    
                    # If benchmark matches but no specific subtask, check for general domain keywords
                    domain_keywords = [domain.lower(), domain.lower().replace(' ', '_')]
                    if any(keyword in task_lower for keyword in domain_keywords):
                        return domain
        
        # Fallback: check for domain keywords in task name
        for domain in self.domain_mappings.keys():
            domain_keywords = [domain.lower(), domain.lower().replace(' ', '_')]
            if any(keyword in task_lower for keyword in domain_keywords):
                return domain
        
        return None
    
    def aggregate_by_domain(self, all_metrics: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Aggregate metrics by domain.
        
        Args:
            all_metrics (Dict): Metrics data from load_all_metrics()
            
        Returns:
            Dict[str, Dict]: Domain-wise aggregated metrics
        """
        domain_data = {}
        
        # Initialize domain data structure
        for domain in self.domain_mappings.keys():
            domain_data[domain] = {
                'tasks': [],
                'total_instances': 0,
                'scores': [],
                'exact_matches': [],
                'total_cost': 0,
                'benchmarks': set()
            }
        
        # Classify and aggregate tasks
        unclassified_tasks = []
        
        for benchmark_name, metrics_list in all_metrics.items():
            for metric in metrics_list:
                # Skip aggregates for individual task classification
                if metric.get('is_aggregate', False):
                    continue
                
                task_name = metric['task_name']
                domain = self.classify_task_by_domain(task_name, benchmark_name)
                
                if domain:
                    domain_data[domain]['tasks'].append(metric)
                    domain_data[domain]['total_instances'] += metric.get('num_instances', 0)
                    domain_data[domain]['benchmarks'].add(benchmark_name)
                    
                    # Collect scores
                    if metric.get('primary_score') is not None:
                        domain_data[domain]['scores'].append(metric['primary_score'])
                    if metric.get('exact_match') is not None:
                        domain_data[domain]['exact_matches'].append(metric['exact_match'])
                    
                    # Aggregate cost
                    cost = metric.get('total_price') or metric.get('calculated_total_cost')
                    if cost:
                        domain_data[domain]['total_cost'] += cost
                else:
                    unclassified_tasks.append((benchmark_name, task_name))
        
        # Calculate domain averages
        for domain, data in domain_data.items():
            if data['scores']:
                data['avg_primary_score'] = sum(data['scores']) / len(data['scores'])
                data['weighted_primary_score'] = self._calculate_weighted_average(data['tasks'], 'primary_score')
            else:
                data['avg_primary_score'] = None
                data['weighted_primary_score'] = None
            
            if data['exact_matches']:
                data['avg_exact_match'] = sum(data['exact_matches']) / len(data['exact_matches'])
                data['weighted_exact_match'] = self._calculate_weighted_average(data['tasks'], 'exact_match')
            else:
                data['avg_exact_match'] = None
                data['weighted_exact_match'] = None
            
            data['num_tasks'] = len(data['tasks'])
            data['num_benchmarks'] = len(data['benchmarks'])
        
        # Print unclassified tasks for debugging
        if unclassified_tasks:
            print(f"\nWarning: {len(unclassified_tasks)} tasks could not be classified into domains:")
            for bench, task in unclassified_tasks[:10]:  # Show first 10
                print(f"  {bench}: {task}")
            if len(unclassified_tasks) > 10:
                print(f"  ... and {len(unclassified_tasks) - 10} more")
        
        return domain_data
    
    def _calculate_weighted_average(self, tasks: List[Dict], metric_field: str) -> Optional[float]:
        """Calculate weighted average by number of instances."""
        total_weighted_sum = 0
        total_instances = 0
        
        for task in tasks:
            metric_value = task.get(metric_field)
            instances = task.get('num_instances', 0)
            
            if metric_value is not None and instances > 0:
                total_weighted_sum += metric_value * instances
                total_instances += instances
        
        return total_weighted_sum / total_instances if total_instances > 0 else None
    
    def display_domain_summary(self, domain_data: Dict[str, Dict], format_type: str = "table"):
        """
        Display domain-wise performance summary.
        
        Args:
            domain_data (Dict): Domain aggregated data from aggregate_by_domain()
            format_type (str): Output format
        """
        print(f"\n{'='*100}")
        print(f"DOMAIN-WISE PERFORMANCE SUMMARY - {self.model_name}")
        print(f"{'='*100}")
        
        # Prepare summary data
        summary_data = []
        
        for domain, data in domain_data.items():
            if data['num_tasks'] == 0:
                continue  # Skip domains with no tasks
            
            # Format scores
            avg_score_str = f"{data['avg_primary_score']:.4f}" if data['avg_primary_score'] is not None else "N/A"
            weighted_score_str = f"{data['weighted_primary_score']:.4f}" if data['weighted_primary_score'] is not None else "N/A"
            avg_exact_str = f"{data['avg_exact_match']:.4f}" if data['avg_exact_match'] is not None else "N/A"
            
            # Format cost
            cost_str = f"${data['total_cost']:.4f}" if data['total_cost'] > 0 else "N/A"
            
            summary_data.append({
                'Domain': domain,
                'Tasks': data['num_tasks'],
                'Benchmarks': data['num_benchmarks'],
                'Instances': data['total_instances'],
                'Avg Score': avg_score_str,
                'Weighted Score': weighted_score_str,
                'Avg Exact Match': avg_exact_str,
                'Total Cost': cost_str
            })
        
        # Sort by weighted score (descending)
        summary_data.sort(key=lambda x: float(x['Weighted Score']) if x['Weighted Score'] != "N/A" else 0, reverse=True)
        
        if format_type == "table":
            print(tabulate(summary_data, headers="keys", tablefmt="grid"))
        else:
            for item in summary_data:
                print(f"{item['Domain']:.<20} Weighted: {item['Weighted Score']:>8} | Avg: {item['Avg Score']:>8} "
                      f"({item['Tasks']} tasks, {item['Instances']} instances)")
    
    def display_domain_details(self, domain_data: Dict[str, Dict], domain_name: str):
        """
        Display detailed breakdown for a specific domain.
        
        Args:
            domain_data (Dict): Domain aggregated data
            domain_name (str): Name of domain to detail
        """
        if domain_name not in domain_data:
            print(f"Domain '{domain_name}' not found.")
            return
        
        data = domain_data[domain_name]
        
        print(f"\n{'='*80}")
        print(f"DETAILED BREAKDOWN - {domain_name.upper()}")
        print(f"{'='*80}")
        
        print(f"Summary:")
        print(f"  Total Tasks: {data['num_tasks']}")
        print(f"  Total Instances: {data['total_instances']}")
        print(f"  Benchmarks: {', '.join(sorted(data['benchmarks']))}")
        print(f"  Average Score: {data['avg_primary_score']:.4f}" if data['avg_primary_score'] else "  Average Score: N/A")
        print(f"  Weighted Score: {data['weighted_primary_score']:.4f}" if data['weighted_primary_score'] else "  Weighted Score: N/A")
        
        print(f"\nIndividual Tasks:")
        task_data = []
        for task in sorted(data['tasks'], key=lambda x: x.get('primary_score', 0), reverse=True):
            score_str = f"{task['primary_score']:.4f}" if task['primary_score'] is not None else "N/A"
            exact_str = f"{task['exact_match']:.4f}" if task['exact_match'] is not None else "N/A"
            
            task_data.append({
                'Benchmark': task['benchmark'],
                'Task': task['task_name'],
                'Primary Score': score_str,
                'Exact Match': exact_str,
                'Instances': task['num_instances']
            })
        
        print(tabulate(task_data, headers="keys", tablefmt="grid"))
    
    def create_domain_comparison_table(self, models_data: Dict[str, Dict[str, List[Dict]]]) -> pd.DataFrame:
        """
        Create a domain comparison table across multiple models.
        
        Args:
            models_data (Dict): Multi-model metrics data
            
        Returns:
            pd.DataFrame: Domain comparison table
        """
        # Aggregate domain data for each model
        model_domain_data = {}
        
        for model_name, model_metrics in models_data.items():
            # Create a temporary analyzer for this model
            temp_analyzer = DomainPerformanceAnalyzer()
            temp_analyzer.model_name = model_name
            temp_analyzer.domain_mappings = self.domain_mappings
            
            domain_data = temp_analyzer.aggregate_by_domain(model_metrics)
            model_domain_data[model_name] = domain_data
        
        # Create comparison matrix
        comparison_data = {}
        all_domains = set()
        
        # Collect all domains
        for model_data in model_domain_data.values():
            all_domains.update(model_data.keys())
        
        # Build comparison data
        for model_name, domain_data in model_domain_data.items():
            model_row = {}
            
            for domain in sorted(all_domains):
                if domain in domain_data and domain_data[domain]['num_tasks'] > 0:
                    # Use weighted score as primary metric
                    score = domain_data[domain]['weighted_primary_score']
                    model_row[domain] = score
                else:
                    model_row[domain] = None
            
            comparison_data[model_name] = model_row
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(comparison_data, orient='index')
        
        # Add overall average (excluding domains with no data)
        avg_scores = []
        for model in df.index:
            model_scores = [score for score in df.loc[model] if score is not None]
            avg_score = sum(model_scores) / len(model_scores) if model_scores else None
            avg_scores.append(avg_score)
        
        df['OVERALL_AVERAGE'] = avg_scores
        
        # Sort columns by domain name, with overall average first
        cols = ['OVERALL_AVERAGE'] + sorted([col for col in df.columns if col != 'OVERALL_AVERAGE'])
        df = df[cols]
        
        return df
    
    def display_domain_comparison(self, models_data: Dict[str, Dict[str, List[Dict]]], 
                                format_type: str = "table", precision: int = 4):
        """
        Display domain comparison across multiple models.
        
        Args:
            models_data (Dict): Multi-model metrics data
            format_type (str): Output format
            precision (int): Number of decimal places
        """
        df = self.create_domain_comparison_table(models_data)
        
        if df.empty:
            print("No domain data available for comparison.")
            return
        
        print(f"\n{'='*120}")
        print(f"DOMAIN-WISE MODEL COMPARISON")
        print(f"(Scores are weighted by number of instances)")
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
        
        # Display rankings
        print(f"\n{'='*60}")
        print("DOMAIN RANKINGS")
        print(f"{'='*60}")
        
        # Overall ranking
        overall_scores = [(model, df.loc[model, 'OVERALL_AVERAGE']) 
                         for model in df.index if pd.notna(df.loc[model, 'OVERALL_AVERAGE'])]
        overall_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("\nOVERALL RANKING:")
        for i, (model, score) in enumerate(overall_scores, 1):
            print(f"{i:2d}. {model:.<30} {score:.{precision}f}")
        
        # Domain-specific rankings
        for domain in df.columns:
            if domain == 'OVERALL_AVERAGE':
                continue
                
            domain_scores = [(model, df.loc[model, domain]) 
                           for model in df.index if pd.notna(df.loc[model, domain])]
            
            if len(domain_scores) >= 2:  # Only show if at least 2 models have data
                domain_scores.sort(key=lambda x: x[1], reverse=True)
                
                print(f"\n{domain.upper()}:")
                for i, (model, score) in enumerate(domain_scores, 1):
                    print(f"{i:2d}. {model:.<25} {score:.{precision}f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze model performance by scientific domains",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('model_dir', nargs='?', help='Path to model directory containing lmeval results')
    parser.add_argument('--benchmark', help='Filter to specific benchmark')
    parser.add_argument('--format', choices=['simple', 'table'], default='table', help='Output format')
    parser.add_argument('--domain', help='Show detailed breakdown for specific domain')
    
    # Comparison mode arguments
    parser.add_argument('--comparison-dir', help='Path to directory containing model subdirectories')
    parser.add_argument('--comparison-mode', action='store_true', help='Show multi-model domain comparison')
    
    # Export arguments
    parser.add_argument('--export', help='Export domain summary to TSV file')
    parser.add_argument('--export-comparison', help='Export domain comparison to TSV file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.comparison_mode and not args.comparison_dir:
        parser.error("--comparison-mode requires --comparison-dir")
    
    if not args.comparison_mode and not args.model_dir:
        parser.error("model_dir is required unless using --comparison-mode")
    
    try:
        analyzer = DomainPerformanceAnalyzer(args.model_dir)
        
        if args.comparison_mode:
            # Multi-model comparison
            models_data = analyzer.load_multi_model_metrics(args.comparison_dir, args.benchmark)
            
            if not models_data:
                print(f"No models with metrics found in {args.comparison_dir}")
                return 1
            
            # For now, just do single model analysis on first model
            print("Multi-model domain comparison not fully implemented yet.")
            print("Showing domain analysis for first model...")
            
            first_model = list(models_data.keys())[0]
            analyzer.model_name = first_model
            all_metrics = models_data[first_model]
            
            domain_data = analyzer.aggregate_by_domain(all_metrics)
            analyzer.display_domain_summary(domain_data, args.format)
        
        else:
            # Single model analysis
            all_metrics = analyzer.load_all_metrics(args.benchmark)
            
            if not all_metrics:
                print(f"No metrics found in {args.model_dir}")
                return 1
            
            # Aggregate by domain
            domain_data = analyzer.aggregate_by_domain(all_metrics)
            
            # Display summary
            analyzer.display_domain_summary(domain_data, args.format)
            
            # Show detailed domain breakdown if requested
            if args.domain:
                analyzer.display_domain_details(domain_data, args.domain)
            
            # Export if requested
            if args.export:
                # Convert domain data to DataFrame for export
                export_data = []
                for domain, data in domain_data.items():
                    if data['num_tasks'] > 0:
                        export_data.append({
                            'Domain': domain,
                            'Num_Tasks': data['num_tasks'],
                            'Num_Benchmarks': data['num_benchmarks'],
                            'Total_Instances': data['total_instances'],
                            'Avg_Primary_Score': data['avg_primary_score'],
                            'Weighted_Primary_Score': data['weighted_primary_score'],
                            'Avg_Exact_Match': data['avg_exact_match'],
                            'Weighted_Exact_Match': data['weighted_exact_match'],
                            'Total_Cost': data['total_cost']
                        })
                
                df = pd.DataFrame(export_data)
                df.to_csv(args.export, sep='\t', index=False)
                print(f"Exported domain summary to {args.export}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 