#!/usr/bin/env python3
"""
Test script for Model Pair Difference Analyzer

This script demonstrates how to use the analyzer to compare model pairs
as specified in the user's request.
"""

import sys
import os
from pathlib import Path

# Add the current directory to path so we can import the analyzer
sys.path.append(str(Path(__file__).parent))

from model_pair_difference_analyzer import ModelPairDifferenceAnalyzer

def test_o3_o4_comparison():
    """Test the specific comparison requested: o3-mini pairs vs o4-mini pairs."""
    
    # Path to the lmeval-api directory
    lmeval_dir = "/home/haoxinl/lmeval-api"
    
    # Initialize the analyzer
    print("Initializing Model Pair Difference Analyzer...")
    analyzer = ModelPairDifferenceAnalyzer(lmeval_dir, dataset_name=None)
    
    # Define the model pairs
    pair1 = ("o3-mini-high", "o3-mini-low")
    pair2 = ("o4-mini-high", "o4-mini-low")
    
    # Run the comparison
    print(f"\nComparing model pairs:")
    print(f"  Pair 1: {pair1[0]} vs {pair1[1]}")
    print(f"  Pair 2: {pair2[0]} vs {pair2[1]}")
    
    results = analyzer.compare_model_pairs(pair1, pair2)
    
    # Print the results
    analyzer.print_results(results)
    
    # Save results to file
    output_file = "o3_o4_mini_comparison_results.json"
    if results:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

def test_individual_difference_sets():
    """Test extracting individual difference sets."""
    
    lmeval_dir = "/home/haoxinl/lmeval-api"
    analyzer = ModelPairDifferenceAnalyzer(lmeval_dir, dataset_name=None)
    
    # Extract difference sets individually for inspection
    print("\n" + "="*60)
    print("INDIVIDUAL DIFFERENCE SET ANALYSIS")
    print("="*60)
    
    # O3 mini pair
    o3_diff_set = analyzer.extract_difference_set("o3-mini-high", "o3-mini-low")
    print(f"\nO3 mini difference set: {len(o3_diff_set)} instances")
    
    # O4 mini pair  
    o4_diff_set = analyzer.extract_difference_set("o4-mini-high", "o4-mini-low")
    print(f"O4 mini difference set: {len(o4_diff_set)} instances")
    
    # Calculate basic overlap
    if o3_diff_set and o4_diff_set:
        intersection = o3_diff_set & o4_diff_set
        union = o3_diff_set | o4_diff_set
        jaccard = len(intersection) / len(union) if union else 0
        
        print(f"\nBasic overlap analysis:")
        print(f"  Intersection: {len(intersection)} instances")
        print(f"  Union: {len(union)} instances") 
        print(f"  Jaccard similarity: {jaccard:.3f}")
        
        # Show some example overlapping instances
        if intersection:
            print(f"\nExample overlapping instances (first 5):")
            for i, (task, doc_id) in enumerate(sorted(intersection)):
                if i >= 5:
                    break
                print(f"    {task}: {doc_id}")

if __name__ == "__main__":
    print("Testing Model Pair Difference Analyzer")
    print("="*50)
    
    try:
        # Test the main comparison
        test_o3_o4_comparison()
        
        # Test individual difference sets
        test_individual_difference_sets()
        
    except Exception as e:
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc() 