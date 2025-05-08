"""
Task handler system for supporting different dataset types
"""
import os
import json
import glob
import re
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datasets import load_dataset

# Registry to hold all task handlers
TASK_HANDLERS = {}

def register_task_handler(dataset_name):
    """Decorator to register task handlers"""
    def decorator(cls):
        TASK_HANDLERS[dataset_name] = cls
        return cls
    return decorator

class BaseTaskHandler(ABC):
    """Base class for task handlers"""
    
    @staticmethod
    def get_handler_for_dataset(dataset_name):
        """Get the appropriate handler for a dataset"""
        if dataset_name in TASK_HANDLERS:
            return TASK_HANDLERS[dataset_name]()
        return DefaultTaskHandler()
    
    @abstractmethod
    def get_domains(self, predictions=None, directory=None):
        """Get domain information for this task"""
        pass
    
    def get_domain_display_name(self):
        """Get display name for domains in this task"""
        return "Domains"
    
    def get_subdomain_display_name(self):
        """Get display name for subdomains in this task"""
        return "Subdomains"
    
    def get_domain_selection_name(self):
        """Get selection name for domain analysis type"""
        return ["High-Level Domains", "Subdomains"]
    
    def process_predictions(self, predictions):
        """Process predictions for this task (if needed)"""
        return predictions
    
    def check_correctness(self, prediction):
        """Check if a prediction is correct"""
        is_correct = False
        
        # Handle missing metrics field
        if not prediction or "metrics" not in prediction:
            return False
        
        metrics = prediction["metrics"]
        
        # Check various metrics fields that might indicate correctness
        if "exact_match" in metrics:
            is_correct = metrics["exact_match"] == 1
        elif "exact_match_flex" in metrics:
            is_correct = metrics["exact_match_flex"] == 1
        elif "accuracy" in metrics:
            is_correct = metrics["accuracy"] == 1
        
        return is_correct

@register_task_handler("gpqa")
class GPQATaskHandler(BaseTaskHandler):
    """Handler for GPQA dataset"""
    
    def get_domains(self, predictions=None, directory=None):
        """Get domain information for GPQA"""
        try:
            gpqa = load_dataset("Idavidrein/gpqa", "gpqa_main")
            
            # Extract domain information
            high_level_domains = {}
            subdomains = {}
            
            for idx, item in enumerate(gpqa['train']):
                # Extract the ID and domains
                high_level_domains[idx] = item.get('High-level domain', 'Unknown')
                subdomains[idx] = item.get('Subdomain', 'Unknown')
            
            # Simplify domain names to high level domains
            for id, domain in list(high_level_domains.items()):
                if domain != 'Unknown':
                    high_level = domain.split('>')[0].strip()
                    high_level_domains[id] = high_level
                    
            return {"high_level": high_level_domains, "subdomain": subdomains}
        except Exception as e:
            print(f"Error loading GPQA dataset: {e}")
            return None

@register_task_handler("mmlu_pro")
class MMLUProTaskHandler(BaseTaskHandler):
    """Handler for MMLU-Pro dataset"""
    
    def get_domains(self, predictions=None, directory=None):
        """Get domain (subtask) information for MMLU-Pro"""
        if not directory:
            return None
            
        # Extract domains from prediction file names
        subtasks = {}
        
        # Find all prediction files for mmlu_pro
        pred_files = glob.glob(os.path.join(directory, "task-*-mmlu_pro*-predictions.jsonl"))
        
        # Map from prediction index to domain
        for pred_file in pred_files:
            # Extract domain from filename like task-000-mmlu_pro_math:cot-predictions.jsonl
            domain_match = re.search(r'mmlu_pro_([^:]+)', os.path.basename(pred_file))
            if domain_match:
                subtask = domain_match.group(1)
                
                # Load predictions to extract native_ids
                with open(pred_file, 'r') as f:
                    for idx, line in enumerate(f):
                        try:
                            pred = json.loads(line)
                            # Use idx as fallback only if native_id is not None
                            native_id = pred.get("native_id")
                            if native_id is None:
                                # For null native_id, create a unique identifier using file and line position
                                native_id = f"{os.path.basename(pred_file)}:{idx}"
                                
                            subtasks[native_id] = subtask
                        except json.JSONDecodeError:
                            continue
        
        # For MMLU-Pro, we only want to show subtasks, not categories
        # Return empty high_level domains to suppress the category tab
        return {"high_level": {}, "subdomain": subtasks}
    
    def get_domain_display_name(self):
        """Get display name for domains in MMLU-Pro"""
        return "Categories"
    
    def get_subdomain_display_name(self):
        """Get display name for subdomains in MMLU-Pro"""
        return "Subtasks"
    
    def get_domain_selection_name(self):
        """Get selection name for domain analysis type"""
        return ["Categories", "Subtasks"]

@register_task_handler("scibench")
class SciBenchTaskHandler(BaseTaskHandler):
    """Handler for SciBench dataset"""
    
    def get_domains(self, predictions=None, directory=None):
        """Get domain information for SciBench dataset"""
        if not predictions:
            return None
            
        # Example classification based on question content keywords
        # In a real implementation, this would use the actual SciBench structure
        subjects = {}
        difficulty_levels = {}
        
        for pred in predictions:
            # Extract question content or ID to determine domain
            native_id = pred.get("native_id", -1)
            
            # Skip entries with None native_id
            if native_id is None:
                continue
                
            # Use simple rule-based classification based on question ID range
            # This is just a demonstration; a real implementation would use dataset metadata
            if native_id % 5 == 0:
                subjects[native_id] = "Physics"
                difficulty_levels[native_id] = "Advanced"
            elif native_id % 5 == 1:
                subjects[native_id] = "Chemistry"
                difficulty_levels[native_id] = "Intermediate"
            elif native_id % 5 == 2:
                subjects[native_id] = "Biology"
                difficulty_levels[native_id] = "Basic"
            elif native_id % 5 == 3:
                subjects[native_id] = "Computer Science"
                difficulty_levels[native_id] = "Advanced"
            else:
                subjects[native_id] = "Mathematics"
                difficulty_levels[native_id] = "Intermediate"
                
        return {"high_level": subjects, "subdomain": difficulty_levels}
    
    def get_domain_display_name(self):
        """Get display name for domains in SciBench"""
        return "Subjects"
    
    def get_subdomain_display_name(self):
        """Get display name for subdomains in SciBench"""
        return "Difficulty Levels"
    
    def get_domain_selection_name(self):
        """Get selection name for domain analysis type"""
        return ["Subjects", "Difficulty Levels"]

class DefaultTaskHandler(BaseTaskHandler):
    """Default handler for datasets without specific implementations"""
    
    def get_domains(self, predictions=None, directory=None):
        """Default implementation for domains"""
        # Without specific domain information, we can't provide domain analysis
        return None 