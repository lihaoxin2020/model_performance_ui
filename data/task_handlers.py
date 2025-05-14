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
        elif "correctness" in metrics:
            is_correct = metrics["correctness"] == 1
        
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

@register_task_handler("olympiadbench")
class OlympiadBenchTaskHandler(BaseTaskHandler):
    """Handler for OlympiadBench dataset"""

    def get_domains(self, predictions=None, directory=None):
        """Get subtask (problem) information for OlympiadBench."""
        if not directory:
            return None

        subtasks = {}
        # Match files like: task-000-olympiadbench_OE_TO_maths_en_COMP-predictions.jsonl
        pattern = os.path.join(directory, "task-*-olympiadbench_*-predictions.jsonl")
        for pred_file in glob.glob(pattern):
            fname = os.path.basename(pred_file)
            # Extract domain between TO_ and _COMP
            m = re.search(r'olympiadbench_[^_]+_TO_(.+?)_COMP', fname)
            if not m:
                continue
            domain = m.group(1)
            with open(pred_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    try:
                        obj = json.loads(line)
                        native_id = obj.get("native_id")
                        if native_id is None:
                            native_id = f"{fname}:{idx}"
                        subtasks[native_id] = domain
                    except json.JSONDecodeError:
                        continue

        # no high_level categories; only subtasks
        return {"high_level": {}, "subdomain": subtasks}

    def get_domain_display_name(self):
        return "Subtasks"

    def get_subdomain_display_name(self):
        return "Problems"

    def get_domain_selection_name(self):
        return ["Subtasks"]

@register_task_handler("ugphysics")
class UGPhysicsTaskHandler(BaseTaskHandler):
    """Handler for UGPhysics dataset"""

    def get_domains(self, predictions=None, directory=None):
        """Get subtask information for UGPhysics."""
        if not directory:
            return None

        subtasks = {}
        # Match files like: task-000-ugphysics_Electrodynamics-predictions.jsonl
        pattern = os.path.join(directory, "task-*-ugphysics_*-predictions.jsonl")
        
        for pred_file in glob.glob(pattern):
            fname = os.path.basename(pred_file)
            
            # Extract the subtask between 'ugphysics_' and '-predictions'
            m = re.search(r"task-\d+-ugphysics_(.+?)-predictions\.jsonl", fname)
            if not m:
                continue
            
            # Get the full subtask name (e.g. 'Electrodynamics')
            subtask = m.group(1)
            
            # Load predictions to extract native_ids
            with open(pred_file, 'r') as f:
                for line in f:
                    try:
                        pred = json.loads(line)
                        # Use native_id from prediction
                        native_id = pred.get("native_id")
                        if native_id is not None:
                            subtasks[native_id] = subtask
                    except json.JSONDecodeError:
                        continue

        # No high_level categories for UGPhysics—only subtasks
        return {"high_level": {}, "subdomain": subtasks}

    def get_domain_display_name(self):
        return "Topics"

    def get_subdomain_display_name(self):
        return "Topics"

    def get_domain_selection_name(self):
        return ["Topics"]

@register_task_handler("lab_bench")
class LabBenchTaskHandler(BaseTaskHandler):
    """Handler for LabBench dataset"""

    def get_domains(self, predictions=None, directory=None):
        """Get subtask information for LabBench."""
        if not directory:
            return None
        
        subtasks = {}
        # Match files like: task-000-lab_bench_CloningScenarios:cot-predictions.jsonl
        pattern = os.path.join(directory, "task-*-lab_bench_*-predictions.jsonl")
        
        for pred_file in glob.glob(pattern):
            fname = os.path.basename(pred_file)
            
            # Extract the subtask between 'lab_bench_' and ':cot-predictions'
            m = re.search(r"task-\d+-lab_bench_(.+?):cot-predictions\.jsonl", fname)
            if not m:
                continue
            
            # Get the full subtask name (e.g. 'CloningScenarios')
            subtask = m.group(1)
            
            # Load predictions to extract native_ids
            with open(pred_file, 'r') as f:
                for idx, line in enumerate(f):
                    try:
                        pred = json.loads(line)
                        # Create unique identifier using file and line position
                        native_id = f"{fname}:{idx}"
                        subtasks[native_id] = subtask
                    except json.JSONDecodeError:
                        continue

        # No high_level categories for LabBench—only subtasks
        return {"high_level": {}, "subdomain": subtasks}

    def process_predictions(self, predictions):
        """Process predictions for LabBench."""
        
        processed = []
        for i, pred in enumerate(predictions):
            # Get filename from metadata
            metadata = pred.get("metadata", {})
            filename = metadata.get("filename", "")
            if not filename:
                # If filename not in metadata, try to extract from directory path
                directory = metadata.get("directory", "")
                if directory:
                    # Find the task-*-lab_bench_* file in the directory
                    pattern = os.path.join(directory, "task-*-lab_bench_*-predictions.jsonl")
                    matches = glob.glob(pattern)
                    if matches:
                        filename = os.path.basename(matches[0])
            
            if not filename:
                print(f"Warning: Could not determine filename for prediction {i}")
                continue
                
            # Create unique identifier using file and line position
            line_number = metadata.get("line_number", i)
            native_id = f"{filename}:{line_number}"
            pred["native_id"] = native_id
            processed.append(pred)

        
        return processed

    def get_domain_display_name(self):
        return "Tasks"

    def get_subdomain_display_name(self):
        return "Tasks"

    def get_domain_selection_name(self):
        return ["Tasks"]

@register_task_handler("scieval")
class SciEvalTaskHandler(BaseTaskHandler):
    """Handler for SciEval dataset"""

    def get_domains(self, predictions=None, directory=None):
        """Get domain information for SciEval."""
        if not directory:
            return None
        
        # Track both subjects (high-level) and combined subtasks
        subjects = {}
        subtasks = {}
        
        # Match files like: task-000-scieval_biology_scientific_calculation-predictions.jsonl
        pattern = os.path.join(directory, "task-*-scieval_*-predictions.jsonl")
        
        for pred_file in glob.glob(pattern):
            fname = os.path.basename(pred_file)
            
            # Extract subject and task type from filename
            # Format: scieval_<subject>_<task_type>
            m = re.search(r"scieval_([^_]+)_(.+?)-predictions\.jsonl", fname)
            if not m:
                continue
            
            subject = m.group(1).capitalize()  # e.g., "biology" -> "Biology"
            task_type = m.group(2).replace("_", " ").title()  # e.g., "scientific_calculation" -> "Scientific Calculation"
            subtask = f"{subject} - {task_type}"  # e.g., "Biology - Scientific Calculation"
            
            # Load predictions to extract doc_ids
            with open(pred_file, 'r') as f:
                for line in f:
                    try:
                        pred = json.loads(line)
                        doc_id = pred.get("doc_id")
                        if doc_id is not None:
                            # Create unique identifier using file and doc_id
                            native_id = f"{fname}:{doc_id}"
                            subjects[native_id] = subject
                            subtasks[native_id] = subtask
                    except json.JSONDecodeError:
                        continue

        # Return both subjects and combined subtasks
        return {"high_level": subjects, "subdomain": subtasks}

    def process_predictions(self, predictions):
        """Process predictions for SciEval."""
        processed = []
        for i, pred in enumerate(predictions):
            # Get filename from metadata
            metadata = pred.get("metadata", {})
            filename = metadata.get("filename", "")
            if not filename:
                # If filename not in metadata, try to extract from directory path
                directory = metadata.get("directory", "")
                if directory:
                    # Find the task-*-scieval_* file in the directory
                    pattern = os.path.join(directory, "task-*-scieval_*-predictions.jsonl")
                    matches = glob.glob(pattern)
                    if matches:
                        filename = os.path.basename(matches[0])
            
            if not filename or pred.get("doc_id") is None:
                continue
                
            # Create unique identifier using file and doc_id
            native_id = f"{filename}:{pred['doc_id']}"
            pred["native_id"] = native_id
            processed.append(pred)
        
        return processed

    def get_domain_display_name(self):
        return "Subjects"

    def get_subdomain_display_name(self):
        return "Tasks"

    def get_domain_selection_name(self):
        return ["Subjects", "Tasks"]

class DefaultTaskHandler(BaseTaskHandler):
    """Default handler for datasets without specific implementations"""
    
    def get_domains(self, predictions=None, directory=None):
        """Default implementation for domains"""
        # Without specific domain information, we can't provide domain analysis
        return None 