#!/usr/bin/env python3
import os
import subprocess
import glob
from pathlib import Path

def main():
    # Base command
    base_cmd = """oe-eval --model qwen-instruct-synthetic_1_qwen_math_only-sft-temp0.8 \
--num-shots 0 \
--model-args '{"model_path": "/home/jovyan/workspace/oe-eval-internal/models/qwen-instruct-synthetic_1_qwen_math_only-sft/model_cur", "tokenizer": "Qwen/Qwen2.5-7B-Instruct", "chat_model": true, "max_length": 65536, "rope_scaling": {"rope_type": "yarn", "factor": 2, "original_max_position_embeddings": 32768}}' \
--gpus 1 \
--num-workers 1 \
--model-type vllm \
--batch-size 64 \
--run-local \
--recompute-metrics \
--task-args '{"chat_overrides": {"generation_kwargs": {"top_p": 0.95, "do_sample": true,"temperature": 0.8, "max_gen_toks": 65536, "truncate_context": false}, "context_kwargs": {"assistant_prefix": "<think>\\n"}}}' \
--task gpqa_diamond:0shot_cot::scillm_maj_at_k"""

    # Path to output directories
    base_dir = "/home/jovyan/workspace/lmeval/qwen-gpqa_diamond-majAtK"
    
    # Get all directories
    directories = [d for d in glob.glob(f"{base_dir}/*") if os.path.isdir(d)]
    
    print(f"Found {len(directories)} directories to process")
    
    # Run command for each directory
    for dir_path in directories:
        dir_name = os.path.basename(dir_path)
        # output_dir = f"/home/jovyan/workspace/outputs/{dir_name}"
        output_dir = dir_path
        # from IPython import embed; embed()
        
        # Create full command
        full_cmd = f"{base_cmd} --output-dir {output_dir}"
        
        print(f"Running evaluation for: {dir_name}")
        print(f"Command: {full_cmd}")
        
        try:
            # Execute the command
            subprocess.run(full_cmd, shell=True, check=True)
            print(f"Successfully completed evaluation for {dir_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation for {dir_name}: {e}")
        
        print("-" * 80)
    
    print("All evaluations completed")

if __name__ == "__main__":
    main() 