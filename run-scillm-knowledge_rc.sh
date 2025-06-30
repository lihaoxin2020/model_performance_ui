#!/bin/bash

# Toggle preemptible settings - set to "true" to make jobs preemptible, "false" to make them non-preemptible
PREEMPTIBLE="true"
# MODEL_TYPE="vllm-b200"
MODEL_TYPE="vllm"
GPUS="1"
NUM_WORKERS="1"
MAX_GEN_TOKS="16000"
# MAX_GEN_TOKS="32000"
BATCH_SIZE="4"

TOP_P="0.95"
# MAX_GEN_TOKS="8192"
# WORKSPACE="ai2/lm-eval"
WORKSPACE="ai2/scillm"
ENV="VLLM_USE_V1=0"
# REPEATS="32"
REPEATS="1"

# Define temperatures to test
TEMPERATURES=(0.6)
# TEMPERATURES=(0.0)
# TEMPERATURES=(0.6 1.0)

# Define model and model path variables
# Define arrays for knowledge sets and seeds to iterate through
KNOWLEDGE_SETS=(
    # "r1"
    # "llama3.1-instruct-synthetic_1"
    # "llama3.1-instruct-synthetic_1_stem_only"
    # "llama3.1-instruct-synthetic_1_math_only"
    # "qwen-insturct-synthetic_1"
    # "qwen-instruct-synthetic_1_stem_only"
    # "qwen-instruct-synthetic_1_math_only"
    # "qwen-insturct-synthetic_1-sft-full"
    # "qwen-instruct-synthetic_1_stem_only-sft-full"
    # "qwen-instruct-synthetic_1_qwen_math_only-sft-full"
    # "llama3.1-instruct-synthetic_1-sft-full"
    # "llama3.1-instruct-synthetic_1_stem_only-full"
    # "llama3.1-instruct-synthetic_1_math_only-full"
    "deepseek-reasoner-full"
    # "poem"
)

# KNOWLEDGE_SOURCE will be set dynamically based on the task
# SEEDS=(-1)
SEEDS=(111)
# SEEDS=(111 222 333 444 555)
# SEEDS=(666 777 888 999 1010)


MODELS1=(
    # Models that DO use the assistant_prefix think tag
    # "Qwen3-0.6B --num-shots 0 --model-args '{\"model_path\": \"Qwen/Qwen3-0.6B\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "OpenThinker3-7B --num-shots 0 --model-args '{\"model_path\": \"open-thoughts/OpenThinker3-7B\", \"chat_model\": true, \"max_length\": ${MAX_GEN_TOKS}, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "OpenR1-Distill-7B --num-shots 0 --model-args '{\"model_path\": \"open-r1/OpenR1-Distill-7B\", \"chat_model\": true, \"max_length\": ${MAX_GEN_TOKS}}'"
    # "llama3.1-instruct-synthetic_1_stem_only_no_reasoning-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/llama3.1-instruct-synthetic_1_stem_only_no_reasoning-sft:model_cur\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 65536}'"
    # "Llama-3.1-Nemotron-Nano-8B-v1-thinking --num-shots 0 --model-args '{\"model_path\": \"nvidia/Llama-3.1-Nemotron-Nano-8B-v1\", \"chat_model\": true, \"max_length\": 65536}'"
    # "deepseek-r1 --num-shots 0 --model-args '{\"model\": \"deepseek/deepseek-reasoner\", \"chat_model\": true}'"
    # "deepseek-v3 --num-shots 0 --model-args '{\"model\": \"deepseek/deepseek-chat\", \"chat_model\": true}'"
    # "claude-3-7-sonnet-20250219-low --num-shots 0 --model-args '{\"model\": \"anthropic/claude-3-7-sonnet-20250219\", \"chat_model\": true}'"
    # "o3-mini --num-shots 0 --model-args '{\"chat_model\": true, \"data_parallel_size\": 1}'"
    # "o4-mini --num-shots 0 --model-args '{\"chat_model\": true, \"data_parallel_size\": 1}'"
    # "Qwen2.5-7B-Instruct --num-shots 0 --model-args '{\"model_path\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "Qwen2.5-7B-Instruct --num-shots 0 --model-args '{\"model_path\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "Qwen3-8B --num-shots 0 --model-args '{\"model_path\": \"Qwen/Qwen3-8B\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}, \"enable_thinking\": false}'"
    # "Qwen3-8B-thinking --num-shots 0 --model-args '{\"model_path\": \"Qwen/Qwen3-8B\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}, \"enable_thinking\": true}'"
    # "Qwen3-32B --num-shots 0 --model-args '{\"model_path\": \"Qwen/Qwen3-32B\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}, \"enable_thinking\": false}'"
    # "Qwen3-32B-thinking --num-shots 0 --model-args '{\"model_path\": \"Qwen/Qwen3-32B\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}, \"enable_thinking\": true}'"
    # "Qwen/Qwen2.5-Math-7B-Instruct --num-shots 0 --model-args '{\"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "PrimeIntellect/SYNTHETIC-1-SFT-7B --num-shots 0 --model-args '{\"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "Uni-SMART/SciLitLLM1.5-7B --num-shots 0 --model-args '{\"chat_model\": true, \"max_length\": 32768}'"
    # "qwen-infinity-instruct-7b --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-infinity-instruct-7b:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "qwen-instruct-7b-tulu3 --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-7b-tulu3:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "qwen-instruct-7b-scimix-tulu3-all-sft-5epochs --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-7b-scimix-tulu3-all-sft-5epochs:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "qwen-infinity-instruct-scimix-v2-7b --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-infinity-instruct-scimix-v2-7b:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "qwen-infinity-instruct-7b --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-infinity-instruct-7b:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "scilitllm1.5-7b-scimix-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/scilitllm1.5-7b-scimix-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "scilitllm1.5-7b-scimix-sft-1epoch --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/scilitllm1.5-7b-scimix-sft-1epoch:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "qwen-infinity-instruct-scimix-v2-7b-synthetic_1_no_think-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-infinity-instruct-scimix-v2-7b-synthetic_1_no_think-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "qwen-infinity-instruct-7b-synthetic_1_no_think-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-infinity-instruct-7b-synthetic_1_no_think-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "scilitllm1.5-7b-instruct --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/scilitllm1.5-7b-instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "scilitllm1.5-7b-cpt --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/scilitllm1.5-7b-cpt\", \"chat_model\": true, \"max_length\": 32768}'"
    # "Llama-3.1-8B-Instruct --num-shots 0 --model-args '{\"model_path\": \"meta-llama/Llama-3.1-8B-Instruct\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 65536}'"
    # "Llama-3.1-8B-Instruct-${KNOWLEDGE_SET}-${SEED} --num-shots 0 --model-args '{\"model_path\": \"meta-llama/Llama-3.1-8B-Instruct\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 65536}'"
    # "llama3.1-instruct-scimix-sft-synthetic_1_no_think-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/llama3.1-instruct-scimix-sft-synthetic_1_no_think-sft:model_cur\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 65536}'"
    # "llama3.1-instruct-scimix-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://01JS7XTXKKGRR6Q28PJV0TQYV1:model_cur\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 65536}'"
    # "qwen-instruct-sciriff-grpo --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-sciriff-grpo\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "qwen-instruct-scimix-v2-7b --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-scimix-v2-7b:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "qwen-instruct-scimix-v2-7b-model --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-scimix-v2-7b:model\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "qwen-instruct-sciriff-grpo-3epoch --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-sciriff-grpo-3epoch\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "scilitllm1.5-7b-synthetic_1_no_think-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/scilitllm1.5-7b-synthetic_1_no_think-sft:model_cur\", \"chat_model\": true, \"max_length\": 32768}'"
    # "qwen-infinity-instruct-scimix-v2-7b --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-infinity-instruct-scimix-v2-7b:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"    
    # "gpt-4o-2024-08-06 --num-shots 0 --model-args '{\"chat_model\": true, \"max_length\": 16384}'"
    # "claude-3-5-sonnet-20241022 --num-shots 0 --model-args '{\"chat_model\": true, \"max_length\": 16384}'"
    # "qwen-instruct-synthetic_1_qwen_stem_only_no_reasoning-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-synthetic_1_qwen_stem_only_no_reasoning-sft:model\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}, \"data_parallel_size\": 8}'"
    # "Qwen/Qwen3-8B --num-shots 0 --model-args '{\"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}, \"data_parallel_size\": 1}'"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --num-shots 0 --model-args '{\"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "Qwen/QwQ-32B --num-shots 0 --model-args '{\"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}, \"data_parallel_size\": 4}'"
)

MODELS2=(
    "qwen-instruct-synthetic_1-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://01JP3HYS4Q12CBBS7K51EPXQ1Z:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "llama3.1-instruct-synthetic_1-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://01JSFQQZNYR1FJT30X2X03PXX7:model_cur\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 65536}'"
    # "qwen-instruct-synthetic_1_qwen_math_only-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-synthetic_1_qwen_math_only-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "qwen-instruct-synthetic_1_stem_only-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-synthetic_1_stem_only-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "llama3.1-instruct-synthetic_1_math_only-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/llama3.1-instruct-synthetic_1_math_only-sft:model_cur\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 65536}'"
    # "llama3.1-instruct-synthetic_1_stem_only-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/llama3.1-instruct-synthetic_1_stem_only-sft:model_cur\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 65536}'"
)

# MODELS2=(
    # "Llama-3.1-Nemotron-Nano-8B-v1 --num-shots 0 --model-args '{\"model_path\": \"nvidia/Llama-3.1-Nemotron-Nano-8B-v1\", \"chat_model\": true, \"max_length\": 65536}'"
    # "qwen-insturct-synthetic_1-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://01JP3HYS4Q12CBBS7K51EPXQ1Z:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "qwen-insturct-synthetic_1-sft-${KNOWLEDGE_SET} --num-shots 0 --model-args '{\"model_path\": \"beaker://01JP3HYS4Q12CBBS7K51EPXQ1Z:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "qwen-insturct-synthetic_1-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://01JP3HYS4Q12CBBS7K51EPXQ1Z:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "qwen-insturct-synthetic_1-sft --num-shots 0 --model-args '{\"model_path\": \"/home/jovyan/workspace/oe-eval-internal/models/model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "scimix-synthetic_1-qwen-sft-sciriff-grpo --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/scimix-synthetic_1-qwen-sft-sciriff-grpo\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "scimix-synthetic_1-qwen-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/scimix-synthetic_1-qwen-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "qwen-instruct-sciriff_r1_sft --num-shots 0 --model-args '{\"model_path\": \"beaker://haoxinl/qwen-instruct-sciriff_r1_sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "qwen-infinity-instruct-scimix-v2-7b-synthetic_1-sft-0.6 --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-infinity-instruct-scimix-v2-7b-synthetic_1-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "qwen-infinity-instruct-7b-synthetic_1-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-infinity-instruct-7b-synthetic_1-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "qwen-insturct-synthetic_1-sft-sciriff-grpo --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-insturct-synthetic_1-sft-sciriff-grpo\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "llama3.1-instruct-scimix-sft-synthetic_1-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/llama3.1-instruct-scimix-sft-synthetic_1-sft:model_cur\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "scilitllm1.5-7b-synthetic_1-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/scilitllm1.5-7b-synthetic_1-sft:model_cur\", \"chat_model\": true, \"max_length\": 32768}'"
    # "scimix-synthetic_1-qwen-sft-v2-cur-sciriff-grpo-v2 --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/scimix-synthetic_1-qwen-sft-v2-cur-sciriff-grpo-v2\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "scimix-synthetic_1-qwen-sft-v2 --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/scimix-synthetic_1-qwen-sft-v2:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}'"
    # "scilitllm1.5-7b-synthetic_1-sft-grpo-sciriff --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/scilitllm1.5-7b-synthetic_1-sft-grpo-sciriff\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "llama3.1-instruct-synthetic_1-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://01JSFQQZNYR1FJT30X2X03PXX7:model_cur\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 65536}'"
    # "llama3.1-instruct-scimix-sft-synthetic_1-sft-v2 --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/llama3.1-instruct-scimix-sft-synthetic_1-sft-v2:model_cur\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 65536}'"
    # "qwen-instruct-synthetic_1-sft-sciriff-r1-distill-filtered-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-synthetic_1-sft-sciriff-r1-distill-filtered-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "scilitllm1.5-7b-synthetic_1-sft-v2 --num-shots 0 --model-args '{\"model_path\": \"beaker://01JSJYKDVNHMPW30898ZJ1P387:model_cur\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "qwen-instruct-synthetic_1_qwen_math_only-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-synthetic_1_qwen_math_only-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "qwen-instruct-synthetic_1_qwen_math_only-sft-${KNOWLEDGE_SET} --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-synthetic_1_qwen_math_only-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "qwen-instruct-synthetic_1_stem_only_no_reasoning-sft_math_only-sft-model --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-synthetic_1_stem_only_no_reasoning-sft_math_only-sft:model\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}, \"data_parallel_size\": 8}'"
    # "qwen-instruct-synthetic_1_grpo-math --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-synthetic_1_grpo-math\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "qwen-instruct-synthetic_1_qwen_math_only-sft-grpo-sciriff --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-synthetic_1_qwen_math_only-sft-grpo-sciriff\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}, \"data_parallel_size\": 8}'"
    # "llama3.1-instruct-synthetic_1-sft-grpo-sciriff --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/llama3.1-instruct-synthetic_1-sft-grpo-sciriff\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"data_parallel_size\": 8}'"
    # "qwen-instruct-synthetic_1_grpo-ifeval_3 --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-synthetic_1_grpo-ifeval_3\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "llama3.1-instruct-synthetic_1_math_only-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/llama3.1-instruct-synthetic_1_math_only-sft:model_cur\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 65536}'"
    # "qwen-instruct-synthetic_1_stem_only-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-synthetic_1_stem_only-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "qwen-instruct-synthetic_1_stem_only-sft-${KNOWLEDGE_SET} --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-synthetic_1_stem_only-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "qwen-instruct-synthetic_1_no_stem_reasoning-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-synthetic_1_no_stem_reasoning-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "llama3.1-instruct-synthetic_1_stem_only_no_reasoning-sft_math_only-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/llama3.1-instruct-synthetic_1_stem_only_no_reasoning-sft_math_only-sft:model_cur\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 65536}'"
    # "llama3.1-instruct-synthetic_1_stem_only-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/llama3.1-instruct-synthetic_1_stem_only-sft:model_cur\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 65536}'"
    # "llama3.1-instruct-s1k-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/llama3.1-instruct-s1k-sft:model_cur\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 65536}'"
    # "llama3.1-instruct-synthetic_1_no_stem_reasoning-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/llama3.1-instruct-synthetic_1_no_stem_reasoning-sft:model_cur\", \"tokenizer\": \"meta-llama/Llama-3.1-8B-Instruct\", \"chat_model\": true, \"max_length\": 65536}'"
    # "qwen-instruct-s1k-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://yixinliu/qwen-instruct-s1k-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "qwen-insturct-synthetic_1-gt_enhance-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://haoxinl/qwen-insturct-synthetic_1-gt_enhance-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "qwen-insturct-synthetic_1-gt_enhance-sft-model --num-shots 0 --model-args '{\"model_path\": \"beaker://haoxinl/qwen-insturct-synthetic_1-gt_enhance-sft\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "qwen-instruct-mot_sci_sciinstruct-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://haoxinl/qwen-instruct-mot_sci_sciinstruct-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "qwen-instruct-mot_sci-sft --num-shots 0 --model-args '{\"model_path\": \"beaker://haoxinl/qwen-instruct-mot_sci-sft:model_cur\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "qwen-instruct-mot_sci_sciinstruct-sft-model --num-shots 0 --model-args '{\"model_path\": \"beaker://haoxinl/qwen-instruct-mot_sci_sciinstruct-sft\", \"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
    # "qwen3-8b-synthetic-sft-model --num-shots 0 --model-args '{\"model_path\": \"beaker://haoxinl/qwen3-8b-synthetic-sft:model\", \"tokenizer\": \"Qwen/Qwen3-8B\", \"chat_model\": true, \"max_length\": 65536, \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2, \"original_max_position_embeddings\": 32768}}'"
# )

# Define an array of tasks
TASKS=(
    # "fos::scillm --limit 1000"
    # "drsm:mc::scillm"
    # "drsm:rc::scillm"
    # "winogrande::scillm"
    # "hellaswag::scillm"
    # "openbookqa::scillm"
    # "arc_challenge::scillm"

    # "gpqa:0shot_cot::scillm"
    # "mmlu_pro:0shot_cot::scillm"
    # "mmlu:0shot_cot::scillm"
    # "gsm8k::scillm"
    # "bbh:cot::scillm"
    # "minerva_math::scillm"

    # "medmcqa:0shot_cot::scillm"
    # "lab_bench:0shot_cot::scillm"
    # "drsm:0shot_cot::scillm"
    # "fos:0shot_cot::scillm"
    # "scibench:0shot_cot::scillm"
    # "sciriff::scillm"
    # "sciknoweval::scillm"
    # "scieval::scillm"
    # "agi_eval_english:0shot_cot::zs_cot"

    # "medmcqa:0shot::scillm"
    # "lab_bench:0shot::scillm"
    # "drsm:0shot::scillm"
    # "fos:0shot::scillm"
    # "scibench:0shot::scillm"
    # "agi_eval_english:0shot::zs"

    # "gpqa_knowledge:0shot_cot::scillm"
    # "gpqa_knowledge_rc:0shot_cot::scillm"
    # "mmlu_pro:cot_knowledge::scillm --limit 200"
    # "mmlu_pro:0shot_cot::scillm_maj_at_k --limit 200"
    # "gpqa_diamond:0shot_cot::scillm"
    # "gpqa:0shot_cot::scillm"
    # "gpqa:0shot_cot::scillm_maj_at_k"
    # "gpqa_pro:0shot_cot::scillm_maj_at_k"
    # "lab_bench:0shot_cot::scillm_maj_at_k"
    # "lab_bench:0shot_cot::scillm"
    # "scibench:0shot_cot::scillm"
    # "sciriff::scillm"
    # "olympiadbench::scillm"
    # "mmlu_pro:0shot_cot::scillm --limit 200"
    # "mmlu_pro:rc_knowledge::scillm --limit 200"
    # "lab_bench:0shot_cot_knowledge::scillm"
    "lab_bench:0shot_rc_knowledge::scillm"
    # "mmlu_pro:0shot_cot::scillm_maj_at_k --limit 200"
    # "sciknoweval::scillm --limit 200"
    # "scieval::scillm --limit 200"
    # "ugphysics::scillm --limit 200"
    # "supergpqa_Science:cot::scillm"
    # "supergpqa_Engineering:cot::scillm"

    # Add more tasks as needed
)

# Set preemptible/not-preemptible flags based on PREEMPTIBLE variable
if [ "$PREEMPTIBLE" = "true" ]; then
    PREEMPTIBLE_ARGS='"preemptible": true, "not-preemptible": false'
else
    PREEMPTIBLE_ARGS='"preemptible": false, "not-preemptible": true'
fi

# MODEL_ARGS="{\"tokenizer\": \"Qwen/Qwen2.5-7B-Instruct\", \"chat_model\": true, \"max_length\": 32768}"
# MODEL_ARGS="{\"chat_model\": true, \"max_length\": 32768}"

# Conditionally add model_path if set
# if [[ -n "$MODEL_PATH" ]]; then
#     MODEL_ARGS=$(echo "$MODEL_ARGS" | jq --arg model_path "$MODEL_PATH" '. + {model_path: $model_path}')
# fi

# Loop through each knowledge set and seed combination
for KNOWLEDGE_SET in "${KNOWLEDGE_SETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "===== Starting experiments with KNOWLEDGE_SET: $KNOWLEDGE_SET, SEED: $SEED ====="
        
        # Loop through each model in MODELS1 (no assistant_prefix)
        for MODEL in "${MODELS1[@]}"; do
            MODEL_BASE=$(echo "$MODEL" | cut -d' ' -f1)
            MODEL_REST=$(echo "$MODEL" | cut -d' ' -f2-)
            
            # Loop through each temperature
            for TEMP in "${TEMPERATURES[@]}"; do
                # Set do_sample based on temperature
                if (( $(echo "$TEMP == 0.0" | bc -l) )); then
                    DO_SAMPLE="false"
                else
                    DO_SAMPLE="true"
                fi
                
                # Create new model name with temperature in it
                MODEL_WITH_TEMP="${MODEL_BASE}-${KNOWLEDGE_SET}-${SEED}-temp${TEMP} ${MODEL_REST}"
                
                echo "===== Running with model (no assistant_prefix): $MODEL_WITH_TEMP ====="
                
                # Set gantry args based on MODEL_TYPE
                # if [ "$MODEL_TYPE" = "vllm-b200" ]; then
                GANTRY_ARGS='{"install": "pip install torch torchvision torchaudio pytorch-triton --pre --index-url https://download.pytorch.org/whl/nightly/cu128 && pip install transformers --upgrade", "hf_token": true, '${PREEMPTIBLE_ARGS}', "env-secret": "OPENAI_API_KEY=OPENAI_API_KEY", "env": "'${ENV}'"}'
                # else
                #     GANTRY_ARGS='{"install": "pip install -e .[gpu]", "hf_token": true, '${PREEMPTIBLE_ARGS}', "env-secret": "OPENAI_API_KEY=OPENAI_API_KEY", "env": "'${ENV}'"}'
                # fi
                
                # Log file for experiment IDs
                MODEL_NAME="${MODEL_BASE}-${KNOWLEDGE_SET}"
                EXPERIMENT_LOG="${MODEL_NAME}-rc.log"

                # Loop through each task and execute the command
                for TASK in "${TASKS[@]}"; do
                    # Set KNOWLEDGE_SOURCE based on the task
                    if [[ "$TASK" == *"mmlu_pro:cot_knowledge::scillm"* || "$TASK" == *"mmlu_pro:rc_knowledge::scillm"* ]]; then
                        KNOWLEDGE_SOURCE="mmlu_pro-r1"
                    elif [[ "$TASK" == *"gpqa_knowledge:0shot_cot::scillm"* || "$TASK" == *"gpqa_knowledge_rc:0shot_cot::scillm"* ]]; then
                        KNOWLEDGE_SOURCE="gpqa-knowledge"
                    elif [[ "$TASK" == *"lab_bench:0shot_cot_knowledge::scillm"* ]]; then
                        KNOWLEDGE_SOURCE="lab_bench-r1"
                    else
                        KNOWLEDGE_SOURCE="mmlu_pro-r1"  # default fallback
                    fi
                    
                    # Base command without the task argument and without assistant_prefix
                    if [ "$MODEL_TYPE" = "vllm-b200" ]; then
                        BASE_CMD="oe-eval --model ${MODEL_WITH_TEMP} --gpus ${GPUS} --num-workers ${NUM_WORKERS} --model-type vllm --batch-size ${BATCH_SIZE} --use-gantry \
                        --gantry-args '${GANTRY_ARGS}' --beaker-image haoxinl/oe-eval-b200 --beaker-workspace ${WORKSPACE} --cluster ai2/titan-cirrascale \
                        --task-args '{\"chat_overrides\": {\"generation_kwargs\": {\"top_p\": ${TOP_P}, \"do_sample\": ${DO_SAMPLE},\"temperature\": ${TEMP}, \"max_gen_toks\": ${MAX_GEN_TOKS}, \"truncate_context\": false}, \"context_kwargs\": {\"knowledge_set\": \"lihaoxin2020/${KNOWLEDGE_SET}-${KNOWLEDGE_SOURCE}\", \"knowledge_shuffling_seed\": \"${SEED}\"}}}'"
                    else
                        BASE_CMD="oe-eval --model ${MODEL_WITH_TEMP} --gpus ${GPUS} --num-workers ${NUM_WORKERS} --model-type vllm --beaker-image haoxinl/oe-eval-b200 --batch-size ${BATCH_SIZE} --use-gantry \
                        --gantry-args '${GANTRY_ARGS}' --beaker-workspace ${WORKSPACE} --cluster ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale,ai2/saturn-cirrascale,ai2/neptune-cirrascale,ai2/titan-cirrascale,ai2/augusta-google-1 \
                        --task-args '{\"chat_overrides\": {\"generation_kwargs\": {\"top_p\": ${TOP_P}, \"do_sample\": ${DO_SAMPLE},\"temperature\": ${TEMP}, \"max_gen_toks\": ${MAX_GEN_TOKS}, \"truncate_context\": false}, \"context_kwargs\": {\"knowledge_set\": \"lihaoxin2020/${KNOWLEDGE_SET}-${KNOWLEDGE_SOURCE}\", \"knowledge_shuffling_seed\": \"${SEED}\"}}}'"
                    fi
                    # \"preemptible\": false, \"not-preemptible\": true, 
                    # , \"context_kwargs\": {\"assistant_prefix\": \"<think>\n\"}
                    # && pip install vllm==0.8.5 && pip install transformers --upgrade
                    # --task-args '{\"chat_overrides\": {\"context_kwargs\": {\"assistant_prefix\": \"<think>\n\"}}}'"
                    # --task-args '{\"chat_overrides\": {\"context_kwargs\": {\"assistant_prefix\": \"<think>\n\"}}}'"
                    # --task-args '{\"chat_overrides\": {\"generation_kwargs\": {\"do_sample\": true ,\"temperature\": 0.6, \"max_gen_toks\": 32768, \"truncate_context\": false}}}'"
                    
                    CMD="$BASE_CMD --task $TASK "
                    echo "Running: $CMD"
                    
                    # Execute command and capture output
                    OUTPUT=$(eval $CMD 2>&1)
                    echo "$OUTPUT"
                    
                    # Extract experiment ID and log it
                    EXPERIMENT_ID=$(echo "$OUTPUT" | grep -o "([A-Z0-9]\{26\})" | tr -d '()')
                    if [[ -n "$EXPERIMENT_ID" ]]; then
                        TASK_NAME=$(echo "$TASK" | cut -d' ' -f1)
                        echo "${MODEL_NAME}-${TASK_NAME}-${SEED}:${EXPERIMENT_ID}" >> "$EXPERIMENT_LOG"
                        echo "Logged experiment ID: ${MODEL_NAME}-${TASK_NAME}-${SEED}:${EXPERIMENT_ID}"
                    fi
                done
                
                echo "===== Finished with model (no assistant_prefix): $MODEL_WITH_TEMP ====="
                echo ""
            done
        done

        # Loop through each model in MODELS2 (with assistant_prefix)
        for MODEL in "${MODELS2[@]}"; do
            MODEL_BASE=$(echo "$MODEL" | cut -d' ' -f1)
            MODEL_REST=$(echo "$MODEL" | cut -d' ' -f2-)
            
            # Loop through each temperature
            for TEMP in "${TEMPERATURES[@]}"; do
                # Set do_sample based on temperature
                if (( $(echo "$TEMP == 0.0" | bc -l) )); then
                    DO_SAMPLE="false"
                else
                    DO_SAMPLE="true"
                fi
                
                # Create new model name with temperature in it
                MODEL_WITH_TEMP="${MODEL_BASE}-${KNOWLEDGE_SET}-${SEED}-temp${TEMP} ${MODEL_REST}"
                
                echo "===== Running with model (with assistant_prefix): $MODEL_WITH_TEMP ====="
                
                # Set gantry args based on MODEL_TYPE
                # if [ "$MODEL_TYPE" = "vllm-b200" ]; then
                GANTRY_ARGS='{"install": "pip install torch torchvision torchaudio pytorch-triton --pre --index-url https://download.pytorch.org/whl/nightly/cu128 && pip install transformers --upgrade", "hf_token": true, '${PREEMPTIBLE_ARGS}', "env-secret": "OPENAI_API_KEY=OPENAI_API_KEY", "env": "'${ENV}'"}'
                # else
                #     GANTRY_ARGS='{"install": "pip install -e .[gpu]", "hf_token": true, '${PREEMPTIBLE_ARGS}', "env-secret": "OPENAI_API_KEY=OPENAI_API_KEY", "env": "'${ENV}'"}'
                # fi
                
                # Log file for experiment IDs
                MODEL_NAME="${MODEL_BASE}-${KNOWLEDGE_SET}"
                EXPERIMENT_LOG="${MODEL_NAME}-rc.log"

                # Loop through each task and execute the command
                for TASK in "${TASKS[@]}"; do
                    # Set KNOWLEDGE_SOURCE based on the task
                    if [[ "$TASK" == *"mmlu_pro:cot_knowledge::scillm"* || "$TASK" == *"mmlu_pro:rc_knowledge::scillm"* ]]; then
                        KNOWLEDGE_SOURCE="mmlu_pro-r1"
                    elif [[ "$TASK" == *"gpqa_knowledge:0shot_cot::scillm"* || "$TASK" == *"gpqa_knowledge_rc:0shot_cot::scillm"* ]]; then
                        KNOWLEDGE_SOURCE="gpqa-knowledge"
                    elif [[ "$TASK" == *"lab_bench:0shot_cot_knowledge::scillm"* ]]; then
                        KNOWLEDGE_SOURCE="lab_bench-r1"
                    else
                        KNOWLEDGE_SOURCE="mmlu_pro-r1"  # default fallback
                    fi
                    
                    # Base command without the task argument and without assistant_prefix
                    if [ "$MODEL_TYPE" = "vllm-b200" ]; then
                        BASE_CMD="oe-eval --model ${MODEL_WITH_TEMP} --gpus ${GPUS} --num-workers ${NUM_WORKERS} --model-type vllm --batch-size ${BATCH_SIZE} --use-gantry \
                        --gantry-args '${GANTRY_ARGS}' --beaker-image haoxinl/oe-eval-b200 --beaker-workspace ${WORKSPACE} --cluster ai2/titan-cirrascale \
                        --task-args '{\"chat_overrides\": {\"generation_kwargs\": {\"repeats\": ${REPEATS}, \"top_p\": ${TOP_P}, \"do_sample\": ${DO_SAMPLE},\"temperature\": ${TEMP}, \"max_gen_toks\": ${MAX_GEN_TOKS}, \"truncate_context\": false}, \"context_kwargs\": {\"assistant_prefix\": \"<think>\nOkay, let'\''s think step by step. The following knowledge pieces are related to the question.\", \"knowledge_set\": \"lihaoxin2020/${KNOWLEDGE_SET}-${KNOWLEDGE_SOURCE}\", \"knowledge_shuffling_seed\": \"${SEED}\"}}}'"
                    else
                        BASE_CMD="oe-eval --model ${MODEL_WITH_TEMP} --gpus ${GPUS} --num-workers ${NUM_WORKERS} --model-type vllm --beaker-image haoxinl/oe-eval-b200 --batch-size ${BATCH_SIZE} --use-gantry \
                        --gantry-args '${GANTRY_ARGS}' --beaker-workspace ${WORKSPACE} --cluster ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale,ai2/saturn-cirrascale,ai2/neptune-cirrascale,ai2/titan-cirrascale,ai2/augusta-google-1,ai2/triton-cirrascale,ai2/prior-elanding,ai2/rhea-cirrascale \
                        --task-args '{\"chat_overrides\": {\"generation_kwargs\": {\"repeats\": ${REPEATS}, \"top_p\": ${TOP_P}, \"do_sample\": ${DO_SAMPLE},\"temperature\": ${TEMP}, \"max_gen_toks\": ${MAX_GEN_TOKS}, \"truncate_context\": false}, \"context_kwargs\": {\"assistant_prefix\": \"<think>\nOkay, let'\''s think step by step. The following knowledge pieces are related to the question.\", \"knowledge_set\": \"lihaoxin2020/${KNOWLEDGE_SET}-${KNOWLEDGE_SOURCE}\", \"knowledge_shuffling_seed\": \"${SEED}\"}}}'"
                    fi
                    # \"preemptible\": false, \"not-preemptible\": true, 
                    # , \"context_kwargs\": {\"assistant_prefix\": \"<think>\n\"}
                    # && pip install vllm==0.8.5 && pip install transformers --upgrade
                    # --task-args '{\"chat_overrides\": {\"context_kwargs\": {\"assistant_prefix\": \"<think>\n\"}}}'"
                    # --task-args '{\"chat_overrides\": {\"context_kwargs\": {\"assistant_prefix\": \"<think>\n\"}}}'"
                    # --task-args '{\"chat_overrides\": {\"generation_kwargs\": {\"do_sample\": true ,\"temperature\": 0.6, \"max_gen_toks\": 32768, \"truncate_context\": false}}}'"
                    
                    CMD="$BASE_CMD --task $TASK "
                    echo "Running: $CMD"
                    
                    # Execute command and capture output
                    OUTPUT=$(eval $CMD 2>&1)
                    echo "$OUTPUT"
                    
                    # Extract experiment ID and log it
                    EXPERIMENT_ID=$(echo "$OUTPUT" | grep -o "([A-Z0-9]\{26\})" | tr -d '()')
                    if [[ -n "$EXPERIMENT_ID" ]]; then
                        TASK_NAME=$(echo "$TASK" | cut -d' ' -f1)
                        echo "${MODEL_NAME}-${TASK_NAME}-${SEED}:${EXPERIMENT_ID}" >> "$EXPERIMENT_LOG"
                        echo "Logged experiment ID: ${MODEL_NAME}-${TASK_NAME}-${SEED}:${EXPERIMENT_ID}"
                    fi
                done
                
                echo "===== Finished with model (with assistant_prefix): $MODEL_WITH_TEMP ====="
                echo ""
            done
        done
        
        echo "===== Finished all experiments for KNOWLEDGE_SET: $KNOWLEDGE_SET, SEED: $SEED ====="
        echo ""
    done
done
