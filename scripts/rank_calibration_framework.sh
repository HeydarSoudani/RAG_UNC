#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --time=3:00:00
#SBATCH --output=script_logging/slurm_%A.out
#SBATCH --exclusive

module load 2022
module load Python/3.10.4-GCCcore-11.3.0
# pip install transformers==4.37.2
# pip install evaluate

model="meta-llama/Llama-2-7b-chat-hf"
prompt_format="q_positive"
dataset="nq"
fraction_of_data_to_use=0.1
num_return_sequences=5
correctness="bert_similarity"


srun python $HOME/RAG_UNC/baselines/rank_calibration/run/run_framework.py \
    --model "$model" \
    --dataset "$dataset" \
    --prompt_format "$prompt_format" \
    --fraction_of_data_to_use "$fraction_of_data_to_use" \
    --correctness "$correctness" \
    --num_return_sequences "$num_return_sequences"


# prompt_format:
    # 'only_q', 'q_positive', 'q_negative'

# Datasets:
    # 'trivia', 'nq', 'squad1', 'webquestions'
    # 'hotpotqa', '2wikimultihopqa', 'musique'
    # 'topicoqa_org', 'topicoqa_his', 'topicoqa_rw',

# Model name:
    # tiny_llama: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # stable_lm2: "stabilityai/stablelm-2-zephyr-1_6b"
    # MiniCPM: "openbmb/MiniCPM-2B-sft-fp32"

    # mistral: "mistralai/Mistral-7B-Instruct-v0.1"
    # zephyr: "HuggingFaceH4/zephyr-7b-beta"
    # llama2: "meta-llama/Llama-2-7b-chat-hf"
    # llama3: "meta-llama/Meta-Llama-3-8B-Instruct"
