#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
# module load Java/21.0.2
# pip install --upgrade transformers



### === Set variables ==========================
model="meta-llama/Llama-2-7b-chat-hf"
dataset="nqgold"
subsec="test"
prompt_format="q_positive"
fraction_of_data_to_use=0.104    # nqgold 0.104 | trivia 0.057 | popqa 0.035


srun python $HOME/RAG_UNC/2_truth_torch_framework/run/run_framework.py \
    --model "$model" \
    --dataset "$dataset" \
    --subsec "$subsec" \
    --prompt_format "$prompt_format" \
    --fraction_of_data_to_use "$fraction_of_data_to_use"



# prompt_format:
    # 'only_q', 'q_negative', 'q_positive',
    # 'bm25_retriever_top1', 'bm25_retriever_top5',
    # 'contriever_retriever_top1', 'contriever_retriever_top5',
    # 'rerank_retriever_top1', 'rerank_retriever_top5'

# Datasets:
    # 'nqgold', 'trivia', 'popqa'

# Model name:
    # llama3.1: "meta-llama/Llama-3.1-8B-Instruct"
    # Qwen2.5: Qwen/Qwen2.5-7B-Instruct
    # llama2: meta-llama/Llama-2-7b-chat-hf
    # Mistral: mistralai/Mistral-7B-Instruct-v0.3
    # Tinyllama: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    

