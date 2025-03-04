#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --time=1:30:00
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load Java/21.0.2
# pip install --upgrade transformers


python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 20 -input "datasets/row_files/corpus" -index "datasets/row_files/corpus/bm25_index" -storePositions -storeDocvectors -storeRaw


# ### === Set variables ==========================
# model="meta-llama/Llama-3.1-8B-Instruct"
# dataset="popqa"
# subsec="test"
# prompt_format="q_positive"
# fraction_of_data_to_use=0.021    # nqgold 0.104 | trivia 0.034 | popqa 0.021


# srun python $HOME/RAG_UNC/2_truth_torch_framework/run/run_framework.py \
#     --model "$model" \
#     --dataset "$dataset" \
#     --subsec "$subsec" \
#     --prompt_format "$prompt_format" \
#     --fraction_of_data_to_use "$fraction_of_data_to_use"



# prompt_format:
    # 'only_q', 'q_negative', 'q_positive',
    # 'bm25_retriever_top1', 'bm25_retriever_top5',
    # 'contriever_retriever_top1', 'contriever_retriever_top5',
    # 'rerank_retriever_top1', 'rerank_retriever_top5'

# Datasets:
    # 'nqgold', 'trivia', 'popqa',
    # '2wikimultihopqa', 'hotpotqa', 'musique',

# Model name:
    # GPT-4o: openai/gpt-4o'
    # GPT-3.5: openai/gpt-3.5-turbo-instruct'
    
    # llama3.2: "meta-llama/Llama-3.2-3B-Instruct"
    # llama3.2: "meta-llama/Llama-3.2-1B-Instruct"
    # llama3.1: "meta-llama/Llama-3.1-8B-Instruct"
    # llama2:   "meta-llama/Llama-2-7b-chat-hf"
    # Qwen2.5:  "Qwen/Qwen2.5-7B-Instruct"

    # DS-7B: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # DS-8B: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    # DS-14B: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    
    # Mistral: mistralai/Mistral-7B-Instruct-v0.3
    

