#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --time=3:00:00
#SBATCH --output=script_logging/slurm_%A.out

# huggingface-cli login

# module load 2024
# module load Python/3.12.3-GCCcore-13.3.0
# pip install evaluate
# pip install tensorflow
# pip install tensorflow-hub
# pip install tensorflow-text


module load 2022
module load Python/3.10.4-GCCcore-11.3.0
# pip install --upgrade transformers

# pip uninstall transformers torch
# pip install transformers torch

# python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 20 -input "datasets/single_hop/corpus" -index "datasets/single_hop/corpus/bm25_index" -storePositions -storeDocvectors -storeRaw
# srun python $HOME/RAG_UNC/processed_datasets/_processing_dataset.py
# srun python $HOME/RAG_UNC/processed_datasets/_corpus_preparation.py


model="meta-llama/Llama-2-7b-chat-hf"
dataset="webquestions"
main_prompt_format="rerank_retriever_top1"
second_prompt_format="only_q"
fraction_of_data_to_use=1.0
run_id="run_0"

srun python $HOME/RAG_UNC/framework/run/run_framework.py \
    --model "$model" \
    --dataset "$dataset" \
    --main_prompt_format "$main_prompt_format" \
    --second_prompt_format "$second_prompt_format" \
    --fraction_of_data_to_use "$fraction_of_data_to_use" \
    --output_file_postfix "$output_file_postfix" \
    --run_id "$run_id"




# prompt_format:
# 'only_q', 'q_negative', 'q_positive',
# 'bm25_retriever_top1', 'bm25_retriever_top5',
# 'rerank_retriever_top1', 'rerank_retriever_top5'

# mode:
# 'seperated', 'combined'

# Datasets:
    # 'webquestions', 'trivia', 'nq', 'squad1',
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
