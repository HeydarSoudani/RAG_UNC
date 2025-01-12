#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --output=script_logging/slurm_%A.out

module load 2022
module load Python/3.10.4-GCCcore-11.3.0
# module load Java/11.0.2
# pip install transformers==4.37.2

# python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 20 -input "datasets/single_hop/corpus" -index "datasets/single_hop/corpus/bm25_index" -storePositions -storeDocvectors -storeRaw
# python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 20 -input "datasets/single_hop/corpus_hf" -index "datasets/single_hop/corpus_hf/bm25_index" -storePositions -storeDocvectors -storeRaw
# srun python $HOME/RAG_UNC/processed_datasets/_processing_dataset.py
# srun python $HOME/RAG_UNC/processed_datasets/_corpus_preparation.py
# srun python processed_datasets/_contriever_retrieval_model.py

model="meta-llama/Llama-2-7b-chat-hf"
dataset="nqgold"
subsec="test"
main_prompt_format="only_q"
second_prompt_format="rerank_retriever_top1"
fraction_of_data_to_use=1.0    # nqgold 0.173 | trivia 0.057 | popqa 0.035
run_id="run_0"
generation_type="normal"
alpha_generation=0.5
alpha_probability=0.5

# model="meta-llama/Llama-2-7b-chat-hf"
# dataset="trivia"
# subsec="dev"
# main_prompt_format="q_positive"
# second_prompt_format="only_q"
# fraction_of_data_to_use=0.340    # nqgold 0.173 | trivia 0.057,  | popqa 0.035
# run_id="run_0"
# generation_type="normal"
# alpha_generation=0.5
# alpha_probability=0.5

# model="meta-llama/Llama-2-7b-chat-hf"
# dataset="popqa"
# subsec="test"
# main_prompt_format="bm25_retriever_top1"
# second_prompt_format="only_q"
# fraction_of_data_to_use=0.205    # nqgold 0.173 | trivia 0.057 | popqa 0.035
# run_id="run_0"
# generation_type="normal"
# alpha_generation=0.5
# alpha_probability=0.5


srun python $HOME/RAG_UNC/framework/run/run_framework.py \
    --model "$model" \
    --dataset "$dataset" \
    --subsec "$subsec" \
    --main_prompt_format "$main_prompt_format" \
    --second_prompt_format "$second_prompt_format" \
    --fraction_of_data_to_use "$fraction_of_data_to_use" \
    --output_file_postfix "$output_file_postfix" \
    --run_id "$run_id" \
    --generation_type "$generation_type" \
    --alpha_generation "$alpha_generation" \
    --alpha_probability "$alpha_probability"



# generation_type:
    # 'normal', 'cad'

# prompt_format:
    # 'only_q', 'q_negative', 'q_positive', 'q_conflict'
    # 'bm25_retriever_top1', 'bm25_retriever_top5',
    # 'rerank_retriever_top1', 'rerank_retriever_top5'

# Datasets:
    # 'nqgold',
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

# mode:
    # 'seperated', 'combined'








# model="meta-llama/Llama-2-7b-chat-hf"
# dataset="nqswap"
# subsec="test"
# main_prompt_format="only_q"
# second_prompt_format="q_conflict"
# fraction_of_data_to_use=0.5    # nqgold 0.173 | trivia 0.057 | popqa 0.035
# run_id="run_0"
# generation_type="normal"
# alpha_generation=0.5
# alpha_probability=0.5