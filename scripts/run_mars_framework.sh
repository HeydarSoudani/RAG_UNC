#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --time=0:30:00
#SBATCH --output=script_logging/slurm_%A.out

module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load Java/11.0.2
# pip install transformers==4.37.2
# pip install --upgrade transformers
# pip install "minicheck[llm] @ git+https://github.com/Liyan06/MiniCheck.git@main"


# python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 20 -input "datasets/single_hop/corpus" -index "datasets/single_hop/corpus/bm25_index" -storePositions -storeDocvectors -storeRaw
# python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 20 -input "datasets/single_hop/corpus_hf" -index "datasets/single_hop/corpus_hf/bm25_index" -storePositions -storeDocvectors -storeRaw
# srun python $HOME/RAG_UNC/processed_datasets/_processing_dataset.py
# srun python $HOME/RAG_UNC/processed_datasets/_corpus_preparation.py
# srun python processed_datasets/_contriever_retrieval_model.py


# prompt_format:
    # 'only_q', 'q_negative', 'q_positive', 'q_conflict'
    # 'bm25_retriever_top1', 'bm25_retriever_top5',
    # 'contriever_retriever_top1', 'contriever_retriever_top5',
    # 'rerank_retriever_top1', 'rerank_retriever_top5'

# Datasets:
    # 'nqgold',
    # 'webquestions', 'trivia', 'nq', 'squad1',
    # 'hotpotqa', '2wikimultihopqa', 'musique'
    # 'topicoqa_org', 'topicoqa_his', 'topicoqa_rw',

# Model name:
    # llama2: meta-llama/Llama-2-7b-chat-hf
    # Mistral: mistralai/Mistral-7B-Instruct-v0.3
    # Tinyllama: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    # Vicuna:  lmsys/vicuna-7b-v1.5

    # Qwen2.5: Qwen/Qwen2.5-7B-Instruct
    # llama3.1: "meta-llama/Llama-3.1-8B-Instruct"  --> pip install --upgrade transformers


model="mistralai/Mistral-7B-Instruct-v0.3"
dataset="nqgold"
subsec="test"
main_prompt_format="q_positive"
second_prompt_format="only_q"
fraction_of_data_to_use=1.0    # nqgold 0.173 | trivia 0.057 | popqa 0.035
run_id="run_0"
generation_type="normal"
alpha_generation=0.5
alpha_probability=0.5

# model="mistralai/Mistral-7B-Instruct-v0.3"
# dataset="trivia"
# subsec="dev"
# main_prompt_format="q_negative"
# second_prompt_format="only_q"
# fraction_of_data_to_use=0.340    # nqgold 0.173 | trivia 0.057/0.340,  | popqa 0.035
# run_id="run_0"
# generation_type="normal"
# alpha_generation=0.5
# alpha_probability=0.5

# model="mistralai/Mistral-7B-Instruct-v0.3"
# dataset="popqa"
# subsec="test"
# main_prompt_format="q_negative"
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










# mode:
    # 'seperated', 'combined'

# generation_type:
    # 'normal', 'cad'




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