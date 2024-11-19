#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --output=script_logging/slurm_%A.out
#SBATCH --exclusive

module load 2022
module load Python/3.10.4-GCCcore-11.3.0
# pip install transformers==4.37.2
# pip install evaluate


model_name="meta-llama/Llama-2-7b-chat-hf"
dataset="trivia"
prompt_format="q_negative"
fraction_of_data_to_use=1.0
output_file_postfix=""
run_id="run_3"


srun python $HOME/RAG_UNC/baselines/MARS/run_framework.py \
    --model_name "$model_name" \
    --dataset "$dataset" \
    --prompt_format "$prompt_format" \
    --fraction_of_data_to_use "$fraction_of_data_to_use" \
    --output_file_postfix "$output_file_postfix" \
    --run_id "$run_id"


# srun python $HOME/RAG_UNC/baselines/MARS/get_affinity_uncertainty.py \
#     --model_name "$model_name" \
#     --dataset "$dataset" \
#     --prompt_format "$prompt_format" \
#     --fraction_of_data_to_use "$fraction_of_data_to_use" \
#     --output_file_postfix "$output_file_postfix" \
#     --run_id "$run_id"


# Datasets:
    # 'trivia', 'nq', 'squad1', 'webquestions'
    # 'hotpotqa', '2wikimultihopqa', 'musique'
    # 'topicoqa_org', 'topicoqa_his', 'topicoqa_rw',

# prompt_format:
    # 'only_q', 'q_positive', 'q_negative'

# Model name:
    # tiny_llama: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # stable_lm2: "stabilityai/stablelm-2-zephyr-1_6b"
    # MiniCPM: "openbmb/MiniCPM-2B-sft-fp32"

    # mistral: "mistralai/Mistral-7B-Instruct-v0.1"
    # zephyr: "HuggingFaceH4/zephyr-7b-beta"
    # llama2: "meta-llama/Llama-2-7b-chat-hf"
    # llama3: "meta-llama/Meta-Llama-3-8B-Instruct"






# srun python $HOME/RAG_UNC/baselines/MARS/1_generate_answers_mg.py \
#     --model_name "$model_name" \
#     --dataset "$dataset" \
#     --prompt_format "$prompt_format"

# srun python $HOME/RAG_UNC/baselines/MARS/1_generate_answers.py \
#     --model_name "$model_name" \
#     --dataset "$dataset" \
#     --prompt_format "$prompt_format"

# srun python $HOME/RAG_UNC/baselines/MARS/2_clean_generations.py \
#     --model_name "$model_name" \
#     --dataset "$dataset" \
#     --prompt_format "$prompt_format"

# srun python $HOME/RAG_UNC/baselines/MARS/3_get_semantic_similarity.py \
#     --model_name "$model_name"\
#     --dataset "$dataset" \
#     --prompt_format "$prompt_format"

# srun python $HOME/RAG_UNC/baselines/MARS/4_get_likelihoods.py \
#     --model_name "$model_name"\
#     --dataset "$dataset" \
#     --prompt_format "$prompt_format"

# srun python $HOME/RAG_UNC/baselines/MARS/5_get_uncertainty.py \
#     --model_name "$model_name"\
#     --dataset "$dataset" \
#     --prompt_format "$prompt_format"

# srun python $HOME/RAG_UNC/baselines/MARS/6_get_auc_results.py \
#     --model_name "$model_name"\
#     --dataset "$dataset" \
#     --prompt_format "$prompt_format"

# srun python $HOME/RAG_UNC/baselines/MARS/6_get_pea_results.py \
#     --model_name "$model_name"\
#     --dataset "$dataset" \
#     --prompt_format "$prompt_format"

# srun python $HOME/RAG_UNC/baselines/MARS/6_get_rce_results.py \
#     --model_name "$model_name"\
#     --dataset "$dataset" \
#     --prompt_format "$prompt_format"


