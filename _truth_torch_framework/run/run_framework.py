#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import argparse

from utils.utils import set_seed
from truth_generation import truth_generation



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--dataset', type=str, default='nqgold', choices=[
        'nqgold', 'trivia', 'popqa',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'webquestions', 'squad1', 'nq', 'nqswap',
        'topicoqa',
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--prompt_format', type=str, default='q_positive', choices=[
        'only_q', 'q_positive', 'q_negative', 'q_conflict',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'contriever_retriever_top1', 'contriever_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    parser.add_argument('--accuracy_metric', type=str, default="model_judge", choices=[
        'exact_match', 'model_judge', 'bem_score', 'bert_score', 'rouge_score'
    ])
    parser.add_argument('--model_eval', type=str, default='gpt-3.5-turbo') # meta-llama/Llama-3.1-8B-Instruct
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    parser.add_argument("--roc_auc_threshold", type=float, default=0.8)
    parser.add_argument('--num_generations', type=int, default=10)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--decoding_method', type=str, default='beam_search')
    parser.add_argument('--temperature', type=float, default='1.0')
    parser.add_argument('--num_beams', type=int, default='1')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_0 (300s-G3.5)')
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    
    ### === Define CUDA device =================== 
    args.output_dir = f"_truth_torch_framework/run_output/{args.run}"
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected.")
        
    
    ### === Run Steps ============================
    
    set_seed(args.seed)
    truth_generation(args)
    
    
    # python _truth_torch_framework/run/run_framework.py

