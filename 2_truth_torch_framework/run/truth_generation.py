#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import wandb
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

import TruthTorchLM as ttlm
from utils.utils import set_seed
from datasets_ import single_hop




def truth_generation(args):
    
    # === Output files ==========================
    model_ = args.model.split('/')[-1]
    overall_results_output_file = f'{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.prompt_format}/overall_results.jsonl'
    
    # === Generation Model ======================
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    
    # === UE Models =============================
    n_generations = 10
    pt = ttlm.truth_methods.PTrue()
    confidence = ttlm.truth_methods.Confidence()
    pe = ttlm.truth_methods.Entropy(number_of_generations=n_generations)
    se = ttlm.truth_methods.SemanticEntropy()
    mars = ttlm.truth_methods.MARS()
    sar = ttlm.truth_methods.SAR()
    sd = ttlm.truth_methods.SelfDetection(number_of_questions=5)
    
    ns = ttlm.truth_methods.NumSemanticSetUncertainty()
    eigv = ttlm.truth_methods.SumEigenUncertainty()
    ecc = ttlm.truth_methods.EccentricityUncertainty()
    deg = ttlm.truth_methods.MatrixDegreeUncertainty()
    
    truth_methods = [pt, confidence, pe, se, mars, sar, sd, ns, eigv, ecc, deg]
    
    
    # === Setup dataset =========================
    dataset = single_hop.get_dataset(args.prompt_format, args.dataset, args.subsec, args.fraction_of_data_to_use)
    if args.prompt_format == "only_q":
        with_rag, user_prompt = False, ttlm.templates.DEFAULT_USER_PROMPT
    else:
        with_rag, user_prompt = True, ttlm.templates.DEFAULT_RAG_USER_PROMPT
    
    sample_index = 0
    print(f"Dataset example {sample_index}:")
    print(f"Id:               {dataset[sample_index]['qid']}")
    print(f"Question:         {dataset[sample_index]['question']}")
    print(f"Answers:          {dataset[sample_index]['ground_truths']}")
    print(f"Context:         \n{dataset[sample_index]['context']}")
    
    # === Generation ============================
    wandb_run = wandb.init(
        project="ue_truthtorch",
        entity="heydar-soudani",
        mode="disabled"
    )

    results = ttlm.evaluate_truth_method(
        dataset=dataset,
        user_prompt=user_prompt,
        # size_of_data=5,
        with_rag=with_rag,
        model=model,
        truth_methods=truth_methods,
        eval_metrics=['auroc', 'spearman', 'auprc', 'prr'],
        tokenizer=tokenizer,
        
        correctness_evaluator=ttlm.evaluators.ExactMatch(),
        max_new_tokens=64,
        seed=args.seed,
        wandb_run=wandb_run,
        # wandb_push_method_details=True
    )
    
    # === Write in the file ====================
    reuslts_dict = {}
    reuslts_dict['correctness'] = {
        'metric': args.accuracy_metric,
        'accuracy': sum(results['output_dict']['generation_correctness'])/len(results['output_dict']['generation_correctness'])
    }
    for i in range(len(results['eval_list'])):
        reuslts_dict[results['output_dict']['truth_methods'][i]] = {
            "Unc_mean": np.mean(results['output_dict'][f'truth_method_{i}']['truth_values']).item(),
            "AUROC": results['eval_list'][i]['auroc'].item(),
            "Spearman": results['eval_list'][i]['spearman'].item(),
            "AUPRC": results['eval_list'][i]['auprc'].item(),
            "PRR": results['eval_list'][i]['prr'].item(),
        }
    
    # === Save Output ===========================
    os.makedirs(os.path.dirname(overall_results_output_file), exist_ok=True)
    with open(overall_results_output_file, 'w') as file:
        json.dump(reuslts_dict, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--dataset', type=str, default='nqgold', choices=[
        'nqgold', 'nqswap', 'trivia', 'popqa',
        'webquestions', 'squad1', 'nq',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa',
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--prompt_format', type=str, default='bm25_retriever_top1', choices=[
        'only_q', 'q_positive', 'q_negative', 'q_conflict',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'contriever_retriever_top1', 'contriever_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    parser.add_argument('--accuracy_metric', type=str, default="exact_match", choices=[
        'exact_match', 'bem_score', 'bert_score', 'rouge_score', 'llama3_score', 'gpt_score'
    ])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.003)
    parser.add_argument("--roc_auc_threshold", type=float, default=0.8)
    parser.add_argument('--num_generations_per_prompt', type=int, default=10)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--decoding_method', type=str, default='beam_search')
    parser.add_argument('--temperature', type=float, default='1.0')
    parser.add_argument('--num_beams', type=int, default='1')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    
    ### === Define CUDA device =================== 
    args.output_dir = "2_truth_torch_framework/run_output"
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
    
    
    # python 2_truth_torch_framework/run/truth_generation.py
