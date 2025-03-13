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
from datasets_ import short_form


def truth_generation(args):
    
    print("\n== Generation with truthfulness ...")
    print(f"""
        Model name:  {args.model}
        Dataset:     {args.dataset}/{args.subsec} ({args.fraction_of_data_to_use})
        Prompt:      {args.prompt_format}
        Correctness: {args.accuracy_metric}
        Seed:        {args.seed}
    """.replace('        ', ''))
    
    # === Output files ==========================
    model_ = args.model.split('/')[-1]
    generations_output_file = f'{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.prompt_format}/generations.jsonl'
    uncertainties_output_file = f'{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.prompt_format}/uncertainties.jsonl'
    overall_results_output_file = f'{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.prompt_format}/overall_results.jsonl'
    
    
    # === Generation Model ======================
    # model = "gpt-4o"
    # model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    
    # === UE Models =============================
    # White-box
    pt = ttlm.truth_methods.PTrue()
    cnf = ttlm.truth_methods.Confidence()
    pe = ttlm.truth_methods.Entropy(number_of_generations=args.num_generations)
    se = ttlm.truth_methods.SemanticEntropy()
    mars = ttlm.truth_methods.MARS()
    lars_co = ttlm.truth_methods.LARS(ue_type='confidence')
    sar = ttlm.truth_methods.SAR()
    # Black-box
    nums = ttlm.truth_methods.NumSemanticSetUncertainty()
    eigv = ttlm.truth_methods.SumEigenUncertainty()
    ecc = ttlm.truth_methods.EccentricityUncertainty()
    deg = ttlm.truth_methods.MatrixDegreeUncertainty()
    verb = ttlm.truth_methods.VerbalizedConfidence()
    inside = ttlm.truth_methods.Inside()
    kere = ttlm.truth_methods.KernelLanguageEntropy()
    
    truth_methods_name = [
        'Pt', 'Conf', 'PE', 'SE', 'MARS', 'SAR', 'LARS_Co', 'INS',
        'NumS', 'EigV', 'ECC', 'Deg', 'Verb', 'KerE'
    ]
    truth_methods = [
        pt, cnf, pe, se, mars, sar, lars_co, inside,
        nums, eigv, ecc, deg, verb, kere
    ]
    
    # === Correctness Evaluator =================
    if args.accuracy_metric == "exact_match":
        correctness_evaluator = ttlm.evaluators.ExactMatch()    
    
    elif args.accuracy_metric == "model_judge":
        if 'gpt' in args.model_eval:
            correctness_evaluator = ttlm.evaluators.ModelJudge(model=args.model_eval, num_retries=3)
        else:
            # correctness_evaluator = ttlm.evaluators.ModelJudge(model=model, tokenizer=tokenizer, num_retries=3)
            # model_eval = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
            model_eval = AutoModelForCausalLM.from_pretrained(args.model_eval, torch_dtype=torch.bfloat16, device_map='auto')
            tokenizer_eval = AutoTokenizer.from_pretrained(args.model_eval, use_fast=False)
            correctness_evaluator = ttlm.evaluators.ModelJudge(model=model_eval, tokenizer=tokenizer_eval, num_retries=3)
    
    
    # === Setup dataset =========================
    is_fewshot = False
    dataset = short_form.get_dataset(args.prompt_format, args.dataset, args.subsec, args.fraction_of_data_to_use)
    if args.prompt_format == "only_q":
        with_rag = False
        user_prompt = short_form.get_few_shot_user_prompt(args.dataset, with_context=with_rag) if is_fewshot else ttlm.templates.DEFAULT_USER_PROMPT
    else:
        with_rag = True
        user_prompt = short_form.get_few_shot_user_prompt(args.dataset, with_context=with_rag) if is_fewshot else ttlm.templates.DEFAULT_RAG_USER_PROMPT
    
    sample_index = 0
    print(f"Dataset example {sample_index}:")
    print(f"Id:               {dataset[sample_index]['qid']}")
    print(f"Question:         {dataset[sample_index]['question']}")
    print(f"Answers:          {dataset[sample_index]['ground_truths']}")
    print(f"Context:        \n{dataset[sample_index]['context']}")
    
    
    # === Generation ============================
    wandb_run = wandb.init(
        project="ue_truthtorch",
        entity="heydar-soudani",
        mode="disabled"
    )

    results = ttlm.evaluate_truth_method(
        dataset=dataset,
        user_prompt=user_prompt,
        with_rag=with_rag,
        model=model,
        truth_methods=truth_methods,
        eval_metrics=['auroc', 'spearman', 'auprc', 'prr'],
        tokenizer=tokenizer,
        correctness_evaluator=correctness_evaluator,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        wandb_run=wandb_run
        # wandb_push_method_details=True
    )
    
    
    # === Save Generations =====================
    os.makedirs(os.path.dirname(generations_output_file), exist_ok=True)
    with open(generations_output_file, 'w') as file:
        for i in range(len(results['output_dict']['generation_correctness'])):
            data = {
                "qid": results['output_dict']['qid'][i],
                "question": results['output_dict']['question_text'][i],
                "ground_truths": results['output_dict']['ground_truths'][i],
                "correctness": results['output_dict']['generation_correctness'][i],
                "generation_text_most_likely": results['output_dict']['generation'][i],
                "samples_generation_text": results['output_dict']['samples_generated_text'][i],
                # "samples_generation_token": results['output_dict']['samples_generated_token'][i],
                # "samples_logprobs": results['output_dict']['samples_logprobs'][i],
                "samples_probs": [torch.exp(torch.tensor(item)).tolist() for item in results['output_dict']['samples_logprobs'][i]]
            }
            file.write(json.dumps(data) + '\n')
    
    # === Save Uncertainty val. =================
    os.makedirs(os.path.dirname(uncertainties_output_file), exist_ok=True)
    with open(uncertainties_output_file, 'w') as file:
        for i in range(len(dataset)):
            
            unc_values = {}
            for j in range(len(truth_methods)):
                unc_values[truth_methods_name[j]] = results['output_dict'][f'truth_method_{j}']['truth_values'][i]
            
            data = {
                "qid": results['output_dict']['qid'][i],
                "question": results['output_dict']['question_text'][i],
                "ground_truths": results['output_dict']['ground_truths'][i],
                "correctness": results['output_dict']['generation_correctness'][i],
                **unc_values
            }
        
            file.write(json.dumps(data) + '\n')
    
    # === Save Evaluation =======================
    reuslts_dict = {}
    
    correctness_values = [max(0, val) for val in results['output_dict']['generation_correctness']]
    reuslts_dict['correctness'] = {
        'metric': args.accuracy_metric,
        'accuracy': sum(correctness_values) / len(correctness_values)
    }
    
    for i in range(len(results['eval_list'])):    
        truth_method = results['output_dict']['truth_methods'][i]
        truth_values = np.array(results['output_dict'][f'truth_method_{i}']['truth_values'])
        if truth_method == "VerbalizedConfidence":
            truth_values = truth_values[truth_values <= 1.0]
        
        reuslts_dict[truth_method] = {
            "Unc_mean": np.nanmean(truth_values).item(),
            "AUROC": np.array(results['eval_list'][i]['auroc']).item(),
            "Spearman": np.array(results['eval_list'][i]['spearman']).item(),
            "AUPRC": np.array(results['eval_list'][i]['auprc']).item(),
            "PRR": np.array(results['eval_list'][i]['prr']).item(),
        }
    
    os.makedirs(os.path.dirname(overall_results_output_file), exist_ok=True)
    with open(overall_results_output_file, 'w') as file:
        json.dump(reuslts_dict, file, indent=4)
    print(f"Results are saved in: {overall_results_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='nqgold', choices=[
        'nqgold', 'trivia', 'popqa',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'webquestions', 'squad1', 'nq', 'nqswap',
        'topicoqa',
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--prompt_format', type=str, default='only_q', choices=[
        'only_q', 'q_positive', 'q_negative', 'q_conflict',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'contriever_retriever_top1', 'contriever_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    parser.add_argument('--accuracy_metric', type=str, default="exact_match", choices=[
        'exact_match', 'model_judge', 'bem_score', 'bert_score', 'rouge_score'
    ])
    parser.add_argument('--model_eval', type=str, default='gpt-3.5-turbo') # meta-llama/Llama-3.1-8B-Instruct
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.01)
    parser.add_argument("--roc_auc_threshold", type=float, default=0.8)
    parser.add_argument('--num_generations', type=int, default=10)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--decoding_method', type=str, default='beam_search')
    parser.add_argument('--temperature', type=float, default='1.0')
    parser.add_argument('--num_beams', type=int, default='1')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_1 (300s-EM)')
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
    
    
    # python _truth_torch_framework/run/truth_generation.py
