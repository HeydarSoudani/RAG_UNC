#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import pickle
import sklearn
import argparse
import numpy as np
import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from utils.utils import set_seed, uncertainty_to_confidence_min_max, uncertainty_to_confidence_gaussian, uncertainty_to_confidence_sigmoid, uncertainty_to_confidence_tanh
from metrics.calibration import plugin_RCE_est, indication_diagram
from metrics.calibration import ECE_estimate


def get_calibration_results(args):
    print("\n--- Step 6: Get Calibration Results ...")
    print(f"""
        Model name:   {args.model}
        Dataset:      {args.dataset}
        Prompt (1st): {args.main_prompt_format}
        Prompt (2ed): {args.second_prompt_format}
        Run id:       {args.run_id}
        Seed:         {args.seed}
    """.replace('        ', ''))
    

    # === Define output files ===================
    model = args.model.split('/')[-1]
    base_dir_output = f'{args.output_dir}/{args.dataset}/{args.run_id}'
    calibration_output_file = f'{base_dir_output}/{args.main_prompt_format}/calibration_results/{model}_{args.temperature}_calibration_results.jsonl'
    
    calibration_output_dir = os.path.dirname(calibration_output_file)
    os.makedirs(calibration_output_dir, exist_ok=True)


    # === Define input files ====================
    correctness_input_file = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_correctness.pkl'
    entropy_input_file = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_new_uncertainty__sec_{args.second_prompt_format}.pkl'

    with open(entropy_input_file, 'rb') as f:
        entropy_dict  = pickle.load(f)
    with open(correctness_input_file, 'rb') as f:
        correctness_dict  = pickle.load(f)
    
    # === 
    # 
    entropy_df = pd.DataFrame.from_dict(entropy_dict, orient='index')
    entropy_df['id'] = entropy_df.index
    # 
    correctness_df = pd.DataFrame(correctness_dict)
    correctness_keys_to_use = ('id', 'bem_score', 'bert_score', 'exact_match', 'rouge_score')
    correctness_small = dict((k, correctness_df[k]) for k in correctness_keys_to_use)
    correctness_df = pd.DataFrame.from_dict(correctness_small)

    result_df = entropy_df.merge(correctness_df, on='id')
    print(result_df)

    def get_correctness(results):
        correctness_results = {}
        correctness_results['selected_metric'] = args.accuracy_metric
        
        if args.accuracy_metric in ['bem_score', 'gpt_score', 'exact_match']:
            correctness_bin = (results[args.accuracy_metric] > args.roc_auc_threshold).astype('int') 
        elif args.accuracy_metric == 'bert_score':
            correctness_bin = (results[args.accuracy_metric].apply(lambda x: x['F1']) > args.roc_auc_threshold).astype('int') 
        elif args.accuracy_metric == 'rouge_score':
            correctness_bin = (results[args.accuracy_metric].apply(lambda x: x['rougeL']) > args.roc_auc_threshold).astype('int') 
        correctness_results['accuracy'] = correctness_bin.mean()
        
        # non-binarized accuracy
        correctness_results['exact_match_mean'] = results['exact_match'].mean()
        correctness_results['bem_score_mean'] = results['bem_score'].mean()
        correctness_results['bert_score_mean'] = results['bert_score'].apply(lambda x: x['F1']).mean()
        correctness_results['rougeL_score_mean'] = results['rouge_score'].apply(lambda x: x['rougeL']).mean()
        if args.accuracy_metric in ['bem_score', 'gpt_score']:
            one_minus_correctness = 1 - results[args.accuracy_metric]
        elif args.accuracy_metric == 'rouge_score':
            one_minus_correctness = 1 - results[args.accuracy_metric].apply(lambda x: x['rougeL'])
        elif args.accuracy_metric == 'bert_score':
            one_minus_correctness = 1 - results[args.accuracy_metric].apply(lambda x: x['F1'])
        elif args.accuracy_metric == 'exact_match':
            one_minus_correctness = 1 - results[args.accuracy_metric].astype('int') 
        
        return correctness_results, correctness_bin, one_minus_correctness


    # === Save the calibration result ============
    result_dict = {}
    uncertainty_model = 'entropy'
    uncertainty_key = 'entropy'
    result_dict[uncertainty_model]= {}
    
    correctness_results, correctness_bin, one_minus_correctness = get_correctness(result_df) 
    correctness = 1 - np.array(one_minus_correctness)
    result_dict['correctness'] = correctness_results
    
    uncertainty_values = result_df[uncertainty_key]
    
    result_dict[uncertainty_model]["AUROC"] = sklearn.metrics.roc_auc_score(1 - correctness_bin, uncertainty_values)

    print(result_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--dataset', type=str, default='nqgold', choices=[
        'trivia', 'nq', 'squad1', 'webquestions',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa_org', 'topicoqa_his', 'topicoqa_rw',
        'nqgold'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--main_prompt_format', type=str, default='q_positive', choices=[
        'only_q', 'q_positive', 'q_negative'
    ])
    parser.add_argument('--second_prompt_format', type=str, default='only_q', choices=[
        'only_q', 'q_positive', 'q_negative'
    ])
    
    parser.add_argument('--accuracy_metric', type=str, default="exact_match", choices=[
        'bem_score', 'exact_match', 'bert_score', 'rouge_score', 'llama3_score', 'gpt_score'
    ])
    parser.add_argument('--model_llama_eval', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.01)
    parser.add_argument("--roc_auc_threshold", type=float, default=0.8)
    parser.add_argument("--output_file_postfix", type=str, default="")
    
    parser.add_argument('--num_generations_per_prompt', type=int, default=10)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--type_of_question', type=str)
    parser.add_argument('--decoding_method', type=str, default='beam_search')
    parser.add_argument('--temperature', type=float, default='1.0')
    parser.add_argument('--num_beams', type=int, default='1')
    parser.add_argument('--top_p', type=float, default=1.0)
    
    parser.add_argument('--generation_type', type=str, default='normal', choices=['normal', 'cad'])
    # parser.add_argument('--with_groundedness', type=str, default='yes', choices=['no', 'yes'])
    parser.add_argument('--affinity_mode', type=str, default='disagreement')
    parser.add_argument('--run_id', type=str, default='run_0')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    
    args.output_dir = "framework/run_output"
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected.")
    
    if args.main_prompt_format != 'only_q':
        args.second_prompt_format == 'only_q'
    
    set_seed(args.seed)
    get_calibration_results(args)
    
    # python framework/run/test_faegheh_idea_calibration.py
    
    