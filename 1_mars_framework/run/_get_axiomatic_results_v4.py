#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import random
import pickle
import sklearn
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
# from minicheck.minicheck import MiniCheck

from utils.utils import set_seed, uncertainty_to_confidence_min_max
from metrics.calibration import ECE_estimate

UNC_THERESHOLD = 1000


def get_axiomatic_results(args):
    print("\n--- Step 6: Get Axiomatic Results V4 ...")
    print(f"""
        Model name:   {args.model}
        Dataset:      {args.dataset} / {args.subsec}
        Prompt (1st): {args.main_prompt_format}
        Prompt (2ed): {args.second_prompt_format}
        Run id:       {args.run_id}
        Seed:         {args.seed}
    """.replace('        ', ''))
    
    # === Define output files ===================
    # archive_500samples
    model = args.model.split('/')[-1]
    generation_type = f"prob_alpha_{str(args.alpha_probability)}"
    base_dir = f'{args.output_dir}/{args.dataset}/{args.run_id}/'
    
    # === Load semantic model ===================
    semantic_model_name = "microsoft/deberta-large-mnli"
    semantic_model = AutoModelForSequenceClassification.from_pretrained(semantic_model_name).to(args.device)
    semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
    semantic_model.eval()
    
    # === Functions =============================
    keys_mapping = {
        'main_prompt': {
            'PE': 'average_predictive_entropy_main_prompt',
            'SE': 'predictive_entropy_over_concepts_main_prompt',
            'PE_MARS': 'average_predictive_entropy_importance_max_main_prompt',
            'SE_MARS': 'predictive_entropy_over_concepts_importance_max_main_prompt'
        },
        'second_prompt': {
            'PE': 'average_predictive_entropy_second_prompt',
            'SE': 'predictive_entropy_over_concepts_second_prompt',
            'PE_MARS': 'average_predictive_entropy_importance_max_second_prompt',
            'SE_MARS': 'predictive_entropy_over_concepts_importance_max_second_prompt'
        },
        'third_prompt': {
            'PE': 'average_predictive_entropy_third_prompt',
            'SE': 'predictive_entropy_over_concepts_third_prompt',
            'PE_MARS': 'average_predictive_entropy_importance_max_third_prompt',
            'SE_MARS': 'predictive_entropy_over_concepts_importance_max_third_prompt'
        },
        'forth_prompt': {
            'PE': 'average_predictive_entropy_forth_prompt',
            'SE': 'predictive_entropy_over_concepts_forth_prompt',
            'PE_MARS': 'average_predictive_entropy_importance_max_forth_prompt',
            'SE_MARS': 'predictive_entropy_over_concepts_importance_max_forth_prompt'
        },
        # 'fifth_prompt': {
        #     'PE': 'average_predictive_entropy_forth_prompt',
        #     'SE': 'predictive_entropy_over_concepts_forth_prompt',
        #     'PE_MARS': 'average_predictive_entropy_importance_max_forth_prompt',
        #     'SE_MARS': 'predictive_entropy_over_concepts_importance_max_forth_prompt'
        # }
         
    }
    
    def create_result_df(main_prompt_format, second_prompt_format):
        
        results_dir = f'{base_dir}/{main_prompt_format}__{second_prompt_format}'
        if not os.path.isdir(results_dir):
            temp = 'bm25_retriever_top1' if args.dataset == 'popqa' else 'q_positive'
            results_dir = f'{base_dir}/{main_prompt_format}__{temp}'
        
        generation_file = f'{results_dir}/{model}_cleaned_generation_{args.generation_type}.pkl'
        similarities_input_file = f'{results_dir}/{model}_similarities_generation.pkl'
        correctness_input_file = f'{results_dir}/{model}_correctness.pkl'
        likelihoods_input_file = f'{results_dir}/{generation_type}/{model}_uncertainty_mars_generation.pkl'
        
        with open(generation_file, 'rb') as infile:
            cleaned_sequences = pickle.load(infile)
        with open(similarities_input_file, 'rb') as f:
            similarities_dict = pickle.load(f)
        with open(likelihoods_input_file, 'rb') as f:
            likelihoods_results  = pickle.load(f)
        with open(correctness_input_file, 'rb') as f:
            correctness_results  = pickle.load(f)
        
        
        # === Read data ============================
        # 
        similarities_df = pd.DataFrame.from_dict(similarities_dict, orient='index')
        similarities_df['id'] = similarities_df.index
        similarities_df['has_semantically_different_answers'] = similarities_df['has_semantically_different_answers'].astype('int')

        # 
        generations_df = pd.DataFrame(cleaned_sequences)
        generations_df['length_of_most_likely_generation'] = generations_df['most_likely_generation'].apply(
            lambda x: len(str(x).split(' ')))
        generations_df['variance_of_length_of_generations'] = generations_df['generated_texts'].apply(
            lambda x: np.var([len(str(y).split(' ')) for y in x]))
        
        # 
        correctness_df = pd.DataFrame(correctness_results)
        correctness_keys_to_use = ('id', 'bem_score', 'bert_score', 'exact_match', 'rouge_score')
        correctness_small = dict((k, correctness_df[k]) for k in correctness_keys_to_use)
        correctness_df = pd.DataFrame.from_dict(correctness_small)
        
        # 
        keys_to_use = (
            'ids',
            'average_predictive_entropy_main_prompt', 'predictive_entropy_over_concepts_main_prompt',
            'average_predictive_entropy_importance_max_main_prompt', 'predictive_entropy_over_concepts_importance_max_main_prompt',
            'average_predictive_entropy_second_prompt', 'predictive_entropy_over_concepts_second_prompt',
            'average_predictive_entropy_importance_max_second_prompt', 'predictive_entropy_over_concepts_importance_max_second_prompt',
            'average_predictive_entropy_third_prompt', 'predictive_entropy_over_concepts_third_prompt',
            'average_predictive_entropy_importance_max_third_prompt', 'predictive_entropy_over_concepts_importance_max_third_prompt',
            'average_predictive_entropy_forth_prompt', 'predictive_entropy_over_concepts_forth_prompt',
            'average_predictive_entropy_importance_max_forth_prompt', 'predictive_entropy_over_concepts_importance_max_forth_prompt',
            # 'average_predictive_entropy_fifth_prompt', 'predictive_entropy_over_concepts_fifth_prompt',
            # 'average_predictive_entropy_importance_max_fifth_prompt', 'predictive_entropy_over_concepts_importance_max_fifth_prompt',
        )
        likelihoods = likelihoods_results
        likelihoods_small = dict((k, likelihoods[k]) for k in keys_to_use)
        for key in likelihoods_small:
            if key == 'average_predictive_entropy_on_subsets':
                likelihoods_small[key].shape
            if type(likelihoods_small[key]) is torch.Tensor:
                likelihoods_small[key] = torch.squeeze(likelihoods_small[key].cpu())
        likelihoods_df = pd.DataFrame.from_dict(likelihoods_small)
        likelihoods_df.rename(columns={'ids': 'id'}, inplace=True) 
        
        # 
        # groundedness_df = pd.DataFrame(groundedness_results)

        # 
        result_df = generations_df.merge(similarities_df, on='id').merge(likelihoods_df, on='id').merge(correctness_df, on='id') # .merge(groundedness_df, on='id')
        result_df['len_most_likely_generation_length'] = result_df['most_likely_generation'].apply(lambda x: len(x.split()))
        return result_df
    
    def run_axiomatic_metrics(prompt_order):
        
        for axiom_num in ['1', '2']:
            print(f"= Axiom: {axiom_num} =======")
            
            if axiom_num == '1':
                selected_main_prompt_df = result_df_main_prompt[(result_df_main_prompt["exact_match"] == True)]
                selected_second_prompt_df = result_df_second_prompt[(result_df_second_prompt["exact_match"] == True)]
                
                selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_main_prompt_df['id'].tolist())]
                selected_second_prompt_correct_df = selected_second_prompt_df[selected_second_prompt_df["exact_match"] == True]
                selected_second_prompt_notcorrect_df = selected_second_prompt_df[selected_second_prompt_df["exact_match"] == False]
                print(f'Axiom 1 -> Main {len(selected_main_prompt_df)}')
                print(f'Axiom 1 -> 2ed, c: {len(selected_second_prompt_correct_df)}')
                print(f'Axiom 1 -> 2ed n: {len(selected_second_prompt_notcorrect_df)}')
                
            elif axiom_num == '2':
                selected_main_prompt_df = result_df_main_prompt[(result_df_main_prompt["exact_match"] == False)]
                selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_main_prompt_df['id'].tolist())]
                selected_second_prompt_correct_df = selected_second_prompt_df[selected_second_prompt_df["exact_match"] == True]
                selected_second_prompt_notcorrect_df = selected_second_prompt_df[selected_second_prompt_df["exact_match"] == False]
                print(f'Axiom 2 -> Main {len(selected_main_prompt_df)}')
                print(f'Axiom 2 -> 2ed, c: {len(selected_second_prompt_correct_df)}')
                print(f'Axiom 2 -> 2ed n: {len(selected_second_prompt_notcorrect_df)}')

                
            for uncertainty_model in ['PE']: # 'PE', 'SE', 'PE_MARS', 'SE_MARS'
                unc_model_key_main_prompt = keys_mapping[f'{prompt_order}_prompt'][uncertainty_model]
                unc_model_key_second_prompt = keys_mapping['main_prompt'][uncertainty_model]
                
                # Get Uncertainty
                uncertainty_values_main_prompt =  selected_main_prompt_df[unc_model_key_main_prompt]
                uncertainty_values_second_prompt_correct = selected_second_prompt_correct_df[unc_model_key_second_prompt]
                uncertainty_values_second_prompt_notcorrect = selected_second_prompt_notcorrect_df[unc_model_key_second_prompt]
                
                uncertainty_values_main_prompt_filtered =  uncertainty_values_main_prompt[uncertainty_values_main_prompt<UNC_THERESHOLD]
                uncertainty_values_second_prompt_correct_filtered = uncertainty_values_second_prompt_correct[uncertainty_values_second_prompt_correct<UNC_THERESHOLD] 
                uncertainty_values_second_prompt_notcorrect_filtered = uncertainty_values_second_prompt_notcorrect[uncertainty_values_second_prompt_notcorrect<UNC_THERESHOLD] 

                print(f"Axiom {axiom_num}, {uncertainty_model}")
                print(f"Unc. corr : {uncertainty_values_second_prompt_correct_filtered.mean():.3f} -> {uncertainty_values_main_prompt_filtered.mean():.3f}")
                print(f"Unc. ncor : {uncertainty_values_second_prompt_notcorrect_filtered.mean():.3f} -> {uncertainty_values_main_prompt_filtered.mean():.3f}")
                print('\n')             

    # ======  
    result_df_main_prompt = create_result_df(args.main_prompt_format, args.second_prompt_format)
    result_df_second_prompt = create_result_df(args.second_prompt_format, args.main_prompt_format)
     
    for prompt_order in ['main']: # 'second', 'third', 'forth'
        print(f"=== {prompt_order} ====================================")
        run_axiomatic_metrics(prompt_order)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--model_llama_eval', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='nqgold', choices=[
        'nqgold', 'nqswap', 'trivia', 'popqa',
        'webquestions', 'squad1', 'nq',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa',
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--main_prompt_format', type=str, default='bm25_retriever_top1', choices=[
        'only_q', 'q_positive', 'q_negative', 'q_negative',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    parser.add_argument('--second_prompt_format', type=str, default='only_q', choices=[
        'only_q', 'q_positive', 'q_negative', 'q_conflict',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    parser.add_argument('--accuracy_metric', type=str, default="exact_match", choices=[
        'exact_match', 'rouge_score', 'bert_score', 'bem_score', 'llama3_score', 'gpt_score'
    ])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
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
    parser.add_argument('--alpha_generation', type=float, default=0.5)
    parser.add_argument('--alpha_probability', type=float, default=0.5)
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
    get_axiomatic_results(args)
    
    # python framework/run/get_axiomatic_results_v4.py
    
    
    