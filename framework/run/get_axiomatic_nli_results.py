#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from utils.significant_testing import wilcoxon_test
from utils.utils import set_seed, uncertainty_to_confidence_min_max

UNC_THERESHOLD = 1000

def get_axiomatic_results(args):
    print("\n--- Step 6: Get Axiomatic Results (NLI) ...")
    print(f"""
        Model name:   {args.model}
        Dataset:      {args.dataset} / {args.subsec}
        Prompt (1st): {args.main_prompt_format}
        Prompt (2ed): {args.second_prompt_format}
        Run id:       {args.run_id}
        Seed:         {args.seed}
    """.replace('        ', ''))
    
    # === Define output files ===================
    model_ = args.model.split('/')[-1]
    base_dir = f'{args.output_dir}/{model_}/{args.dataset}/{args.subsec}/{args.run_id}/'
    generation_type = f"prob_alpha_{str(args.alpha_probability)}"
    
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
        # 'third_prompt': {
        #     'PE': 'average_predictive_entropy_third_prompt',
        #     'SE': 'predictive_entropy_over_concepts_third_prompt',
        #     'PE_MARS': 'average_predictive_entropy_importance_max_third_prompt',
        #     'SE_MARS': 'predictive_entropy_over_concepts_importance_max_third_prompt'
        # },
        # 'forth_prompt': {
        #     'PE': 'average_predictive_entropy_forth_prompt',
        #     'SE': 'predictive_entropy_over_concepts_forth_prompt',
        #     'PE_MARS': 'average_predictive_entropy_importance_max_forth_prompt',
        #     'SE_MARS': 'predictive_entropy_over_concepts_importance_max_forth_prompt'
        # },
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
        
        generation_file = f'{results_dir}/cleaned_generation_{args.generation_type}.pkl'
        similarities_input_file = f'{results_dir}/similarities_generation.pkl'
        correctness_input_file = f'{results_dir}/correctness.pkl'
        uncertainty_mars_input_file = f'{results_dir}/{generation_type}/uncertainty_mars_generation.pkl'
        uncertainty_bb_input_file = f'{results_dir}/uncertainty_bb_generation.pkl'
        axiomatic_variables_input_file = f'{results_dir}/{generation_type}/axiomatic_variables.pkl'
        
        with open(generation_file, 'rb') as infile:
            cleaned_sequences = pickle.load(infile)
        with open(similarities_input_file, 'rb') as f:
            similarities_dict = pickle.load(f)
        with open(correctness_input_file, 'rb') as f:
            correctness_results  = pickle.load(f)
        with open(uncertainty_mars_input_file, 'rb') as f:
            uncertainty_mars_results  = pickle.load(f)
        with open(uncertainty_bb_input_file, 'rb') as f:
            uncertainty_bb_results  = pickle.load(f)
        
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
            # 'average_predictive_entropy_third_prompt', 'predictive_entropy_over_concepts_third_prompt',
            # 'average_predictive_entropy_importance_max_third_prompt', 'predictive_entropy_over_concepts_importance_max_third_prompt',
            # 'average_predictive_entropy_forth_prompt', 'predictive_entropy_over_concepts_forth_prompt',
            # 'average_predictive_entropy_importance_max_forth_prompt', 'predictive_entropy_over_concepts_importance_max_forth_prompt',
            # 'average_predictive_entropy_fifth_prompt', 'predictive_entropy_over_concepts_fifth_prompt',
            # 'average_predictive_entropy_importance_max_fifth_prompt', 'predictive_entropy_over_concepts_importance_max_fifth_prompt',
        )
        uncertainty_mars = uncertainty_mars_results
        uncertainty_mars_small = dict((k, uncertainty_mars[k]) for k in keys_to_use)
        for key in uncertainty_mars_small:
            if key == 'average_predictive_entropy_on_subsets':
                uncertainty_mars_small[key].shape
            if type(uncertainty_mars_small[key]) is torch.Tensor:
                uncertainty_mars_small[key] = torch.squeeze(uncertainty_mars_small[key].cpu())
        uncertainty_mars_df = pd.DataFrame.from_dict(uncertainty_mars_small)
        uncertainty_mars_df.rename(columns={'ids': 'id'}, inplace=True) 
        # 
        uncertainty_bb_df = pd.DataFrame(uncertainty_bb_results)
        uncertainty_bb_keys_to_use = ('id', 'degree_u', 'ecc_u', 'spectral_u')
        uncertainty_bb_small = dict((k, uncertainty_bb_df[k]) for k in uncertainty_bb_keys_to_use)
        uncertainty_bb_df = pd.DataFrame.from_dict(uncertainty_bb_small)
        
        # 
        if main_prompt_format != 'only_q':
            with open(axiomatic_variables_input_file, 'rb') as f:
                axiomatic_variables_results  = pickle.load(f)
            axiomatic_variables_df = pd.DataFrame(axiomatic_variables_results)
            result_df = generations_df.merge(similarities_df, on='id').merge(uncertainty_mars_df, on='id').merge(uncertainty_bb_df, on='id').merge(correctness_df, on='id').merge(axiomatic_variables_df, on='id')
        else:
            result_df = generations_df.merge(similarities_df, on='id').merge(uncertainty_mars_df, on='id').merge(uncertainty_bb_df, on='id').merge(correctness_df, on='id')
        
        # 
        result_df['len_most_likely_generation_length'] = result_df['most_likely_generation'].apply(lambda x: len(x.split()))
        return result_df
    
    def get_axiomatic_coef(answer_equality_nli, nli_main, nli_sec, coefs=(0.33, 0.33, 0.33)):
        C1, C2, C3 = coefs[0], coefs[1], coefs[2]
        return C1*answer_equality_nli[1] + C2*nli_main[1] + C3*nli_sec[1]
        # return C1*answer_equality_nli[1] + C2*nli_main[1]
    
    def get_correctness(results):
        correctness_results = {}
        
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
        # correctness_results['rougeL_score_mean'] = results['rouge_score'].apply(lambda x: x['rougeL']).mean()
        if args.accuracy_metric in ['bem_score', 'gpt_score']:
            one_minus_correctness = 1 - results[args.accuracy_metric]
        elif args.accuracy_metric == 'rouge_score':
            one_minus_correctness = 1 - results[args.accuracy_metric].apply(lambda x: x['rougeL'])
        elif args.accuracy_metric == 'bert_score':
            one_minus_correctness = 1 - results[args.accuracy_metric].apply(lambda x: x['F1'])
        elif args.accuracy_metric == 'exact_match':
            one_minus_correctness = 1 - results[args.accuracy_metric].astype('int') 
        
        return correctness_results, correctness_bin, one_minus_correctness
    
    def run_axiomatic_metrics(prompt_order):
        
        for uncertainty_model in ['spectral_u']: # 'PE', 'SE', 'PE_MARS', 'SE_MARS', 'spectral_u', 'ecc_u', 'degree_u'
            
            # =============================================
            print(f"Unc. Model: {uncertainty_model}")
            if uncertainty_model in ['PE', 'SE', 'PE_MARS', 'SE_MARS']:
                unc_model_key_main_prompt = keys_mapping[f'{prompt_order}_prompt'][uncertainty_model]
                unc_model_key_second_prompt = keys_mapping['main_prompt'][uncertainty_model]
            elif uncertainty_model in ['degree_u', 'ecc_u', 'spectral_u']:
                unc_model_key_main_prompt = uncertainty_model
                unc_model_key_second_prompt = uncertainty_model
            
            if uncertainty_model == 'degree_u':
                result_df_main_prompt[unc_model_key_main_prompt] = result_df_main_prompt[unc_model_key_main_prompt] + 0.9
                result_df_second_prompt[unc_model_key_second_prompt] = result_df_second_prompt[unc_model_key_second_prompt] + 0.9
            
            
            # === 2) Get Axiomatic Coef. ==================
            result_df_main_prompt['axiomatic_coef'] = [
                get_axiomatic_coef(answer_equality_nli, nli_main, nli_sec, coefs=(0.1, 0.8, 0.1))
                for answer_equality_nli, nli_main, nli_sec in tqdm(zip(
                    result_df_main_prompt['answer_equality_nli'],
                    result_df_main_prompt['nli_relation_main'],
                    result_df_main_prompt['nli_relation_second']
                ), desc='Getting axiomatic coef. ...')
            ]
            
            filtered_df = result_df_main_prompt[result_df_main_prompt['axiom_num_nli'].isin(['1', '2', '4', '5', 'others'])]
            mean_value = filtered_df['axiomatic_coef'].mean()
            std_value = filtered_df['axiomatic_coef'].std()
            C4 = 1.0 + mean_value
            print(f"axiomatic_coef -> mean: {mean_value}, std:{std_value}")
            
            
            # ============================================                
            result_df_main_prompt[f"{unc_model_key_main_prompt}_cal"] = (C4 - result_df_main_prompt['axiomatic_coef']) * result_df_main_prompt[unc_model_key_main_prompt]
            
            result_df_main_prompt_filtered = result_df_main_prompt[result_df_main_prompt[unc_model_key_main_prompt] <UNC_THERESHOLD]
            total_rows = len(result_df_main_prompt_filtered)
            result = result_df_main_prompt_filtered.groupby('axiom_num_nli').agg(
                accuracy=('exact_match', lambda x: x.sum() / len(x)),
                average_uncertainty=(unc_model_key_main_prompt, 'mean'),
                n_samples=(unc_model_key_main_prompt, 'count'),
                p_samples=(unc_model_key_main_prompt, lambda x: len(x) / total_rows),
                coef_mean=('axiomatic_coef', 'mean'),
                coef_unc_mean=(f'{unc_model_key_main_prompt}_cal', 'mean')
            ).reset_index()
            print(result)
            
            # result = result_df_main_prompt_filtered.groupby('axiom_num_correctness').agg(
            #     true_ratio=('exact_match', lambda x: x.sum() / len(x)),
            #     average_uncertainty=(unc_model_key_main_prompt, 'mean'),
            #     row_count=(unc_model_key_main_prompt, 'count'),
            #     coef_mean=('axiomatic_coef', 'mean'),
            #     coef_unc_mean=(f'{unc_model_key_main_prompt}_cal', 'mean')
            # ).reset_index()
            # print(result)
            
            # ==========================================
            for axiom_num in ['1', '2', '4', '5', 'others']: # '1', '2', '4', '5', 'other'
                print(f"== Axiom: {axiom_num} ===")
                
                # === Get samples =======
                selected_main_prompt_df = result_df_main_prompt[result_df_main_prompt['axiom_num_nli'] == axiom_num]
                selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_main_prompt_df['id'].tolist())]
                print(f'# Samples: {len(selected_main_prompt_df)}')
                print(f'# Samples: {len(selected_second_prompt_df)}')
                
                # === Get Uncertainty ===
                for type_ in ['normal', 'calibrated']: # 
                    uncertainty_values_main_prompt = selected_main_prompt_df[f"{unc_model_key_main_prompt}_cal"] if type_ == 'calibrated' else selected_main_prompt_df[unc_model_key_main_prompt]
                    
                    uncertainty_values_second_prompt = selected_second_prompt_df[unc_model_key_second_prompt]    
                    uncertainty_values_main_prompt_filtered =  uncertainty_values_main_prompt[uncertainty_values_main_prompt<UNC_THERESHOLD]
                    uncertainty_values_second_prompt_filtered = uncertainty_values_second_prompt[uncertainty_values_second_prompt<UNC_THERESHOLD]
                    stat, p_value, is_significant = wilcoxon_test(uncertainty_values_main_prompt.tolist(), uncertainty_values_second_prompt.tolist())
                    print(f"Type: {type_}")
                    print(f"Uncertainty: {uncertainty_values_second_prompt_filtered.mean():.3f} -> {uncertainty_values_main_prompt_filtered.mean():.3f}")
                    print(f"Is it significant? {is_significant}")
                    print('\n')
                print('\n')
        
    # ======
    result_df_main_prompt = create_result_df(args.main_prompt_format, args.second_prompt_format)    
    result_df_second_prompt = create_result_df(args.second_prompt_format, args.main_prompt_format)
    
    for prompt_order in ['main']: # main, 'second', 'third', 'forth'
        print(f"=== {prompt_order} ====================================")
        print(f"Main: {len(result_df_main_prompt)}")
        print(f"2ed:  {len(result_df_second_prompt)}")
        run_axiomatic_metrics(prompt_order)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--dataset', type=str, default='trivia', choices=[
        'nqgold', 'trivia', 'popqa', 'nqswap',
        'webquestions', 'squad1', 'nq',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa',
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--main_prompt_format', type=str, default='q_positive', choices=[
        'only_q', 'q_positive', 'q_negative', 'q_conflict',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'contriever_retriever_top1', 'contriever_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    parser.add_argument('--second_prompt_format', type=str, default='only_q', choices=[
        'only_q', 'q_positive', 'q_negative', 'q_conflict',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'contriever_retriever_top1', 'contriever_retriever_top5',
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
    
    
    # python framework/run/get_axiomatic_nli_results.py
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # ===================================
    # === Grid search: For testing ......
    
    # === 1) Get Coefs.
    # values = np.arange(0, 1.05, 0.05)
    # valid_combinations = [(a, b) for a, b in itertools.product(values, repeat=2) if np.isclose(a + b, 1)]
    # best_coefs = None
    # best_auroc = float('-inf')

    # prompt_order = 'main'
    # uncertainty_model = 'PE'
    # unc_model_key_main_prompt = keys_mapping[f'{prompt_order}_prompt'][uncertainty_model]
    # filtered_df = result_df_main_prompt[result_df_main_prompt['axiom_num'].isin(['1', '2', '4', '5'])]
    # _, correctness_bin, _ = get_correctness(result_df_main_prompt)
    
    # for c1, c2 in tqdm(valid_combinations, desc="Searching coefs ..."):
    #     temp_axiomatic_coef = [
    #         get_axiomatic_coef(answer_equality_nli, nli_main, nli_sec, coefs=(c1, c2))
    #         for answer_equality_nli, nli_main, nli_sec in tqdm(zip(
    #             result_df_main_prompt['answer_equality_nli'],
    #             result_df_main_prompt['nli_relation_main'],
    #             result_df_main_prompt['nli_relation_second']
    #         ), desc='Getting axiomatic coef. ...')
    #     ]
    #     temp_axiomatic_coef_series = pd.Series(temp_axiomatic_coef, index=result_df_main_prompt.index)
        
    #     mean_value = filtered_df['axiomatic_coef'].mean()
    #     std_value = filtered_df['axiomatic_coef'].std()
        
    #     uncertainty_values = ((1.0+mean_value - temp_axiomatic_coef_series)*result_df_main_prompt[unc_model_key_main_prompt])
    #     temp_auroc = sklearn.metrics.roc_auc_score(1 - correctness_bin, uncertainty_values)
    
    #     if temp_auroc > best_auroc:
    #         best_auroc = temp_auroc
    #         best_coefs = (c1, c2)
    # print(f"Best parameters: {best_coefs} with score: {best_auroc}")

    
    
    
