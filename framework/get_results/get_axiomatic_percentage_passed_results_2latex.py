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
            axiomatic_variables_oe_input_file = f'{results_dir}/{generation_type}/axiomatic_variables_oe.pkl'
            axiomatic_variables_gn_input_file = f'{results_dir}/{generation_type}/axiomatic_variables_gn.pkl'
            axiomatic_variables_gk_input_file = f'{results_dir}/{generation_type}/axiomatic_variables_gk.pkl'
            axiomatic_variables_gm_input_file = f'{results_dir}/{generation_type}/axiomatic_variables_gm.pkl'
            with open(axiomatic_variables_oe_input_file, 'rb') as f:
                axiomatic_variables_oe_results  = pickle.load(f)
            with open(axiomatic_variables_gn_input_file, 'rb') as f:
                axiomatic_variables_gn_results  = pickle.load(f)
            with open(axiomatic_variables_gk_input_file, 'rb') as f:
                axiomatic_variables_gk_results  = pickle.load(f)
            with open(axiomatic_variables_gm_input_file, 'rb') as f:
                axiomatic_variables_gm_results  = pickle.load(f)
            
            axiomatic_variables_oe_df = pd.DataFrame(axiomatic_variables_oe_results)
            axiomatic_variables_gn_df = pd.DataFrame(axiomatic_variables_gn_results)
            axiomatic_variables_gk_df = pd.DataFrame(axiomatic_variables_gk_results)
            axiomatic_variables_gm_df = pd.DataFrame(axiomatic_variables_gm_results)
            axiomatic_variables_oe_df = pd.DataFrame.from_dict(dict((k, axiomatic_variables_oe_df[k]) for k in ('id', 'output_equality_em', 'output_equality_nli', 'axiom_num_correctness')))
            axiomatic_variables_gn_df = pd.DataFrame.from_dict(dict((k, axiomatic_variables_gn_df[k]) for k in ('id', 'groundedness_nli_main', 'groundedness_nli_second', 'axiom_num_nli')))
            axiomatic_variables_gk_df = pd.DataFrame.from_dict(dict((k, axiomatic_variables_gk_df[k]) for k in ('id', 'groundedness_kldiv_main', 'groundedness_kldiv_second')))
            axiomatic_variables_gm_df = pd.DataFrame.from_dict(dict((k, axiomatic_variables_gm_df[k]) for k in ('id', 'groundedness_minicheck_main', 'groundedness_minicheck_second', 'axiom_num_minicheck')))
            
            result_df = generations_df.merge(similarities_df, on='id')\
                .merge(uncertainty_mars_df, on='id')\
                .merge(uncertainty_bb_df, on='id')\
                .merge(correctness_df, on='id')\
                .merge(axiomatic_variables_oe_df, on='id')\
                .merge(axiomatic_variables_gn_df, on='id')\
                .merge(axiomatic_variables_gk_df, on='id')\
                .merge(axiomatic_variables_gm_df, on='id')
        
        else:
            result_df = generations_df.merge(similarities_df, on='id')\
                .merge(uncertainty_mars_df, on='id')\
                .merge(uncertainty_bb_df, on='id')\
                .merge(correctness_df, on='id')

        return result_df
    
    def get_axiomatic_coef(answer_equality_nli, nli_main, nli_sec, coefs=(0.33, 0.33, 0.33)):
        C1, C2, C3 = coefs[0], coefs[1], coefs[2]
        return C1*answer_equality_nli[1] + C2*nli_main[1] + C3*nli_sec[1]
        # return C1*answer_equality_nli[1] + C2*nli_main[1]
    
    def run_axiomatic_metrics(prompt_order, axiomatic_evalator, uncertainty_model, coefs=(0.33, 0.33, 0.33)):
        
        # === 1) Set inputs df =========================
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
            get_axiomatic_coef(answer_equality_nli, nli_main, nli_sec, coefs=coefs)
            for answer_equality_nli, nli_main, nli_sec in tqdm(zip(
                result_df_main_prompt['output_equality_nli'],
                result_df_main_prompt[f'groundedness_{axiomatic_evalator}_main'],
                result_df_main_prompt[f'groundedness_{axiomatic_evalator}_second']
            ), desc='Getting axiomatic coef. ...')
        ]
        
        filtered_df = result_df_main_prompt[result_df_main_prompt[f'axiom_num_correctness'].isin(['1', '2', '4', '5', 'others'])]
        mean_value = filtered_df['axiomatic_coef'].mean()
        std_value = filtered_df['axiomatic_coef'].std()
        C4 = 1.0 + mean_value
        print(f"axiomatic_coef -> mean: {mean_value}, std:{std_value}")
        result_df_main_prompt[f"{unc_model_key_main_prompt}_cal"] = (C4 - result_df_main_prompt['axiomatic_coef']) * result_df_main_prompt[unc_model_key_main_prompt]
        
        # ===
        result_df_main_prompt_filtered = result_df_main_prompt[result_df_main_prompt[unc_model_key_main_prompt] <UNC_THERESHOLD]
        total_rows = len(result_df_main_prompt_filtered)
        result = result_df_main_prompt_filtered.groupby(f'axiom_num_correctness').agg(
            accuracy=('exact_match', lambda x: x.sum() / len(x)),
            average_uncertainty=(unc_model_key_main_prompt, 'mean'),
            n_samples=(unc_model_key_main_prompt, 'count'),
            p_samples=(unc_model_key_main_prompt, lambda x: len(x) / total_rows),
            coef_mean=('axiomatic_coef', 'mean'),
            coef_unc_mean=(f'{unc_model_key_main_prompt}_cal', 'mean')
        ).reset_index()
        print(result)
        
        # === 3) Get values =========================
        axiom_outputs = []
        for axiom_num in ['1', '2', '4', '5']: # '1', '2', '4', '5', 'others'
            print(f"== Axiom: {axiom_num} ===")
            
            # === Get samples =======
            selected_main_prompt_df = result_df_main_prompt[result_df_main_prompt[f'axiom_num_correctness'] == axiom_num]
            selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_main_prompt_df['id'].tolist())]
            print(f'# Samples: {len(selected_main_prompt_df)}')
            print(f'# Samples: {len(selected_second_prompt_df)}')
            
            # === Get Uncertainty ===
            passing_percent = []
            for uncertainty_type in ['normal', 'calibrated']:
                uncertainty_values_main_prompt = (
                    selected_main_prompt_df[f"{unc_model_key_main_prompt}_cal"]
                    if uncertainty_type == 'calibrated'
                    else selected_main_prompt_df[unc_model_key_main_prompt]
                )
                uncertainty_values_second_prompt = selected_second_prompt_df[unc_model_key_second_prompt]            
                uncertainty_values_main_prompt = uncertainty_values_main_prompt.reset_index(drop=True)
                uncertainty_values_second_prompt = uncertainty_values_second_prompt.reset_index(drop=True)

                if axiom_num in ['1', '4']:
                    passing_percentage = (uncertainty_values_second_prompt > uncertainty_values_main_prompt).mean() * 100
                elif axiom_num in ['2', '5']:
                    passing_percentage = (uncertainty_values_second_prompt < uncertainty_values_main_prompt).mean() * 100
                passing_percent.append(passing_percentage)
                
            print('\n')
            
            axiom_outputs.append((
                f"{passing_percent[0]:.3f}",
                f"{passing_percent[1]:.3f}"
            ))
        
        return axiom_outputs
        
        
    # ===========================
    prompt_order = 'main' # main, 'second', 'third', 'forth'
    axiomatic_evalator = 'minicheck' # 'correctness', 'kldiv', 'nli', 'minicheck'
    
    
    # coefs = { # For Llama2
    #     'kldiv': (0.05, 0.75, 0.2),
    #     'nli': (0.1, 0.9, 0.0),
    #     'minicheck': (0.05, 0.95, 0.0)
    # }
    coefs = { # For Mistral
        'kldiv': (0.05, 0.75, 0.2),
        'nli': (0.05, 0.95, 0.0),
        'minicheck': (0.05, 0.95, 0.0)
    }
    
    result_df_main_prompt = create_result_df(args.main_prompt_format, args.second_prompt_format)    
    result_df_second_prompt = create_result_df(args.second_prompt_format, args.main_prompt_format)
    
    print(f"=== {prompt_order} ====================================")
    print(f"=== {axiomatic_evalator} =============================")
    print(f"Main: {len(result_df_main_prompt)}")
    print(f"2ed:  {len(result_df_second_prompt)}")
    
    dict_output = {}
    for uncertainty_model in ['PE', 'SE', 'PE_MARS', 'SE_MARS', 'spectral_u', 'ecc_u', 'degree_u']: 
        dict_output[uncertainty_model] = run_axiomatic_metrics(prompt_order, axiomatic_evalator, uncertainty_model, coefs=coefs[axiomatic_evalator])
        
    return dict_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.3')
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
    
    # if args.main_prompt_format != 'only_q':
    #     args.second_prompt_format == 'only_q'
    
    set_seed(args.seed)
    
    print("\n--- Step 6: Get Axiomatic Results (NLI) ...")
    print(f"""
        Model name:   {args.model}
        Run id:       {args.run_id}
        Seed:         {args.seed}
    """.replace('        ', ''))
    
    # 
    axiom2desc = {
        'Axiom 1': {"desc": "Positively Consistent", "mark": r"$\downarrow$"},
        'Axiom 2': {"desc": "Negatively Consistent", "mark": r"$\uparrow$"},
        'Axiom 3': {"desc": "Positively Changed", "mark": r"$\downarrow$"},
        'Axiom 4': {"desc": "Negatively Changed", "mark": r"$\uparrow$"},
    }
    unc2title = {
        'PE': 'PE',
        'SE': 'SE',
        'PE_MARS': 'PE+M',
        'SE_MARS': 'SE+M',
        'spectral_u': 'EigV',
        'ecc_u': 'ECC',
        'degree_u': 'Deg'
    }
    
    
    # variables to loop
    datasets = [
        {"dataset": 'nqgold', "subsec": 'test'},
        {"dataset": 'trivia', "subsec": 'dev'},
        {"dataset": 'popqa', "subsec": 'test'}
    ]
    retrievers = [ 
        'bm25_retriever_top1',
        'contriever_retriever_top1',
        'rerank_retriever_top1',
        'q_positive'
    ]
    uncertianty_methods = ['PE', 'SE', 'PE_MARS', 'SE_MARS', 'spectral_u', 'ecc_u', 'degree_u'] # 
    axioms = ['Axiom 1', 'Axiom 2', 'Axiom 3', 'Axiom 4']
    
    def get_color(input, axiom_title):
        sec_value, main_value, is_sig = input
        color, density = 'magenta', '20'
        
        if axiom_title in ['Axiom 1', 'Axiom 3']:
            if sec_value > main_value and is_sig:
                color, density = 'green', '10'
            elif sec_value > main_value and not is_sig:
                color, density = 'magenta', '5'
            elif sec_value < main_value and not is_sig:
                color, density = 'magenta', '10'
            elif sec_value < main_value and is_sig:
                color, density = 'magenta', '20'
                
        if axiom_title in ['Axiom 2', 'Axiom 4']:
            if sec_value > main_value and is_sig:
                color, density = 'magenta', '20'
            elif sec_value > main_value and not is_sig:
                color, density = 'magenta', '10'
            elif sec_value < main_value and not is_sig:
                color, density = 'magenta', '5'
            elif sec_value < main_value and is_sig:
                color, density = 'green', '10'
        
        return color, density
    
    dict2save = {}
    for axiom in axioms:
        dict2save[axiom] = {}
        for unc in uncertianty_methods:
            dict2save[axiom][unc] = []
    
    for dataset in datasets:
        args.dataset = dataset['dataset']
        args.subsec = dataset['subsec']
        
        for retriever in retrievers:
            
            # if args.dataset == 'popqa' and retriever=='q_positive':
            #     continue
            
            args.main_prompt_format = retriever
            outputs = get_axiomatic_results(args)
            
            for unc_method, axioms_value in outputs.items():
                for i, axiom_value in enumerate(axioms_value):
                    dict2save[f"Axiom {i+1}"][unc_method].append(axiom_value)
    
    
    print(dict2save)
    latex_table = ""    
    for axiom_title, axiom_values in dict2save.items():
        latex_table += fr"\multicolumn{{6}}{{l}}{{\textbf{{{axiom_title}:}} {axiom2desc[axiom_title]['desc']} {axiom2desc[axiom_title]['mark']}}} \\ \hline "
        for unc_title, unc_values in axiom_values.items():
            latex_table += f"{unc2title[unc_title]} &"
            for un_idx, unc_value in enumerate(unc_values):
                
                color = 'magenta' if (float(unc_value[1]) < float(unc_value[0])) else 'green'
                
                if un_idx == len(unc_values)-1:
                    latex_table += fr"\colorbox{{{color}!10}}{{{unc_value[0]} $\rightarrow$ {unc_value[1]}}} \\"
                else:
                    latex_table += fr"\colorbox{{{color}!10}}{{{unc_value[0]} $\rightarrow$ {unc_value[1]}}} &"
        
        latex_table += "\hline"
            
    

    axiomatic_evaluation_results = f"framework/get_results/files/axiomatic_evaluation_results_{args.model.split('/')[-1]}.txt"
    with open(axiomatic_evaluation_results, "w") as f:
        f.write(latex_table)
    
    # python framework/get_results/get_axiomatic_percentage_passed_results_2latex.py
    