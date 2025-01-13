#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.significant_testing import wilcoxon_test
from utils.utils import set_seed, uncertainty_to_confidence_min_max

UNC_THERESHOLD = 1000


def get_axiomatic_results(args):
    print("\n--- Step 6: Get Axiomatic Results V3 ...")
    print(f"""
        Model name:   {args.model}
        Dataset:      {args.dataset} / {args.subsec}
        Prompt (1st): {args.main_prompt_format}
        Prompt (2ed): {args.second_prompt_format}
        Run id:       {args.run_id}
        Seed:         {args.seed}
    """.replace('        ', ''))
    
    # === Define output files ===================
    model = args.model.split('/')[-1]
    base_dir = f'{args.output_dir}/{args.dataset}/{args.run_id}/'
    generation_type = f"prob_alpha_{str(args.alpha_probability)}"
    
    
    # === For getting equal outputs =============
    sequence_input_main = f'{base_dir}/{args.main_prompt_format}__{args.second_prompt_format}/{model}_cleaned_generation_{args.generation_type}.pkl'
    if os.path.isdir(f'{base_dir}/{args.second_prompt_format}__{args.main_prompt_format}'):
        sequence_input_secondry = f'{base_dir}/{args.second_prompt_format}__{args.main_prompt_format}/{model}_cleaned_generation_normal.pkl'
    else:
        temp = 'bm25_retriever_top1' if args.dataset == 'popqa' else 'q_positive'
        sequence_input_secondry = f'{base_dir}/{args.second_prompt_format}__{temp}/{model}_cleaned_generation_normal.pkl'
    
    with open(sequence_input_main, 'rb') as infile:
        sequences_main = pickle.load(infile)
    with open(sequence_input_secondry, 'rb') as infile:
        sequences_secondry = pickle.load(infile)
        
    
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
        result_df = generations_df.merge(similarities_df, on='id').merge(likelihoods_df, on='id').merge(correctness_df, on='id') # .merge(groundedness_df, on='id')
        result_df['len_most_likely_generation_length'] = result_df['most_likely_generation'].apply(lambda x: len(x.split()))
        return result_df
    
    def compute_answer_equality_em(sequence_1, sequence_2):
        sequence_2_ = {}
        for sample in sequence_2:
            sequence_2_[sample['id']] = sample
        
        semantically_similar_list = []
        semantically_not_similar_list = []
        with open(answers_equality_output_jsonl_file, 'w') as jl_ofile:
            for i, sample in tqdm(enumerate(sequence_1)):
                id_ = sample['id']
                is_equal = False
                seq1 = sample['cleaned_most_likely_generation'].strip()
                
                if id_ in sequence_2_:
                    seq2 = sequence_2_[id_]['cleaned_most_likely_generation'].strip()
        
                    # if seq1=='\n' or seq2=='\n':
                    #     is_equal = False
                    # elif seq1=='' or seq2=='':
                    #     is_equal = False
                    # else:
                    if seq1 == seq2 or seq1.lower() == seq2 or seq1.capitalize() == seq2:
                        is_equal = True
                    if seq2 == seq1 or seq2.lower() == seq1 or seq2.capitalize() == seq1:
                        is_equal = True
            
                    if is_equal:
                        semantically_similar_list.append(id_)
                    else:
                        semantically_not_similar_list.append(id_)

                else:
                    print(f"\nQuery {id_} is not common between two sequences !!!")
                    seq2 = ""
                    semantically_not_similar_list.append(id_)

                result_item = {
                    'id': id_,
                    'question': sample['question'],
                    'answers': sample['answers'],
                    'generation_seq_1': seq1,
                    'generation_seq_2': seq2,
                    'is_equal': is_equal
                }
                jl_ofile.write(json.dumps(result_item) + '\n')

        return semantically_similar_list, semantically_not_similar_list

    def run_axiomatic_metrics(prompt_order):
        
        ### === Step1: Check if answer1 is equal to answer2 ===
        def get_output_equality():
            if os.path.isfile(answers_equality_output_jsonl_file):
                print(f"{answers_equality_output_jsonl_file} exists.")
                answer_equal_list, answer_not_equal_list = [], []
                with open(answers_equality_output_jsonl_file, 'r') as file:
                    for line in file:
                        if line.strip():
                            item = json.loads(line)
                            if item['is_equal']:
                                answer_equal_list.append(item['id'])
                            else:
                                answer_not_equal_list.append(item['id'])
            else:
                print("Computing similarity ...")
                answer_equal_list, answer_not_equal_list = compute_answer_equality_em(sequences_main, sequences_secondry)
            
            print(f"Answer equal: {len(answer_equal_list)}")
            print(f"Answer not equal: {len(answer_not_equal_list)}")
            return answer_equal_list, answer_not_equal_list
        
        answer_equal_list, answer_not_equal_list = get_output_equality()

        ### === Step2: Compute Axioms =========================
        for uncertainty_model in ['PE']: # 'PE', 'SE', 'PE_MARS', 'SE_MARS'
            print(f"Unc. Model: {uncertainty_model}")
            unc_model_key_main_prompt = keys_mapping[f'{prompt_order}_prompt'][uncertainty_model]
            unc_model_key_second_prompt = keys_mapping['main_prompt'][uncertainty_model]
        
            all_axioms_ids = []
            for axiom_num in ['1', '2', '4', '5']:
                print(f"== Axiom: {axiom_num} ===")
                
                # Get samples
                if axiom_num == '1':
                    # selected_list_ = answer_equal_list
                    # answer_equal_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_)]
                    # selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_main_prompt_df['id'].tolist())]
                    correct_main_prompt_df = result_df_main_prompt[(result_df_main_prompt["exact_match"] == True)]
                    correct_second_prompt_df = result_df_second_prompt[(result_df_second_prompt["exact_match"] == True)]
                    selected_main_prompt_df = result_df_main_prompt[(result_df_main_prompt["exact_match"] == True) & (result_df_main_prompt['id'].isin(correct_second_prompt_df['id'].tolist()))]
                    selected_second_prompt_df = result_df_second_prompt[(result_df_second_prompt["exact_match"] == True) & (result_df_second_prompt['id'].isin(correct_main_prompt_df['id'].tolist()))]
                    print(f'# Samples: {len(selected_main_prompt_df)}')
                    print(f'# Samples: {len(selected_second_prompt_df)}')
                    all_axioms_ids.extend(selected_main_prompt_df['id'].tolist())
                    
                elif axiom_num == '2':
                    selected_list_ = answer_equal_list
                    answer_equal_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_)]
                    selected_main_prompt_df = answer_equal_main_prompt_df[(answer_equal_main_prompt_df["exact_match"] == False)]
                    selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_main_prompt_df['id'].tolist())]
                    print(f'# Samples: {len(selected_main_prompt_df)}')
                    print(f'# Samples: {len(selected_second_prompt_df)}')
                    all_axioms_ids.extend(selected_main_prompt_df['id'].tolist())
                
                elif axiom_num == '4':
                    selected_list_ = answer_not_equal_list
                    answer_not_equal_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_)]
                    selected_main_prompt_df = answer_not_equal_main_prompt_df[(answer_not_equal_main_prompt_df["exact_match"] == True)]
                    
                    selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_main_prompt_df['id'].tolist())]
                    selected_second_prompt_correct_df = selected_second_prompt_df[selected_second_prompt_df["exact_match"] == True]
                    selected_second_prompt_notcorrect_df = selected_second_prompt_df[selected_second_prompt_df["exact_match"] == False]
                    selected_main_prompt_notcorrect_df_ = selected_main_prompt_df[selected_main_prompt_df['id'].isin(selected_second_prompt_notcorrect_df['id'].tolist())] 
                    
                    print(f'Main:   {len(selected_main_prompt_notcorrect_df_)}')
                    print(f'2ed nc: {len(selected_second_prompt_notcorrect_df)}')
                    all_axioms_ids.extend(selected_main_prompt_notcorrect_df_['id'].tolist())

                elif axiom_num == '5':
                    selected_list_ = answer_not_equal_list
                    answer_not_equal_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_)]
                    selected_main_prompt_df = answer_not_equal_main_prompt_df[(answer_not_equal_main_prompt_df["exact_match"] == False)]
                
                    selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_main_prompt_df['id'].tolist())]
                    selected_second_prompt_correct_df = selected_second_prompt_df[selected_second_prompt_df["exact_match"] == True]
                    selected_second_prompt_notcorrect_df = selected_second_prompt_df[selected_second_prompt_df["exact_match"] == False]
                    selected_main_prompt_correct_df_ = selected_main_prompt_df[selected_main_prompt_df['id'].isin(selected_second_prompt_correct_df['id'].tolist())] 
                    
                    print(f'Main:  {len(selected_main_prompt_correct_df_)}')
                    print(f'2ed c: {len(selected_second_prompt_correct_df)}')
                    all_axioms_ids.extend(selected_main_prompt_correct_df_['id'].tolist())

                # Get Uncertainty
                if axiom_num in ['1', '2']:
                    uncertainty_values_main_prompt =  selected_main_prompt_df[unc_model_key_main_prompt]
                    uncertainty_values_second_prompt = selected_second_prompt_df[unc_model_key_second_prompt]
                    uncertainty_values_main_prompt_filtered =  uncertainty_values_main_prompt[uncertainty_values_main_prompt<UNC_THERESHOLD]
                    uncertainty_values_second_prompt_filtered = uncertainty_values_second_prompt[uncertainty_values_second_prompt<UNC_THERESHOLD]
                    stat, p_value, is_significant = wilcoxon_test(uncertainty_values_main_prompt.tolist(), uncertainty_values_second_prompt.tolist())
                    print(f"Uncertainty: {uncertainty_values_second_prompt_filtered.mean():.3f} -> {uncertainty_values_main_prompt_filtered.mean():.3f}")
                    print(f"Is it significant? {is_significant}")
                
                elif axiom_num in ['4']:
                    uncertainty_values_main_prompt = selected_main_prompt_notcorrect_df_[unc_model_key_main_prompt]
                    uncertainty_values_second_prompt_notcorrect = selected_second_prompt_notcorrect_df[unc_model_key_second_prompt]
                    uncertainty_values_main_prompt_filtered =  uncertainty_values_main_prompt[uncertainty_values_main_prompt<UNC_THERESHOLD]
                    uncertainty_values_second_prompt_notcorrect_filtered = uncertainty_values_second_prompt_notcorrect[uncertainty_values_second_prompt_notcorrect<UNC_THERESHOLD] 
                    stat, p_value, is_significant = wilcoxon_test(uncertainty_values_main_prompt.tolist(), uncertainty_values_second_prompt_notcorrect.tolist())
                    print(f"Uncertainty: {uncertainty_values_second_prompt_notcorrect_filtered.mean():.3f} -> {uncertainty_values_main_prompt_filtered.mean():.3f}")
                    print(f"Is it significant? {is_significant}")

                elif axiom_num in ['5']:
                    uncertainty_values_main_prompt = selected_main_prompt_correct_df_[unc_model_key_main_prompt]
                    uncertainty_values_second_prompt_correct = selected_second_prompt_correct_df[unc_model_key_second_prompt]
                    uncertainty_values_main_prompt_filtered =  uncertainty_values_main_prompt[uncertainty_values_main_prompt<UNC_THERESHOLD]
                    uncertainty_values_second_prompt_correct_filtered = uncertainty_values_second_prompt_correct[uncertainty_values_second_prompt_correct<UNC_THERESHOLD] 
                    stat, p_value, is_significant = wilcoxon_test(uncertainty_values_main_prompt.tolist(), uncertainty_values_second_prompt_correct.tolist())
                    print(f"Uncertainty: {uncertainty_values_second_prompt_correct_filtered.mean():.3f} -> {uncertainty_values_main_prompt_filtered.mean():.3f}")
                    print(f"Is it significant? {is_significant}")
           
                print('\n')
            
            # For ids not in axioms
            print(f"= Not in Axioms =======")
            result_df_main_prompt_not_in_axioms = result_df_main_prompt[~result_df_main_prompt['id'].isin(all_axioms_ids)]
            result_df_second_prompt_not_in_axioms = result_df_second_prompt[(result_df_second_prompt['id'].isin(result_df_main_prompt_not_in_axioms['id'].tolist()))]
            result_df_main_prompt_not_in_axioms_ = result_df_main_prompt_not_in_axioms[result_df_main_prompt_not_in_axioms['id'].isin(result_df_second_prompt_not_in_axioms['id'].tolist())] 
            print(f'# Samples: {len(result_df_main_prompt_not_in_axioms_)}')
            print(f'# Samples: {len(result_df_second_prompt_not_in_axioms)}')
            
            uncertainty_values_main_prompt =  result_df_main_prompt_not_in_axioms_[unc_model_key_main_prompt]
            uncertainty_values_second_prompt = result_df_second_prompt_not_in_axioms[unc_model_key_second_prompt]
            uncertainty_values_main_prompt_filtered =  uncertainty_values_main_prompt[uncertainty_values_main_prompt<UNC_THERESHOLD]
            uncertainty_values_second_prompt_filtered = uncertainty_values_second_prompt[uncertainty_values_second_prompt<UNC_THERESHOLD]
            stat, p_value, is_significant = wilcoxon_test(uncertainty_values_main_prompt.tolist(), uncertainty_values_second_prompt.tolist())
            print(f"Uncertainty: {uncertainty_values_second_prompt_filtered.mean():.3f} -> {uncertainty_values_main_prompt_filtered.mean():.3f}")
            print(f"Is it significant? {is_significant}")
            print('\n')
            

    # ======
    result_df_main_prompt = create_result_df(args.main_prompt_format, args.second_prompt_format)    
    result_df_second_prompt = create_result_df(args.second_prompt_format, args.main_prompt_format)
        
    for prompt_order in ['main']: # 'third', 'second', 'forth'
        print(f"=== {prompt_order} ====================================")
        print(f"Main: {len(result_df_main_prompt)}")
        print(f"2ed:  {len(result_df_second_prompt)}")
        
        answers_equality_output_jsonl_file = f'{base_dir}/{args.main_prompt_format}__{args.second_prompt_format}/{generation_type}/axiomatic_results_{prompt_order}/{model}_answers_equality_output.jsonl'
        answers_equality_output_dir = os.path.dirname(answers_equality_output_jsonl_file)
        os.makedirs(answers_equality_output_dir, exist_ok=True)
        
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
    parser.add_argument('--main_prompt_format', type=str, default='q_positive', choices=[
        'only_q', 'q_positive', 'q_negative', 'q_conflict',
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
    
    # python framework/run/get_axiomatic_correctness_results.py
    
    
    