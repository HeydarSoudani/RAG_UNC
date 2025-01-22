#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import itertools
import json
import torch
import pickle
import sklearn
import sklearn.metrics
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    
    # === For getting equal outputs =============
    sequence_input_main = f'{base_dir}/{args.main_prompt_format}__{args.second_prompt_format}/cleaned_generation_{args.generation_type}.pkl'
    if os.path.isdir(f'{base_dir}/{args.second_prompt_format}__{args.main_prompt_format}'):
        sequence_input_secondry = f'{base_dir}/{args.second_prompt_format}__{args.main_prompt_format}/cleaned_generation_normal.pkl'
    else:
        temp = 'bm25_retriever_top1' if args.dataset == 'popqa' else 'q_positive'
        sequence_input_secondry = f'{base_dir}/{args.second_prompt_format}__{temp}/cleaned_generation_normal.pkl'
    
    with open(sequence_input_main, 'rb') as infile:
        sequences_main = pickle.load(infile)
    with open(sequence_input_secondry, 'rb') as infile:
        sequences_secondry = pickle.load(infile)
        
    # === Load semantic model ===================
    # - Labels: {0: Contradiction, 1: Neutral, 2: Entailment}
    # semantic_model_name = "microsoft/deberta-v2-xxlarge-mnli"
    # semantic_model_name = 'facebook/bart-large-mnli'
    # semantic_model_name = "tals/albert-xlarge-vitaminc-mnli"
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
        likelihoods_input_file = f'{results_dir}/{generation_type}/uncertainty_mars_generation.pkl'
        
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
            # 'average_predictive_entropy_third_prompt', 'predictive_entropy_over_concepts_third_prompt',
            # 'average_predictive_entropy_importance_max_third_prompt', 'predictive_entropy_over_concepts_importance_max_third_prompt',
            # 'average_predictive_entropy_forth_prompt', 'predictive_entropy_over_concepts_forth_prompt',
            # 'average_predictive_entropy_importance_max_forth_prompt', 'predictive_entropy_over_concepts_importance_max_forth_prompt',
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
        if main_prompt_format != 'only_q':
            axiomatic_variables_input_file = f'{results_dir}/{generation_type}/axiomatic_variables.pkl'
            with open(axiomatic_variables_input_file, 'rb') as f:
                axiomatic_variables_results  = pickle.load(f)
            axiomatic_variables_df = pd.DataFrame(axiomatic_variables_results)
            result_df = generations_df.merge(similarities_df, on='id').merge(likelihoods_df, on='id').merge(correctness_df, on='id').merge(axiomatic_variables_df, on='id')
        else:
            result_df = generations_df.merge(similarities_df, on='id').merge(likelihoods_df, on='id').merge(correctness_df, on='id')
        
        # 
        result_df['len_most_likely_generation_length'] = result_df['most_likely_generation'].apply(lambda x: len(x.split()))
        return result_df
    
    def get_axiomatic_coef(answer_equality_nli, nli_main, nli_sec, coefs=(0.33, 0.33)):
        C1, C2 = coefs[0], coefs[1]
        
        # switch_main = 1.0 if nli_main[0]==2 else 0.0
        # switch_2ed = 1.0 if nli_sec[0]==2 else 0.0
        # return C1*answer_equality_nli[1] + C2*switch_main*nli_main[1] + C3*switch_2ed*nli_sec[1]
        return C1*answer_equality_nli[1] + C2*nli_main[1]
    
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
    
    def compute_answer_equality_em(sequence_1, sequence_2):
        sequence_2_ = {}
        for sample in sequence_2:
            sequence_2_[sample['id']] = sample
        
        semantically_similar_list = []
        semantically_not_similar_list = []
        with open(answers_equality_output_file, 'w') as jl_ofile:
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

    def get_document_output_relation_axiom12(queries_list, axiom_output_file):
        
        if os.path.isfile(axiom_output_file):
            print(f"{axiom_output_file} exists.")
            with open(axiom_output_file, 'r') as file:
                relation_main = json.load(file)['main'] 
        
        else:
            relation_main = {'entailment': [], 'neutral': [], 'contradiction': []}
            for idx, sample in tqdm(enumerate(sequences_main)):
                id_ = sample['id']
                if id_ in queries_list:
                    
                    # === Prapare inputs
                    question = sample['question']
                    generated_text_most_likely = sample['most_likely_generation']
                    prompt_text = sample['prompt_text']
                    doc_text = prompt_text.split('Document:')[-1].split('Question:')[0]
                    answer_ = f"{question} {generated_text_most_likely}"
                    
                    # === Common NLI: Similar to semantic semilarity
                    input = doc_text + ' [SEP] ' + answer_
                    encoded_input = semantic_tokenizer.encode(input, padding=True)
                    prediction = semantic_model(torch.tensor(torch.tensor([encoded_input]), device=args.device))['logits']
                    predicted_label = torch.argmax(prediction, dim=1)
                    
                    reverse_input = answer_ + ' [SEP] ' + doc_text
                    encoded_reverse_input = semantic_tokenizer.encode(reverse_input, padding=True)
                    reverse_prediction = semantic_model(torch.tensor(torch.tensor([encoded_reverse_input]), device=args.device))['logits']
                    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
                    
                    item = (id_, torch.softmax(prediction, dim=1).tolist()[0], torch.softmax(reverse_prediction, dim=1).tolist()[0])
                    if 0 in predicted_label or 0 in reverse_predicted_label:
                        relation_main['contradiction'].append(item)
                    else:
                        relation_main['entailment'].append(item)
            
            # Write to file 
            axiom_output = {'main': relation_main}
            with open(axiom_output_file, 'w') as file:
                json.dump(axiom_output, file, indent=4)
            
        axiom_1_items_main = relation_main['entailment']
        axiom_2_items_main = relation_main['contradiction']
            
        return axiom_1_items_main, axiom_2_items_main        
        
    def get_document_output_relation_axiom45(queries_list, axiom_output_file):
        
        sequence_main_ = {}
        for sample in sequences_main:
            sequence_main_[sample['id']] = sample
        
        if os.path.isfile(axiom_output_file):
            print(f"{axiom_output_file} exists.")
            with open(axiom_output_file, 'r') as file:
                data = json.load(file)
                relation_main = data['main'] 
                relation_2ed = data['second']
        
        else:
            relation_main = {'entailment': [], 'neutral': [], 'contradiction': []}
            relation_2ed = {'entailment': [], 'neutral': [], 'contradiction': []}
        
            ### === Loop on main ====================
            for idx, sample in tqdm(enumerate(sequences_main)):
                id_ = sample['id']
                if id_ in queries_list:
                    # === Prapare inputs
                    question = sample['question']
                    generated_text_most_likely = sample['most_likely_generation']
                    prompt_text = sample['prompt_text']
                    doc_text = prompt_text.split('Document:')[-1].split('Question:')[0]
                    answer_ = f"{question} {generated_text_most_likely}"
            
                    # === Common NLI: Similar to semantic semilarity
                    input = doc_text + ' [SEP] ' + answer_
                    encoded_input = semantic_tokenizer.encode(input, padding=True)
                    prediction = semantic_model(torch.tensor(torch.tensor([encoded_input]), device=args.device))['logits']
                    predicted_label = torch.argmax(prediction, dim=1)
                    
                    reverse_input = answer_ + ' [SEP] ' + doc_text
                    encoded_reverse_input = semantic_tokenizer.encode(reverse_input, padding=True)
                    reverse_prediction = semantic_model(torch.tensor(torch.tensor([encoded_reverse_input]), device=args.device))['logits']
                    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
                    
                    item = (id_, torch.softmax(prediction, dim=1).tolist()[0], torch.softmax(reverse_prediction, dim=1).tolist()[0])
                    if 0 in predicted_label or 0 in reverse_predicted_label:
                        relation_main['contradiction'].append(item)
                    else:
                        relation_main['entailment'].append(item)
            
            ### === Loop on second ====================
            for idx, sample2 in tqdm(enumerate(sequences_secondry)):
                id_ = sample2['id']
                if id_ in queries_list:
                    
                    # === Prapare inputs
                    question = sample2['question']
                    generated_text_most_likely = sample2['most_likely_generation']
                    prompt_text = sequence_main_[id_]['prompt_text']
                    doc_text = prompt_text.split('Document:')[-1].split('Question:')[0]
                    answer_ = f"{question} {generated_text_most_likely}"
            
                    # === Common NLI: Similar to semantic semilarity
                    input = doc_text + ' [SEP] ' + answer_
                    encoded_input = semantic_tokenizer.encode(input, padding=True)
                    prediction = semantic_model(torch.tensor(torch.tensor([encoded_input]), device=args.device))['logits']
                    predicted_label = torch.argmax(prediction, dim=1)
                    
                    reverse_input = answer_ + ' [SEP] ' + doc_text
                    encoded_reverse_input = semantic_tokenizer.encode(reverse_input, padding=True)
                    reverse_prediction = semantic_model(torch.tensor(torch.tensor([encoded_reverse_input]), device=args.device))['logits']
                    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
                    
                    item = (id_, torch.softmax(prediction, dim=1).tolist()[0], torch.softmax(reverse_prediction, dim=1).tolist()[0])
                    if 0 in predicted_label or 0 in reverse_predicted_label:
                        relation_2ed['contradiction'].append(item)
                    else:
                        relation_2ed['entailment'].append(item)

            # Write to file 
            axiom_output = {'main': relation_main, 'second': relation_2ed}
            with open(axiom_output_file, 'w') as file:
                json.dump(axiom_output, file, indent=4)

        # Axiom 4
        main_entailment_ids = {item[0] for item in relation_main['entailment']}
        second_contradiction_ids = {item[0] for item in relation_2ed['contradiction']}
        axiom_4_ids = main_entailment_ids & second_contradiction_ids
        axiom_4_items_main = [item for item in relation_main['entailment'] if item[0] in axiom_4_ids]
        axiom_4_items_2ed = [item for item in relation_2ed['contradiction'] if item[0] in axiom_4_ids]

        # Axiom 5
        main_contradiction_ids = {item[0] for item in relation_main['contradiction']}
        second_entailment_ids = {item[0] for item in relation_2ed['entailment']}
        axiom_5_ids = main_contradiction_ids & second_entailment_ids
        axiom_5_items_main = [item for item in relation_main['contradiction'] if item[0] in axiom_5_ids]
        axiom_5_items_2ed = [item for item in relation_2ed['entailment'] if item[0] in axiom_5_ids]
        
        # Others
        items_to_remove = axiom_4_ids | axiom_5_ids 
        other_ids = [item for item in queries_list if item not in items_to_remove]
        other_items = [item for item in relation_main['entailment'] if item[0] in other_ids]
        other_items.extend([item for item in relation_main['contradiction'] if item[0] in other_ids])

        return axiom_4_items_main, axiom_5_items_main, other_items
    
    def run_axiomatic_metrics(prompt_order):
        
        ### === Step1: Check if answer1 is equal to answer2 ===
        # def get_output_equality():
        #     if os.path.isfile(answers_equality_output_file):
        #         print(f"{answers_equality_output_file} exists.")
        #         answer_equal_list, answer_not_equal_list = [], []
        #         with open(answers_equality_output_file, 'r') as file:
        #             for line in file:
        #                 if line.strip():
        #                     item = json.loads(line)
        #                     if item['is_equal']:
        #                         answer_equal_list.append(item['id'])
        #                     else:
        #                         answer_not_equal_list.append(item['id'])
        #     else:
        #         print("Computing similarity ...")
        #         answer_equal_list, answer_not_equal_list = compute_answer_equality_em(sequences_main, sequences_secondry)
            
        #     print(f"Answer equal: {len(answer_equal_list)}")
        #     print(f"Answer not equal: {len(answer_not_equal_list)}")
        #     return answer_equal_list, answer_not_equal_list
        # answer_equal_list, answer_not_equal_list = get_output_equality()

        ### === Step2: Compute Axioms =========================
        for uncertainty_model in ['PE']: # 'PE', 'SE', 'PE_MARS', 'SE_MARS'
            print(f"Unc. Model: {uncertainty_model}")
            unc_model_key_main_prompt = keys_mapping[f'{prompt_order}_prompt'][uncertainty_model]
            unc_model_key_second_prompt = keys_mapping['main_prompt'][uncertainty_model]
            
            # === 2) Get Axiomatic Coef.
            result_df_main_prompt['axiomatic_coef'] = [
                get_axiomatic_coef(answer_equality_nli, nli_main, nli_sec, coefs=(0.0, 1.0))
                for answer_equality_nli, nli_main, nli_sec in tqdm(zip(
                    result_df_main_prompt['answer_equality_nli'],
                    result_df_main_prompt['nli_relation_main'],
                    result_df_main_prompt['nli_relation_second']
                ), desc='Getting axiomatic coef. ...')
            ]
            
            filtered_df = result_df_main_prompt[result_df_main_prompt['axiom_num_nli'].isin(['1', '2', '4', '5'])]
            mean_value = filtered_df['axiomatic_coef'].mean()
            std_value = filtered_df['axiomatic_coef'].std()
            print(f"axiomatic_coef -> mean: {mean_value}, std:{std_value}")
            
            result_df_main_prompt[f"{unc_model_key_main_prompt}_cal"] = (1.0+mean_value - result_df_main_prompt['axiomatic_coef']) * result_df_main_prompt[unc_model_key_main_prompt]
            
            result_df_main_prompt_filtered = result_df_main_prompt[result_df_main_prompt[unc_model_key_main_prompt] <UNC_THERESHOLD]
            result = result_df_main_prompt_filtered.groupby('axiom_num_nli').agg(
                true_ratio=('exact_match', lambda x: x.sum() / len(x)),
                average_uncertainty=(unc_model_key_main_prompt, 'mean'),
                row_count=(unc_model_key_main_prompt, 'count'),
                coef_mean=('axiomatic_coef', 'mean'),
                coef_unc_mean=(f'{unc_model_key_main_prompt}_cal', 'mean')
            ).reset_index()
            print(result)
            
            all_axioms_ids = []
            for axiom_num in ['1', '2', '4', '5', 'others']: # '1', '2', '4', '5', 'other'
                print(f"== Axiom: {axiom_num} ===")
                
                # === Get samples (v1) =====================
                # if axiom_num in ['1', '2']:
                #     axiom1_items, axiom2_items = get_document_output_relation_axiom12(answer_equal_list, axioms12_output_file)
                #     selected_list = axiom1_items if axiom_num=='1' else axiom2_items
                
                # elif axiom_num in ['4', '5', 'other']:
                #     axiom4_items, axiom5_items, other_items = get_document_output_relation_axiom45(answer_not_equal_list, axioms45_output_file)
                #     selected_list = axiom4_items if axiom_num=='4' else axiom5_items if axiom_num=='5' else other_items
                    
                # if len(selected_list) > 0:
                #     selected_list_ = [tup[0] for tup in selected_list]
                #     selected_main_prompt_df_ = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_)]
                #     selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_list_)]
                #     selected_main_prompt_df = selected_main_prompt_df_[selected_main_prompt_df_['id'].isin(selected_second_prompt_df['id'].tolist())] 
                #     print(f'# Samples: {len(selected_main_prompt_df)}')
                #     print(f'# Samples: {len(selected_main_prompt_df)}')
                #     all_axioms_ids.extend(selected_main_prompt_df['id'].tolist())
                    
                # === Get samples (v2) =====================    
                selected_main_prompt_df = result_df_main_prompt[result_df_main_prompt['axiom_num_nli'] == axiom_num]
                selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_main_prompt_df['id'].tolist())]
                print(f'# Samples: {len(selected_main_prompt_df)}')
                print(f'# Samples: {len(selected_second_prompt_df)}')
                
                # === Get Uncertainty =====================
                for type_ in ['normal', 'calibrated']: # 'calibrated'
                    
                    if type_ == 'calibrated':
                        # uncertainty_values_main_prompt = (1.15 - selected_main_prompt_df['axiomatic_coef']) * selected_main_prompt_df[unc_model_key_main_prompt]
                        # uncertainty_values_second_prompt = (1.5 - selected_main_prompt_df['axiomatic_coef']) * selected_second_prompt_df[unc_model_key_second_prompt]
                        uncertainty_values_main_prompt = selected_main_prompt_df[f"{unc_model_key_main_prompt}_cal"]
                        
                    else:
                        uncertainty_values_main_prompt =  selected_main_prompt_df[unc_model_key_main_prompt]
                    
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
    
    # === For testing ......
    
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

    
    for prompt_order in ['main']: # main, 'second', 'third', 'forth'
        print(f"=== {prompt_order} ====================================")
        print(f"Main: {len(result_df_main_prompt)}")
        print(f"2ed:  {len(result_df_second_prompt)}")
        
        # axioms12_output_file = f'{base_dir}/{args.main_prompt_format}__{args.second_prompt_format}/{generation_type}/axiomatic_results_{prompt_order}/{model}_axioms12_output.json'
        # axioms45_output_file = f'{base_dir}/{args.main_prompt_format}__{args.second_prompt_format}/{generation_type}/axiomatic_results_{prompt_order}/{model}_axiom45_output.json'
        # answers_equality_output_file = f'{base_dir}/{args.main_prompt_format}__{args.second_prompt_format}/{generation_type}/axiomatic_results_{prompt_order}/{model}_answers_equality_output.jsonl'
        # answers_equality_output_dir = os.path.dirname(answers_equality_output_file)
        # os.makedirs(answers_equality_output_dir, exist_ok=True)
        
        run_axiomatic_metrics(prompt_order)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='nqgold', choices=[
        'nqgold', 'nqswap', 'trivia', 'popqa',
        'webquestions', 'squad1', 'nq',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa',
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--main_prompt_format', type=str, default='only_q', choices=[
        'only_q', 'q_positive', 'q_negative', 'q_conflict',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    parser.add_argument('--second_prompt_format', type=str, default='q_positive', choices=[
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
    
    # python framework/run/get_axiomatic_nli_results.py
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        #     # Get Confidence
        # uncertainty_values_all_main_prompt = result_df_main_prompt[unc_model_key_main_prompt]
        # uncertainty_values_all_second_prompt = result_df_second_prompt[unc_model_key_second_prompt]
        # uncertainty_values_all_main_prompt_filtered = np.array(uncertainty_values_all_main_prompt[uncertainty_values_all_main_prompt<UNC_THERESHOLD])
        # uncertainty_values_all_second_prompt_filtered = np.array(uncertainty_values_all_second_prompt[uncertainty_values_all_second_prompt<UNC_THERESHOLD])
        # min_all_main, max_all_main = np.min(uncertainty_values_all_main_prompt_filtered), np.max(uncertainty_values_all_main_prompt_filtered)
        # min_all_second, max_all_second = np.min(uncertainty_values_all_second_prompt_filtered), np.max(uncertainty_values_all_second_prompt_filtered)
        # confidence_values_main_prompt = uncertainty_to_confidence_min_max(uncertainty_values_main_prompt_filtered, min_val=min_all_main, max_val=max_all_main)
        # confidence_values_second_prompt = uncertainty_to_confidence_min_max(uncertainty_values_second_prompt_filtered, min_val=min_all_second, max_val=max_all_second)
            
    
    
    


        # Method 1) Common NLI: based on label
    # encoded_input = semantic_tokenizer.encode_plus(
    #     doc_text,
    #     answer_,
    #     padding=True,
    #     truncation=True,
    #     max_length=512,
    #     return_tensors="pt",
    #     truncation_strategy="only_first"
    # )
    # encoded_input = {key: val.to(args.device) for key, val in encoded_input.items()}
    # prediction = semantic_model(**encoded_input).logits
    # predicted_label = prediction.argmax(dim=1).item()

    # item = (id_, predicted_label)
    # if predicted_label==0: # entailment
    #     relation_queries['entailment'].append(item)
    # elif predicted_label==1: # neutral
    #     relation_queries['neutral'].append(item)
    # elif predicted_label==2: # contradiction
    #     relation_queries['contradiction'].append(item)
    # else:
    #     relation_queries['neutral'].append(item)
    ### === For BART  
    # if predicted_label==0: # entailment
    #     relation_queries['contradiction'].append(item)
    # elif predicted_label==1: # neutral
    #     relation_queries['neutral'].append(item)
    # elif predicted_label==2: # contradiction
    #     relation_queries['entailment'].append(item)
    # else:
    #     relation_queries['neutral'].append(item)


    # Method 2) MiniCheck
    # pred_label, predicted_score_, _, _ = minicheck_factual_scorer.score(docs=[doc_text], claims=[answer_])
    # predicted_score = predicted_score_[0]
    
    
    # Method 3) 
    # predicted_score_ = hallucination_detector.predict([(doc_text, answer_)])
    # predicted_score = predicted_score_.item()
    
    # item = (id_, predicted_score)
    # if predicted_score > 0.50: # entailment
    #     relation_queries['entailment'].append(item)
    # elif predicted_score < 0.50: # contradiction
    #     relation_queries['contradiction'].append(item)
    # else:
    #     relation_queries['neutral'].append(item)


    # Method 4) Long NLI 
    # predicted_score_ = long_nli_model([dict(text=doc_text, text_pair=answer_)])
    # item = (id_, predicted_score_[0]['score'])
    # relation_queries[predicted_score_[0]['label']].append(item)
    
    
    
    # def compute_answer_equality(sequence_1, sequence_2, threshold=0.5):
        
    #     sequence_2_ = {}
    #     for sample in sequence_2:
    #         sequence_2_[sample['id']] = sample
        
    #     agree_list = []
    #     non_agree_list = []
    #     with open(similarity_output_jsonl_file, 'w') as jl_ofile:
            
    #         for i, sample in tqdm(enumerate(sequence_1)):
    #             id_ = sample['id']
                
    #             if id_ in sequence_2_:
    #                 generation_most_likely_seq1 = sample['cleaned_most_likely_generation']
    #                 generation_most_likely_seq2 = sequence_2_[id_]['cleaned_most_likely_generation']
                    
    #                 # print(generation_most_likely_seq1)
    #                 # print(generation_most_likely_seq2)
                    
    #                 similarity = similarity_model.predict([generation_most_likely_seq1, generation_most_likely_seq2])
                    
    #                 if similarity > threshold:
    #                     agree_list.append(id_)
    #                 else:
    #                     non_agree_list.append(id_)
                    
    #                 # print(similarity)
    #                 result_item = {
    #                     'id': id_,
    #                     'question': sample['question'],
    #                     'generation_seq_1': generation_most_likely_seq1,
    #                     'generation_seq_2': generation_most_likely_seq2,
    #                     'sim_score': float(similarity)
    #                 }
    #                 jl_ofile.write(json.dumps(result_item) + '\n')
                    
                
    #             else:
    #                 print(f"\nQuery {id_} is not common between two sequences !!!")
            
    #     return agree_list, non_agree_list
       
    # def compute_answer_equality_nli(sequence_1, sequence_2):
    #     sequence_2_ = {}
    #     for sample in sequence_2:
    #         sequence_2_[sample['id']] = sample
        
    #     semantically_similar_list = []
    #     semantically_not_similar_list = []
        
    #     ### Similar to semantic similarity
    #     with open(answers_equality_output_file, 'w') as jl_ofile:
    #         for i, sample in tqdm(enumerate(sequence_1)):
    #             id_ = sample['id']
    #             question = sample['question']
    #             is_equal = False
                
    #             if id_ in sequence_2_:
    #                 generation_most_likely_seq1 = sample['cleaned_most_likely_generation']
    #                 generation_most_likely_seq2 = sequence_2_[id_]['cleaned_most_likely_generation']
    #                 qa_1 = question + ' ' + generation_most_likely_seq1
    #                 qa_2 = question + ' ' + generation_most_likely_seq2
                    
    #                 input = generation_most_likely_seq1 + ' [SEP] ' + generation_most_likely_seq2
    #                 encoded_input = semantic_tokenizer.encode(input, padding=True)
    #                 prediction = semantic_model(torch.tensor(torch.tensor([encoded_input]), device=args.device))['logits']
    #                 predicted_label = torch.argmax(prediction, dim=1)

    #                 reverse_input = generation_most_likely_seq2 + ' [SEP] ' + generation_most_likely_seq1
    #                 encoded_reverse_input = semantic_tokenizer.encode(reverse_input, padding=True)
    #                 reverse_prediction = semantic_model(torch.tensor(torch.tensor([encoded_reverse_input]), device=args.device))['logits']
    #                 reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
                    
    #                 if 0 in predicted_label or 0 in reverse_predicted_label:
    #                     semantically_not_similar_list.append(id_)
    #                     is_equal = False
    #                 else:
    #                     semantically_similar_list.append(id_)
    #                     is_equal = True

    #                 result_item = {
    #                     'id': id_,
    #                     'question': sample['question'],
    #                     'answers': sample['answers'],
    #                     'generation_seq_1': generation_most_likely_seq1,
    #                     'generation_seq_2': generation_most_likely_seq2,
    #                     'is_equal': is_equal
    #                 }
    #                 jl_ofile.write(json.dumps(result_item) + '\n')

    #             else:
    #                 print(f"\nQuery {id_} is not common between two sequences !!!")

    #     return semantically_similar_list, semantically_not_similar_list

    
    
    
    # (2), (3), (4) -> Src: https://arxiv.org/pdf/2410.03461
    # 2) MiniCheck (EMNLP24)
    # - MiniCheck: https://huggingface.co/lytang/MiniCheck-Flan-T5-Large
    # - pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
    # minicheck_factual_scorer = MiniCheck(model_name='flan-t5-large', cache_dir='./ckpts')
    
    # 2) Hallucination Detector
    # - Vectara: https://huggingface.co/vectara/hallucination_evaluation_model
    # - It needs updated version of transformer: pip install --upgrade transformers
    # hallucination_detector = AutoModelForSequenceClassification.from_pretrained(
    #     'vectara/hallucination_evaluation_model', trust_remote_code=True)

    # 3) Long NLI: 
    # - Tasksource: https://huggingface.co/tasksource/deberta-base-long-nli
    # long_nli_model = pipeline("text-classification", model="tasksource/deberta-base-long-nli", device=args.device)

    
    
    
    # def in_doc_existence(sequences, ids):
    #     samples = [item for item in sequences if item['id'] in ids]
        
    #     doc_exist, doc_not_exist = [], []
    #     for idx, sample in enumerate(samples):
    #         answer = sample['cleaned_most_likely_generation']
    #         prompt_text = sample['prompt_text']
    #         doc_text = prompt_text.split('Document:')[-1].split('Question:')[0]
            
    #         # def is_answer_in_doc(answer, doc, threshold=0.8):
    #         #     return SequenceMatcher(None, answer, doc).ratio() > threshold
            
    #         # if answer in doc_text:
    #         #     print('1')
    #         if answer.lower() in doc_text.lower():
    #             doc_exist.append(sample['id'])
    #         else:
    #             doc_not_exist.append(sample['id'])
    #         # if re.search(r'\b' + re.escape(answer) + r'\b', doc_text, re.IGNORECASE):
    #         #     print('3')
    #         # if is_answer_in_doc(answer, doc_text):
    #         #     print('4')
    #         # else:
    #         #     print('5')
    #     return doc_exist, doc_not_exist
    
    
    
        # print("================= Axiom: 4 =================")
    # axiom_num = '4'
    # axiom_4 = get_entail_contradict_relations_nli(axiom_num, non_agree_list)
    # print(f"Entailment:    {len(axiom_4['entailment'])} ({(len(axiom_4['entailment']) / len(sequences_main))*100:.2f}%)")
    # # print(f"Contradiction: {len(axiom_4['contradiction'])} ({len(axiom_4['contradiction']) / len(sequences_main)*100:.2f}%)")
    # # print(f"Neutral:       {len(axiom_4['neutral'])} ({len(axiom_4['neutral']) / len(sequences_main)*100:.2f}%)")
    # print('\n')
    # relation_key = 'entailment'
    # selected_list = axiom_4[relation_key]
    # selected_list_ = [tup[0] for tup in selected_list]
    
    # if len(selected_list) > 0:
    #     for uncertainty_model in uncertainty_methods:
            
    #         if uncertainty_model in ['PE', 'PE_MARS']:
    #             result_df_main_prompt = result_df_main_filtered_pe
    #             result_df_second_prompt = result_df_second_prompt_filtered_pe
    #         elif uncertainty_model in ['SE', 'SE_MARS']:
    #             result_df_main_prompt = result_df_main_filtered_se
    #             result_df_second_prompt = result_df_second_prompt_filtered_se
            
    #         unc_model_key_main_prompt = keys_mapping['main_prompt'][uncertainty_model]
    #         unc_model_key_second_prompt = keys_mapping['second_prompt'][uncertainty_model]

    #         selected_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_)]
    #         # selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_list_)]
            
    #         _, correctness_main_prompt_bin, one_minus_correctness_main_prompt = get_correctness(selected_main_prompt_df)
    #         correctness_main_prompt = 1 - np.array(one_minus_correctness_main_prompt)
    #         # _, correctness_second_prompt_bin, one_minus_correctness_second_prompt = get_correctness(selected_second_prompt_df)
    #         # correctness_second_prompt = 1 - np.array(one_minus_correctness_second_prompt)
            
    #         uncertainty_main_prompt_values =  selected_main_prompt_df[unc_model_key_main_prompt]
    #         uncertainty_second_prompt_values = selected_main_prompt_df[unc_model_key_second_prompt]
            
    #         if len(set(correctness_main_prompt_bin)) == 1:
    #             print("Warning: Only one class present in y_true. ROC AUC score is not defined.")
    #             auroc_main_prompt = 0.5
    #             auroc_second_prompt = 0.5
    #         else:
    #             auroc_main_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_main_prompt_values)
    #             auroc_second_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_second_prompt_values)
    #         print(f"{uncertainty_model}, Axiom 4: {relation_key}")
    #         print(f"Uncertainty: {uncertainty_second_prompt_values[uncertainty_second_prompt_values<1000].mean():.3f} -> {uncertainty_main_prompt_values[uncertainty_main_prompt_values<1000].mean():.3f}")
    #         print(f"Acc. ({args.accuracy_metric}): {round(correctness_main_prompt.mean()*100, 2)}")
    #         print(f"AUROC:       {round(auroc_second_prompt, 3)} -> {round(auroc_main_prompt, 3)}")
    #         print('\n')
                
    # else: 
    #     print(f"{relation_key} does not contain data!!!")
    #     print('\n')


    # print("================= Axiom: 5 =================")
    # axiom_num = '5'
    # axiom_5 = get_entail_contradict_relations_nli(axiom_num, non_agree_list)
    # print(f"Contradiction: {len(axiom_5['contradiction'])} ({len(axiom_5['contradiction']) / len(sequences_main)*100:.2f}%)")
    # # print(f"Entailment:    {len(axiom_5['entailment'])} ({(len(axiom_5['entailment']) / len(sequences_main))*100:.2f}%)")
    # # print(f"Neutral:       {len(axiom_4['neutral'])} ({len(axiom_4['neutral']) / len(sequences_main)*100:.2f}%)")
    
    # print('\n')
    # relation_key = 'contradiction'
    # selected_list = axiom_5[relation_key]
    # selected_list_ = [tup[0] for tup in selected_list]
    
    # if len(selected_list) > 0:
    #     for uncertainty_model in uncertainty_methods: 
        
    #         if uncertainty_model in ['PE', 'PE_MARS']:
    #             result_df_main_prompt = result_df_main_filtered_pe
    #             result_df_second_prompt = result_df_second_prompt_filtered_pe
    #         elif uncertainty_model in ['SE', 'SE_MARS']:
    #             result_df_main_prompt = result_df_main_filtered_se
    #             result_df_second_prompt = result_df_second_prompt_filtered_se
            
    #         unc_model_key_main_prompt = keys_mapping['main_prompt'][uncertainty_model]
    #         unc_model_key_second_prompt = keys_mapping['second_prompt'][uncertainty_model]

    #         selected_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_)]
    #         selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_list_)]
            
    #         _, correctness_main_prompt_bin, one_minus_correctness_main_prompt = get_correctness(selected_main_prompt_df)
    #         correctness_main_prompt = 1 - np.array(one_minus_correctness_main_prompt)
    #         _, correctness_second_prompt_bin, one_minus_correctness_second_prompt = get_correctness(selected_second_prompt_df)
    #         correctness_second_prompt = 1 - np.array(one_minus_correctness_second_prompt)
            
    #         uncertainty_main_prompt_values = selected_second_prompt_df[unc_model_key_second_prompt]
    #         uncertainty_second_prompt_values =  selected_second_prompt_df[unc_model_key_main_prompt]
    #         auroc_main_prompt = sklearn.metrics.roc_auc_score(1 - correctness_second_prompt_bin, uncertainty_main_prompt_values)
    #         auroc_second_prompt = sklearn.metrics.roc_auc_score(1 - correctness_second_prompt_bin, uncertainty_second_prompt_values)
    #         print(f"{uncertainty_model}, Axiom 5: {relation_key}")
    #         print(f"Uncertainty: {uncertainty_second_prompt_values[uncertainty_second_prompt_values<1000].mean():.3f} -> {uncertainty_main_prompt_values[uncertainty_main_prompt_values<1000].mean():.3f}")
    #         print(f"Acc. ({args.accuracy_metric}): {round(correctness_second_prompt.mean()*100, 2)}")
    #         print(f"AUROC:       {round(auroc_second_prompt, 3)} -> {round(auroc_main_prompt, 3)}")
    #         print('\n')

    # else: 
    #     print(f"{relation_key} does not contain data!!!")
    #     print('\n')


    # print("================= Axiom: 6 =================")
    # selected_list_1 = axioms_123['entailment']
    # selected_list_2 = axiom_4['entailment']
    # selected_list_1_ = [tup[0] for tup in selected_list_1]
    # selected_list_2_ = [tup[0] for tup in selected_list_2]
    
    # if len(selected_list_1) > 0:
    #     for uncertainty_model in uncertainty_methods:
            
    #         if uncertainty_model in ['PE', 'PE_MARS']:
    #             result_df_main_prompt = result_df_main_filtered_pe
    #             result_df_second_prompt = result_df_second_prompt_filtered_pe
    #         elif uncertainty_model in ['SE', 'SE_MARS']:
    #             result_df_main_prompt = result_df_main_filtered_se
    #             result_df_second_prompt = result_df_second_prompt_filtered_se
    
    #         unc_model_key_main_prompt = keys_mapping['main_prompt'][uncertainty_model]
    #         # unc_model_key_second_prompt = keys_mapping['second_prompt'][uncertainty_model]

    #         selected_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_1_)]
    #         selected_second_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_2_)]
            
    #         uncertainty_main_prompt_values = selected_main_prompt_df[unc_model_key_main_prompt]
    #         uncertainty_second_prompt_values =  selected_second_prompt_df[unc_model_key_main_prompt]
            
    #         print(f"{uncertainty_model}, Axiom 6")
    #         print(f"Uncertainty: {uncertainty_second_prompt_values.mean():.3f} -> {uncertainty_main_prompt_values.mean():.3f}")
    #         print('\n')
    # else: 
    #     # print(f"{relation_key} does not contain data!!!")
    #     print('\n')

    # Axiom 6
    # relation_key = 'contradiction'
    # selected_list = axioms_456[relation_key]
    # selected_list_ = [tup[0] for tup in selected_list]
    
    # if len(selected_list) > 0:
    #     axiom6_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_)]
    #     axiom6_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_list_)]
        
    #     _, correctness_main_prompt_bin, one_minus_correctness_main_prompt = get_correctness(axiom6_main_prompt_df)
    #     correctness_main_prompt = 1 - np.array(one_minus_correctness_main_prompt)
    #     _, correctness_second_prompt_bin, one_minus_correctness_second_prompt = get_correctness(axiom6_second_prompt_df)
    #     correctness_second_prompt = 1 - np.array(one_minus_correctness_second_prompt)
        
    #     uncertainty_main_prompt_values =  axiom6_main_prompt_df[unc_model_key_main_prompt]
    #     uncertainty_second_prompt_values = axiom6_second_prompt_df[unc_model_key_main_prompt]
    #     auroc_main_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_main_prompt_values)
    #     auroc_second_prompt = sklearn.metrics.roc_auc_score(1 - correctness_second_prompt_bin, uncertainty_second_prompt_values)
            
    #     print(f"{uncertainty_model}, Axiom 6: {relation_key}")
    #     print(f"Uncertainty: {uncertainty_second_prompt_values.mean():.3f} -> {uncertainty_main_prompt_values.mean():.3f}")
    #     print(f"Acc. ({args.accuracy_metric}):  {round(correctness_second_prompt.mean()*100, 2)} -> {round(correctness_main_prompt.mean()*100, 2)}")
    #     print(f"AUROC:       {round(auroc_second_prompt, 3)} -> {round(auroc_main_prompt, 3)}")
    #     print('\n')

    # else: 
    #     print(f"{relation_key} does not contain data!!!")
    #     print('\n')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # print(id_)
    
    # input =  answer_ + ' [SEP] ' + doc_text
    # print(input)
    # encoded_input = semantic_tokenizer.encode(input, padding=True)
    # prediction = semantic_model(torch.tensor(torch.tensor([encoded_input]), device=args.device))['logits']
    # predicted_label = torch.argmax(prediction, dim=1)
    # print(predicted_label)

    # reverse_input = doc_text + ' [SEP] ' + answer_
    # encoded_reverse_input = semantic_tokenizer.encode(reverse_input, padding=True)
    # reverse_prediction = semantic_model(torch.tensor(torch.tensor([encoded_reverse_input]), device=args.device))['logits']
    # reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
    # print(reverse_predicted_label)
    
    # print(doc_text)
    # print(generated_text_most_likely)
    # encoded_input = semantic_tokenizer(
    #     doc_text,
    #     generated_text_most_likely,
    #     return_tensors="pt",
    #     padding="max_length",
    #     truncation=True,
    #     max_length=1024  # Adjust based on your needs
    # )

    
    ### === For random testing =================
    # print(ids)
    # num_samples = 40
    # random_selection = random.sample(ids, num_samples)
    # selected_list = random_selection
    
    ### === Second check: answer2 exist in doc =
    # agree_list_doc_exist, agree_list_doc_not_exist = in_doc_existence(sequences_main, agree_list)
    # selected_list = agree_list_doc_exist # Axiom 1 (positive passage)
    # selected_list = agree_list_doc_not_exist # Axiom 2 (positive passage)
    
    # Axiom 4
    # non_agree_list_doc_exist, non_agree_list_doc_not_exist = in_doc_existence(sequences_main, non_agree_list)
    # selected_list = non_agree_list_doc_exist
    
    # print(f"# samples: {len(selected_list)} ({round((len(selected_list)/len(sequences_main))*100 , 2)}%)")
    
    # for uncertainty_model in ['PE', 'SE']: # , 'PE_MARS', 'SE_MARS' 
    #     # unc_model_key = keys_mapping[uncertainty_model]
    #     unc_model_key_main_prompt = keys_mapping['main_prompt'][uncertainty_model]
    #     unc_model_key_second_prompt = keys_mapping['second_prompt'][uncertainty_model]
        
    #     if uncertainty_model in ['PE', 'PE_MARS']:
    #         result_df_main_prompt = result_df_main_filtered_pe
    #         result_df_second_prompt = result_df_second_prompt_filtered_pe
    #     elif uncertainty_model in ['SE', 'SE_MARS']:
    #         result_df_main_prompt = result_df_main_filtered_se
    #         result_df_second_prompt = result_df_second_prompt_filtered_se
    
    #     ### For whole samples (main prompt)
    #     _, correctness_main_prompt_bin_, one_minus_correctness_main_prompt_ = get_correctness(result_df_main_prompt)
    #     correctness_main_prompt_ = 1 - np.array(one_minus_correctness_main_prompt_)
    #     uncertainty_main_prompt_values_ = result_df_main_prompt[unc_model_key_main_prompt]
        
    #     # test2
    #     # uncertainty_main_prompt_values_ = np.where(
    #     #     result_df_main_prompt['id'].isin(selected_list),
    #     #     result_df_main_prompt[unc_model_key_main_prompt] * 1.2,
    #     #     result_df_main_prompt[unc_model_key_main_prompt]
    #     # )
    #     # test3: combine
    #     # uncertainty_main_prompt_values_ = np.where(
    #     #     result_df_main_prompt['id'].isin(selected_list),
    #     #     result_df_main_prompt[unc_model_key_main_prompt] * 0.5,
    #     #     abs(result_df_main_prompt[unc_model_key_main_prompt] - result_df_main_prompt[unc_model_key_second_prompt])
    #     # )
    #     confidence_main_prompt_values_ = uncertainty_to_confidence_min_max(uncertainty_main_prompt_values_)
        
    #     ### For Axiom1 samples (second prompt)
    #     agree_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list)]
    #     agree_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_list)]
        
    #     _, correctness_main_prompt_bin, one_minus_correctness_main_prompt = get_correctness(agree_main_prompt_df)
    #     _, correctness_second_prompt_bin, one_minus_correctness_second_prompt = get_correctness(agree_second_prompt_df)
    #     correctness_main_prompt = 1 - np.array(one_minus_correctness_main_prompt)
    #     correctness_second_prompt = 1 - np.array(one_minus_correctness_second_prompt)
        
    #     uncertainty_main_prompt_values =  agree_main_prompt_df[unc_model_key_main_prompt] # 0.5*
    #     uncertainty_second_prompt_values = agree_second_prompt_df[unc_model_key_main_prompt] # Axioms: 1, 2, 3
        
    #     confidence_main_prompt_values = uncertainty_to_confidence_min_max(uncertainty_main_prompt_values)
    #     confidence_second_prompt_values = uncertainty_to_confidence_min_max(uncertainty_second_prompt_values)
        
    #     auroc_main_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_main_prompt_values)
    #     auroc_second_prompt = sklearn.metrics.roc_auc_score(1 - correctness_second_prompt_bin, uncertainty_second_prompt_values)
    #     ece_main_prompt = ece_estimate(correctness_main_prompt, confidence_main_prompt_values)
    #     ece_second_prompt = ece_estimate(correctness_second_prompt, confidence_second_prompt_values)
    #     print(f"{uncertainty_model}, Axioms:")
    #     print(f"Acc. (bem):  {correctness_second_prompt.mean()} -> {correctness_main_prompt.mean()}")
    #     print(f"Uncertainty: {uncertainty_second_prompt_values.mean()} -> {uncertainty_main_prompt_values.mean()}")
    #     print(f"Confidence:  {confidence_second_prompt_values.mean()} -> {confidence_main_prompt_values.mean()}")
    #     print(f"AUROC:       {auroc_second_prompt} -> {auroc_main_prompt}")
    #     print(f"ECE:         {ece_second_prompt} -> {ece_main_prompt}")
        
        # Axiom 4-1
        # uncertainty_second_prompt_values = agree_main_prompt_df[unc_model_key_second_prompt] 
        # confidence_second_prompt_values = uncertainty_to_confidence_min_max(uncertainty_second_prompt_values)
        # auroc_second_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_second_prompt_values)
        # ece_second_prompt = ece_estimate(correctness_main_prompt, confidence_second_prompt_values)
        # print(f"{uncertainty_model}, Axioms:")
        # print(f"Acc. (bem):  {correctness_main_prompt.mean()}")
        # print(f"Uncertainty: {uncertainty_second_prompt_values.mean()} -> {uncertainty_main_prompt_values.mean()}")
        # print(f"Confidence:  {confidence_second_prompt_values.mean()} -> {confidence_main_prompt_values.mean()}")
        # print(f"AUROC:       {auroc_second_prompt} -> {auroc_main_prompt}")
        # print(f"ECE:         {ece_second_prompt} -> {ece_main_prompt}")
        
        ### === For test one: decrease the uncertainty of samples with same answer with/wo docs
        # auroc_test1 = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin_, uncertainty_main_prompt_values_)
        # ece_test1 = ece_estimate(correctness_main_prompt_, confidence_main_prompt_values_)
        # plot_correctness_vs_uncertainty_for_axioms(
        #     correctness_main_prompt_, uncertainty_main_prompt_values_,
        #     correctness_main_prompt, uncertainty_main_prompt_values,
        #     f'AUROC: {round(auroc_test1, 4)}\nECE: {round(ece_test1, 4)}',
        #     f'{uncertainty_model}_axiom1', num_bins=40
        # )
        # print(f"AUROC:       {auroc_test1}")
        # print(f"ECE:         {ece_test1}")

