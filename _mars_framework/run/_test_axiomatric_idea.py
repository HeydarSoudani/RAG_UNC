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
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline

from utils.utils import set_seed, uncertainty_to_confidence_min_max


def test_axiomatic_idea(args):
    print("\n--- Step 6: Test Axiomatic Idea ...")
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
    base_dir_output = f'{args.output_dir}/{args.dataset}/{args.run_id}/'
    similarity_output_jsonl_file = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_similarity_output__sec_{args.second_prompt_format}.jsonl'
    axioms123_output_jsonl_file = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_axioms123_output__sec_{args.second_prompt_format}.json'
    axiom4_output_jsonl_file = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_axiom4_output__sec_{args.second_prompt_format}.json'
    axiom5_output_jsonl_file = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_axiom5_output__sec_{args.second_prompt_format}.json'
    
    sequence_input_main = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
    sequence_input_secondry = f'{base_dir_output}/{args.second_prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
    with open(sequence_input_main, 'rb') as infile:
        sequences_main = pickle.load(infile)
    with open(sequence_input_secondry, 'rb') as infile:
        sequences_secondry = pickle.load(infile)
        
    
    # === Load semantic model ===================
    long_nli_model = pipeline("text-classification", model="tasksource/deberta-base-long-nli", device=args.device)

    # === Functions =============================
    def create_result_df(prompt_format):
        
        similarities_input_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_similarities_generation.pkl'
        likelihoods_input_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_uncertainty_generation.pkl'
        correctness_input_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_correctness.pkl'
        generation_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
        # groundedness_input_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_groundedness_generation__sec_{args.second_prompt_format}.pkl'
        
        with open(generation_file, 'rb') as infile:
            cleaned_sequences = pickle.load(infile)
        with open(similarities_input_file, 'rb') as f:
            similarities_dict = pickle.load(f)
        with open(likelihoods_input_file, 'rb') as f:
            likelihoods_results  = pickle.load(f)
        with open(correctness_input_file, 'rb') as f:
            correctness_results  = pickle.load(f)
        # with open(groundedness_input_file, 'rb') as f:
        #     groundedness_results  = pickle.load(f)
        
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
    
    def get_aggreement(sequence_1, sequence_2, threshold=0.5):
        
        sequence_2_ = {}
        for sample in sequence_2:
            sequence_2_[sample['id']] = sample
        
        agree_list = []
        non_agree_list = []
        with open(similarity_output_jsonl_file, 'w') as jl_ofile:
            
            for i, sample in tqdm(enumerate(sequence_1)):
                id_ = sample['id']
                
                if id_ in sequence_2_:
                    generation_most_likely_seq1 = sample['cleaned_most_likely_generation']
                    generation_most_likely_seq2 = sequence_2_[id_]['cleaned_most_likely_generation']
                    
                    # print(generation_most_likely_seq1)
                    # print(generation_most_likely_seq2)
                    
                    similarity = similarity_model.predict([generation_most_likely_seq1, generation_most_likely_seq2])
                    
                    if similarity > threshold:
                        agree_list.append(id_)
                    else:
                        non_agree_list.append(id_)
                    
                    # print(similarity)
                    result_item = {
                        'id': id_,
                        'question': sample['question'],
                        'generation_seq_1': generation_most_likely_seq1,
                        'generation_seq_2': generation_most_likely_seq2,
                        'sim_score': float(similarity)
                    }
                    jl_ofile.write(json.dumps(result_item) + '\n')
                    
                
                else:
                    print(f"\nQuery {id_} is not common between two sequences !!!")
            
        return agree_list, non_agree_list
    
    def get_nli_relations(axioms, queries_list):
        # Src: https://huggingface.co/vectara/hallucination_evaluation_model
        # Input: a list of pairs of (premise, hypothesis)
        # It returns a score between 0 and 1 for each pair where
        # 0 means that the hypothesis is not evidenced at all by the premise and
        # 1 means the hypothesis is fully supported by the premise.
        
        if axioms == "123":
            results_file = axioms123_output_jsonl_file
            sequences = sequences_main
        elif axioms == "4":
            results_file = axiom4_output_jsonl_file
            sequences = sequences_main
        elif axioms == "5":
            results_file = axiom5_output_jsonl_file
            sequences = sequences_secondry
        else:
            print(f"No valid axiom number !!!")
        
        if os.path.isfile(results_file):
            print(f"{results_file} exists.")
            with open(results_file, 'r') as file:
                relation_queries = json.load(file) 
        else:
            relation_queries = {
                'entailment': [],
                'neutral': [],
                'contradiction': []
            }
        
            with torch.no_grad():
                for idx, sample in tqdm(enumerate(sequences)):
                    id_ = sample['id']
                    
                    if id_ in queries_list:
                        
                        # Get and prepare variables
                        question = sample['question']
                        generated_text_most_likely = sample['most_likely_generation']
                        prompt_text = sample['prompt_text']
                        doc_text = prompt_text.split('Document:')[-1].split('Question:')[0]
                        answer_ = f"{question} {generated_text_most_likely}"
            
                        # Method 4) Long NLI 
                        predicted_score_ = long_nli_model([dict(text=doc_text, text_pair=answer_)])
                        item = (id_, predicted_score_[0]['score'])
                        relation_queries[predicted_score_[0]['label']].append(item)
            
            
            # Write to file 
            with open(results_file, 'w') as file:
                json.dump(relation_queries, file, indent=4)
        
        return relation_queries

    # ======
    uncertainty_methods = ['PE', 'SE'] #  'PE_MARS', 'SE_MARS'
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
        } 
    }
    
    result_df_main = create_result_df(args.main_prompt_format)
    result_df_main_filtered_pe = result_df_main[result_df_main['average_predictive_entropy_main_prompt'] <= 100]
    result_df_main_filtered_se = result_df_main[result_df_main['predictive_entropy_over_concepts_main_prompt'] <= 100]
    
    result_df_second_prompt = create_result_df(args.second_prompt_format)
    result_df_second_prompt_filtered_pe = result_df_second_prompt[result_df_second_prompt['average_predictive_entropy_main_prompt'] <= 100]
    result_df_second_prompt_filtered_se = result_df_second_prompt[result_df_second_prompt['predictive_entropy_over_concepts_main_prompt'] <= 100]
    
    # First check: answer1 is equal to answer2
    if os.path.isfile(similarity_output_jsonl_file):
        print(f"{similarity_output_jsonl_file} exists.")
        threshold = 0.5
        agree_list, non_agree_list = [], []
        with open(similarity_output_jsonl_file, 'r') as file:
            ids = []
            for line in file:
                sample = json.loads(line.strip())
                ids.append(sample['id'])
                if sample['sim_score'] > threshold:
                    agree_list.append(sample['id'])
                else:
                    non_agree_list.append(sample['id'])
    else:
        print("Computing similarity ...")
        similarity_model_name = "cross-encoder/stsb-roberta-large"
        similarity_model = CrossEncoder(model_name=similarity_model_name, num_labels=1)
        similarity_model.model.to(args.device)
        agree_list, non_agree_list = get_aggreement(sequences_main, sequences_secondry)
    
    # === Main Computation ========================
    # print("================= Axioms: 1, 2, 3 ==========")
    # axioms_num = '123'
    # axioms_123 = get_nli_relations(axioms_num, agree_list)
    # print(f"Entailment:    {len(axioms_123['entailment'])} ({(len(axioms_123['entailment']) / len(sequences_main))*100:.2f}%)")
    # print(f"Contradiction: {len(axioms_123['contradiction'])} ({len(axioms_123['contradiction']) / len(sequences_main)*100:.2f}%)")
    # print(f"Neutral:       {len(axioms_123['neutral'])} ({len(axioms_123['neutral']) / len(sequences_main)*100:.2f}%)")
    # print('\n')
    
    # for uncertainty_model in uncertainty_methods: 
        
    #     if uncertainty_model in ['PE', 'PE_MARS']:
    #         result_df_main_prompt = result_df_main_filtered_pe
    #         result_df_second_prompt = result_df_second_prompt_filtered_pe
    #     elif uncertainty_model in ['SE', 'SE_MARS']:
    #         result_df_main_prompt = result_df_main_filtered_se
    #         result_df_second_prompt = result_df_second_prompt_filtered_se
        
    #     unc_model_key_main_prompt = keys_mapping['main_prompt'][uncertainty_model]
    #     unc_model_key_second_prompt = keys_mapping['second_prompt'][uncertainty_model]
        
        
        # ==== Test axiom 1 ===================
        # relation_key = 'entailment'
        # selected_list = axioms_123[relation_key]
        
        # if len(selected_list) > 0:
        #     _, correctness_main_prompt_bin, one_minus_correctness_main_prompt = get_correctness(result_df_main_prompt)
        #     _, correctness_second_prompt_bin, one_minus_correctness_second_prompt = get_correctness(result_df_second_prompt)
        #     correctness_main_prompt = 1 - np.array(one_minus_correctness_main_prompt)
        #     correctness_second_prompt = 1 - np.array(one_minus_correctness_second_prompt)
            
        #     selected_list_ = [tup[0] for tup in selected_list]
        #     uncertainty_main_prompt_values_test = np.where(
        #         result_df_main_prompt['id'].isin(selected_list_),
        #         result_df_main_prompt[unc_model_key_main_prompt] * 0.2,
        #         result_df_main_prompt[unc_model_key_main_prompt]
        #     )
        #     uncertainty_main_prompt_values =  result_df_main_prompt[unc_model_key_main_prompt]
        #     uncertainty_second_prompt_values = result_df_second_prompt[unc_model_key_main_prompt]
            
        #     if len(set(correctness_main_prompt_bin)) == 1:
        #         print("Warning: Only one class present in y_true. ROC AUC score is not defined.")
        #         auroc_main_prompt_test = 0.5
        #         auroc_main_prompt = 0.5
        #         auroc_second_prompt = 0.5
        #     else:
        #         auroc_main_prompt_test = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_main_prompt_values_test)
        #         auroc_main_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_main_prompt_values)
        #         auroc_second_prompt = sklearn.metrics.roc_auc_score(1 - correctness_second_prompt_bin, uncertainty_second_prompt_values)

        #     print(f"{uncertainty_model}, Axiom1: {relation_key}")
        #     print(f"Uncertainty: {uncertainty_second_prompt_values.mean():.3f} -> {uncertainty_main_prompt_values.mean():.3f} | {uncertainty_main_prompt_values_test.mean():.3f}")
        #     print(f"Acc. ({args.accuracy_metric}):  {round(correctness_second_prompt.mean()*100, 2)} -> {round(correctness_main_prompt.mean()*100, 2)}")
        #     print(f"AUROC:       {round(auroc_second_prompt, 3)} -> {round(auroc_main_prompt, 3)} | {round(auroc_main_prompt_test, 3)}")
        #     print('\n')             
            

        # ==== Test axiom 2 ===================
        # relation_key = 'contradiction'
        # selected_list = axioms_123[relation_key]
        
        # if len(selected_list) > 0:
        #     _, correctness_main_prompt_bin, one_minus_correctness_main_prompt = get_correctness(result_df_main_prompt)
        #     _, correctness_second_prompt_bin, one_minus_correctness_second_prompt = get_correctness(result_df_second_prompt)
        #     correctness_main_prompt = 1 - np.array(one_minus_correctness_main_prompt)
        #     correctness_second_prompt = 1 - np.array(one_minus_correctness_second_prompt)
            
        #     selected_list_ = [tup[0] for tup in selected_list]
        #     uncertainty_main_prompt_values_test = np.where(
        #         result_df_main_prompt['id'].isin(selected_list_),
        #         result_df_main_prompt[unc_model_key_main_prompt] * 2.0,
        #         result_df_main_prompt[unc_model_key_main_prompt]
        #     )
        #     uncertainty_main_prompt_values =  result_df_main_prompt[unc_model_key_main_prompt]
        #     uncertainty_second_prompt_values = result_df_second_prompt[unc_model_key_main_prompt]
            
        #     if len(set(correctness_main_prompt_bin)) == 1:
        #         print("Warning: Only one class present in y_true. ROC AUC score is not defined.")
        #         auroc_main_prompt_test = 0.5
        #         auroc_main_prompt = 0.5
        #         auroc_second_prompt = 0.5
        #     else:
        #         auroc_main_prompt_test = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_main_prompt_values_test)
        #         auroc_main_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_main_prompt_values)
        #         auroc_second_prompt = sklearn.metrics.roc_auc_score(1 - correctness_second_prompt_bin, uncertainty_second_prompt_values)

        #     print(f"{uncertainty_model}, Axiom1: {relation_key}")
        #     print(f"Uncertainty: {uncertainty_second_prompt_values.mean():.3f} -> {uncertainty_main_prompt_values.mean():.3f} | {uncertainty_main_prompt_values_test.mean():.3f}")
        #     print(f"Acc. ({args.accuracy_metric}):  {round(correctness_second_prompt.mean()*100, 2)} -> {round(correctness_main_prompt.mean()*100, 2)}")
        #     print(f"AUROC:       {round(auroc_second_prompt, 3)} -> {round(auroc_main_prompt, 3)} | {round(auroc_main_prompt_test, 3)}")
        #     print('\n') 


        # ==== Test axiom 3 ===================
        # relation_key = 'neutral'
        # selected_list = axioms_123[relation_key]
        # selected_list_ = [tup[0] for tup in selected_list]
        
        # if len(selected_list) > 0:
        #     _, correctness_main_prompt_bin, one_minus_correctness_main_prompt = get_correctness(result_df_main_prompt)
        #     _, correctness_second_prompt_bin, one_minus_correctness_second_prompt = get_correctness(result_df_second_prompt)
        #     correctness_main_prompt = 1 - np.array(one_minus_correctness_main_prompt)
        #     correctness_second_prompt = 1 - np.array(one_minus_correctness_second_prompt)
            
        #     uncertainty_main_prompt_values =  result_df_main_prompt[unc_model_key_main_prompt]
        #     uncertainty_second_prompt_values = result_df_second_prompt[unc_model_key_main_prompt]
            
        #     uncertainty_main_prompt_values_test = [
        #         result_df_second_prompt.loc[result_df_second_prompt['id'] == row_id, unc_model_key_main_prompt].values[0]
        #         if row_id in selected_list_ and not result_df_second_prompt.loc[result_df_second_prompt['id'] == row_id].empty
        #         else main_value
        #         for row_id, main_value in zip(result_df_main_prompt['id'], result_df_main_prompt[unc_model_key_main_prompt])
        #     ]
        #     uncertainty_main_prompt_values_test = pd.Series(uncertainty_main_prompt_values_test)

        #     if len(set(correctness_main_prompt_bin)) == 1:
        #         print("Warning: Only one class present in y_true. ROC AUC score is not defined.")
        #         auroc_main_prompt_test = 0.5
        #         auroc_main_prompt = 0.5
        #         auroc_second_prompt = 0.5
        #     else:
        #         auroc_main_prompt_test = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_main_prompt_values_test)
        #         auroc_main_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_main_prompt_values)
        #         auroc_second_prompt = sklearn.metrics.roc_auc_score(1 - correctness_second_prompt_bin, uncertainty_second_prompt_values)

        #     print(f"{uncertainty_model}, Axiom1: {relation_key}")
        #     print(f"Uncertainty: {uncertainty_second_prompt_values.mean():.3f} -> {uncertainty_main_prompt_values.mean():.3f} | {uncertainty_main_prompt_values_test.mean():.3f}")
        #     print(f"Acc. ({args.accuracy_metric}):  {round(correctness_second_prompt.mean()*100, 2)} -> {round(correctness_main_prompt.mean()*100, 2)}")
        #     print(f"AUROC:       {round(auroc_second_prompt, 3)} -> {round(auroc_main_prompt, 3)} | {round(auroc_main_prompt_test, 3)}")
        #     print('\n') 


    print("================= Axiom: 4 =================")
    axiom_num = '4'
    axiom_4 = get_nli_relations(axiom_num, non_agree_list)
    print(f"Entailment:    {len(axiom_4['entailment'])} ({(len(axiom_4['entailment']) / len(sequences_main))*100:.2f}%)")

    print('\n')
    relation_key = 'entailment'
    selected_list = axiom_4[relation_key]
    selected_list_ = [tup[0] for tup in selected_list]
    
    if len(selected_list) > 0:
        for uncertainty_model in uncertainty_methods:
            
            if uncertainty_model in ['PE', 'PE_MARS']:
                result_df_main_prompt = result_df_main_filtered_pe
                result_df_second_prompt = result_df_second_prompt_filtered_pe
            elif uncertainty_model in ['SE', 'SE_MARS']:
                result_df_main_prompt = result_df_main_filtered_se
                result_df_second_prompt = result_df_second_prompt_filtered_se
                
            unc_model_key_main_prompt = keys_mapping['main_prompt'][uncertainty_model]
            unc_model_key_second_prompt = keys_mapping['second_prompt'][uncertainty_model]

    
            _, correctness_main_prompt_bin, one_minus_correctness_main_prompt = get_correctness(result_df_main_prompt)
            correctness_main_prompt = 1 - np.array(one_minus_correctness_main_prompt)

            uncertainty_prompt_values =  result_df_main_prompt[['id', unc_model_key_main_prompt, unc_model_key_second_prompt]]
            uncertainty_main_prompt_values = uncertainty_prompt_values[unc_model_key_main_prompt]
            uncertainty_second_prompt_values = uncertainty_prompt_values[unc_model_key_second_prompt]
            uncertainty_main_prompt_values_test = [
                row[unc_model_key_second_prompt] if row['id'] in selected_list_ else row[unc_model_key_main_prompt]
                for _, row in uncertainty_prompt_values.iterrows()
            ]
            uncertainty_main_prompt_values_test = pd.Series(uncertainty_main_prompt_values_test)
            
            if len(set(correctness_main_prompt_bin)) == 1:
                print("Warning: Only one class present in y_true. ROC AUC score is not defined.")
                auroc_main_prompt_test = 0.5
                auroc_main_prompt = 0.5
                auroc_second_prompt = 0.5
            else:
                auroc_main_prompt_test = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_main_prompt_values_test)
                auroc_main_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_main_prompt_values)
                auroc_second_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_second_prompt_values)

            print(f"{uncertainty_model}, Axiom1: {relation_key}")
            print(f"Uncertainty: {uncertainty_second_prompt_values.mean():.3f} -> {uncertainty_main_prompt_values.mean():.3f} | {uncertainty_main_prompt_values_test.mean():.3f}")
            print(f"Acc. ({args.accuracy_metric}): {round(correctness_main_prompt.mean()*100, 2)}")
            print(f"AUROC:       {round(auroc_second_prompt, 3)} -> {round(auroc_main_prompt, 3)} | {round(auroc_main_prompt_test, 3)}")
            print('\n') 
            
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--model_llama_eval', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='trivia', choices=[
        'webquestions', 'nq', 'trivia', 'squad1',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa_org', 'topicoqa_his', 'topicoqa_rw',
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--main_prompt_format', type=str, default='q_negative', choices=[
        'only_q', 'q_positive', 'q_negative',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    parser.add_argument('--second_prompt_format', type=str, default='only_q', choices=[
        'only_q', 'q_positive', 'q_negative',
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
    
    # parser.add_argument('--with_groundedness', type=str, default='yes', choices=['no', 'yes'])
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
    test_axiomatic_idea(args)
    
    # python framework/run/test_axiomatric_idea.py
    
