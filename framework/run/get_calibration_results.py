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
import sklearn.metrics
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sentence_transformers.cross_encoder import CrossEncoder

from utils import set_seed, uncertainty_to_confidence_min_max, uncertainty_to_confidence_gaussian, uncertainty_to_confidence_sigmoid, uncertainty_to_confidence_tanh
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
    base_dir_output = f'{args.output_dir}/{args.dataset}/{args.run_id}/'
    similarity_output_jsonl_file = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_similarity_output__sec_{args.second_prompt_format}.jsonl'
    calibration_output_file = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_calibration_results.jsonl'

    sequence_input_main = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
    sequence_input_secondry = f'{base_dir_output}/{args.second_prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
    with open(sequence_input_main, 'rb') as infile:
        sequences_main = pickle.load(infile)
    with open(sequence_input_secondry, 'rb') as infile:
        sequences_secondry = pickle.load(infile)
    
    
    def create_result_df(prompt_format):
        
        similarities_input_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_similarities_generation.pkl'
        likelihoods_input_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_uncertainty_generation.pkl'
        correctness_input_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_correctness.pkl'
        generation_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
        
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
        correctness_keys_to_use = ('id', 'bem_score', 'bert_score', 'exact_match')
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
        result_df = generations_df.merge(similarities_df, on='id').merge(likelihoods_df, on='id').merge(correctness_df, on='id')
        result_df['len_most_likely_generation_length'] = result_df['most_likely_generation'].apply(lambda x: len(x.split()))
        return result_df
    
    
    # === Filtering samples with very high entropy
    result_df_main = create_result_df(args.main_prompt_format)
    result_df_main_filtered_pe = result_df_main[result_df_main['average_predictive_entropy_main_prompt'] <= 100]
    result_df_main_filtered_se = result_df_main[result_df_main['predictive_entropy_over_concepts_main_prompt'] <= 100]
    
    # === 
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
        } 
    }
    ece_estimate = ECE_estimate()
    
    def get_correctness(results):
        correctness_results = {}
        
        if args.accuracy_metric in ['bem_score', 'rouge_score', 'gpt_score', 'exact_match']:
            correctness_bin = (results[args.accuracy_metric] > args.roc_auc_threshold).astype('int') 
        elif args.accuracy_metric == 'bert_score':
            correctness_bin = (results[args.accuracy_metric].apply(lambda x: x['F1']) > args.roc_auc_threshold).astype('int') 
        correctness_results['accuracy'] = correctness_bin.mean()
        
        # non-binarized accuracy
        correctness_results['exact_match_mean'] = results['exact_match'].mean()
        correctness_results['bem_score_mean'] = results['bem_score'].mean()
        correctness_results['bert_score_mean'] = results['bert_score'].apply(lambda x: x['F1']).mean()
        if args.accuracy_metric in ['bem_score', 'rouge_score', 'gpt_score']:
            one_minus_correctness = 1 - results[args.accuracy_metric]
        elif args.accuracy_metric == 'bert_score':
            one_minus_correctness = 1 - results[args.accuracy_metric].apply(lambda x: x['F1'])
        elif args.accuracy_metric == 'exact_match':
            one_minus_correctness = 1 - results[args.accuracy_metric].astype('int') 
        
        return correctness_results, correctness_bin, one_minus_correctness

    def plot_correctness_vs_confidence(correctness, confidence, ece_text, prefix, num_bins=10):
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_means = []  # Store mean correctness per bin
        bin_centers = []  # Store bin centers for plotting
        for i in range(num_bins):
            # Define bin range
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]
            in_bin = (confidence >= bin_lower) & (confidence < bin_upper)
            
            indices_in_bin = np.where(in_bin)[0]
            # print(indices_in_bin)

            # Calculate mean correctness in the bin
            if np.any(in_bin):
                bin_mean_correctness = correctness[in_bin].mean()
                bin_means.append(bin_mean_correctness)
            else:
                # Append NaN or zero if no samples in bin
                bin_means.append(0)
                
            bin_centers.append((bin_lower + bin_upper) / 2)

        # Plot the binned mean correctness against confidence
        plt.figure(figsize=(8, 6))
        plt.bar(bin_centers, bin_means, width=1/num_bins, color='b', alpha=0.7, edgecolor='black')
        plt.plot([0, 1], [0, 1], 'g--', label="Perfect Calibration")
        plt.text(0.05, 0.95, f'ECE: {round(ece_text, 4)}', fontsize=14, color='red', ha='left', va='top', transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round", facecolor="lightgrey", edgecolor="lightgrey"))
        
        plt.xlabel('Confidence')
        plt.ylabel('Correctness')
        plt.title(f'Correctness vs Confidence')
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.xticks(np.linspace(0.1, 1.0, 10))  # Set x-ticks as specified
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{base_dir_output}/{args.main_prompt_format}/ECE_correctness_vs_confidence_{prefix}.png')
    
    def plot_correctness_vs_uncertainty(correctness, uncertainty, metric_text, prefix, num_bins=10):
        max_uncertainty_val = 20
        
        bin_edges = np.linspace(0, max_uncertainty_val, num_bins + 1)
        bin_means = []  # Store mean correctness per bin
        bin_centers = []  # Store bin centers for plotting
        for i in range(num_bins):
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]
            in_bin = (uncertainty >= bin_lower) & (uncertainty < bin_upper)

            # Calculate mean correctness in the bin
            if np.any(in_bin):
                bin_mean_correctness = correctness[in_bin].mean()
                bin_means.append(bin_mean_correctness)
            else:
                # Append NaN or zero if no samples in bin
                bin_means.append(0)
                
            bin_centers.append((bin_lower + bin_upper) / 2)

        # Plot the binned mean correctness against confidence
        plt.figure(figsize=(8, 6))
        plt.bar(bin_centers, bin_means, width=max_uncertainty_val/num_bins, color='salmon', alpha=0.7, edgecolor='black')
        plt.plot([0, max_uncertainty_val], [1, 0], 'g--', label="Perfect Calibration")
        plt.text(0.65, 0.95, metric_text, fontsize=18, color='red', ha='left', va='top', transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round", facecolor="lightgrey", edgecolor="lightgrey"))
        
        plt.xlabel('Uncertainty')
        plt.ylabel('Correctness')
        plt.title(f'Correctness vs Uncertainty')
        plt.ylim(0, 1)
        plt.xlim(0, max_uncertainty_val)
        plt.xticks(range(0, max_uncertainty_val+1, int(max_uncertainty_val/10)))  # Set x-ticks as specified
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{base_dir_output}/{args.main_prompt_format}/ECE_correctness_vs_uncertainty_{prefix}.png')
    
    def plot_correctness_vs_uncertainty_for_axioms(correctness, uncertainty, axiom_correctness, axiom_uncertainty, metric_text, prefix, num_bins=10):
        max_uncertainty_val = 20
        bin_edges = np.linspace(0, max_uncertainty_val, num_bins + 1)
        bin_means = []
        bin_means_axiom = []
        bin_centers = []
        
        for i in range(num_bins):
            
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]
            bin_centers.append((bin_lower + bin_upper) / 2)
            in_bin = (uncertainty >= bin_lower) & (uncertainty < bin_upper)
            in_bin_axiom = (axiom_uncertainty >= bin_lower) & (axiom_uncertainty < bin_upper)
            
            if np.any(in_bin):
                bin_mean_correctness = correctness[in_bin].mean()
                bin_means.append(bin_mean_correctness)
            else:
                bin_means.append(0)
        
            if np.any(in_bin_axiom):
                bin_mean_correctness_axiom = sum(axiom_correctness[in_bin_axiom]) / len(correctness[in_bin])
                bin_means_axiom.append(bin_mean_correctness_axiom)
            else:
                bin_means_axiom.append(0)
        
        plt.figure(figsize=(8, 6))
        plt.bar(bin_centers, bin_means, width=max_uncertainty_val/num_bins, color='salmon', alpha=0.7, edgecolor='black')
        plt.bar(bin_centers, bin_means_axiom, width=max_uncertainty_val/num_bins, color='yellowgreen', alpha=0.7, edgecolor='black')
        plt.plot([0, max_uncertainty_val], [1, 0], 'g--', label="Perfect Calibration")
        plt.text(0.65, 0.95, metric_text, fontsize=18, color='red', ha='left', va='top', transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round", facecolor="lightgrey", edgecolor="lightgrey"))
        
        plt.xlabel('Uncertainty')
        plt.ylabel('Correctness')
        plt.title(f'Correctness vs Uncertainty')
        plt.ylim(0, 1)
        plt.xlim(0, max_uncertainty_val)
        plt.xticks(range(0, max_uncertainty_val+1))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{base_dir_output}/{args.main_prompt_format}/ECE_correctness_vs_uncertainty_{prefix}.png')
    
    def plot_correctness_vs_uncertainties_indication_diagram(correctness, uncertainty, rce_text, prefix, num_bins=10):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax = indication_diagram(correctness=correctness, uncertainties=uncertainty, fig=fig, ax=ax, num_bins=num_bins)
        ax.legend(loc='upper right', frameon=False, fontsize=18)
        ax.set_xlabel(f'Uncertainty', fontsize=18)
        ax.set_ylabel('Correctness', fontsize=18)
        
        plt.text(
            0.8, 0.8, f'RCE: {round(rce_text, 4)}',
            fontsize=16, color='red', ha='left', va='top', transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round", facecolor="lightgrey", edgecolor="lightgrey")
        )
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{base_dir_output}/{args.main_prompt_format}/RCE_correctness_vs_uncertainty_{prefix}.png')
    
    def run_calibration_metrics(uncertainty_model):
        
        if uncertainty_model in ['PE', 'PE_MARS']:
            result_df = result_df_main_filtered_pe
        elif uncertainty_model in ['SE', 'SE_MARS']:
            result_df = result_df_main_filtered_se
        
        correctness_results, correctness_bin, one_minus_correctness = get_correctness(result_df) 
        correctness = 1 - np.array(one_minus_correctness)
        result_dict['correctness'] = correctness_results
        
        ### === For main prompt
        result_dict[uncertainty_model]= {}
        unc_model_key_main_prompt = keys_mapping['main_prompt'][uncertainty_model]
        uncertainty_values = result_df[unc_model_key_main_prompt]
        # print(uncertainty_values.nsmallest(10))
        # print(uncertainty_values.nlargest(10))
        
        result_dict[uncertainty_model]["Unc._value"] = uncertainty_values.mean()
        result_dict[uncertainty_model]["AUROC"] = sklearn.metrics.roc_auc_score(1 - correctness_bin, uncertainty_values)
        
        ln_predictive_entropy_pearson_corr, ln_predictive_entropy_pearson_p_value = pearsonr(one_minus_correctness, uncertainty_values)
        ln_predictive_entropy_spearman_corr, ln_predictive_entropy_spearman_p_value = spearmanr(one_minus_correctness, uncertainty_values)
        result_dict[uncertainty_model]['pearson_corr'] = ln_predictive_entropy_pearson_corr
        result_dict[uncertainty_model]['pearson_p_value'] = ln_predictive_entropy_pearson_p_value
        result_dict[uncertainty_model]['spearman_corr'] = ln_predictive_entropy_spearman_corr
        result_dict[uncertainty_model]['spearman_p_value'] = ln_predictive_entropy_spearman_p_value
        
        confidence_values = uncertainty_to_confidence_min_max(uncertainty_values)
        result_dict[uncertainty_model]['conf._normalized'] = confidence_values.mean()
        
        result_dict[uncertainty_model]['ECE'] = ece_estimate(correctness, confidence_values)
        result_dict[uncertainty_model]['RCE'] = plugin_RCE_est(correctness, uncertainty_values)
        
        plot_correctness_vs_confidence(
            correctness, confidence_values,
            result_dict[uncertainty_model]['ECE'], prefix=uncertainty_model,
            num_bins=40
        )
        plot_correctness_vs_uncertainty(
            correctness, uncertainty_values,
            f"AUROC: {round(result_dict[uncertainty_model]['AUROC'], 4)}\nECE: {round(result_dict[uncertainty_model]['ECE'], 4)}",
            prefix=uncertainty_model, num_bins=40
        )
        plot_correctness_vs_uncertainties_indication_diagram(
            correctness, uncertainty_values,
            result_dict[uncertainty_model]['RCE'], prefix=uncertainty_model,
            num_bins=30
        )
        
        
        ### === For second prompt
        unc_model_key_second_prompt = keys_mapping['second_prompt'][uncertainty_model]
        uncertainty_values_second_prompt = result_df[f"{unc_model_key_second_prompt}"]
        # print(uncertainty_only_q_values.nsmallest(10))
        # print(uncertainty_only_q_values.nlargest(10))
        auroc_test2 = sklearn.metrics.roc_auc_score(1 - correctness_bin, uncertainty_values_second_prompt)
        confidence_values_second_prompt = uncertainty_to_confidence_min_max(uncertainty_values_second_prompt)
        ece_test2 = ece_estimate(correctness, confidence_values_second_prompt)
        
        plot_correctness_vs_uncertainty(
            correctness, uncertainty_values_second_prompt,
            f'AUROC: {round(auroc_test2, 4)}\nECE: {round(ece_test2, 4)}',
            prefix=f"{uncertainty_model}_second_prompt", num_bins=40
        )
        
        ### === Combime first & second prompts 
        
        # 1)
        uncertainty_multiply_values = uncertainty_values * uncertainty_values_second_prompt
        # print(uncertainty_multiply_values.nsmallest(10))
        # print(uncertainty_multiply_values.nlargest(10))
        
        auroc_test2 = sklearn.metrics.roc_auc_score(1 - correctness_bin, uncertainty_multiply_values)
        confidence_multiply_values = uncertainty_to_confidence_min_max(uncertainty_multiply_values)
        ece_test2 = ece_estimate(correctness, confidence_multiply_values)
        plot_correctness_vs_uncertainty(
            correctness, uncertainty_multiply_values,
            f'AUROC: {round(auroc_test2, 4)}\nECE: {round(ece_test2, 4)}',
            prefix=f"{uncertainty_model}_multiply", num_bins=40
        )
        
        # 2)
        uncertainty_sum_values = uncertainty_values + uncertainty_values_second_prompt
        # print(uncertainty_sum_values.nsmallest(10))
        # print(uncertainty_sum_values.nlargest(10))
        
        auroc_test2 = sklearn.metrics.roc_auc_score(1 - correctness_bin, uncertainty_sum_values)
        confidence_sum_values = uncertainty_to_confidence_min_max(uncertainty_sum_values)
        ece_test2 = ece_estimate(correctness, confidence_sum_values)
        plot_correctness_vs_uncertainty(
            correctness, uncertainty_sum_values,
            f'AUROC: {round(auroc_test2, 4)}\nECE: {round(ece_test2, 4)}',
            prefix=f"{uncertainty_model}_sum", num_bins=40
        )
        
        # 3)
        uncertainty_abs_values = abs(uncertainty_values - uncertainty_values_second_prompt)
        # print(uncertainty_abs_values.nsmallest(10))
        # print(uncertainty_abs_values.nlargest(10))
        
        auroc_test2 = sklearn.metrics.roc_auc_score(1 - correctness_bin, uncertainty_abs_values)
        confidence_abs_values = uncertainty_to_confidence_min_max(uncertainty_abs_values)
        ece_test2 = ece_estimate(correctness, confidence_abs_values)
        plot_correctness_vs_uncertainty(
            correctness, uncertainty_abs_values,
            f'AUROC: {round(auroc_test2, 4)}\nECE: {round(ece_test2, 4)}',
            prefix=f"{uncertainty_model}_abs", num_bins=40
        )

    result_dict = {}
    for uncertainty_model in ['PE', 'SE']: #, 'PE_MARS', 'SE_MARS'
        run_calibration_metrics(uncertainty_model)
    
    ### === Save the calibration result ============
    print(result_dict)
    def convert_to_serializable(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(calibration_output_file, 'w') as file:
        json.dump(result_dict, file, indent=4, default=convert_to_serializable)
    
    
    ### ===========================================
    ### ===========================================
    # === Axiomatic runs ==========================
    # print('\n')
    # def get_aggreement(sequence_1, sequence_2, threshold=0.5):
        
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
    
    # ### =================
    # # == Here both main and second come from the main files 
    # result_df_second_prompt = create_result_df(args.second_prompt_format)
    # result_df_second_prompt_filtered_pe = result_df_second_prompt[result_df_second_prompt['average_predictive_entropy_main_prompt'] <= 100]
    # result_df_second_prompt_filtered_se = result_df_second_prompt[result_df_second_prompt['predictive_entropy_over_concepts_main_prompt'] <= 100]
    
    # # First check: answer1 is equal to answer2
    # if os.path.isfile(similarity_output_jsonl_file):
    #     print(f"{similarity_output_jsonl_file} exists.")
    #     threshold = 0.5
    #     agree_list, non_agree_list = [], []
    #     with open(similarity_output_jsonl_file, 'r') as file:
    #         ids = []
    #         for line in file:
    #             sample = json.loads(line.strip())
    #             ids.append(sample['id'])
    #             if sample['sim_score'] > threshold:
    #                 agree_list.append(sample['id'])
    #             else:
    #                 non_agree_list.append(sample['id'])
    # else:
    #     print("Computing similarity ...")
    #     similarity_model_name = "cross-encoder/stsb-roberta-large"
    #     similarity_model = CrossEncoder(model_name=similarity_model_name, num_labels=1)
    #     similarity_model.model.to(args.device)
    #     agree_list, non_agree_list = get_aggreement(sequences_main, sequences_secondry)
    
    # selected_list = agree_list    # Axiom 3 (negative passage)
    # selected_list = non_agree_list
    
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

    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--model_llama_eval', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='nq', choices=[
        'trivia', 'nq', 'squad1', 'webquestions',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa_org', 'topicoqa_his', 'topicoqa_rw',
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--main_prompt_format', type=str, default='q_positive', choices=[
        'only_q', 'q_positive', 'q_negative', 'bm25_retriever', 'rerank_retriever_top5'
    ])
    parser.add_argument('--second_prompt_format', type=str, default='only_q', choices=[
        'only_q', 'q_positive', 'q_negative', 'bm25_retriever', 'rerank_retriever'
    ])
    parser.add_argument('--accuracy_metric', type=str, default="bem_score", choices=[
        'bem_score', 'exact_match', 'bert_score', 'rouge_score', 'llama3_score', 'gpt_score'
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
    get_calibration_results(args)
    
    # python framework/run/get_calibration_results.py
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # keys_to_use = ('ids', 'predictive_entropy', 'mutual_information', 'average_predictive_entropy',\
    #             'average_pointwise_mutual_information', 'average_neg_log_likelihood_of_most_likely_gen',\
    #              'neg_log_likelihood_of_most_likely_gen',\
    #             'predictive_entropy_over_concepts', 'number_of_semantic_sets', 'unnormalised_entropy_over_concepts',\
    #             'scores_importance_mean','scores_importance_max','scores_importance_min','scores_prob',\
    #             'predictive_entropy_over_concepts_importance_mean','predictive_entropy_over_concepts_importance_max','predictive_entropy_over_concepts_importance_min'\
    #             , 'average_predictive_entropy_importance_mean', 'average_predictive_entropy_importance_max', 'average_predictive_entropy_importance_min',
    #             )
    
    ### === Plot acc & uncertainty level ===========
    # fig, ax = plt.subplots(figsize=(12, 6))
    
    # ax.bar(result_df.index, 1-correctness, color='orange', alpha=0.5, label=args.accuracy_metric)
    # # ax.plot(result_df.index, np.array(min_max_normalize(se_uncertainty)), color='blue')
    # ax.plot(result_df.index, np.array(se_uncertainty), color='blue')

    # average_se_uncertainty = np.mean(se_uncertainty)
    # ax.axhline(average_se_uncertainty, color='green', linestyle='--', label='Average SE Uncertainty')


    # ax.set_xlabel('Query ID')
    # ax.set_ylabel(f"{args.accuracy_metric} / SE")
    # ax.tick_params(axis='y')

    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(f'{base_dir_output}/uncertainty_accuracy.png')

    
    ### ===  boxplot for KL ========================
    # scores = []
    # for qid, item in verification_results.items():
    #     for generation in item['groundedness_scores']:
    #         scores.extend(generation[1])
    
    # mean_score = np.mean(scores)
    # std_dev_score = np.std(scores)
    # threshold = mean_score + std_dev_score
    # binary_scores = [1 if score > threshold else 0 for score in scores]
    # # print("Mean Score:", mean_score)
    # # print("Standard Deviation:", std_dev_score)
    # # print("Threshold:", threshold)

    
    # fig, ax = plt.subplots(figsize=(6, 8))  # Customize figure size as needed
    # ax.boxplot(scores, vert=True, patch_artist=True, widths=0.6,
    #         boxprops=dict(facecolor="skyblue", color="blue"),  # Fill color of the box
    #         medianprops=dict(color="red"),  # Median line color
    #         whiskerprops=dict(color="blue"),  # Whisker color
    #         capprops=dict(color="blue"),  # Caps color
    #         flierprops=dict(marker='o', color='black', alpha=0.5))  # Outliers style

    # ax.set_title(f"{args.prompt_format}")
    # ax.set_ylabel("KL-div Score")
    # ax.set_xticks([1])  # Position of the x-tick
    # ax.set_xticklabels(["WebQuestions"])  # Label of the x-tick

    # plt.grid()
    # plt.gca().yaxis.set_major_locator(MultipleLocator(2.5))
    # plt.savefig(f'{base_dir_output}/kl_threshold.png')
    # # plt.show() 