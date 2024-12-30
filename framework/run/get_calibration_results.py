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
    
    def create_result_df(prompt_format):
        
        generation_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_cleaned_generation_{args.generation_type}.pkl'
        similarities_input_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_similarities_generation.pkl'
        uncertainty_mars_input_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_uncertainty_mars_generation.pkl'
        uncertainty_bb_input_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_uncertainty_bb_generation.pkl'
        correctness_input_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_correctness.pkl'
        # groundedness_input_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_groundedness_generation__sec_{args.second_prompt_format}.pkl'
        
        with open(generation_file, 'rb') as infile:
            cleaned_sequences = pickle.load(infile)
        with open(similarities_input_file, 'rb') as f:
            similarities_dict = pickle.load(f)
        with open(uncertainty_mars_input_file, 'rb') as f:
            uncertainty_mars_results  = pickle.load(f)
        # with open(uncertainty_bb_input_file, 'rb') as f:
        #     uncertainty_bb_results  = pickle.load(f)
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
            
            'average_predictive_entropy_third_prompt', 'predictive_entropy_over_concepts_third_prompt',
            'average_predictive_entropy_importance_max_third_prompt', 'predictive_entropy_over_concepts_importance_max_third_prompt',
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
        # uncertainty_bb_df = pd.DataFrame(uncertainty_bb_results)
        # uncertainty_bb_keys_to_use = ('id', 'degree_u', 'ecc_u', 'spectral_u')
        # uncertainty_bb_small = dict((k, uncertainty_bb_df[k]) for k in uncertainty_bb_keys_to_use)
        # uncertainty_bb_df = pd.DataFrame.from_dict(uncertainty_bb_small)
        
        # 
        # groundedness_df = pd.DataFrame(groundedness_results)

        # 
        result_df = generations_df.merge(similarities_df, on='id').merge(uncertainty_mars_df, on='id').merge(correctness_df, on='id') #.merge(uncertainty_bb_df, on='id') # .merge(groundedness_df, on='id')
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
            'SE_MARS': 'predictive_entropy_over_concepts_importance_max_main_prompt',
            'EigV': 'spectral_u',
            'Ecc': 'ecc_u',
            'Deg': 'degree_u',
            
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
    ece_estimate = ECE_estimate()
    
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
        plt.savefig(f'{base_dir_output}/{args.main_prompt_format}/calibration_results/ECE_correctness_vs_confidence_{prefix}.png')
    
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
        plt.savefig(f'{base_dir_output}/{args.main_prompt_format}/calibration_results/ECE_correctness_vs_uncertainty_{prefix}.png')
    
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
        plt.savefig(f'{base_dir_output}/{args.main_prompt_format}/calibration_results/ECE_correctness_vs_uncertainty_{prefix}.png')
    
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
        plt.savefig(f'{base_dir_output}/{args.main_prompt_format}/calibration_results/RCE_correctness_vs_uncertainty_{prefix}.png')
    
    def run_calibration_metrics(uncertainty_model):
        
        if uncertainty_model in ['PE', 'PE_MARS']:
            result_df = result_df_main_filtered_pe
        elif uncertainty_model in ['SE', 'SE_MARS']:
            result_df = result_df_main_filtered_se
        else:
            result_df = result_df_main
        
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
        
        # coef_1 = result_df['most_likely_kl_main_second'].apply(lambda x: x.get('max', None))
        # coef_2 = result_df['most_likely_kl_second_third'].apply(lambda x: x.get('max', None))
        # coef_3 = result_df['most_likely_kl_main_third'].apply(lambda x: x.get('max', None)) 
        # coef_1_ = coef_1 / (coef_1+coef_2)
        # coef_2_ = coef_2 / (coef_1+coef_2)
        
        ### === For second prompt
        unc_model_key_second_prompt = keys_mapping['second_prompt'][uncertainty_model]
        uncertainty_values_second_prompt = result_df[f"{unc_model_key_second_prompt}"]        
        auroc_test2 = sklearn.metrics.roc_auc_score(1 - correctness_bin, uncertainty_values_second_prompt)
        confidence_values_second_prompt = uncertainty_to_confidence_min_max(uncertainty_values_second_prompt)
        ece_test2 = ece_estimate(correctness, confidence_values_second_prompt)
        plot_correctness_vs_uncertainty(
            correctness, uncertainty_values_second_prompt,
            f'AUROC: {round(auroc_test2, 4)}\nECE: {round(ece_test2, 4)}',
            prefix=f"{uncertainty_model}_second_prompt", num_bins=40
        )
        
        ### === For third prompt
        unc_model_key_third_prompt = keys_mapping['third_prompt'][uncertainty_model]
        uncertainty_values_third_prompt = result_df[f"{unc_model_key_third_prompt}"]
        # print(uncertainty_values_third_prompt.nsmallest(10))
        # print(uncertainty_values_third_prompt.nlargest(10))
        auroc_test2 = sklearn.metrics.roc_auc_score(1 - correctness_bin, uncertainty_values_third_prompt)
        confidence_values_third_prompt = uncertainty_to_confidence_min_max(uncertainty_values_third_prompt)
        ece_test2 = ece_estimate(correctness, confidence_values_third_prompt)
        plot_correctness_vs_uncertainty(
            correctness, uncertainty_values_third_prompt,
            f'AUROC: {round(auroc_test2, 4)}\nECE: {round(ece_test2, 4)}',
            prefix=f"{uncertainty_model}_third_prompt", num_bins=40
        )
        
        
        ### === Combine first & second prompts 
        
        # 1)
        # uncertainty_multiply_values = uncertainty_values * uncertainty_values_second_prompt
        # auroc_test2 = sklearn.metrics.roc_auc_score(1 - correctness_bin, uncertainty_multiply_values)
        # confidence_multiply_values = uncertainty_to_confidence_min_max(uncertainty_multiply_values)
        # ece_test2 = ece_estimate(correctness, confidence_multiply_values)
        # plot_correctness_vs_uncertainty(
        #     correctness, uncertainty_multiply_values,
        #     f'AUROC: {round(auroc_test2, 4)}\nECE: {round(ece_test2, 4)}',
        #     prefix=f"{uncertainty_model}_multiply", num_bins=40
        # )
        
        # 2)
        # uncertainty_sum_values = (uncertainty_values + uncertainty_values_second_prompt)/2
        # # uncertainty_sum_values = (coef_1_*uncertainty_values + coef_2_*uncertainty_values_second_prompt)/2
        # auroc_test2 = sklearn.metrics.roc_auc_score(1 - correctness_bin, uncertainty_sum_values)
        # confidence_sum_values = uncertainty_to_confidence_min_max(uncertainty_sum_values)
        # ece_test2 = ece_estimate(correctness, confidence_sum_values)
        # plot_correctness_vs_uncertainty(
        #     correctness, uncertainty_sum_values,
        #     f'AUROC: {round(auroc_test2, 4)}\nECE: {round(ece_test2, 4)}',
        #     prefix=f"{uncertainty_model}_sum", num_bins=40
        # )
        
        # 3)
        # uncertainty_abs_values = abs(uncertainty_values - uncertainty_values_second_prompt)
        # auroc_test2 = sklearn.metrics.roc_auc_score(1 - correctness_bin, uncertainty_abs_values)
        # confidence_abs_values = uncertainty_to_confidence_min_max(uncertainty_abs_values)
        # ece_test2 = ece_estimate(correctness, confidence_abs_values)
        # plot_correctness_vs_uncertainty(
        #     correctness, uncertainty_abs_values,
        #     f'AUROC: {round(auroc_test2, 4)}\nECE: {round(ece_test2, 4)}',
        #     prefix=f"{uncertainty_model}_abs", num_bins=40
        # )
        
        # # 4)
        # uncertainty_sum_values = 0.5*uncertainty_values + 0.4*uncertainty_values_second_prompt + 0.1*uncertainty_values_third_prompt
        # auroc_test2 = sklearn.metrics.roc_auc_score(1 - correctness_bin, uncertainty_sum_values)
        # confidence_sum_values = uncertainty_to_confidence_min_max(uncertainty_sum_values)
        # ece_test2 = ece_estimate(correctness, confidence_sum_values)
        # plot_correctness_vs_uncertainty(
        #     correctness, uncertainty_sum_values,
        #     f'AUROC: {round(auroc_test2, 4)}\nECE: {round(ece_test2, 4)}',
        #     prefix=f"{uncertainty_model}_sum3", num_bins=40
        # )
        

    result_dict = {}
    for uncertainty_model in ['PE', 'SE']: #, 'PE_MARS', 'SE_MARS', 'EigV', 'Ecc', 'Deg'
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
    
    parser.add_argument('--accuracy_metric', type=str, default="bert_score", choices=[
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