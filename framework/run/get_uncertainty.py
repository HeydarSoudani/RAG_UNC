#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import pickle
import logging
import argparse
import numpy as np

from utils import set_seed


def get_uncertainty(args):
    
    print("\n--- Step 4: Get uncertainty ...")
    print(f"""
        Model name: {args.model}
        Dataset: {args.dataset}
        Prompt format: {args.prompt_format}
        Run id: {args.run_id}
        Seed: {args.seed}
    """.replace('   ', ''))

    llh_shift = torch.tensor(5.0) # does not effect anything
    
    # === Define output files ========================
    # === Read the likelihoods data ==================
    model = args.model.split('/')[-1]
    uncertainty_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_uncertainty_generation.pkl'
    likelihoods_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_likelihoods_generation.pkl'
    
    with open(likelihoods_file, 'rb') as infile:
        result = pickle.load(infile)

    # === Functions ==================================
    def get_overall_log_likelihoods(list_of_results):
        """Compute log likelihood of all generations under their given context.
        list_of_results: list of dictionaries with keys:
        returns: dictionary with keys: 'neg_log_likelihoods', 'average_neg_log_likelihoods'
                that contains tensors of shape (num_models, num_generations, num_samples_per_generation)
        """

        result_dict = {}
        geometric_dict ={}

        # list_of_keys = ['neg_log_likelihoods', 'average_neg_log_likelihoods',\
        #                 'pointwise_mutual_information', 'average_neg_log_likelihood_of_most_likely_gen',\
        #                 'neg_log_likelihood_of_most_likely_gen', 'semantic_set_ids', \
        #                 'average_neg_log_likelihoods_importance_mean', 'average_neg_log_likelihoods_importance_max', 'average_neg_log_likelihoods_importance_min',\
        #                 'most_likely_neg_log_likelihoods', 
        #                 'most_likely_neg_log_likelihoods_importance_mean', 'most_likely_neg_log_likelihoods_importance_max', 'most_likely_neg_log_likelihoods_importance_min']
        list_of_keys = [
            'average_neg_log_likelihoods',
            'average_neg_log_likelihoods_importance_mean',
            'average_neg_log_likelihoods_importance_max',
            'average_neg_log_likelihoods_importance_min',
            'most_likely_neg_log_likelihoods', 
            'most_likely_neg_log_likelihoods_importance_mean',
            'most_likely_neg_log_likelihoods_importance_max',
            'most_likely_neg_log_likelihoods_importance_min',
            'similarity_score',
            'semantic_set_ids',
        ]


        geometric_keys = ['has_different_answers','unique_answers_indices']

        for key in geometric_keys:
            overall_results = []
            for sample in list_of_results:
                overall_results.append(sample[key])
            geometric_dict[key]  = overall_results

        for key in list_of_keys:
            list_of_ids = []
            overall_results = []
            results_per_model = []
            for sample in list_of_results:
                average_neg_log_likelihoods = sample[key]
                list_of_ids.append(sample['id'])
                results_per_model.append(average_neg_log_likelihoods)

            results_per_model = [torch.tensor(x) if isinstance(x, int) else x for x in results_per_model]
            results_per_model = torch.stack(results_per_model)

            overall_results.append(results_per_model)

            if key not in ['meaning_vectors', 'meaning_vectors_only_answer','has_different_answers']:
                overall_results = torch.stack(overall_results)

            result_dict[key] = overall_results

        result_dict['ids'] = list_of_ids
        return result_dict, geometric_dict
    
    def get_mutual_information(log_likelihoods):
        """Compute confidence measure for a given set of likelihoods"""

        mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
        tiled_mean = mean_across_models.tile(log_likelihoods.shape[0], 1, 1)
        diff_term = torch.exp(log_likelihoods) * log_likelihoods - torch.exp(tiled_mean) * tiled_mean
        f_j = torch.div(torch.sum(diff_term, dim=0), diff_term.shape[0])
        mutual_information = torch.div(torch.sum(torch.div(f_j, mean_across_models), dim=1), f_j.shape[-1])

        return mutual_information
    
    def get_log_likelihood_variance(neg_log_likelihoods):
        """Compute log likelihood variance of approximate posterior predictive"""
        mean_across_models = torch.mean(neg_log_likelihoods, dim=0)
        variance_of_neg_log_likelihoods = torch.var(mean_across_models, dim=1)

        return variance_of_neg_log_likelihoods
    
    def get_log_likelihood_mean(neg_log_likelihoods):
        """Compute softmax variance of approximate posterior predictive"""
        mean_across_models = torch.mean(neg_log_likelihoods, dim=0)
        mean_of_neg_log_likelihoods = torch.mean(mean_across_models, dim=1)

        return mean_of_neg_log_likelihoods

    def get_mean_of_poinwise_mutual_information(pointwise_mutual_information):
        """Compute mean of pointwise mutual information"""
        mean_across_models = torch.mean(pointwise_mutual_information, dim=0)
        return torch.mean(mean_across_models, dim=1)

    def get_predictive_entropy(log_likelihoods):
        """Compute predictive entropy of approximate posterior predictive"""
        #log_likelihoods = log_likelihoods[:,:,:1]
        mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
        entropy = -torch.sum(mean_across_models, dim=1) / torch.tensor(mean_across_models.shape[1])
        return entropy

    def get_predictive_entropy_over_concepts(log_likelihoods, semantic_set_ids):
        """Compute the semantic entropy"""
        #log_likelihoods = log_likelihoods[:,:,:1]
        mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
        # This is ok because all the models have the same semantic set ids
        semantic_set_ids = semantic_set_ids[0]
        entropies = []
        for row_index in range(mean_across_models.shape[0]):
            aggregated_likelihoods = []
            row = mean_across_models[row_index]
            semantic_set_ids_row = semantic_set_ids[row_index]
            #semantic_set_ids_row = semantic_set_ids_row[:1]
            for semantic_set_id in torch.unique(semantic_set_ids_row):
                aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0))
            aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - llh_shift
            entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
            entropies.append(entropy)

        return torch.tensor(entropies)

    def get_margin_probability_uncertainty_measure(log_likelihoods):
        """Compute margin probability uncertainty measure"""
        mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
        topk_likelihoods, indices = torch.topk(mean_across_models, 2, dim=1, sorted=True)
        margin_probabilities = np.exp(topk_likelihoods[:, 0]) - np.exp(topk_likelihoods[:, 1])

        return margin_probabilities

    def get_number_of_unique_elements_per_row(tensor):
        assert len(tensor.shape) == 2
        return torch.count_nonzero(torch.sum(torch.nn.functional.one_hot(tensor), dim=1), dim=1)


    # === Main part ==================
    overall_results, geometric_results = get_overall_log_likelihoods(result)
    average_pointwise_mutual_information = get_mean_of_poinwise_mutual_information(overall_results['pointwise_mutual_information'])

    mutual_information = get_mutual_information(overall_results['neg_log_likelihoods'])
    predictive_entropy = get_predictive_entropy(-overall_results['neg_log_likelihoods'])
    predictive_entropy_over_concepts = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods'],
        overall_results['semantic_set_ids']
    )

    unnormalised_entropy_over_concepts = get_predictive_entropy_over_concepts(
        -overall_results['neg_log_likelihoods'],
        overall_results['semantic_set_ids']
    ) # proposed algorithm
    margin_measures = get_margin_probability_uncertainty_measure(-overall_results['average_neg_log_likelihoods'])
    unnormalised_margin_measures = get_margin_probability_uncertainty_measure(-overall_results['neg_log_likelihoods'])

    scores_prob = overall_results['most_likely_neg_log_likelihoods']
    scores_importance_mean = overall_results['most_likely_neg_log_likelihoods_importance_mean']
    scores_importance_max = overall_results['most_likely_neg_log_likelihoods_importance_max']
    scores_importance_min = overall_results['most_likely_neg_log_likelihoods_importance_min']

    predictive_entropy_over_concepts_importance_mean = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_mean'],
        overall_results['semantic_set_ids']
    )    
    predictive_entropy_over_concepts_importance_max = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_max'],
        overall_results['semantic_set_ids']
    )    
    predictive_entropy_over_concepts_importance_min = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_min'],
        overall_results['semantic_set_ids']
    ) 
    
    number_of_semantic_sets = get_number_of_unique_elements_per_row(overall_results['semantic_set_ids'][0])
    average_predictive_entropy = get_predictive_entropy(-overall_results['average_neg_log_likelihoods'])

    average_predictive_entropy_importance_mean = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_mean'])
    average_predictive_entropy_importance_max = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_max'])
    average_predictive_entropy_importance_min = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_min'])

    overall_results['mutual_information'] = mutual_information
    overall_results['predictive_entropy'] = predictive_entropy
    overall_results['predictive_entropy_over_concepts'] = predictive_entropy_over_concepts
    overall_results['unnormalised_entropy_over_concepts'] = unnormalised_entropy_over_concepts
    overall_results['number_of_semantic_sets'] = number_of_semantic_sets
    overall_results['margin_measures'] = margin_measures
    overall_results['unnormalised_margin_measures'] = unnormalised_margin_measures

    overall_results['scores_prob'] = scores_prob
    overall_results['scores_importance_mean'] = scores_importance_mean
    overall_results['scores_importance_max'] = scores_importance_max
    overall_results['scores_importance_min'] = scores_importance_min

    overall_results['average_predictive_entropy'] = average_predictive_entropy
    overall_results['average_pointwise_mutual_information'] = average_pointwise_mutual_information

    overall_results['average_predictive_entropy_importance_mean'] = average_predictive_entropy_importance_mean
    overall_results['average_predictive_entropy_importance_max'] = average_predictive_entropy_importance_max
    overall_results['average_predictive_entropy_importance_min'] = average_predictive_entropy_importance_min

    overall_results['predictive_entropy_over_concepts_importance_mean'] = predictive_entropy_over_concepts_importance_mean
    overall_results['predictive_entropy_over_concepts_importance_max'] = predictive_entropy_over_concepts_importance_max
    overall_results['predictive_entropy_over_concepts_importance_min'] = predictive_entropy_over_concepts_importance_min


    ### === Save the uncertainty result ============
    with open(uncertainty_output_file, 'wb') as ofile:
        pickle.dump(overall_results, ofile)
    print(f"Results saved to {uncertainty_output_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--dataset', type=str, default='webquestions', choices=[
        'trivia', 'nq', 'squad1', 'webquestions',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa_org', 'topicoqa_his', 'topicoqa_rw',
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--prompt_format', type=str, default='only_q', choices=[
        'only_q', 'q_positive', 'q_negative'
    ])
    parser.add_argument('--accuracy_metric', type=str, default="bem_score", choices=[
        'bem_score', 'exact_match', 'bert_score', 'rouge_score', 'llama3_score', 'gpt_score'
    ])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.05)
    parser.add_argument("--roc_auc_threshold", type=float, default=0.8)
    parser.add_argument("--output_file_postfix", type=str, default="")
    
    parser.add_argument('--num_generations_per_prompt', type=int, default=10)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--type_of_question', type=str)
    parser.add_argument('--decoding_method', type=str, default='beam_search')
    parser.add_argument('--temperature', type=float, default='1.0')
    parser.add_argument('--num_beams', type=int, default='1')
    parser.add_argument('--top_p', type=float, default=1.0)
    
    parser.add_argument('--with_groundedness', type=str, default='yes', choices=['no', 'yes'])
    parser.add_argument('--run_id', type=str, default='run_15')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    args.output_dir = "framework/run_output"
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected.")
    
    get_uncertainty(args)
    
    # python framework/run/get_uncertainty.py
    
   
