#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pickle
import argparse

from utils.utils import set_seed


def get_uncertainty_mars(args):
    print("\n--- Step 3-2: Get Uncertainty ...")
    print(f"""
        Model name: {args.model}
        Dataset: {args.dataset}
        Prompt (1st): {args.main_prompt_format}
        Prompt (2ed): {args.second_prompt_format}
        Run id: {args.run_id}
        Seed: {args.seed}
    """.replace('   ', ''))
    llh_shift = torch.tensor(5.0) # does not effect anything

    # === Define/Read In/Out files ========================
    model = args.model.split('/')[-1]
    generation_type = f"prob_alpha_{str(args.alpha_probability)}"
    base_dir = f'{args.output_dir}/{args.dataset}/{args.subsec}/{args.run_id}/{args.main_prompt_format}__{args.second_prompt_format}'
    # inputs
    likelihoods_file = f'{base_dir}/{generation_type}/{model}_likelihoods_generation.pkl'
    # outputs
    uncertainty_output_file = f'{base_dir}/{generation_type}/{model}_uncertainty_mars_generation.pkl'

    with open(likelihoods_file, 'rb') as infile:
        likelihoods = pickle.load(infile)

    # === Functions =======================================
    def get_predictive_entropy(log_likelihoods):
        """Compute predictive entropy of approximate posterior predictive"""
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

        # print(torch.tensor(entropies))

        return torch.tensor(entropies)
    
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
            'average_neg_log_likelihoods_main_prompt',
            'average_neg_log_likelihoods_importance_mean_main_prompt',
            'average_neg_log_likelihoods_importance_max_main_prompt',
            'average_neg_log_likelihoods_importance_min_main_prompt',
            'most_likely_neg_log_likelihoods_main_prompt', 
            'most_likely_neg_log_likelihoods_importance_mean_main_prompt',
            'most_likely_neg_log_likelihoods_importance_max_main_prompt',
            'most_likely_neg_log_likelihoods_importance_min_main_prompt',
            
            'average_neg_log_likelihoods_second_prompt',
            'average_neg_log_likelihoods_importance_mean_second_prompt',
            'average_neg_log_likelihoods_importance_max_second_prompt',
            'average_neg_log_likelihoods_importance_min_second_prompt',
            'most_likely_neg_log_likelihoods_second_prompt', 
            'most_likely_neg_log_likelihoods_importance_mean_second_prompt',
            'most_likely_neg_log_likelihoods_importance_max_second_prompt',
            'most_likely_neg_log_likelihoods_importance_min_second_prompt',
            
            'average_neg_log_likelihoods_third_prompt',
            'average_neg_log_likelihoods_importance_mean_third_prompt',
            'average_neg_log_likelihoods_importance_max_third_prompt',
            'average_neg_log_likelihoods_importance_min_third_prompt',
            'most_likely_neg_log_likelihoods_third_prompt', 
            'most_likely_neg_log_likelihoods_importance_mean_third_prompt',
            'most_likely_neg_log_likelihoods_importance_max_third_prompt',
            'most_likely_neg_log_likelihoods_importance_min_third_prompt',
            
            'average_neg_log_likelihoods_forth_prompt',
            'average_neg_log_likelihoods_importance_mean_forth_prompt',
            'average_neg_log_likelihoods_importance_max_forth_prompt',
            'average_neg_log_likelihoods_importance_min_forth_prompt',
            'most_likely_neg_log_likelihoods_forth_prompt', 
            'most_likely_neg_log_likelihoods_importance_mean_forth_prompt',
            'most_likely_neg_log_likelihoods_importance_max_forth_prompt',
            'most_likely_neg_log_likelihoods_importance_min_forth_prompt',
            
            'average_neg_log_likelihoods_fifth_prompt',
            'average_neg_log_likelihoods_importance_mean_fifth_prompt',
            'average_neg_log_likelihoods_importance_max_fifth_prompt',
            'average_neg_log_likelihoods_importance_min_fifth_prompt',
            'most_likely_neg_log_likelihoods_fifth_prompt', 
            'most_likely_neg_log_likelihoods_importance_mean_fifth_prompt',
            'most_likely_neg_log_likelihoods_importance_max_fifth_prompt',
            'most_likely_neg_log_likelihoods_importance_min_fifth_prompt',
            
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
    
    
    ### === Main loop =====================================
    overall_results, geometric_results = get_overall_log_likelihoods(likelihoods)

    # === Main prompt ======== 
    # PE & SE
    average_predictive_entropy_main_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_main_prompt'])
    predictive_entropy_over_concepts_main_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_main_prompt'],
        overall_results['semantic_set_ids']
    )
    # With MARS
    average_predictive_entropy_importance_mean_main_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_mean_main_prompt'])
    average_predictive_entropy_importance_max_main_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_max_main_prompt'])
    average_predictive_entropy_importance_min_main_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_min_main_prompt'])
    predictive_entropy_over_concepts_importance_mean_main_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_mean_main_prompt'],
        overall_results['semantic_set_ids']
    )    
    predictive_entropy_over_concepts_importance_max_main_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_max_main_prompt'],
        overall_results['semantic_set_ids']
    )    
    predictive_entropy_over_concepts_importance_min_main_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_min_main_prompt'],
        overall_results['semantic_set_ids']
    ) 
    
    # === Second prompt ======== 
    # PE & SE
    average_predictive_entropy_second_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_second_prompt'])
    predictive_entropy_over_concepts_second_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_second_prompt'],
        overall_results['semantic_set_ids']
    )
    # With MARS
    average_predictive_entropy_importance_mean_second_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_mean_second_prompt'])
    average_predictive_entropy_importance_max_second_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_max_second_prompt'])
    average_predictive_entropy_importance_min_second_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_min_second_prompt'])
    predictive_entropy_over_concepts_importance_mean_second_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_mean_second_prompt'],
        overall_results['semantic_set_ids']
    )    
    predictive_entropy_over_concepts_importance_max_second_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_max_second_prompt'],
        overall_results['semantic_set_ids']
    )    
    predictive_entropy_over_concepts_importance_min_second_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_min_second_prompt'],
        overall_results['semantic_set_ids']
    ) 
    
    # === Third prompt ======== 
    # PE & SE
    average_predictive_entropy_third_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_third_prompt'])
    predictive_entropy_over_concepts_third_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_third_prompt'],
        overall_results['semantic_set_ids']
    )
    # With MARS
    average_predictive_entropy_importance_mean_third_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_mean_third_prompt'])
    average_predictive_entropy_importance_max_third_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_max_third_prompt'])
    average_predictive_entropy_importance_min_third_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_min_third_prompt'])
    predictive_entropy_over_concepts_importance_mean_third_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_mean_third_prompt'],
        overall_results['semantic_set_ids']
    )    
    predictive_entropy_over_concepts_importance_max_third_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_max_third_prompt'],
        overall_results['semantic_set_ids']
    )    
    predictive_entropy_over_concepts_importance_min_third_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_min_third_prompt'],
        overall_results['semantic_set_ids']
    )
    
    # === forth prompt ======== 
    # PE & SE
    average_predictive_entropy_forth_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_forth_prompt'])
    predictive_entropy_over_concepts_forth_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_forth_prompt'],
        overall_results['semantic_set_ids']
    )
    # With MARS
    average_predictive_entropy_importance_mean_forth_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_mean_forth_prompt'])
    average_predictive_entropy_importance_max_forth_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_max_forth_prompt'])
    average_predictive_entropy_importance_min_forth_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_min_forth_prompt'])
    predictive_entropy_over_concepts_importance_mean_forth_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_mean_forth_prompt'],
        overall_results['semantic_set_ids']
    )    
    predictive_entropy_over_concepts_importance_max_forth_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_max_forth_prompt'],
        overall_results['semantic_set_ids']
    )    
    predictive_entropy_over_concepts_importance_min_forth_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_min_forth_prompt'],
        overall_results['semantic_set_ids']
    )
    
    # === Fifth prompt ======== 
    # PE & SE
    average_predictive_entropy_fifth_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_fifth_prompt'])
    predictive_entropy_over_concepts_fifth_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_fifth_prompt'],
        overall_results['semantic_set_ids']
    )
    # With MARS
    average_predictive_entropy_importance_mean_fifth_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_mean_fifth_prompt'])
    average_predictive_entropy_importance_max_fifth_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_max_fifth_prompt'])
    average_predictive_entropy_importance_min_fifth_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_min_fifth_prompt'])
    predictive_entropy_over_concepts_importance_mean_fifth_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_mean_fifth_prompt'],
        overall_results['semantic_set_ids']
    )    
    predictive_entropy_over_concepts_importance_max_fifth_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_max_fifth_prompt'],
        overall_results['semantic_set_ids']
    )    
    predictive_entropy_over_concepts_importance_min_fifth_prompt = get_predictive_entropy_over_concepts(
        -overall_results['average_neg_log_likelihoods_importance_min_fifth_prompt'],
        overall_results['semantic_set_ids']
    )
    
    
    # === Write in variables ==================
    # = Main prompt ===
    overall_results['average_predictive_entropy_main_prompt'] = average_predictive_entropy_main_prompt
    overall_results['predictive_entropy_over_concepts_main_prompt'] = predictive_entropy_over_concepts_main_prompt
    overall_results['average_predictive_entropy_importance_mean_main_prompt'] = average_predictive_entropy_importance_mean_main_prompt
    overall_results['average_predictive_entropy_importance_max_main_prompt'] = average_predictive_entropy_importance_max_main_prompt
    overall_results['average_predictive_entropy_importance_min_main_prompt'] = average_predictive_entropy_importance_min_main_prompt
    overall_results['predictive_entropy_over_concepts_importance_mean_main_prompt'] = predictive_entropy_over_concepts_importance_mean_main_prompt
    overall_results['predictive_entropy_over_concepts_importance_max_main_prompt'] = predictive_entropy_over_concepts_importance_max_main_prompt
    overall_results['predictive_entropy_over_concepts_importance_min_main_prompt'] = predictive_entropy_over_concepts_importance_min_main_prompt
    
    # = Second prompt ===
    overall_results['average_predictive_entropy_second_prompt'] = average_predictive_entropy_second_prompt
    overall_results['predictive_entropy_over_concepts_second_prompt'] = predictive_entropy_over_concepts_second_prompt
    overall_results['average_predictive_entropy_importance_mean_second_prompt'] = average_predictive_entropy_importance_mean_second_prompt
    overall_results['average_predictive_entropy_importance_max_second_prompt'] = average_predictive_entropy_importance_max_second_prompt
    overall_results['average_predictive_entropy_importance_min_second_prompt'] = average_predictive_entropy_importance_min_second_prompt
    overall_results['predictive_entropy_over_concepts_importance_mean_second_prompt'] = predictive_entropy_over_concepts_importance_mean_second_prompt
    overall_results['predictive_entropy_over_concepts_importance_max_second_prompt'] = predictive_entropy_over_concepts_importance_max_second_prompt
    overall_results['predictive_entropy_over_concepts_importance_min_second_prompt'] = predictive_entropy_over_concepts_importance_min_second_prompt
    
    # = Third prompt ===
    overall_results['average_predictive_entropy_third_prompt'] = average_predictive_entropy_third_prompt
    overall_results['predictive_entropy_over_concepts_third_prompt'] = predictive_entropy_over_concepts_third_prompt
    overall_results['average_predictive_entropy_importance_mean_third_prompt'] = average_predictive_entropy_importance_mean_third_prompt
    overall_results['average_predictive_entropy_importance_max_third_prompt'] = average_predictive_entropy_importance_max_third_prompt
    overall_results['average_predictive_entropy_importance_min_third_prompt'] = average_predictive_entropy_importance_min_third_prompt
    overall_results['predictive_entropy_over_concepts_importance_mean_third_prompt'] = predictive_entropy_over_concepts_importance_mean_third_prompt
    overall_results['predictive_entropy_over_concepts_importance_max_third_prompt'] = predictive_entropy_over_concepts_importance_max_third_prompt
    overall_results['predictive_entropy_over_concepts_importance_min_third_prompt'] = predictive_entropy_over_concepts_importance_min_third_prompt
    
    # = forth prompt ===
    overall_results['average_predictive_entropy_forth_prompt'] = average_predictive_entropy_forth_prompt
    overall_results['predictive_entropy_over_concepts_forth_prompt'] = predictive_entropy_over_concepts_forth_prompt
    overall_results['average_predictive_entropy_importance_mean_forth_prompt'] = average_predictive_entropy_importance_mean_forth_prompt
    overall_results['average_predictive_entropy_importance_max_forth_prompt'] = average_predictive_entropy_importance_max_forth_prompt
    overall_results['average_predictive_entropy_importance_min_forth_prompt'] = average_predictive_entropy_importance_min_forth_prompt
    overall_results['predictive_entropy_over_concepts_importance_mean_forth_prompt'] = predictive_entropy_over_concepts_importance_mean_forth_prompt
    overall_results['predictive_entropy_over_concepts_importance_max_forth_prompt'] = predictive_entropy_over_concepts_importance_max_forth_prompt
    overall_results['predictive_entropy_over_concepts_importance_min_forth_prompt'] = predictive_entropy_over_concepts_importance_min_forth_prompt
    
    # = Fifth prompt ===
    overall_results['average_predictive_entropy_fifth_prompt'] = average_predictive_entropy_fifth_prompt
    overall_results['predictive_entropy_over_concepts_fifth_prompt'] = predictive_entropy_over_concepts_fifth_prompt
    overall_results['average_predictive_entropy_importance_mean_fifth_prompt'] = average_predictive_entropy_importance_mean_fifth_prompt
    overall_results['average_predictive_entropy_importance_max_fifth_prompt'] = average_predictive_entropy_importance_max_fifth_prompt
    overall_results['average_predictive_entropy_importance_min_fifth_prompt'] = average_predictive_entropy_importance_min_fifth_prompt
    overall_results['predictive_entropy_over_concepts_importance_mean_fifth_prompt'] = predictive_entropy_over_concepts_importance_mean_fifth_prompt
    overall_results['predictive_entropy_over_concepts_importance_max_fifth_prompt'] = predictive_entropy_over_concepts_importance_max_fifth_prompt
    overall_results['predictive_entropy_over_concepts_importance_min_fifth_prompt'] = predictive_entropy_over_concepts_importance_min_fifth_prompt
    
    ### === Save the uncertainty result ============
    with open(uncertainty_output_file, 'wb') as ofile:
        pickle.dump(overall_results, ofile)
    print(f"Results saved to {uncertainty_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--dataset', type=str, default='trivia', choices=[
        'nqgold', 'trivia', 'popqa',
        'webquestions', 'squad1', 'nq',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa',
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--main_prompt_format', type=str, default='bm25_retriever_top1', choices=[
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
    get_uncertainty_mars(args)
    
    # python framework/run/get_uncertainty_mars.py
    
