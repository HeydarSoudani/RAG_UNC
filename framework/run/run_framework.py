#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import argparse

from answers_generation import generation
from answers_generation_cad import generation_cad
from get_semantic_similarity import get_similarity
from get_probabilities import get_probability
from get_likelihoods_mars import get_likelihoods_mars

from get_uncertainty_mars import get_uncertainty_mars
from get_uncertainty_blackbox import get_uncertainty_bb
from get_uncertainty_sar import get_uncertainty_sar

from get_axiomatic_variables import get_axiomatic_variables
from get_correctness import get_correctness
from get_calibration_results import get_calibration_results
from get_axiomatic_nli_results import get_axiomatic_results

from utils.utils import set_seed

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
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test', 'validation'])
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
    
    parser.add_argument('--generation_type', type=str, default='normal', choices=['normal', 'cad'])
    parser.add_argument('--alpha_generation', type=float, default=0.5)
    parser.add_argument('--alpha_probability', type=float, default=0.5)
    parser.add_argument('--affinity_mode', type=str, default='disagreement')
    parser.add_argument('--run_id', type=str, default='run_0')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    
    ### === Define CUDA device =================== 
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
    
    
    ### === Run Steps ============================
    set_seed(args.seed)
    ## === Phase 1: answer generation & cleaning
    # if args.generation_type == 'normal':
    #     generation(args)
    # elif args.generation_type == 'cad':
    #     generation_cad(args)
    
    # ## === Phase 2: Uncertainty computation
    get_similarity(args)       # this generates importance score | # works with: pip install transformers==4.37.2
    get_probability(args)
    get_likelihoods_mars(args)
    
    get_uncertainty_mars(args)
    # get_uncertainty_bb(args)
    # get_uncertainty_sar(args) # TODO: sar_uncertainty
    
    ## === Phase 3: correctness and calibration results
    get_correctness(args)
    get_axiomatic_variables(args)
    get_calibration_results(args)
    # get_axiomatic_results(args)
    
    
    # python framework/run/run_framework.py












    # For grid search
    # model = args.model.split('/')[-1]
    # greedy_output_jsonl_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_{args.mode}_greedy_results_1.jsonl'
    # values = np.arange(0, 1.1, 0.1)
    # # values = np.arange(0, 1.05, 0.05)
    # best_result = 0.0
    # with open(greedy_output_jsonl_file, 'w') as jl_ofile:
    #     for lambda1 in values:
    #         for lambda2 in values:
    #             for lambda3 in values:
    #                 # Check if the sum of lambdas equals 1
    #                 if np.isclose(lambda1 + lambda2 + lambda3, 1.0):
    #                     print(f'Current lambda: {lambda1}, {lambda2}, {lambda3} ...')
    #                     args.landa_1 = lambda1
    #                     args.landa_2 = lambda2
    #                     args.landa_3 = lambda3
                        
    #                     # get_probability(args)
    #                     get_likelihoods(args)
    #                     result_dict = get_calibration_results(args)
    #                     SE_auroc = result_dict['entropy_over_concepts_auroc']
                        
    #                     if SE_auroc > best_result:
    #                         best_result = SE_auroc
    #                         best_landas = (args.landa_1, args.landa_2, args.landa_3)
                        
    #                     result_item = {
    #                         'landa_1': args.landa_1,
    #                         'landa_2': args.landa_2,
    #                         'landa_3': args.landa_3,
    #                         "PE_auroc": result_dict['ln_predictive_entropy_auroc'],
    #                         "SE_auroc": result_dict['entropy_over_concepts_auroc'],
    #                         "SE_M_auroc": result_dict['entropy_over_concepts_auroc_importance_max'],
    #                     }
    #                     jl_ofile.write(json.dumps(result_item) + '\n')
        
    #     print(f'Best SE result: {best_result}')
    #     print(f'Best landas result: {best_landas}')



