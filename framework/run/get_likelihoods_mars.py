#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pickle
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.utils import set_seed


def get_likelihoods_mars(args):
    print("\n--- Step 3-2: Get Likelihoods ...")
    print(f"""
        Model name: {args.model}
        Dataset: {args.dataset}
        Prompt (1st): {args.main_prompt_format}
        Prompt (2ed): {args.second_prompt_format}
        Run id: {args.run_id}
        Seed: {args.seed}
    """.replace('   ', ''))
    llh_shift = torch.tensor(5.0) # does not effect anything


    # === Define output file ========================
    # === Read the generation & similarities data ===
    model = args.model.split('/')[-1]
    likelihoods_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_likelihoods_generation.pkl'
    similarities_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_similarities_generation.pkl'
    # probabilities_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_probabilities_generation.pkl'
    probabilities_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_probabilities_generation__sec_{args.second_prompt_format}.pkl'
    sequence_input = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'

    with open(sequence_input, 'rb') as infile:
        sequences = pickle.load(infile)    
    with open(similarities_file, 'rb') as infile:
        similarities_dict = pickle.load(infile)
    with open(probabilities_file, 'rb') as infile:
        probabilities_dict = pickle.load(infile)


    # === Load model =================================
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token_id = 1 # Very crucial don't forget
    
    # === Functions ==================================
    IGNORE_INDEX = -100
    def softmax_with_temperature(logits, temperature):
        scaled_logits = logits / temperature
        softmax_probs = torch.nn.functional.softmax(scaled_logits, dim=0)
        return softmax_probs
        
    def compute_token_nll(probs):
        neg_log_likelihoods = -torch.log(probs.reshape(-1)) 
        neg_log_likelihoods = neg_log_likelihoods.reshape(-1)
        loss = torch.mean(neg_log_likelihoods)
        
        if torch.isnan(loss):
            loss = 100000
        
        return loss
    
    def compute_token_nll_importance_phrase(
        generation, probs, importance_scores, phrases, mode='mean'
    ):
        importance_vector = importance_scores.to(args.device)
        
        neg_log_likelihoods = -torch.log(probs.reshape(-1)) 
        neg_log_likelihoods = neg_log_likelihoods.reshape(-1)
        
        #find probabilities of each word
        ids = generation
        neg_log_likelihoods_word = []
        token_idx = 0
        merged_importance_vector  = []
        i = 0
        while i < len(phrases):
            found = False
            while found == False:
                for k in range(1,len(phrases)-i+1):
                    word  = "".join(phrases[i:i+k])
                    #print(word)
                    last_token = -1
                    for j in range(token_idx+1, len(ids)+1):#importance should be summed I guess
                        if tokenizer.decode(ids[token_idx:j]).strip().replace(" ", "").lower() == word.strip().replace(" ", "").lower():
                            last_token = j
                        
                    if last_token != -1:
                        if mode == 'mean':
                            neg_log_likelihoods_word.append(torch.mean(neg_log_likelihoods[token_idx:last_token]))
                            merged_importance_vector.append(torch.mean(importance_vector[i:i+k]))
                        elif mode == 'max':
                            neg_log_likelihoods_word.append(torch.max(neg_log_likelihoods[token_idx:last_token]))
                            merged_importance_vector.append(torch.mean(importance_vector[i:i+k]))
                        elif mode == 'min':
                            neg_log_likelihoods_word.append(torch.min(neg_log_likelihoods[token_idx:last_token]))
                            merged_importance_vector.append(torch.mean(importance_vector[i:i+k]))
                        
                        found = True
                        i += k
                        token_idx = last_token 
                        break
        
        neg_log_likelihoods_word = torch.tensor(neg_log_likelihoods_word).to(args.device)
        merged_importance_vector = torch.tensor(merged_importance_vector).to(args.device)
        merged_importance_vector = merged_importance_vector/torch.sum(merged_importance_vector)
        
        if 'medical' in args.run_id:
            merged_importance_vector = softmax_with_temperature(merged_importance_vector,0.001) # Only for medical dataset
            score = 0.5 * torch.sum(merged_importance_vector * neg_log_likelihoods_word) + 0.5 * torch.mean(neg_log_likelihoods)
        else:
            score = 0.5 * torch.sum(merged_importance_vector * neg_log_likelihoods_word) + 0.5 * torch.mean(neg_log_likelihoods)

        if torch.isnan(score):
            score = 100000
        
        return score

    ### === Main loop ==================================
    result = []
    ids = []
    for i, sample in tqdm(enumerate(sequences)):
        # if i == 50:
        #     break
        
        result_dict = {}
        id_ = sample['id']
        ids.append(id_)
        generations = sample['cleaned_generations'].to(args.device)
        generation_most_likely = sample['cleaned_most_likely_generation_ids'].to(args.device)
        importance_scores = similarities_dict[id_]['importance_scores']
        importance_score_most_likely = similarities_dict[id_]['importance_vector']
        probabilities_generations = probabilities_dict[id_]['probabilities']
        probabilities_most_likely = probabilities_dict[id_]['probability_most_likely']
        # groundedness_score_most_likely = groundedness_dict[id_]['groundedness_score_most_likely'][1]
        # groundedness_scores = groundedness_dict[id_]['groundedness_scores']

        # === Define variables ===================
        # First prompt format
        average_neg_log_likelihoods_main_prompt = torch.zeros((generations.shape[0],))
        average_neg_log_likelihoods_importance_mean_main_prompt = torch.zeros((generations.shape[0],))
        average_neg_log_likelihoods_importance_max_main_prompt = torch.zeros((generations.shape[0],))
        average_neg_log_likelihoods_importance_min_main_prompt = torch.zeros((generations.shape[0],))
        # Second prompt format
        average_neg_log_likelihoods_second_prompt = torch.zeros((generations.shape[0],))
        average_neg_log_likelihoods_importance_mean_second_prompt = torch.zeros((generations.shape[0],))
        average_neg_log_likelihoods_importance_max_second_prompt = torch.zeros((generations.shape[0],))
        average_neg_log_likelihoods_importance_min_second_prompt = torch.zeros((generations.shape[0],))
        # Third prompt format
        average_neg_log_likelihoods_third_prompt = torch.zeros((generations.shape[0],))
        average_neg_log_likelihoods_importance_mean_third_prompt = torch.zeros((generations.shape[0],))
        average_neg_log_likelihoods_importance_max_third_prompt = torch.zeros((generations.shape[0],))
        average_neg_log_likelihoods_importance_min_third_prompt = torch.zeros((generations.shape[0],))
        
        ### = For generations ===============================
        for generation_index in range(generations.shape[0]):
            generation = generations[generation_index][generations[generation_index] != tokenizer.pad_token_id]
            importance_score = importance_scores[generation_index][0]
            phrases = importance_scores[generation_index][1]
            
            ### === Main prompt ============================= 
            probs = probabilities_generations[generation_index][2]
            model_output_loss = compute_token_nll(probs) 
            model_output_loss_importance_mean = compute_token_nll_importance_phrase(
                generation, probs,
                importance_score, phrases, mode='mean'
            )
            model_output_loss_importance_max = compute_token_nll_importance_phrase(
                generation, probs,
                importance_score, phrases, mode='max'
            )
            model_output_loss_importance_min = compute_token_nll_importance_phrase(
                generation, probs,
                importance_score, phrases, mode='min'
            )
            
            average_neg_log_likelihoods_main_prompt[generation_index] = model_output_loss
            average_neg_log_likelihoods_importance_mean_main_prompt[generation_index] = model_output_loss_importance_mean
            average_neg_log_likelihoods_importance_max_main_prompt[generation_index] = model_output_loss_importance_max
            average_neg_log_likelihoods_importance_min_main_prompt[generation_index] = model_output_loss_importance_min
            
            ### === Second prompt =============================
            probs_second = probabilities_generations[generation_index][3]
            if len(probs_second) > 0:
                model_output_loss_second = compute_token_nll(probs_second) 
                model_output_loss_importance_mean_second = compute_token_nll_importance_phrase(
                    generation, probs_second,
                    importance_score, phrases, mode='mean'
                )
                model_output_loss_importance_max_second = compute_token_nll_importance_phrase(
                    generation, probs_second,
                    importance_score, phrases, mode='max'
                )
                model_output_loss_importance_min_second = compute_token_nll_importance_phrase(
                    generation, probs_second,
                    importance_score, phrases, mode='min'
                )
                
                average_neg_log_likelihoods_second_prompt[generation_index] = model_output_loss_second
                average_neg_log_likelihoods_importance_mean_second_prompt[generation_index] = model_output_loss_importance_mean_second
                average_neg_log_likelihoods_importance_max_second_prompt[generation_index] = model_output_loss_importance_max_second
                average_neg_log_likelihoods_importance_min_second_prompt[generation_index] = model_output_loss_importance_min_second
            
            else: 
                score = 100000
                average_neg_log_likelihoods_second_prompt[generation_index] = score
                average_neg_log_likelihoods_importance_mean_second_prompt[generation_index] = score
                average_neg_log_likelihoods_importance_max_second_prompt[generation_index] = score
                average_neg_log_likelihoods_importance_min_second_prompt[generation_index] = score

            ### === Third prompt =============================
            probs_third = probabilities_generations[generation_index][4]
            if len(probs_third) > 0:
                model_output_loss_third = compute_token_nll(probs_third) 
                model_output_loss_importance_mean_third = compute_token_nll_importance_phrase(
                    generation, probs_third,
                    importance_score, phrases, mode='mean'
                )
                model_output_loss_importance_max_third = compute_token_nll_importance_phrase(
                    generation, probs_third,
                    importance_score, phrases, mode='max'
                )
                model_output_loss_importance_min_third = compute_token_nll_importance_phrase(
                    generation, probs_third,
                    importance_score, phrases, mode='min'
                )
                
                average_neg_log_likelihoods_third_prompt[generation_index] = model_output_loss_third
                average_neg_log_likelihoods_importance_mean_third_prompt[generation_index] = model_output_loss_importance_mean_third
                average_neg_log_likelihoods_importance_max_third_prompt[generation_index] = model_output_loss_importance_max_third
                average_neg_log_likelihoods_importance_min_third_prompt[generation_index] = model_output_loss_importance_min_third
            
            else: 
                score = 100000
                average_neg_log_likelihoods_third_prompt[generation_index] = score
                average_neg_log_likelihoods_importance_mean_third_prompt[generation_index] = score
                average_neg_log_likelihoods_importance_max_third_prompt[generation_index] = score
                average_neg_log_likelihoods_importance_min_third_prompt[generation_index] = score
            

        ### = For most-likely ===============================
        if len(sample['cleaned_most_likely_generation_ids']) > 0:
            _generation_most_likely = generation_most_likely[generation_most_likely != tokenizer.pad_token_id]
            phrases = importance_score_most_likely[1]
            
            # === Main prompt ===============================
            probs = probabilities_most_likely[2]
            most_likely_model_output_loss = compute_token_nll(probs)
            most_likely_model_output_loss_importance_mean = compute_token_nll_importance_phrase(
                _generation_most_likely, probs,
                importance_score_most_likely[0], phrases, mode='mean'
            )
            most_likely_model_output_loss_importance_max = compute_token_nll_importance_phrase(
                _generation_most_likely, probs,
                importance_score_most_likely[0], phrases, mode='max'
            )
            most_likely_model_output_loss_importance_min = compute_token_nll_importance_phrase(
                _generation_most_likely, probs,
                importance_score_most_likely[0], phrases, mode='min'
            )
            
            most_likely_model_output_loss_main_prompt = most_likely_model_output_loss.cpu()
            most_likely_model_output_loss_importance_mean_main_prompt = most_likely_model_output_loss_importance_mean.cpu()
            most_likely_model_output_loss_importance_max_main_prompt = most_likely_model_output_loss_importance_max.cpu()
            most_likely_model_output_loss_importance_min_main_prompt = most_likely_model_output_loss_importance_min.cpu()
            
            # === Second prompt =============================
            probs_second = probabilities_most_likely[3]
            if len(probs_second) > 0:
            
                most_likely_model_output_loss_second_prompt = compute_token_nll(probs_second)
                most_likely_model_output_loss_importance_mean_second_prompt = compute_token_nll_importance_phrase(
                    _generation_most_likely, probs_second,
                    importance_score_most_likely[0], phrases, mode='mean'
                )
                most_likely_model_output_loss_importance_max_second_prompt = compute_token_nll_importance_phrase(
                    _generation_most_likely, probs_second,
                    importance_score_most_likely[0], phrases, mode='max'
                )
                most_likely_model_output_loss_importance_min_second_prompt = compute_token_nll_importance_phrase(
                    _generation_most_likely, probs_second,
                    importance_score_most_likely[0], phrases, mode='min'
                )
                most_likely_model_output_loss_second_prompt = most_likely_model_output_loss_second_prompt.cpu()
                most_likely_model_output_loss_importance_mean_second_prompt = most_likely_model_output_loss_importance_mean_second_prompt.cpu()
                most_likely_model_output_loss_importance_max_second_prompt = most_likely_model_output_loss_importance_max_second_prompt.cpu()
                most_likely_model_output_loss_importance_min_second_prompt = most_likely_model_output_loss_importance_min_second_prompt.cpu()

            else: 
                score = 100000
                most_likely_model_output_loss_second_prompt = score
                most_likely_model_output_loss_importance_mean_second_prompt = score
                most_likely_model_output_loss_importance_max_second_prompt = score
                most_likely_model_output_loss_importance_min_second_prompt = score

            # === third prompt ===============================
            probs_third = probabilities_most_likely[4]
            
            if len(probs_third) > 0:
                most_likely_model_output_loss_third_prompt = compute_token_nll(probs_third)
                most_likely_model_output_loss_importance_mean_third_prompt = compute_token_nll_importance_phrase(
                    _generation_most_likely, probs_third,
                    importance_score_most_likely[0], phrases, mode='mean'
                )
                most_likely_model_output_loss_importance_max_third_prompt = compute_token_nll_importance_phrase(
                    _generation_most_likely, probs_third,
                    importance_score_most_likely[0], phrases, mode='max'
                )
                most_likely_model_output_loss_importance_min_third_prompt = compute_token_nll_importance_phrase(
                    _generation_most_likely, probs_third,
                    importance_score_most_likely[0], phrases, mode='min'
                )
                most_likely_model_output_loss_third_prompt = most_likely_model_output_loss_third_prompt.cpu()
                most_likely_model_output_loss_importance_mean_third_prompt = most_likely_model_output_loss_importance_mean_third_prompt.cpu()
                most_likely_model_output_loss_importance_max_third_prompt = most_likely_model_output_loss_importance_max_third_prompt.cpu()
                most_likely_model_output_loss_importance_min_third_prompt = most_likely_model_output_loss_importance_min_third_prompt.cpu()

            else: 
                score = 100000
                most_likely_model_output_loss_third_prompt = score
                most_likely_model_output_loss_importance_mean_third_prompt = score
                most_likely_model_output_loss_importance_max_third_prompt = score
                most_likely_model_output_loss_importance_min_third_prompt = score
            
        else:
            score = 100000
            most_likely_model_output_loss_main_prompt = score
            most_likely_model_output_loss_importance_mean_main_prompt = score
            most_likely_model_output_loss_importance_max_main_prompt = score
            most_likely_model_output_loss_importance_min_main_prompt = score
            
            most_likely_model_output_loss_second_prompt = score
            most_likely_model_output_loss_importance_mean_second_prompt = score
            most_likely_model_output_loss_importance_max_second_prompt = score
            most_likely_model_output_loss_importance_min_second_prompt = score
            
            most_likely_model_output_loss_third_prompt = score
            most_likely_model_output_loss_importance_mean_third_prompt = score
            most_likely_model_output_loss_importance_max_third_prompt = score
            most_likely_model_output_loss_importance_min_third_prompt = score
            
            
        ### = Write to file =========================
        result_dict['id'] = id_
        result_dict['generations'] = generations.cpu()
        result_dict['similarity_score'] = sample['similarity_score']
        result_dict['semantic_set_ids'] = torch.tensor(similarities_dict[id_]['semantic_set_ids'], device='cpu')
        result_dict['has_different_answers'] = similarities_dict[id_]['has_different_answers']
        result_dict['unique_answers_indices'] = similarities_dict[id_]['unique_answers_indices']
        
        # Main prompt
        result_dict['average_neg_log_likelihoods_main_prompt'] = average_neg_log_likelihoods_main_prompt.cpu()
        result_dict['average_neg_log_likelihoods_importance_mean_main_prompt'] = average_neg_log_likelihoods_importance_mean_main_prompt.cpu()
        result_dict['average_neg_log_likelihoods_importance_max_main_prompt'] = average_neg_log_likelihoods_importance_max_main_prompt.cpu()
        result_dict['average_neg_log_likelihoods_importance_min_main_prompt'] = average_neg_log_likelihoods_importance_min_main_prompt.cpu()
        result_dict['most_likely_neg_log_likelihoods_main_prompt'] = most_likely_model_output_loss_main_prompt
        result_dict['most_likely_neg_log_likelihoods_importance_mean_main_prompt'] = most_likely_model_output_loss_importance_mean_main_prompt
        result_dict['most_likely_neg_log_likelihoods_importance_max_main_prompt'] = most_likely_model_output_loss_importance_max_main_prompt
        result_dict['most_likely_neg_log_likelihoods_importance_min_main_prompt'] = most_likely_model_output_loss_importance_min_main_prompt
        
        # Second prompt
        result_dict['average_neg_log_likelihoods_second_prompt'] = average_neg_log_likelihoods_second_prompt.cpu()
        result_dict['average_neg_log_likelihoods_importance_mean_second_prompt'] = average_neg_log_likelihoods_importance_mean_second_prompt.cpu()
        result_dict['average_neg_log_likelihoods_importance_max_second_prompt'] = average_neg_log_likelihoods_importance_max_second_prompt.cpu()
        result_dict['average_neg_log_likelihoods_importance_min_second_prompt'] = average_neg_log_likelihoods_importance_min_second_prompt.cpu()
        result_dict['most_likely_neg_log_likelihoods_second_prompt'] = most_likely_model_output_loss_second_prompt
        result_dict['most_likely_neg_log_likelihoods_importance_mean_second_prompt'] = most_likely_model_output_loss_importance_mean_second_prompt
        result_dict['most_likely_neg_log_likelihoods_importance_max_second_prompt'] = most_likely_model_output_loss_importance_max_second_prompt
        result_dict['most_likely_neg_log_likelihoods_importance_min_second_prompt'] = most_likely_model_output_loss_importance_min_second_prompt
        
        # Third prompt
        result_dict['average_neg_log_likelihoods_third_prompt'] = average_neg_log_likelihoods_third_prompt.cpu()
        result_dict['average_neg_log_likelihoods_importance_mean_third_prompt'] = average_neg_log_likelihoods_importance_mean_third_prompt.cpu()
        result_dict['average_neg_log_likelihoods_importance_max_third_prompt'] = average_neg_log_likelihoods_importance_max_third_prompt.cpu()
        result_dict['average_neg_log_likelihoods_importance_min_third_prompt'] = average_neg_log_likelihoods_importance_min_third_prompt.cpu()
        result_dict['most_likely_neg_log_likelihoods_third_prompt'] = most_likely_model_output_loss_third_prompt
        result_dict['most_likely_neg_log_likelihoods_importance_mean_third_prompt'] = most_likely_model_output_loss_importance_mean_third_prompt
        result_dict['most_likely_neg_log_likelihoods_importance_max_third_prompt'] = most_likely_model_output_loss_importance_max_third_prompt
        result_dict['most_likely_neg_log_likelihoods_importance_min_third_prompt'] = most_likely_model_output_loss_importance_min_third_prompt
        
        result.append(result_dict)
    
    ### === Save the likelihoods result ==============
    with open(likelihoods_output_file, 'wb') as ofile:
        pickle.dump(result, ofile)
    print(f"Results saved to {likelihoods_output_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--model_llama_eval', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='webquestions', choices=[
        'trivia', 'nq', 'squad1', 'webquestions',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa_org', 'topicoqa_his', 'topicoqa_rw',
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--main_prompt_format', type=str, default='q_positive', choices=[
        'only_q', 'q_positive', 'q_negative',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    parser.add_argument('--second_prompt_format', type=str, default='only_q', choices=[
        'only_q', 'q_positive', 'q_negative',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
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
    parser.add_argument('--run_id', type=str, default='run_1')
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
    get_likelihoods_mars(args)
    
    # python framework/run/get_likelihoods_mars.py
    
   
   
   
   
   
   
   
   
   
   
   
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    #   def get_overall_log_likelihoods(list_of_results):
    #     """Compute log likelihood of all generations under their given context.
    #     list_of_results: list of dictionaries with keys:
    #     returns: dictionary with keys: 'neg_log_likelihoods', 'average_neg_log_likelihoods'
    #             that contains tensors of shape (num_models, num_generations, num_samples_per_generation)
    #     """

    #     result_dict = {}
    #     geometric_dict ={}

    #     # list_of_keys = ['neg_log_likelihoods', 'average_neg_log_likelihoods',\
    #     #                 'pointwise_mutual_information', 'average_neg_log_likelihood_of_most_likely_gen',\
    #     #                 'neg_log_likelihood_of_most_likely_gen', 'semantic_set_ids', \
    #     #                 'average_neg_log_likelihoods_importance_mean', 'average_neg_log_likelihoods_importance_max', 'average_neg_log_likelihoods_importance_min',\
    #     #                 'most_likely_neg_log_likelihoods', 
    #     #                 'most_likely_neg_log_likelihoods_importance_mean', 'most_likely_neg_log_likelihoods_importance_max', 'most_likely_neg_log_likelihoods_importance_min']
    #     list_of_keys = [
    #         'average_neg_log_likelihoods_main_prompt',
    #         'average_neg_log_likelihoods_importance_mean_main_prompt',
    #         'average_neg_log_likelihoods_importance_max_main_prompt',
    #         'average_neg_log_likelihoods_importance_min_main_prompt',
    #         'most_likely_neg_log_likelihoods_main_prompt', 
    #         'most_likely_neg_log_likelihoods_importance_mean_main_prompt',
    #         'most_likely_neg_log_likelihoods_importance_max_main_prompt',
    #         'most_likely_neg_log_likelihoods_importance_min_main_prompt',
            
    #         'average_neg_log_likelihoods_second_prompt',
    #         'average_neg_log_likelihoods_importance_mean_second_prompt',
    #         'average_neg_log_likelihoods_importance_max_second_prompt',
    #         'average_neg_log_likelihoods_importance_min_second_prompt',
    #         'most_likely_neg_log_likelihoods_second_prompt', 
    #         'most_likely_neg_log_likelihoods_importance_mean_second_prompt',
    #         'most_likely_neg_log_likelihoods_importance_max_second_prompt',
    #         'most_likely_neg_log_likelihoods_importance_min_second_prompt',
            
    #         'average_neg_log_likelihoods_third_prompt',
    #         'average_neg_log_likelihoods_importance_mean_third_prompt',
    #         'average_neg_log_likelihoods_importance_max_third_prompt',
    #         'average_neg_log_likelihoods_importance_min_third_prompt',
    #         'most_likely_neg_log_likelihoods_third_prompt', 
    #         'most_likely_neg_log_likelihoods_importance_mean_third_prompt',
    #         'most_likely_neg_log_likelihoods_importance_max_third_prompt',
    #         'most_likely_neg_log_likelihoods_importance_min_third_prompt',
            
    #         'similarity_score',
    #         'semantic_set_ids',
            
    #     ]

    #     geometric_keys = ['has_different_answers','unique_answers_indices']

    #     for key in geometric_keys:
    #         overall_results = []
    #         for sample in list_of_results:
    #             overall_results.append(sample[key])
    #         geometric_dict[key]  = overall_results

    #     for key in list_of_keys:
    #         list_of_ids = []
    #         overall_results = []
    #         results_per_model = []
    #         for sample in list_of_results:
    #             average_neg_log_likelihoods = sample[key]
    #             list_of_ids.append(sample['id'])
    #             results_per_model.append(average_neg_log_likelihoods)

    #         results_per_model = [torch.tensor(x) if isinstance(x, int) else x for x in results_per_model]
    #         results_per_model = torch.stack(results_per_model)
    #         overall_results.append(results_per_model)

    #         if key not in ['meaning_vectors', 'meaning_vectors_only_answer','has_different_answers']:
    #             overall_results = torch.stack(overall_results)

    #         result_dict[key] = overall_results

    #     result_dict['ids'] = list_of_ids
    #     return result_dict, geometric_dict
    
    # def get_predictive_entropy(log_likelihoods):
    #     """Compute predictive entropy of approximate posterior predictive"""
    #     mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    #     entropy = -torch.sum(mean_across_models, dim=1) / torch.tensor(mean_across_models.shape[1])
        
    #     return entropy
    
    # def get_predictive_entropy_over_concepts(log_likelihoods, semantic_set_ids):
    #     """Compute the semantic entropy"""
    #     #log_likelihoods = log_likelihoods[:,:,:1]
    #     mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    #     # This is ok because all the models have the same semantic set ids
    #     semantic_set_ids = semantic_set_ids[0]
    #     entropies = []
    #     for row_index in range(mean_across_models.shape[0]):
    #         aggregated_likelihoods = []
    #         row = mean_across_models[row_index]
    #         semantic_set_ids_row = semantic_set_ids[row_index]
    #         #semantic_set_ids_row = semantic_set_ids_row[:1]
    #         for semantic_set_id in torch.unique(semantic_set_ids_row):
    #             aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0))
    #         aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - llh_shift
    #         entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
    #         entropies.append(entropy)

    #     # print(torch.tensor(entropies))

    #     return torch.tensor(entropies)
    
  
   # ### === Attach uncertainty =======================
    # overall_results, geometric_results = get_overall_log_likelihoods(result)
    
    
    # # === Main prompt ======== 
    # # PE & SE
    # average_predictive_entropy_main_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_main_prompt'])
    # predictive_entropy_over_concepts_main_prompt = get_predictive_entropy_over_concepts(
    #     -overall_results['average_neg_log_likelihoods_main_prompt'],
    #     overall_results['semantic_set_ids']
    # )
    # # With MARS
    # average_predictive_entropy_importance_mean_main_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_mean_main_prompt'])
    # average_predictive_entropy_importance_max_main_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_max_main_prompt'])
    # average_predictive_entropy_importance_min_main_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_min_main_prompt'])
    # predictive_entropy_over_concepts_importance_mean_main_prompt = get_predictive_entropy_over_concepts(
    #     -overall_results['average_neg_log_likelihoods_importance_mean_main_prompt'],
    #     overall_results['semantic_set_ids']
    # )    
    # predictive_entropy_over_concepts_importance_max_main_prompt = get_predictive_entropy_over_concepts(
    #     -overall_results['average_neg_log_likelihoods_importance_max_main_prompt'],
    #     overall_results['semantic_set_ids']
    # )    
    # predictive_entropy_over_concepts_importance_min_main_prompt = get_predictive_entropy_over_concepts(
    #     -overall_results['average_neg_log_likelihoods_importance_min_main_prompt'],
    #     overall_results['semantic_set_ids']
    # ) 
    
    # # === Second prompt ======== 
    # # PE & SE
    # average_predictive_entropy_second_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_second_prompt'])
    # predictive_entropy_over_concepts_second_prompt = get_predictive_entropy_over_concepts(
    #     -overall_results['average_neg_log_likelihoods_second_prompt'],
    #     overall_results['semantic_set_ids']
    # )
    # # With MARS
    # average_predictive_entropy_importance_mean_second_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_mean_second_prompt'])
    # average_predictive_entropy_importance_max_second_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_max_second_prompt'])
    # average_predictive_entropy_importance_min_second_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_min_second_prompt'])
    # predictive_entropy_over_concepts_importance_mean_second_prompt = get_predictive_entropy_over_concepts(
    #     -overall_results['average_neg_log_likelihoods_importance_mean_second_prompt'],
    #     overall_results['semantic_set_ids']
    # )    
    # predictive_entropy_over_concepts_importance_max_second_prompt = get_predictive_entropy_over_concepts(
    #     -overall_results['average_neg_log_likelihoods_importance_max_second_prompt'],
    #     overall_results['semantic_set_ids']
    # )    
    # predictive_entropy_over_concepts_importance_min_second_prompt = get_predictive_entropy_over_concepts(
    #     -overall_results['average_neg_log_likelihoods_importance_min_second_prompt'],
    #     overall_results['semantic_set_ids']
    # ) 
    
    # # === Third prompt ======== 
    # # PE & SE
    # average_predictive_entropy_third_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_third_prompt'])
    # predictive_entropy_over_concepts_third_prompt = get_predictive_entropy_over_concepts(
    #     -overall_results['average_neg_log_likelihoods_third_prompt'],
    #     overall_results['semantic_set_ids']
    # )
    # # With MARS
    # average_predictive_entropy_importance_mean_third_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_mean_third_prompt'])
    # average_predictive_entropy_importance_max_third_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_max_third_prompt'])
    # average_predictive_entropy_importance_min_third_prompt = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_importance_min_third_prompt'])
    # predictive_entropy_over_concepts_importance_mean_third_prompt = get_predictive_entropy_over_concepts(
    #     -overall_results['average_neg_log_likelihoods_importance_mean_third_prompt'],
    #     overall_results['semantic_set_ids']
    # )    
    # predictive_entropy_over_concepts_importance_max_third_prompt = get_predictive_entropy_over_concepts(
    #     -overall_results['average_neg_log_likelihoods_importance_max_third_prompt'],
    #     overall_results['semantic_set_ids']
    # )    
    # predictive_entropy_over_concepts_importance_min_third_prompt = get_predictive_entropy_over_concepts(
    #     -overall_results['average_neg_log_likelihoods_importance_min_third_prompt'],
    #     overall_results['semantic_set_ids']
    # )
    
    # # === Write in variables ==================
    # # = Main prompt ===
    # overall_results['average_predictive_entropy_main_prompt'] = average_predictive_entropy_main_prompt
    # overall_results['predictive_entropy_over_concepts_main_prompt'] = predictive_entropy_over_concepts_main_prompt
    # overall_results['average_predictive_entropy_importance_mean_main_prompt'] = average_predictive_entropy_importance_mean_main_prompt
    # overall_results['average_predictive_entropy_importance_max_main_prompt'] = average_predictive_entropy_importance_max_main_prompt
    # overall_results['average_predictive_entropy_importance_min_main_prompt'] = average_predictive_entropy_importance_min_main_prompt
    # overall_results['predictive_entropy_over_concepts_importance_mean_main_prompt'] = predictive_entropy_over_concepts_importance_mean_main_prompt
    # overall_results['predictive_entropy_over_concepts_importance_max_main_prompt'] = predictive_entropy_over_concepts_importance_max_main_prompt
    # overall_results['predictive_entropy_over_concepts_importance_min_main_prompt'] = predictive_entropy_over_concepts_importance_min_main_prompt
    
    # # = Second prompt ===
    # overall_results['average_predictive_entropy_second_prompt'] = average_predictive_entropy_second_prompt
    # overall_results['predictive_entropy_over_concepts_second_prompt'] = predictive_entropy_over_concepts_second_prompt
    # overall_results['average_predictive_entropy_importance_mean_second_prompt'] = average_predictive_entropy_importance_mean_second_prompt
    # overall_results['average_predictive_entropy_importance_max_second_prompt'] = average_predictive_entropy_importance_max_second_prompt
    # overall_results['average_predictive_entropy_importance_min_second_prompt'] = average_predictive_entropy_importance_min_second_prompt
    # overall_results['predictive_entropy_over_concepts_importance_mean_second_prompt'] = predictive_entropy_over_concepts_importance_mean_second_prompt
    # overall_results['predictive_entropy_over_concepts_importance_max_second_prompt'] = predictive_entropy_over_concepts_importance_max_second_prompt
    # overall_results['predictive_entropy_over_concepts_importance_min_second_prompt'] = predictive_entropy_over_concepts_importance_min_second_prompt
    
    # # = Third prompt ===
    # overall_results['average_predictive_entropy_third_prompt'] = average_predictive_entropy_third_prompt
    # overall_results['predictive_entropy_over_concepts_third_prompt'] = predictive_entropy_over_concepts_third_prompt
    # overall_results['average_predictive_entropy_importance_mean_third_prompt'] = average_predictive_entropy_importance_mean_third_prompt
    # overall_results['average_predictive_entropy_importance_max_third_prompt'] = average_predictive_entropy_importance_max_third_prompt
    # overall_results['average_predictive_entropy_importance_min_third_prompt'] = average_predictive_entropy_importance_min_third_prompt
    # overall_results['predictive_entropy_over_concepts_importance_mean_third_prompt'] = predictive_entropy_over_concepts_importance_mean_third_prompt
    # overall_results['predictive_entropy_over_concepts_importance_max_third_prompt'] = predictive_entropy_over_concepts_importance_max_third_prompt
    # overall_results['predictive_entropy_over_concepts_importance_min_third_prompt'] = predictive_entropy_over_concepts_importance_min_third_prompt
    
    
    # ### === Save the uncertainty result ============
    # with open(uncertainty_output_file, 'wb') as ofile:
    #     pickle.dump(overall_results, ofile)
    # print(f"Results saved to {uncertainty_output_file}")

  
  
    # def doc_entropy(probabilities):
    #     # entropy = -probabilities * torch.log(probabilities) - (1 - probabilities) * torch.log(1 - probabilities)
    #     entropy = -probabilities * torch.log(probabilities)
    #     return entropy
  
  
    # def get_predictive_entropy_v2(log_likelihoods, generation_level_weight, generation_level_all):
    #     """Compute predictive entropy of approximate posterior predictive"""
    #     mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))

    #     # ====
    #     lambda_1  = 0.4
    #     lambda_2  = 0.4
    #     generation_level_weight_ = generation_level_weight[0]
    #     generation_level_all_ = generation_level_all[0]
    #     generation_level_all_expanded = generation_level_all_.view(-1, 1)
        
    #     final_weights = torch.where(
    #         generation_level_all_expanded > 5,
    #         1 - lambda_1 * (generation_level_weight_ / generation_level_all_expanded),  # Apply (1 + x/n) when A > 5
    #         1 + lambda_2 * (generation_level_weight_ / generation_level_all_expanded)   # Apply (1 - x/n) when A <= 5
    #     )
    #     # =====
    #     entropy_bf = -torch.sum(mean_across_models, dim=1) / torch.tensor(mean_across_models.shape[1])
        
    #     weighted_mean = mean_across_models * final_weights
    #     entropy = -torch.sum(weighted_mean, dim=1) / torch.tensor(weighted_mean.shape[1])

    #     return entropy
  
  
    # uncertainty_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_uncertainty_generation.pkl'
    # groundedness_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_verification_generation.pkl'
    # only_query_semantic_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_only_query_semantic_generation.pkl'
    # with open(only_query_semantic_file, 'rb') as infile:
    #     only_query_semantic_dict = pickle.load(infile)    
    # if args.with_groundedness == 'yes':
    #     with open(groundedness_file, 'rb') as infile:
    #         groundedness_dict = pickle.load(infile)
    
    
    # def compute_token_nll_v2(probs, probs_only_query, groundedness):
        
    #     if args.prompt_format != 'only_q':
    #         if probs_only_query != None and groundedness != None:
    #             probs_doc = torch.nn.functional.sigmoid(torch.tensor(groundedness)).to(args.device)
    #             final_prob = args.landa_1 * probs + args.landa_2 * probs_only_query + args.landa_3 * probs_doc
    #             probs = final_prob

    #     neg_log_likelihoods = -torch.log(probs.reshape(-1)) 
    #     neg_log_likelihoods = neg_log_likelihoods.reshape(-1)
    #     loss = torch.mean(neg_log_likelihoods)
        
    #     if torch.isnan(loss):
    #         loss = 100000
        
    #     return loss
    
    # def compute_token_nll_v3(probs, probs_only_q):
        
    #     bin_probs = torch.zeros_like(probs)
    #     threshold1 = 0.3
    #     threshold2 = 0.5
    
    #     diff = probs - probs_only_q
    #     bin_probs[diff > threshold1] = 1  # Condition 1: |P1 - P2| > threshold1
    #     bin_probs[(diff <= threshold1) & (probs > threshold2) & (probs_only_q > threshold2)] = 1  # Condition 2

    #     final_prob = 0.5 * probs + 0.5 * bin_probs
    #     probs = final_prob
        
    #     neg_log_likelihoods = -torch.log(probs.reshape(-1)) 
    #     neg_log_likelihoods = neg_log_likelihoods.reshape(-1)
    #     loss = torch.mean(neg_log_likelihoods)
        
    #     if torch.isnan(loss):
    #         loss = 100000
        
    #     return loss
    
    # ======================
    # _doc_entropy = doc_entropy(overall_results['similarity_score'])
    # average_predictive_entropy_cur = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_cur'])
    # average_predictive_entropy_only_q = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_only_q'])
    # average_predictive_entropy_v3 = get_predictive_entropy(-overall_results['average_neg_log_likelihoods_v3'])    
    # predictive_entropy_over_concepts_cur = get_predictive_entropy_over_concepts(
    #     -overall_results['average_neg_log_likelihoods_cur'],
    #     overall_results['semantic_set_ids']
    # )
    # predictive_entropy_over_concepts_only_q = get_predictive_entropy_over_concepts(
    #     -overall_results['average_neg_log_likelihoods_only_q'],
    #     overall_results['semantic_set_ids']
    # )
    # predictive_entropy_over_concepts_v3 = get_predictive_entropy_over_concepts(
    #     -overall_results['average_neg_log_likelihoods_v3'],
    #     overall_results['semantic_set_ids']
    # )
    # overall_results['doc_entropy'] = _doc_entropy
    # overall_results['average_predictive_entropy_cur'] = average_predictive_entropy_cur
    # overall_results['average_predictive_entropy_only_q'] = average_predictive_entropy_only_q
    # overall_results['average_predictive_entropy_v3'] = average_predictive_entropy_v3
    
    # overall_results['predictive_entropy_over_concepts_cur'] = predictive_entropy_over_concepts_cur
    # overall_results['predictive_entropy_over_concepts_only_q'] = predictive_entropy_over_concepts_only_q
    # overall_results['predictive_entropy_over_concepts_v3'] = predictive_entropy_over_concepts_v3
    # ======================
    
    # print('PE')
    # average_predictive_entropy_v4 = get_predictive_entropy_v2(
    #     -overall_results['average_neg_log_likelihoods'],
    #     overall_results['only_query_semantic_vector'],
    #     overall_results['only_query_semantic_all']
    # )
    # overall_results['average_predictive_entropy_v4'] = average_predictive_entropy_v4
    # predictive_entropy_over_concepts = get_predictive_entropy_v2_over_concepts(
    #     -overall_results['average_neg_log_likelihoods'],
    #     overall_results['semantic_set_ids']
    # )