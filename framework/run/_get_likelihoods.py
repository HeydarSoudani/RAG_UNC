#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import pickle
import logging
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from framework.utils.utils import set_seed


def get_likelihoods(args):
    
    print("\n--- Step 3: Get Likelihoods ...")
    print(f"""
        Model name: {args.model}
        Dataset: {args.dataset}
        Prompt format: {args.prompt_format}
        Run id: {args.run_id}
        Seed: {args.seed}
    """.replace('   ', ''))
    
    # === Define output file ========================
    # === Read the generation & similarities data ===
    model = args.model.split('/')[-1]
    
    if args.prompt_format == 'only_q':
        likelihoods_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_likelihoods_generation.pkl'
        likelihoods_output_jsonl_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_likelihoods_generation.jsonl'
        similarities_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_similarities_generation.pkl'
        groundedness_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_verification_generation.pkl'
        probabilities_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_{args.mode}_probabilities_generation.pkl'
    else:
        likelihoods_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_{args.mode}_likelihoods_generation.pkl'
        likelihoods_output_jsonl_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_{args.mode}_likelihoods_generation.jsonl'
        similarities_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_{args.mode}_similarities_generation.pkl'
        groundedness_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_{args.mode}_verification_generation.pkl'
        probabilities_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_{args.mode}_probabilities_generation.pkl'
        
    if args.mode == 'seperated':
        sequence_input = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
        sequence_input_only_q = f'{args.output_dir}/{args.dataset}/{args.run_id}/only_q/{model}_{args.temperature}_cleaned_generation.pkl'
    elif args.mode == 'combined':
        sequence_input = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_combined_generation.pkl'
        sequence_input_only_q = f'{args.output_dir}/{args.dataset}/{args.run_id}/only_q/{model}_{args.temperature}_cleaned_generation.pkl'
    else:
        print('mode is not defined')
    
    with open(sequence_input, 'rb') as infile:
        sequences = pickle.load(infile)
    with open(sequence_input_only_q, 'rb') as infile:
        sequences_only_q = pickle.load(infile)
    
    with open(similarities_file, 'rb') as infile:
        similarities_dict = pickle.load(infile)
    with open(probabilities_file, 'rb') as infile:
        probabilities_dict = pickle.load(infile)
    
    if args.with_groundedness == 'yes':
        with open(groundedness_file, 'rb') as infile:
            groundedness_dict = pickle.load(infile)
    
    # Added for Faegheh's experiment
    _sequences_only_q = {}
    for item in sequences_only_q:
        _sequences_only_q[item['id']] = item
        
    
    # === Load model =================================
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token_id = 1 # Very crucial don't forget
    
    # === Functions ==================================        
    def softmax_with_temperature(logits, temperature):
        scaled_logits = logits / temperature
        softmax_probs = torch.nn.functional.softmax(scaled_logits, dim=0)
        return softmax_probs
    
    IGNORE_INDEX = -100
    def compute_token_nll(
        model_output, prompt_len, generation, probs_only_query,
        groundedness_vector=None
    ):
        if args.prompt_format == 'q_positive':
            landa_1 = 0.6
            landa_2 = 0.2
            landa_3 = 0.2
        elif args.prompt_format == 'q_negative':
            landa_1 = 0.0
            landa_2 = 1.0
            landa_3 = 0.0
        # # log probabilities of the target words
        # # Just in case the loss is not NLL for the model
        # # assert len(generation.shape) == 1
        # _logits = model_output['logits'][0, prompt_len-1:-1]
        # criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='mean')
        # loss = criterion(_logits, generation[prompt_len:])
        # if torch.isnan(loss):
        #     loss = 100000
           
            
        ### == Mine =============
        # Current 
        _logits = model_output['logits'][0, prompt_len-1:-1]
        _logits = _logits.float()
        ids = generation[prompt_len:]
        probs = torch.nn.functional.softmax(_logits, dim=1)  #(gen_tokens, vocab_size)
        probs = torch.gather(probs, dim=1, index=ids.view(-1, 1))
        
        # == Sum
        if probs_only_query != None and groundedness_vector != None:
            probs_doc = torch.nn.functional.sigmoid(torch.tensor(groundedness_vector)).to(args.device)
            # probs_doc = torch.nn.functional.tanh(torch.tensor(groundedness_vector)).to(args.device)
            final_prob = landa_1*probs + landa_2*probs_only_query + landa_3*probs_doc
            probs = final_prob
        
        neg_log_likelihoods = -torch.log(probs.reshape(-1)) 
        neg_log_likelihoods = neg_log_likelihoods.reshape(-1)
        loss = torch.mean(neg_log_likelihoods)
        
        
        if torch.isnan(loss):
            loss = 100000
        return loss

    def compute_token_nll_importance_phrase(
        model_output, prompt_len, generation,
        importance_vector, phrases, probs_only_query,
        mode='mean', groundedness_vector=None
    ):
        if args.prompt_format == 'q_positive':
            landa_1 = 0.6
            landa_2 = 0.2
            landa_3 = 0.2
        elif args.prompt_format == 'q_negative':
            landa_1 = 0.0
            landa_2 = 1.0
            landa_3 = 0.0
        importance_vector = importance_vector.to(args.device)
        
        # if groundedness_vector != None:
        #     _groundedness_vector = [torch.tensor(item).to(args.device) for item in groundedness_vector]
        #     _groundedness_vector = torch.tensor([torch.max(item) for item in _groundedness_vector]).to(args.device)
        #     merged_groundedness_vector  = []
        
        ### ==== For current generation ==============
        _logits = model_output['logits'][0, prompt_len-1:-1]
        _logits = _logits.float()
        ids = generation[prompt_len:]
        probs = torch.nn.functional.softmax(_logits, dim=1)  #(gen_tokens, vocab_size)
        probs = torch.gather(probs, dim=1, index=ids.view(-1, 1))
        
        ### Mine: to add only query generation ======= 
        if probs_only_query != None and groundedness_vector != None:
            probs_doc = torch.nn.functional.sigmoid(torch.tensor(groundedness_vector)).to(args.device)
            # probs_doc = torch.nn.functional.tanh(torch.tensor(groundedness_vector)).to(args.device)
            final_prob = landa_1*probs + landa_2*probs_only_query + landa_3*probs_doc
            probs = final_prob
        
        neg_log_likelihoods = -torch.log(probs.reshape(-1)) 
        neg_log_likelihoods = neg_log_likelihoods.reshape(-1)
        #find probabilities of each word
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
                        
                        # if groundedness_vector != None:
                        #     merged_groundedness_vector.append(torch.max(_groundedness_vector[i:i+k]))
                        
                        found = True
                        i += k
                        token_idx = last_token 
                        break
        
        neg_log_likelihoods_word = torch.tensor(neg_log_likelihoods_word).to(args.device)
        merged_importance_vector = torch.tensor(merged_importance_vector).to(args.device)
        merged_importance_vector = merged_importance_vector/torch.sum(merged_importance_vector)
        
        # if groundedness_vector != None:
        #     merged_groundedness_vector = torch.tensor(merged_groundedness_vector).to(args.device)
        #     merged_groundedness_vector = merged_groundedness_vector/torch.sum(merged_groundedness_vector)       
                 
        # if args.prompt_format == "q_positive":
        #     power_values = torch.where(merged_groundedness_vector < 1, torch.tensor(1.0), torch.where(merged_groundedness_vector <= 2, 1/merged_groundedness_vector, torch.tensor(0.25))) 
        #     power_values = torch.tensor(power_values).to(args.device)
        # elif args.prompt_format == "q_negative":
        #     power_values = torch.where(merged_groundedness_vector < 2, torch.tensor(1.0), torch.tensor(4.0)) 
        # else:
        #     power_values = torch.ones_like(merged_groundedness_vector)
        # powered_merged_importance_vector = torch.pow(merged_importance_vector, power_values)
        # powered_merged_importance_vector = powered_merged_importance_vector/torch.sum(powered_merged_importance_vector)

        if 'medical' in args.run_id:
            merged_importance_vector = softmax_with_temperature(merged_importance_vector,0.001) # Only for medical dataset
            score = 0.5 * torch.sum(merged_importance_vector * neg_log_likelihoods_word) + 0.5 * torch.mean(neg_log_likelihoods)
        else:
            
            # if groundedness_vector != None:
            #     if args.prompt_format == 'q_negative':
            #         score = (1/3) * torch.mean(neg_log_likelihoods)\
            #             + (1/3) * torch.sum(merged_importance_vector * neg_log_likelihoods_word)\
            #             - (1/3) * torch.sum(torch.min(merged_importance_vector, merged_groundedness_vector) * neg_log_likelihoods_word)
            #     else:
            #         score = (1/3) * torch.mean(neg_log_likelihoods)\
            #             + (1/3) * torch.sum(merged_importance_vector * neg_log_likelihoods_word)\
            #             + (1/3) * torch.sum(merged_groundedness_vector * neg_log_likelihoods_word)
            
            # else:
            score = 0.5 * torch.sum(merged_importance_vector * neg_log_likelihoods_word) + 0.5 * torch.mean(neg_log_likelihoods)
                # score = 0.5 * torch.sum(powered_merged_importance_vector * neg_log_likelihoods_word) + 0.5 * torch.mean(neg_log_likelihoods)
            
        
        if torch.isnan(score):
            score = 100000
        
        return score

    # === Main loop ==================================
    result = []
    model.eval()
    with open(likelihoods_output_jsonl_file, 'w') as jl_ofile:
        with torch.no_grad():
            
            for i, sample in tqdm(enumerate(sequences)):
                result_dict = {}
                id_ = sample['id']
                prompt = sample['prompt']
                generations = sample['cleaned_generations'].to(args.device)
            
                importance_vector_most_likely = similarities_dict[id_]['importance_vector'][0]
                phrases_most_likely = similarities_dict[id_]['importance_vector'][1]
                importance_scores = similarities_dict[id_]['importance_scores']
                
                if args.with_groundedness == 'yes':
                    groundedness_vector_most_likely = groundedness_dict[id_]['groundedness_score_most_likely'][1]
                    groundedness_scores = groundedness_dict[id_]['groundedness_scores']
        
                average_neg_log_likelihoods = torch.zeros((generations.shape[0],))
                average_neg_log_likelihoods_importance_mean = torch.zeros((generations.shape[0],))
                average_neg_log_likelihoods_importance_max = torch.zeros((generations.shape[0],))
                average_neg_log_likelihoods_importance_min = torch.zeros((generations.shape[0],))
                average_unconditioned_neg_log_likelihoods = torch.zeros((generations.shape[0],))
                neg_log_likelihoods = torch.zeros((generations.shape[0],))
                neg_unconditioned_log_likelihoods = torch.zeros((generations.shape[0],))
                pointwise_mutual_information = torch.zeros((generations.shape[0],))
                
                result_json = {
                    'id': id_,
                    'question': sample['question'],
                    'answers': sample['answers'],
                    'outputs': []
                }
                
                ### === Mine: for only query generation =============
                if id_ in _sequences_only_q:
                    prompt_only_q = _sequences_only_q[id_]['prompt'].to(args.device)
                    prompt_only_q = prompt_only_q[prompt_only_q != tokenizer.pad_token_id]
                    len_prompt_only_q = len(prompt_only_q)
                
                for generation_index in range(generations.shape[0]):
                    
                    ### === For current generation (Q + P) ==========
                    prompt = prompt[prompt != tokenizer.pad_token_id]
                    _generation = generations[generation_index][generations[generation_index] != tokenizer.pad_token_id] # generation includes prompt
                    generation = torch.cat((prompt, _generation), dim=0) # Mine: generation does not include prompt
                    # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
                    target_ids = generation.clone()
                    target_ids[:len(prompt)] = -100
                    model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=False)
                    
                    ### === Mine: for only query generation =========
                    probs_only_query = None
                    if id_ in _sequences_only_q:
                        generation_only_q = torch.cat((prompt_only_q, _generation), dim=0) # Mine: generation does not include prompt
                        target_ids_only_q = generation_only_q.clone()
                        target_ids_only_q[:len(prompt_only_q)] = -100
                        model_output_only_q = model(torch.reshape(generation_only_q, (1, -1)), labels=target_ids_only_q, output_hidden_states=False)
                    
                        _logits_only_query = model_output_only_q['logits'][0, len_prompt_only_q-1:-1]
                        _logits_only_query = _logits_only_query.float()
                        ids_only_query = generation_only_q[len_prompt_only_q:]
                        probs_only_query = torch.nn.functional.softmax(_logits_only_query, dim=1)
                        probs_only_query = torch.gather(probs_only_query, dim=1, index=ids_only_query.view(-1, 1))
                    
                    # ===============================================
                    if args.with_groundedness == 'yes':
                        groundedness_score = groundedness_scores[generation_index][1] # mine
                    
                    generation_only = generation.clone()[(len(prompt) - 1):]
                    unconditioned_model_output = model(
                        torch.reshape(generation_only, (1, -1)),
                        labels=generation_only,
                        output_hidden_states=False
                    ) # Ignore prompt to get unconditioned model output
                    model_output_loss = compute_token_nll(
                        model_output, len(prompt), generation.reshape(-1), probs_only_query,
                        groundedness_vector=groundedness_score
                    ) # using this is more safe  
                    unconditioned_model_output_loss = compute_token_nll(
                        unconditioned_model_output, 1, generation_only.reshape(-1), probs_only_query,
                        groundedness_vector=groundedness_score
                    ) # using this is more safe


                    importance_score = importance_scores[generation_index][0]
                    phrases = importance_scores[generation_index][1]
                    
                    if args.with_groundedness == 'yes':
                        model_output_loss_importance_mean = compute_token_nll_importance_phrase(
                            model_output, len(prompt), generation.reshape(-1),
                            importance_score, phrases, probs_only_query,
                            mode='mean', groundedness_vector=groundedness_score
                        )
                        
                        model_output_loss_importance_max = compute_token_nll_importance_phrase(
                            model_output, len(prompt), generation.reshape(-1),
                            importance_score, phrases, probs_only_query,
                            mode='max', groundedness_vector=groundedness_score
                        )
                        model_output_loss_importance_min = compute_token_nll_importance_phrase(
                            model_output, len(prompt), generation.reshape(-1),
                            importance_score, phrases, probs_only_query,
                            mode='min', groundedness_vector=groundedness_score
                        )
                    else:
                        model_output_loss_importance_mean = compute_token_nll_importance_phrase(
                            model_output, len(prompt), generation.reshape(-1),
                            importance_score, phrases, probs_only_query,
                            mode='mean'
                        )
                        model_output_loss_importance_max = compute_token_nll_importance_phrase(
                            model_output, len(prompt), generation.reshape(-1),
                            importance_score, phrases, probs_only_query,
                            mode='max'
                        )
                        model_output_loss_importance_min = compute_token_nll_importance_phrase(
                            model_output, len(prompt), generation.reshape(-1),
                            importance_score, phrases, probs_only_query,
                            mode='min'
                        )

                    average_neg_log_likelihoods_importance_mean[generation_index] = model_output_loss_importance_mean
                    average_neg_log_likelihoods_importance_max[generation_index] = model_output_loss_importance_max
                    average_neg_log_likelihoods_importance_min[generation_index] = model_output_loss_importance_min
                    
                    average_neg_log_likelihood = model_output_loss
            
                    average_unconditioned_neg_log_likelihood = unconditioned_model_output_loss
                    average_neg_log_likelihoods[generation_index] = average_neg_log_likelihood
                    average_unconditioned_neg_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood

                    # total neg lok likelihoods
                    neg_log_likelihoods[generation_index] = average_neg_log_likelihood * (len(generation) - len(prompt))
                    neg_unconditioned_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood * (
                        len(generation) - len(prompt))

                    pointwise_mutual_information[generation_index] = -neg_log_likelihoods[
                        generation_index] + neg_unconditioned_log_likelihoods[generation_index]

                ### ====================================================================== 
                # do the same thing above to first and second most likely generations
                # Here (the following three lines, I changed the code)
                
                if len(sample['cleaned_most_likely_generation_ids']) > 0:
                    _most_likely_generation = sample['cleaned_most_likely_generation_ids'].to(args.device)
                    _most_likely_generation = _most_likely_generation[_most_likely_generation != tokenizer.pad_token_id]
                    most_likely_generation = torch.cat((prompt, _most_likely_generation), dim=0) # Mine: generation does not include prompt
                    target_ids = most_likely_generation.clone()
                    target_ids[:len(prompt)] = -100
                    model_output = model(torch.reshape(most_likely_generation, (1, -1)), labels=target_ids, output_hidden_states=False)
                    
                    # Mine: For only_q 
                    probs_only_query = None
                    if id_ in _sequences_only_q:
                        most_likely_generation_only_q = torch.cat((prompt_only_q, _most_likely_generation), dim=0) # Mine: generation does not include prompt
                        target_ids_only_q = most_likely_generation_only_q.clone()
                        target_ids_only_q[:len_prompt_only_q] = -100
                        model_output_only_q = model(torch.reshape(most_likely_generation_only_q, (1, -1)), labels=target_ids_only_q, output_hidden_states=False)
                    
                        _logits_only_query = model_output_only_q['logits'][0, len_prompt_only_q-1:-1]
                        _logits_only_query = _logits_only_query.float()
                        ids_only_query = most_likely_generation_only_q[len_prompt_only_q:]
                        probs_only_query = torch.nn.functional.softmax(_logits_only_query, dim=1)
                        probs_only_query = torch.gather(probs_only_query, dim=1, index=ids_only_query.view(-1, 1))
                    
                    model_output_loss = compute_token_nll(
                        model_output, len(prompt), target_ids.reshape(-1), probs_only_query,
                        groundedness_vector=groundedness_vector_most_likely
                    ) # using this is more safe
                    
                    if args.with_groundedness == 'yes':
                        model_output_loss_importance_mean = compute_token_nll_importance_phrase(
                            model_output, len(prompt), target_ids.reshape(-1),
                            importance_vector_most_likely, phrases_most_likely, probs_only_query,
                            mode='mean', groundedness_vector=groundedness_vector_most_likely
                        )
                        model_output_loss_importance_max = compute_token_nll_importance_phrase(
                            model_output, len(prompt), target_ids.reshape(-1),
                            importance_vector_most_likely, phrases_most_likely, probs_only_query,
                            mode='max', groundedness_vector=groundedness_vector_most_likely
                        )
                        model_output_loss_importance_min = compute_token_nll_importance_phrase(
                            model_output, len(prompt), target_ids.reshape(-1),
                            importance_vector_most_likely, phrases_most_likely, probs_only_query,
                            mode='min', groundedness_vector=groundedness_vector_most_likely
                        )
                    else:
                        model_output_loss_importance_mean = compute_token_nll_importance_phrase(
                            model_output, len(prompt), target_ids.reshape(-1),
                            importance_vector_most_likely, phrases_most_likely, probs_only_query,
                            mode='mean'
                        )
                        model_output_loss_importance_max = compute_token_nll_importance_phrase(
                            model_output, len(prompt), target_ids.reshape(-1),
                            importance_vector_most_likely, phrases_most_likely, probs_only_query,
                            mode='max'
                        )
                        model_output_loss_importance_min = compute_token_nll_importance_phrase(
                            model_output, len(prompt), target_ids.reshape(-1),
                            importance_vector_most_likely, phrases_most_likely, probs_only_query,
                            mode='min'
                        )
                   
                    model_output_loss = model_output_loss.cpu()
                    result_dict['most_likely_neg_log_likelihoods'] = model_output_loss
                    result_dict['most_likely_neg_log_likelihoods_importance_mean'] = model_output_loss_importance_mean.cpu()
                    result_dict['most_likely_neg_log_likelihoods_importance_max'] = model_output_loss_importance_max.cpu()
                    result_dict['most_likely_neg_log_likelihoods_importance_min'] = model_output_loss_importance_min.cpu()
                   
                else:
                    score = 100000
                    model_output_loss = score
                    model_output_loss_importance_mean = score
                    model_output_loss_importance_max = score
                    model_output_loss_importance_min = score
                    
                    result_dict['most_likely_neg_log_likelihoods'] = model_output_loss
                    result_dict['most_likely_neg_log_likelihoods_importance_mean'] = model_output_loss_importance_mean
                    result_dict['most_likely_neg_log_likelihoods_importance_max'] = model_output_loss_importance_max
                    result_dict['most_likely_neg_log_likelihoods_importance_min'] = model_output_loss_importance_min
                
                # second_most_likely_generation = sample['second_most_likely_generation_ids'].to(args.device)
                # target_ids = second_most_likely_generation.clone()
                # target_ids[:len_prompt] = -100
                # model_output = model(torch.reshape(second_most_likely_generation, (1, -1)),
                #                         labels=target_ids,
                #                         output_hidden_states=False)
                # average_neg_log_likelihood_of_second_most_likely_gen = model_output['loss']
                
                average_neg_log_likelihood_of_most_likely_gen = model_output_loss
                neg_log_likelihood_of_most_likely_gen = average_neg_log_likelihood_of_most_likely_gen * (len(most_likely_generation) - len(prompt))
                ### ======================================================================
                
                result_dict['id'] = id_
                result_dict['prompt'] = prompt.cpu()
                result_dict['generations'] = generations.cpu()
                result_dict['average_neg_log_likelihoods'] = average_neg_log_likelihoods.cpu()
                result_dict['average_neg_log_likelihoods_importance_mean'] = average_neg_log_likelihoods_importance_mean.cpu()
                result_dict['average_neg_log_likelihoods_importance_max'] = average_neg_log_likelihoods_importance_max.cpu()
                result_dict['average_neg_log_likelihoods_importance_min'] = average_neg_log_likelihoods_importance_min.cpu()
            
                result_dict['neg_log_likelihoods'] = neg_log_likelihoods.cpu()
                result_dict['average_unconditioned_neg_log_likelihoods'] = average_unconditioned_neg_log_likelihoods.cpu()
                result_dict['neg_unconditioned_log_likelihoods'] = neg_unconditioned_log_likelihoods.cpu()
                result_dict['pointwise_mutual_information'] = pointwise_mutual_information.cpu()
                result_dict['average_neg_log_likelihood_of_most_likely_gen'] = average_neg_log_likelihood_of_most_likely_gen
                # result_dict['average_neg_log_likelihood_of_second_most_likely_gen'] = average_neg_log_likelihood_of_second_most_likely_gen.cpu()
                result_dict['neg_log_likelihood_of_most_likely_gen'] = neg_log_likelihood_of_most_likely_gen
                result_dict['semantic_set_ids'] = torch.tensor(similarities_dict[id_]['semantic_set_ids'], device='cpu')
                
                result_dict['has_different_answers'] = similarities_dict[id_]['has_different_answers']
                result_dict['unique_answers_indices'] = similarities_dict[id_]['unique_answers_indices']
                
                result.append(result_dict)
                jl_ofile.write(json.dumps(result_json) + '\n')
                
    
    ### === Save the likelihoods result ============
    with open(likelihoods_output_file, 'wb') as ofile:
        pickle.dump(result, ofile)
    print(f"Results saved to {likelihoods_output_file}")

    # print(result_json)
    # with open(likelihoods_output_jsonl_file, 'w') as json_file:
    #     json.dump(result_json, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--dataset', type=str, default='webquestions', choices=[
        'trivia', 'nq', 'squad1', 'webquestions',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa_org', 'topicoqa_his', 'topicoqa_rw',
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--prompt_format', type=str, default='q_positive', choices=[
        'only_q', 'q_positive', 'q_negative'
    ])
    parser.add_argument('--accuracy_metric', type=str, default="exact_match", choices=[
        'exact_match', 'bert_score', 'bem_score', 'rouge_score', 'llama3_score', 'gpt_score'
    ])
    
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.1)
    parser.add_argument("--roc_auc_threshold", type=float, default=0.9)
    parser.add_argument("--output_file_postfix", type=str, default="")
    
    parser.add_argument('--num_generations_per_prompt', type=int, default=10)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--type_of_question', type=str)
    parser.add_argument('--decoding_method', type=str, default='beam_search')
    parser.add_argument('--temperature', type=float, default='1.0')
    parser.add_argument('--num_beams', type=int, default='1')
    parser.add_argument('--top_p', type=float, default=1.0)
    
    parser.add_argument('--with_groundedness', type=str, default='yes', choices=['no', 'yes'])
    parser.add_argument('--mode', type=str, default='seperated', choices=['seperated', 'combined'])
    parser.add_argument('--run_id', type=str, default='run_4')
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
    
    get_likelihoods(args)
    
    # python framework/run/get_likelihoods.py
    
   
 