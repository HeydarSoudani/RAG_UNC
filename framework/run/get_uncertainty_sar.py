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
from tqdm import tqdm
import torch.nn as nn
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import set_seed
import utils.clustering as pc 


def get_uncertainty_sar(args):
    print("\n--- Phase 2: Get SAR Uncertainty ...")
    print(f"""
        Model name: {args.model}
        Dataset: {args.dataset}
        Prompt (1st): {args.main_prompt_format}
        Prompt (2ed): {args.second_prompt_format}
        Run id: {args.run_id}
        Seed: {args.seed}
    """.replace('   ', ''))
    
    # === Define IN/OUT files ========================
    model = args.model.split('/')[-1]
    generation_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
    probabilities_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_probabilities_generation__sec_{args.second_prompt_format}.pkl'
    uncertainty_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_sar_uncertainty.pkl'
    uncertainty_output_jsonl_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_sar_uncertainty.jsonl'

    with open(generation_file, 'rb') as infile:
        sequences = pickle.load(infile)
    with open(probabilities_file, 'rb') as infile:
        probabilities_dict = pickle.load(infile)

    # === Load models ================================
    # measure_model_choices = ['cross-encoder/stsb-roberta-large', 'cross-encoder/stsb-distilroberta-base']
    measure_model_name = 'cross-encoder/stsb-roberta-large'
    measure_model = CrossEncoder(model_name=measure_model_name, num_labels=1)
    measure_tokenizer = AutoTokenizer.from_pretrained(measure_model_name, use_fast=False)

    # === Functions/Classes ==========================
    def get_tokenwise_importance(generated_text):
        tokenized = torch.tensor(measure_tokenizer.encode(generated_text, add_special_tokens=False))
        token_importance = []
        for token in tokenized:
            similarity_to_original = measure_model.predict(
                [
                    question + generated_text,
                    question + generated_text.replace(measure_tokenizer.decode(token, skip_special_tokens=True),'')
                ]
            )
            token_importance.append(1 - torch.tensor(similarity_to_original))

        token_importance = torch.tensor(token_importance).reshape(-1)
        return token_importance
    
    def get_sentence_similarities(question, generations):
        similarities = {}
        for i in range(len(generations)):
            similarities[i] = []

        for i in range(len(generations)):
            for j in range(i+1, len(generations)):
                gen_i = question + generations[i]
                gen_j = question + generations[j]
                similarity_i_j = measure_model.predict([gen_i, gen_j])
                similarities[i].append(similarity_i_j)
                similarities[j].append(similarity_i_j)

        return similarities

    
    def get_token_sar(probabilities_generations, importance_scores):
        scores = []
        for i in range(len(probabilities_generations)):
            probs = probabilities_generations[i][2].to(args.device).reshape(-1)
            tokens_importance = importance_scores[i].to(args.device)
        
            if len(tokens_importance) == len(probs):
                weighted_probs = ((tokens_importance / tokens_importance.sum()) * probs)
                scores.append(torch.tensor(weighted_probs).sum())
            else:
                scores.append(torch.tensor(0.0))
        
        return torch.tensor(scores)
 
    def get_sentence_sar(probabilities_generations, sentence_similarities, t=0.001):
        
        def semantic_weighted_log(similarities, entropies, t):
            probs = torch.exp(-1 * entropies)
            weighted_entropy = []
            for idx, (prob, ent) in enumerate(zip(probs, entropies)):
                w_ent = - torch.log(
                    prob + ((torch.tensor(similarities[idx]) / t) * torch.cat([probs[:idx], probs[idx + 1:]])).sum())
                weighted_entropy.append(w_ent)
            return torch.tensor(weighted_entropy)
    
        scores = []
        for i in range(len(probabilities_generations)):
            probs = probabilities_generations[i][2].reshape(-1)
            scores.append(torch.tensor(probs).sum())
        
        gen_scores = torch.tensor(scores)
        gen_scores = semantic_weighted_log(sentence_similarities, gen_scores, t=t)
        return gen_scores.mean()
        
    def get_sar():
        pass

    # === Main loop ==================================
    result_dict = {}
    for idx, sample in tqdm(enumerate(sequences)):
        if idx == 1:
            break
        
        id_ = sample['id']
        question = sample['question']
        generations = sample['cleaned_generated_texts']
        
        # Generations, Get tokenwise_importance 
        importance_scores_generations = []
        for generation_index in range(len(generations)):
            generated_text = generations[generation_index]
            token_importance_list = get_tokenwise_importance(generated_text)
            importance_scores_generations.append(token_importance_list)
            
        # Most_likelihoods, Get tokenwise_importance 
        generated_text_most_likelihoods = sample['cleaned_most_likely_generation']
        importance_scores_generation_most_likelihoods = get_tokenwise_importance(generated_text_most_likelihoods)

        # Get sentence similarity 
        sentence_similarity_generations = get_sentence_similarities(question, generations)
        
        result_dict[id_] = {
            "importance_scores": importance_scores_generations,
            "importance_scores_most_likelihoods": importance_scores_generation_most_likelihoods,
            "sentence_similarities": sentence_similarity_generations
        }
        

    # == Compute Uncertainty =========================
    for idx, sample in tqdm(enumerate(sequences)):
        if idx == 1:
            break
        
        id_ = sample['id']
        generations = sample['cleaned_generations'].to(args.device)
        probabilities_generations = probabilities_dict[id_]['probabilities']
            
        average_neg_log_likelihoods_token_sar = get_token_sar(
            probabilities_generations,
            result_dict[id_]['importance_scores']
        )
        average_neg_log_likelihoods_sentence_sar = get_sentence_sar(
            probabilities_generations,
            result_dict[id_]['sentence_similarities']
        )
        
        
        result_dict[id_]['token_sar_uncertainty'] = average_neg_log_likelihoods_token_sar.mean()
        result_dict[id_]['sentence_sar_uncertainty'] = average_neg_log_likelihoods_sentence_sar.mean()
        # result_dict[id_]['sar_uncertainty'] = average_neg_log_likelihoods_sar.mean()
            
    
    # === Save the uncertainty result ================
    with open(uncertainty_output_file, 'wb') as ofile:
        pickle.dump(result_dict, ofile)
    print(f"Results saved to {uncertainty_output_file}")


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
    
    parser.add_argument('--affinity_mode', type=str, default='disagreement')
    
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
    get_uncertainty_sar(args)
    
    # python framework/run/get_uncertainty_sar.py