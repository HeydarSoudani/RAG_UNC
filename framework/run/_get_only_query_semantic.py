#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import json
import torch
import pickle
import logging
import argparse
from tqdm import tqdm
from transformers import BertTokenizerFast 
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import set_seed





def get_only_query_semantic(args):
    
    print("\n--- Step 2-1: Getting Cluster Semantic ...")
    print(f"""
        Model name: {args.model}
        Dataset: {args.dataset}
        Prompt format: {args.prompt_format}
        Run id: {args.run_id}
        Seed: {args.seed}
    """.replace('   ', ''))    

    # === Define output files =============
    model = args.model.split('/')[-1]
    
    if args.prompt_format == 'only_q':
        only_query_semantic_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_only_query_semantic_generation.pkl'
        similarities_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_similarities_generation.pkl'
    else:
        only_query_semantic_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_{args.mode}_only_query_semantic_generation.pkl'
        similarities_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_{args.mode}_similarities_generation.pkl'

    # === Read the generated data =========
    only_q_generation_input = f'{args.output_dir}/{args.dataset}/{args.run_id}/only_q/{model}_{args.temperature}_cleaned_generation.pkl'
    if args.mode == 'seperated':
        generation_input = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
    elif args.mode == 'combined':
        generation_input = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_combined_generation.pkl'
    else:
        print('mode is not defined')
    
    
    with open(generation_input, 'rb') as infile:
        sequences = pickle.load(infile)
    with open(only_q_generation_input, 'rb') as infile:
        only_q_sequences = pickle.load(infile)
    with open(similarities_file, 'rb') as infile:
        similarities_dict = pickle.load(infile)

    only_q_sequences_obj = {}
    for sample in only_q_sequences:
        only_q_sequences_obj[sample['id']] = sample

    # === Load model tokenizer ============
    semantic_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    semantic_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(args.device)

    # === Functions =======================


    # === Main loop, on sequence ==========
    # model.eval()
    result_dict = {}
    with torch.no_grad():
        for idx, sample in tqdm(enumerate(sequences)):
            # if idx == 5:
            #     break
            
            id_ = sample['id']
            question = sample['question']
            
            if id_ not in only_q_sequences_obj.keys():
                result_dict[id_] = ([0 for _ in range(args.num_generations_per_prompt)], 1)
            
            else:
                generated_texts = sample['cleaned_generated_texts']
                only_q_generated_texts = only_q_sequences_obj[id_]['cleaned_generated_texts']
                filtered_only_q_generated_texts = [item for item in only_q_generated_texts if item != '\n']
                
                num_similar_generations = []
                for i, generation_text in enumerate(generated_texts):
                    
                    per_generation_counter = 0
                    # for only_q_generated_text in only_q_generated_texts:
                    for only_q_generated_text in filtered_only_q_generated_texts:
                        
                        qa_1 = question + ' ' + generation_text
                        qa_2 = question + ' ' + only_q_generated_text
                        
                        input = qa_1 + ' [SEP] ' + qa_2
                        encoded_input = semantic_tokenizer.encode(input, padding=True)
                        prediction = semantic_model(torch.tensor(torch.tensor([encoded_input]), device=args.device))['logits']
                        predicted_label = torch.argmax(prediction, dim=1)

                        reverse_input = qa_2 + ' [SEP] ' + qa_1
                        encoded_reverse_input = semantic_tokenizer.encode(reverse_input, padding=True)
                        reverse_prediction = semantic_model(torch.tensor(torch.tensor([encoded_reverse_input]), device=args.device))['logits']
                        reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
                        
                        if 0 in predicted_label or 0 in reverse_predicted_label:
                            pass
                        else:
                            per_generation_counter +=1
                    
                    num_similar_generations.append(per_generation_counter)
                result_dict[id_] = (num_similar_generations, len(filtered_only_q_generated_texts))
        
        
        # print(result_dict)
        ### === Save the sequences result ======
        with open(only_query_semantic_output_file, 'wb') as ofile:
            pickle.dump(result_dict, ofile)
        print(f"Results saved to {only_query_semantic_output_file}")
                


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
    parser.add_argument('--accuracy_metric', type=str, default="bem_score", choices=[
        'bem_score', 'exact_match', 'bert_score', 'rouge_score', 'llama3_score', 'gpt_score'
    ])
    
    parser.add_argument('--landa_1', type=float, default=1.0)
    parser.add_argument('--landa_2', type=float, default=0.0)
    parser.add_argument('--landa_3', type=float, default=0.0)
    
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.1)
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
    
    get_only_query_semantic(args)
    
    
    
    # python framework/run/get_only_query_semantic.py
    