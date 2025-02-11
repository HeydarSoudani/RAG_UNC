#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import pickle
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

from utils.utils import set_seed


def create_new_probability_dist(args):
    print("\n--- Step 3-1: Creating New Probability ...")
    print(f"""
        Model name:    {args.model}
        Dataset:       {args.dataset}
        Prompt (1st):  {args.main_prompt_format}
        Prompt (2ed):  {args.second_prompt_format}
        Run id:        {args.run_id}
        Seed:          {args.seed}
    """.replace('      ', ''))
    
    # === Files ======================================
    model = args.model.split('/')[-1]
    # inputs
    sequence_input_main = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_cleaned_generation_{args.generation_type}.pkl'
    sequence_input_secondry = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.second_prompt_format}/{model}_{args.temperature}_cleaned_generation_{args.generation_type}.pkl'
    # outputs
    new_uncertainty_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_new_uncertainty__sec_{args.second_prompt_format}.pkl'
    new_uncertainty_output_jsonl_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_new_uncertainty__sec_{args.second_prompt_format}.jsonl' 

    with open(sequence_input_main, 'rb') as infile:
        sequences_main = pickle.load(infile)
    with open(sequence_input_secondry, 'rb') as infile:
        sequences_secondry = pickle.load(infile)


    # === Model definition ===========================
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary Size: {vocab_size}")


    # === Prapare second prompt in an object =========
    _sequences_secondry = {}
    for item in sequences_secondry:
        _sequences_secondry[item['id']] = item


    # === Main loop, on sequence =====================
    probabilities = {}
    overall_results = {}
    with open(new_uncertainty_output_jsonl_file, 'w') as jl_ofile:
        for idx, sample in tqdm(enumerate(sequences_main)):
            
            # if idx == 10:
            #     break
            id_ = sample['id']
            question = sample['question']
            answers = sample['answers']
            probabilities[id_] = torch.zeros((1, vocab_size))
            
            # === Create the probability ==
            generated_ids = sample['cleaned_generations']
            for i in range(generated_ids.shape[0]):
                for j in range(generated_ids.shape[1]):
                    if generated_ids[i,j] != 1:
                        probabilities[id_][0, generated_ids[i,j]] +=1
            
            # if id_ in _sequences_secondry:
            #     generated_ids_sec = _sequences_secondry[id_]['cleaned_generations']
            #     for i in range(generated_ids_sec.shape[0]):
            #         for j in range(generated_ids_sec.shape[1]):
            #             if generated_ids_sec[i,j] != 1:
            #                 probabilities[id_][0, generated_ids_sec[i,j]] +=1
                
            # === Calculate entropy =====
            probs = probabilities[id_][0]
            probs = probs / probs.sum()
            entropy = -torch.sum(probs * torch.log(probs + 1e-9))
            
            
            # === Save results
            overall_results[id_] = {
                'entropy': entropy.item()
            }
            
            result_item = {
                'id': id_,
                'entropy': entropy.item(),
                'question': question,
                'answers': answers
            }
            jl_ofile.write(json.dumps(result_item) + '\n')
        
        
    ### === Save the uncertainty result ============
    with open(new_uncertainty_output_file, 'wb') as ofile:
        pickle.dump(overall_results, ofile)
    print(f"Results saved to {new_uncertainty_output_file}")
        
        
    
    # Plot
    # id_ = sequences_main[0]['id']
    # values = probabilities[id_].squeeze(0)  # Removes the first dimension
    # print(values)
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(vocab_size), values.numpy(), marker='o', linestyle='-', label="Vocab Values")
    # plt.xlabel("Vocab Index", fontsize=12)
    # plt.ylabel("Value", fontsize=12)
    # plt.title("Tensor Values by Vocab Index", fontsize=14)
    # plt.grid(True)
    # plt.legend()
    
    # output_file = "tensor_plot.png"  # Specify the output file name and format
    # plt.savefig(output_file, dpi=300)  # Save with high resolution
    # plt.close()
        
        


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
    create_new_probability_dist(args)
    
    # python framework/run/test_faegheh_idea.py