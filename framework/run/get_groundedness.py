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

def get_groundedness(args):
    print("\n--- Step 3: Get Groundedness ...")
    print(f"""
        Model name:   {args.model}
        Dataset:      {args.dataset}
        Prompt (1st): {args.main_prompt_format}
        Prompt (2ed): {args.second_prompt_format}
        Run id:       {args.run_id}
        Seed:         {args.seed}
    """.replace('        ', ''))

    # === Files ======================================
    model = args.model.split('/')[-1]
    # inputs
    sequence_input_main = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
    sequence_input_secondry = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.second_prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
    similarities_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_similarities_generation.pkl'
    # outputs
    groundedness_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_groundedness_generation__sec_{args.second_prompt_format}.pkl'
    groundedness_output_jsonl_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_groundedness_generation__sec_{args.second_prompt_format}.jsonl' 
    
    with open(sequence_input_main, 'rb') as infile:
        sequences_main = pickle.load(infile)
    with open(sequence_input_secondry, 'rb') as infile:
        sequences_secondry = pickle.load(infile)
    with open(similarities_file, 'rb') as infile:
        similarities_dict = pickle.load(infile)

    # === Prapare second prompt in an object =========
    _sequences_secondry = {}
    for item in sequences_secondry:
        _sequences_secondry[item['id']] = item

    # === Load model tokenizer =======================
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token_id = 1 # Very crucial don't forget
    
    # === Functions ==================================
    def get_difference(first_prompt, second_prompt, generation, only_generation=False):
        
        _generation = generation[generation != tokenizer.pad_token_id]
        
        first_prompt = first_prompt[first_prompt != tokenizer.pad_token_id]
        len_first_prompt = len(first_prompt)
        second_prompt = second_prompt[second_prompt != tokenizer.pad_token_id]
        len_second_prompt = len(second_prompt)
    
        # first prompt
        p1_generation = torch.cat((first_prompt, _generation), dim=0)
        p1_target_ids = p1_generation.clone()
        p1_target_ids[:len_first_prompt] = -100
        p1_model_output = model(torch.reshape(p1_generation, (1, -1)), labels=p1_target_ids, output_hidden_states=False)
        _p1_logits = p1_model_output['logits'][0, len_first_prompt-1:-1]
        _p1_logits = _p1_logits.float()
        p1_probs = torch.nn.functional.softmax(_p1_logits, dim=1)
    
        # second prompt
        p2_generation = torch.cat((second_prompt, _generation), dim=0)
        if only_generation:
            p2_generation_only = p2_generation.clone()[len_second_prompt-1:]
            p2_model_output = model(torch.reshape(p2_generation_only, (1, -1)), labels=p2_generation_only, output_hidden_states=False)
            _p2_logits = p2_model_output['logits'][0, :-1]
        else:
            p2_target_ids = p2_generation.clone()
            p2_target_ids[:len_second_prompt] = -100
            p2_model_output = model(torch.reshape(p2_generation, (1, -1)), labels=p2_target_ids, output_hidden_states=False)
            _p2_logits = p2_model_output['logits'][0, len_second_prompt-1:-1]
        _p2_logits = _p2_logits.float()
        p2_probs = torch.nn.functional.softmax(_p2_logits, dim=1)
    
        # calculate kl
        p1_probs = p1_probs / p1_probs.sum(dim=1, keepdim=True)
        p2_probs = p2_probs / p2_probs.sum(dim=1, keepdim=True)
        kl_divergence_values = torch.sum(p1_probs * torch.log(p1_probs / p2_probs), dim=1, keepdim=True)
        kl_divergence_values = kl_divergence_values.squeeze(dim=1) #(num_token, )
        
        
        if kl_divergence_values.numel() == 0:
            difference_score = {
                'mean': 0.0,
                'max': 0.0,
                'min': 0.0
            }
        else:
            difference_score = {
                'mean': kl_divergence_values.mean().item(),
                'max': kl_divergence_values.max().item(),
                'min': kl_divergence_values.min().item()
            }

        return difference_score


    # === Main loop, on sequence =====================
    result = []
    model.eval()
    with open(groundedness_output_jsonl_file, 'w') as jl_ofile:
        with torch.no_grad():
            for idx, sample in tqdm(enumerate(sequences_main)):
                
                # if idx == 1:
                #     break
                
                id_ = sample['id']
                question = sample['question']
                answers = sample['answers']
                generations = sample['cleaned_generations'].to(args.device)
                generations_text = sample['cleaned_generated_texts']
                prompt = sample['prompt']
                
                generation_most_likely = sample['cleaned_most_likely_generation_ids'].to(args.device)
                generation_text_most_likely = sample['cleaned_most_likely_generation']
                generation_tokens_most_likely = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in generation_most_likely if token_id != 1]
                
                if id_ in _sequences_secondry:
                    prompt_secondry = _sequences_secondry[id_]['prompt'].to(args.device)
                
                result_dict = {}
                result_dict['id'] = id_
                
                # = For generations ====
                generation_kl_main_second = []
                generation_kl_main_third = []
                generation_kl_second_third = []
                for generation_index in range(generations.shape[0]):
                    cur_generation = generations[generation_index]
                    
                    # main-third
                    difference_score__main_third = get_difference(prompt, prompt, cur_generation, only_generation=True)
                    
                    if id_ in _sequences_secondry:
                        # main-second
                        difference_score__main_second = get_difference(prompt, prompt_secondry, cur_generation)
                        # second-third
                        difference_score__second_third = get_difference(prompt_secondry, prompt, cur_generation, only_generation=True)
                    else:
                        difference_score__main_second = {'mean':0.0, 'max':0.0, 'min':0.0}
                        difference_score__second_third = {'mean':0.0, 'max':0.0, 'min':0.0}
                    
                    generation_kl_main_second.append(difference_score__main_second)
                    generation_kl_main_third.append(difference_score__main_third)
                    generation_kl_second_third.append(difference_score__second_third)
                
                result_dict['generation_kl_main_second'] = generation_kl_main_second
                result_dict['generation_kl_main_third'] = generation_kl_main_third
                result_dict['generation_kl_second_third'] = generation_kl_second_third
                
                
                # = For most-likely ====
                if len(generation_most_likely) > 0:
                    # main second
                    difference_score__main_second_most_likely = get_difference(prompt, prompt_secondry, generation_most_likely)
                    # main third
                    difference_score__main_third_most_likely = get_difference(prompt, prompt, generation_most_likely, only_generation=True)
                    # second third
                    difference_score__second_third_most_likely = get_difference(prompt_secondry, prompt, generation_most_likely, only_generation=True)

                    result_dict['most_likely_kl_main_second'] = difference_score__main_second_most_likely
                    result_dict['most_likely_kl_main_third'] = difference_score__main_third_most_likely
                    result_dict['most_likely_kl_second_third'] = difference_score__second_third_most_likely

                else:
                    result_dict['most_likely_kl_main_second'] = {'mean':0.0, 'max':0.0, 'min':0.0}
                    result_dict['most_likely_kl_main_third'] = {'mean':0.0, 'max':0.0, 'min':0.0}
                    result_dict['most_likely_kl_second_third'] = {'mean':0.0, 'max':0.0, 'min':0.0}
                
                result.append(result_dict)
                jl_ofile.write(json.dumps(result_dict) + '\n')
       
    ### === Save the sequences result ======
    with open(groundedness_output_file, 'wb') as ofile:
        pickle.dump(result, ofile)
    print(f"Results saved to {groundedness_output_file}")
                


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
    parser.add_argument('--main_prompt_format', type=str, default='rerank_retriever_top5', choices=[
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
    get_groundedness(args)
    
    # python framework/run/get_groundedness.py