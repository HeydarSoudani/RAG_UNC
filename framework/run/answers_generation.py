#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import json
import torch
import random
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.utils import set_seed
from dataset import single_hop


def generation(args):
    
    print("\n--- Step 1: Answers generation ...")
    print(f"""
        Model name:    {args.model}
        Dataset:       {args.dataset} ({args.fraction_of_data_to_use})
        Prompt (1st):  {args.main_prompt_format}
        Prompt (2ed):  {args.second_prompt_format}
        Run id:        {args.run_id}
        Seed:          {args.seed}
    """.replace('        ', ''))
    
    
    # === Define output files ===================
    model = args.model.split('/')[-1]
    sequences_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_generation_{args.generation_type}.pkl'
    cleaned_sequences_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_cleaned_generation_{args.generation_type}.pkl'
    os.makedirs(os.path.dirname(sequences_output_file), exist_ok=True)
    
    
    # === Model definition ======================
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    if tokenizer.__class__.__name__ == 'LlamaTokenizer':
        #eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.']] + [29889]  # seems to be '.' as well
        eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']] + [29889]  # seems to be '.' as well
        if 'mistral' in args.model:
            eos_token_id += [28723]
            print('added additional eos token')
        #eos_token_id = [tokenizer(_)['input_ids'] for _ in ['\n', ',', '.']]
    elif tokenizer.__class__.__name__ == 'GPT2Tokenizer':
        eos_token_id = [tokenizer.encode(_)[1] for _ in ['.', '\n']]
    elif tokenizer.__class__.__name__ == 'PreTrainedTokenizerFast':
        eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']]
        eos_token_id += [691]
    elif tokenizer.__class__.__name__ == 'CodeGenTokenizer':
        eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.']]
        #eos_token_id += [691]
    else:
        raise NotImplementedError
    
    eos_token_id += [tokenizer.eos_token_id]
    period_token_id = tokenizer('. ')['input_ids'][1]
    eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    question_framing_ids = [[tokenizer(eos_token)['input_ids'][-1]] for eos_token in eos_tokens]

    
    # === Setup dataset ==========================
    Dataset = single_hop.RAGDataset(tokenizer, args.main_prompt_format, args.dataset, args.subsec)
    dataset = Dataset.get_dataset()
    
    sample_index = 0
    print(f"Dataset example {sample_index}:")
    print(f"Id:               {dataset[sample_index]['question_id']}")
    print(f"Similarity Score: {dataset[sample_index]['similarity_score']}")
    print(f"Question:         {dataset[sample_index]['question']}")
    print(f"Answers:          {dataset[sample_index]['answers']}")
    print(f"Prompt:         \n{dataset[sample_index]['prompt']}")
    
    if args.fraction_of_data_to_use < 1.0:
        train_dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=args.seed)['train']
    else:
        train_dataset = dataset

    questions = train_dataset
    dataloader = torch.utils.data.DataLoader(questions, batch_size=1)
    
    
    ### === Generation loop ====================== 
    with torch.no_grad():
        sequences = []
        for idx, batch in tqdm(enumerate(dataloader)):        
            
            # === Generate multiple time ==========
            generations = torch.ones(
                (args.num_generations_per_prompt, args.max_new_tokens), # input_length + max_length_of_generated_sequence
                dtype=torch.long,
                device=args.device
            )
            
            input_ids = batch['input_ids'].to(args.device).reshape(1, -1)
            for i in range(args.num_generations_per_prompt):
                
                generation = model.generate(
                    input_ids,
                    do_sample=True,
                    num_return_sequences=1,
                    num_beams=args.num_beams,
                    max_new_tokens = args.max_new_tokens,
                    # max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                    #eos_token_id=period_token_id,
                    eos_token_id =eos_token_id,
                    temperature=args.temperature,
                    pad_token_id =tokenizer.eos_token_id,
                    bad_words_ids=question_framing_ids,
                    top_p=args.top_p
                )
                generated_len = generation.shape[1] - len(input_ids[0])
                generations[i, :generated_len] = generation[0, len(input_ids[0]):]
            
            prompt_text = tokenizer.decode(input_ids[0])
            sequence_dict = {
                'id': batch['question_id'][0],
                'question': batch['question'][0], 
                'answers': [ans[0] for ans in batch['answers']],
                'similarity_score': batch['similarity_score'][0],
                'prompt': input_ids[0],
                'prompt_text': prompt_text,
                'generations': generations
            }
                
            generated_texts = []
            for generation in generations:
                generated_texts.append(
                    tokenizer.decode(generation, skip_special_tokens=True)
                ) # We already skip special tokens
            sequence_dict['generated_texts'] = generated_texts
            
            # === Generate most likely =============
            input_ids = batch['input_ids'].to(args.device).reshape(1, -1)
            if args.decoding_method == 'beam_search':
                most_likely_generation = model.generate(
                    input_ids,
                    num_beams=1,
                    num_return_sequences=1,
                    do_sample=False,
                    max_new_tokens = args.max_new_tokens,
                    eos_token_id = eos_token_id,
                    pad_token_id =tokenizer.eos_token_id,
                    bad_words_ids=question_framing_ids
                    # max_length=input_ids.shape[1]+max_length_of_generated_sequence,
                    #eos_token_id=period_token_id,
                )
            elif args.decoding_method == 'greedy':
                most_likely_generation = model.generate(
                    input_ids,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens = args.max_new_tokens,
                    eos_token_id = eos_token_id,
                    bad_words_ids=question_framing_ids,
                    # max_length=input_ids.shape[1]+max_length_of_generated_sequence,
                    #eos_token_id=period_token_id,
                )
                
            generated_len = most_likely_generation.shape[1] - len(input_ids[0])
            sequence_dict['most_likely_generation_ids'] = most_likely_generation[0, len(input_ids[0]):].to('cpu')
            sequence_dict['most_likely_generation'] = tokenizer.decode(
                most_likely_generation[0, len(input_ids[0]):], skip_special_tokens=True
            )
            sequences.append(sequence_dict)
            # print(sequence_dict['generated_texts'])

            ### === Save the result ====================== 
            if idx % 50 == 0:
                with open(sequences_output_file, 'wb') as ofile:
                    pickle.dump(sequences, ofile)
                print(f"Results saved to {sequences_output_file}")

    ### === Save the sequences result ============
    with open(sequences_output_file, 'wb') as ofile:
        pickle.dump(sequences, ofile)
    print(f"Results saved to {sequences_output_file}")


    ### === Loop for cleaning the generated data =
    # = Second file in the main code =  
    print('Cleaning the generated data ...')
    cleaned_sequences = []
    for sample in tqdm(sequences):
        discard = False
        cleaned_generations = torch.ones_like(sample['generations'])
        question = sample['question']
        generated_texts = sample['generated_texts']
        cleaned_generated_texts = []
        
        max_len_of_generations = cleaned_generations.shape[-1]
        generated_text = sample['most_likely_generation']
        generated_text_cleaned = re.sub(r'[^\x00-\x7f]',r'', generated_text)
        
        if generated_text_cleaned == generated_text:
            if tokenizer.__class__.__name__=='PreTrainedTokenizerFast':
                clean_ids = torch.tensor(tokenizer(generated_text)['input_ids'][0:], device=args.device)
            else:
                clean_ids = torch.tensor(tokenizer(generated_text)['input_ids'][1:], device=args.device)

            sample['cleaned_most_likely_generation'] = generated_text_cleaned
            sample['cleaned_most_likely_generation_ids'] =  clean_ids

            for i, generated_text in enumerate(generated_texts):

                generated_text_cleaned = re.sub(r'[^\x00-\x7f]',r'', generated_text)
                if generated_text_cleaned != generated_text:
                    discard = True
                    break

                cleaned_generated_texts.append(generated_text_cleaned)
                if tokenizer.__class__.__name__=='PreTrainedTokenizerFast':
                    clean_ids = torch.tensor(tokenizer(generated_text)['input_ids'][0:], device=args.device)
                else:
                    clean_ids = torch.tensor(tokenizer(generated_text)['input_ids'][1:], device=args.device)
                cleaned_generations[i, :min(len(clean_ids), max_len_of_generations)] = clean_ids[:max_len_of_generations]

            if not discard:
                sample['cleaned_generated_texts'] = cleaned_generated_texts
                sample['cleaned_generations'] = cleaned_generations
                cleaned_sequences.append(sample)
    
    print(len(cleaned_sequences))
    ### === Save the sequences result ============
    with open(cleaned_sequences_output_file, 'wb') as ofile:
        pickle.dump(cleaned_sequences, ofile)
    print(f"Results saved to {cleaned_sequences_output_file}")
    


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
    parser.add_argument('--main_prompt_format', type=str, default='only_q', choices=[
        'only_q', 'q_positive', 'q_negative'
    ])
    parser.add_argument('--second_prompt_format', type=str, default='q_positive', choices=[
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
    
    set_seed(args.seed)
    generation(args)
    
    # python framework/run/answers_generation.py
    
    