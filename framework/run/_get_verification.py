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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import set_seed

def get_verification(args):
    
    print("\n--- Step 2: Getting Verification ...")
    print(f"""
        Model name: {args.model}
        Dataset: {args.dataset}
        Prompt format: {args.prompt_format}
        Run id: {args.run_id}
        Seed: {args.seed}
    """.replace('   ', ''))
    
    # === Define output files =============
    model = args.model.split('/')[-1]
    # model = 'Llama-2-7b-chat-hf'
    
    if args.prompt_format == 'only_q':
        verification_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_verification_generation.pkl'
        verification_output_jsonl_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_verification_generation.jsonl'
        similarities_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_similarities_generation.pkl'
    else:
        verification_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_{args.mode}_verification_generation.pkl'
        verification_output_jsonl_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_{args.mode}_verification_generation.jsonl'
        similarities_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_{args.mode}_similarities_generation.pkl'
    
    # === Read the generated data =========
    if args.mode == 'seperated':
        generation_input = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
    elif args.mode == 'combined':
        generation_input = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_combined_generation.pkl'
    else:
        print('mode is not defined')
    
    with open(generation_input, 'rb') as infile:
        sequences = pickle.load(infile)
    with open(similarities_file, 'rb') as infile:
        similarities_dict = pickle.load(infile)
        
        
    # === Load model tokenizer ============
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token_id = 1 # Very crucial don't forget
    

    # === Functions =======================
    # ref: https://stackoverflow.com/questions/76397904/generate-the-probabilities-of-all-the-next-possible-word-for-a-given-text
    def get_next_word_probabilities(model, tokenizer, input_prompt, top_k=500):
        tokenized_input = tokenizer.encode(input_prompt, return_tensors='pt').to(args.device)
        outputs = model(tokenized_input)
        predictions = outputs[0]
        next_token_candidates_tensor = predictions[0, -1, :]
        
        topk_candidates_indexes = torch.topk(next_token_candidates_tensor, top_k).indices.tolist()
        all_candidates_probabilities = torch.nn.functional.softmax(next_token_candidates_tensor, dim=-1)
        topk_candidates_probabilities = all_candidates_probabilities[topk_candidates_indexes].tolist()
        topk_candidates_tokens = [tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]
        
        return all_candidates_probabilities, list(zip(topk_candidates_tokens, topk_candidates_probabilities))
    
    def inference(model, tokenizer, sequence):
        words = re.findall(r'\w+|[^\w\s]', sequence['generated_answer'])
        
        query = sequence['question']
        q_instruction = "You are given a question and you MUST respond the answer (max 10 tokens) from your memorized knowledge.\n"
        q_d_instruction = "You are given a question and you MUST respond the answer (max 10 tokens) either from the provided document or your memorized knowledge.\n"
        
        match = re.search(r"Documents:(.*?)Question:", sequence['prompt'], re.DOTALL)
        if match:
            document_text = match.group(1).strip()
            # print("Extracted Document Text:", document_text)
        else:
            print("No document text found between 'Documents:' and 'Question:'")
            document_text = ""

        groundedness_scores = []
        for idx, word in enumerate(words):
            # print(idx)
            # if idx == 3:
            #     break
            
            previous_words = " ".join(words[:idx])
            q_input = f"{q_instruction}Question: {query} Answer: {previous_words}"
            q_d_input = f"{q_d_instruction}Documents: {document_text} \nQuestion: {query} Answer: {previous_words}"
  
            q_probabilities, q_top_k = get_next_word_probabilities(model, tokenizer, q_input)
            q_d_probabilities, q_d_top_k = get_next_word_probabilities(model, tokenizer, q_d_input)
            q_probabilities_top_k = torch.tensor([item[1] for item in q_top_k])
            q_d_probabilities_top_k = torch.tensor([item[1] for item in q_d_top_k])

            kl_divergence_value = torch.sum(q_d_probabilities * torch.log(q_d_probabilities / q_probabilities))

            groundedness_scores.append(kl_divergence_value.item())

        # group to phrases
        counter = 0
        phrase_groundedness_scores = []
        for idx, phrase in enumerate(sequence['phrases']):
            phrase_words = re.findall(r'\w+|[^\w\s]', phrase)
            len_phrase_words = len(phrase_words)
            
            phrase_groundedness_scores.append(
                groundedness_scores[counter:counter+len_phrase_words]
            )
            counter += len_phrase_words
            
        return sequence['phrases'], phrase_groundedness_scores
    
    def get_groundedness_vector(sequence, pooling='max'):
        # sequence.keys(): 'q_prompt', 'q_d_prompt', 'generated_answer', 'phrases',
        _generation = sequence['generated_answer'].to(args.device)
        _generation = _generation[_generation != tokenizer.pad_token_id]
        
        # For Q only
        q_prompt_len = len(sequence['q_prompt'])
        q_generation = torch.cat((sequence['q_prompt'], _generation), dim=0)
        q_target_ids = q_generation.clone()
        q_target_ids[:q_prompt_len] = -100
        
        q_output = model(torch.reshape(q_generation, (1, -1)), labels=q_target_ids, output_hidden_states=False)
        _q_logits = q_output['logits'][0, q_prompt_len-1:-1]
        _q_logits = _q_logits.float()
        q_probs = torch.nn.functional.softmax(_q_logits, dim=1)
        # q_ids = q_generation[q_prompt_len:]
        # probs = torch.gather(q_probs, dim=1, index=q_ids.view(-1, 1))
        
        # For Q+D
        q_d_prompt_len = len(sequence['q_d_prompt'])
        q_d_generation = torch.cat((sequence['q_d_prompt'], _generation), dim=0)
        q_d_target_ids = q_d_generation.clone()
        q_d_target_ids[:q_d_prompt_len] = -100
        q_d_output = model(torch.reshape(q_d_generation, (1, -1)), labels=q_d_target_ids, output_hidden_states=False)
        _q_d_logits = q_d_output['logits'][0, q_d_prompt_len-1:-1]
        _q_d_logits = _q_d_logits.float()
        q_d_probs = torch.nn.functional.softmax(_q_d_logits, dim=1)
        # q_d_ids = q_d_generation[q_d_prompt_len:]
        # probs = torch.gather(q_d_probs, dim=1, index=q_d_ids.view(-1, 1))
        
        # Ensure probabilities are normalized along the last dimension to avoid issues with division
        q_probs = q_probs / q_probs.sum(dim=1, keepdim=True)
        q_d_probs = q_d_probs / q_d_probs.sum(dim=1, keepdim=True)
        kl_divergence_values = torch.sum(q_d_probs * torch.log(q_d_probs / q_probs), dim=1, keepdim=True)
        groundedness_scores = kl_divergence_values.squeeze(dim=1)
        
        # group to phrases
        counter = 0
        phrase_groundedness_scores = []
        for idx, phrase in enumerate(sequence['phrases']):
            phrase_words = re.findall(r'\w+|[^\w\s]', phrase)
            len_phrase_words = len(phrase_words)
            
            phrase_slice = groundedness_scores[counter:counter+len_phrase_words]
            if pooling == 'mean':
                phrase_groundedness_score = sum(phrase_slice)/len_phrase_words
            elif pooling == 'max':
                phrase_groundedness_score = torch.max(phrase_slice) if phrase_slice.numel() > 0 else torch.tensor(0.0)
            elif pooling == 'min':
                phrase_groundedness_score = min(phrase_slice)

            # phrase_groundedness_scores.append(phrase_groundedness_score.item())
            phrase_groundedness_scores.append(round(phrase_groundedness_score.item(), 6))    
            counter += len_phrase_words
        
    
        _groundedness_scores = [round(item.item(), 6) for item in groundedness_scores]
        
        return (sequence['generated_answer_text'], _groundedness_scores)
        # return (sequence['phrases'], phrase_groundedness_scores)
    
    # === Main loop, on sequence ==========
    result_dict = {}
    model.eval()
    with open(verification_output_jsonl_file, 'w') as jl_ofile:
        with torch.no_grad():
            for idx, sample in tqdm(enumerate(sequences)):
                id_ = sample['id']
                
                if id_ not in similarities_dict:
                    continue
                
                question = sample['question']
                prompt = sample['prompt']
                phrases = [item[1] for item in similarities_dict[id_]["importance_scores"]]
                
                result_dict[id_] = {
                    'question': question,
                    'answers': sample['answers'],
                }
                
                ### === Create prompts =====================
                prompt_text = tokenizer.decode(prompt)
                match = re.search(r"Documents:(.*?)Question:", prompt_text, re.DOTALL)
                if match:
                    document_text = match.group(1).strip()
                    # print("Extracted Document Text:", document_text)
                else:
                    print("No document text found between 'Documents:' and 'Question:'")
                    document_text = " "
                
                q_prompt_text = f"""Answer the question:\nQuestion: {question}\nAnswer: """.replace('   ', '')
                q_d_prompt_text = f"""You are given a question and you MUST respond the answer (max 10 tokens) either from the provided document or your memorized knowledge.\n\nDocuments: {document_text}\n\nQuestion: {question}\n\nAnswer: """.replace('   ', '')
                
                q_prompt_tokenized = tokenizer(q_prompt_text, return_tensors='pt')['input_ids'].to(args.device)
                q_d_prompt_tokenized = tokenizer(q_d_prompt_text, return_tensors='pt')['input_ids'].to(args.device)
                q_prompt_tokenized = q_prompt_tokenized[q_prompt_tokenized != tokenizer.pad_token_id]
                q_d_prompt_tokenized = q_d_prompt_tokenized[q_d_prompt_tokenized != tokenizer.pad_token_id]
                
                # = For most-likely =======================
                if len(sample['cleaned_most_likely_generation_ids']) > 0:
                    sequence = {
                        'q_prompt': q_prompt_tokenized,
                        'q_d_prompt': q_d_prompt_tokenized,
                        'generated_answer_text': sample['cleaned_most_likely_generation'],
                        'generated_answer': sample['cleaned_most_likely_generation_ids'],
                        'phrases': similarities_dict[id_]["importance_vector"][1],
                    }    
                    groundedness_score_most_likely = get_groundedness_vector(sequence)
                else:
                    groundedness_score_most_likely = (sample['cleaned_most_likely_generation'], [])
                
                result_dict[id_]['groundedness_score_most_likely'] = groundedness_score_most_likely
                
                # = For generations ========================
                groundedness_scores = []
                for i, generation in enumerate(sample['cleaned_generations'].to(args.device)):
                    sequence = {
                        'q_prompt': q_prompt_tokenized,
                        'q_d_prompt': q_d_prompt_tokenized,
                        'generated_answer_text': sample['cleaned_generated_texts'][i],
                        'generated_answer': generation,
                        'phrases': phrases[i]
                    }
                    groundedness_scores.append(get_groundedness_vector(sequence))                
                result_dict[id_]['groundedness_scores'] = groundedness_scores
                
                
                # = Write in the json file
                result_item = {
                    'id': id_,
                    'question': question,
                    'answers': sample['answers'],   
                    'groundedness_score_most_likely': groundedness_score_most_likely,
                    'groundedness_scores': groundedness_scores   
                }
                
                jl_ofile.write(json.dumps(result_item) + '\n')
                
    ### === Save the sequences result ======
    with open(verification_output_file, 'wb') as ofile:
        pickle.dump(result_dict, ofile)
    print(f"Results saved to {verification_output_file}")


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
    
    get_verification(args)
    
    
    # python framework/run/get_verification.py
    