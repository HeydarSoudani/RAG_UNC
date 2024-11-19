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

from utils import set_seed


def get_probability(args):
    print("\n--- Step 3-1: Getting Probability ...")
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
    sequence_input_main = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
    sequence_input_secondry = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.second_prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
    # outputs
    probabilities_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_probabilities_generation__sec_{args.second_prompt_format}.pkl'
    probabilities_output_jsonl_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_probabilities_generation__sec_{args.second_prompt_format}.jsonl' 
    
    with open(sequence_input_main, 'rb') as infile:
        sequences_main = pickle.load(infile)
    with open(sequence_input_secondry, 'rb') as infile:
        sequences_secondry = pickle.load(infile)
    
    # === Prapare second prompt in an object =========
    _sequences_secondry = {}
    for item in sequences_secondry:
        _sequences_secondry[item['id']] = item

    # === Load model =================================
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token_id = 1 # Very crucial don't forget
    
    # === Functions ==================================
    def get_probability(prompt, generation):
        
        prompt = prompt[prompt != tokenizer.pad_token_id]
        len_prompt = len(prompt)
        
        _generation = generation[generation != tokenizer.pad_token_id]
        p_generation = torch.cat((prompt, _generation), dim=0)
        target_ids = p_generation.clone()
        target_ids[:len_prompt] = -100
        model_output = model(torch.reshape(p_generation, (1, -1)), labels=target_ids, output_hidden_states=False)
        
        _logits = model_output['logits'][0, len_prompt-1:-1]
        _logits = _logits.float()
        ids = p_generation[len_prompt:]
        probs = torch.nn.functional.softmax(_logits, dim=1)
        probs = torch.gather(probs, dim=1, index=ids.view(-1, 1))
        
        return probs
    
    # === Main loop, on sequence =====================
    result_dict = {}
    model.eval()
    with open(probabilities_output_jsonl_file, 'w') as jl_ofile:
        with torch.no_grad():
            for idx, sample in tqdm(enumerate(sequences_main)):
                id_ = sample['id']
                question = sample['question']
                answers = sample['answers']
                generations = sample['cleaned_generations'].to(args.device)
                generations_text = sample['cleaned_generated_texts']
                prompt = sample['prompt']
                
                generation_most_likely = sample['cleaned_most_likely_generation_ids'].to(args.device)
                generation_text_most_likely = sample['cleaned_most_likely_generation']
                generation_tokens_most_likely = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in generation_most_likely if token_id != 1]
                
                result_dict[id_] = {
                    'question': question,
                    'answers': answers,
                }
                                
                if id_ in _sequences_secondry:
                    prompt_secondry = _sequences_secondry[id_]['prompt'].to(args.device)
                
                # = For generations ====
                probabilities = []
                for generation_index in range(generations.shape[0]):
                    cur_generation = generations[generation_index]
                    cur_generation_tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in cur_generation if token_id != 1]
                    probs = get_probability(prompt, cur_generation)
                    
                    if id_ in _sequences_secondry:
                        probs_secondry = get_probability(prompt_secondry, cur_generation)
                        probabilities.append((generations_text[generation_index], cur_generation_tokens, probs, probs_secondry))
                    else:
                        probabilities.append((generations_text[generation_index], cur_generation_tokens, probs, torch.tensor([])))
                result_dict[id_]['probabilities'] = probabilities
                
                # = For most-likely ====
                if len(generation_most_likely) > 0:
                    probs_most_likely = get_probability(prompt, generation_most_likely)
                    
                    if id_ in _sequences_secondry:
                        probs_secondry_most_likely = get_probability(prompt_secondry, generation_most_likely)
                    else:
                        probs_secondry_most_likely = torch.tensor([])
                else:
                    probs_most_likely = torch.tensor([])
                    probs_secondry_most_likely = torch.tensor([])
                    
                probability_most_likely = (generation_text_most_likely, generation_tokens_most_likely, probs_most_likely, probs_secondry_most_likely)
                result_dict[id_]['probability_most_likely'] = probability_most_likely
                
                
                # = Write to the jsonl file
                probabilities_jsl = [(
                    prob[0],
                    prob[1],
                    [round(i, 4) for i in prob[2].reshape(1, -1).tolist()[0]],
                    [round(i, 4) for i in prob[3].reshape(1, -1).tolist()[0]]
                    ) for prob in probabilities]
                
                probability_most_likely_jsl = (
                    probability_most_likely[0],
                    probability_most_likely[1],
                    [round(i, 4) for i in probability_most_likely[2].reshape(1, -1).tolist()[0]],
                    [round(i, 4) for i in probability_most_likely[3].reshape(1, -1).tolist()[0]]
                )
                result_item = {
                    'id': id_,
                    'question': question,
                    'answers': answers,
                    'probabilities': probabilities_jsl,
                    'probability_most_likely': probability_most_likely_jsl   
                }
                jl_ofile.write(json.dumps(result_item) + '\n')
               
    ### === Save the sequences result ======
    with open(probabilities_output_file, 'wb') as ofile:
        pickle.dump(result_dict, ofile)
    print(f"Results saved to {probabilities_output_file}")
               

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
    parser.add_argument('--main_prompt_format', type=str, default='q_negative', choices=[
        'only_q', 'q_positive', 'q_negative', 'bm25_retriever', 'rerank_retriever'
    ])
    parser.add_argument('--second_prompt_format', type=str, default='only_q', choices=[
        'only_q', 'q_positive', 'q_negative', 'bm25_retriever', 'rerank_retriever'
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
    get_probability(args)
    
    # python framework/run/get_probabilities.py