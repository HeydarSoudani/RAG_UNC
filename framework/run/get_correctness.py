#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import pickle
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.utils import set_seed
from metrics.correctness import BemScore, BertScore, ExactMatch, RougeScore, LLamaScore


def get_correctness(args):
    print("\n--- Step 5: Get Correctness ...")
    print(f"""
        Model name: {args.model}
        Dataset: {args.dataset}
        Prompt (1st): {args.main_prompt_format}
        Prompt (2ed): {args.second_prompt_format}
        Run id: {args.run_id}
        Seed: {args.seed}
    """.replace('   ', ''))

    # === Define output file ========================
    model = args.model.split('/')[-1]
    # inputs
    sequence_input = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_cleaned_generation_{args.generation_type}.pkl'
    similarities_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_similarities_generation.pkl'
    # outputs
    correctness_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_correctness.pkl'
    correctness_output_jsonl_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_correctness.jsonl'
    
    with open(sequence_input, 'rb') as infile:
        sequences = pickle.load(infile)
    with open(similarities_file, 'rb') as f:
        similarities_dict = pickle.load(f)


    # === Define correctness scores =================
    bem_score = BemScore()
    bert_score = BertScore()
    # rouge_score = RougeScore()
    exact_match_score = ExactMatch()
    # llama_score = LLamaScore(args.llama_eval_model, args.device)
    
    
    # === Main loop =================================
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    correctness_sequences = []
        
    with open(correctness_output_jsonl_file, 'w') as jl_ofile:    
        for idx, sample in tqdm(enumerate(sequences)):
            question_id = sample['id']
            question = sample['question']
            reference_answers = sample['answers']
            prompt_text = tokenizer.decode(sample['prompt'])
            sequence_dict = {
                'id': question_id,
                'question': question,
                'answers': reference_answers,
                'generated_texts': sample['generated_texts'],
                'most_likely_generation': sample['most_likely_generation'],
                'generations': sample['generations'],
                'most_likely_generation_ids': sample['most_likely_generation_ids'],
                'prompt': sample['prompt'],
                # 'second_most_likely_generation': sample['second_most_likely_generation'],
                # 'second_most_likely_generation_ids': sample['second_most_likely_generation_ids']
            }
            
            # === Calculate correctness score
            candidate = sample['most_likely_generation'].lstrip()
            bem_score_ = bem_score(question, reference_answers, candidate)
            bert_score_ = bert_score(reference_answers, candidate)
            exact_match_ = exact_match_score(reference_answers, candidate)
            # rouge_score_ = rouge_score(reference_answers, candidate)
            # llama_score_ = llama_score(question, reference_answers, candidate)
            sequence_dict['bem_score'] = bem_score_
            sequence_dict['bert_score'] = bert_score_
            sequence_dict['exact_match'] = exact_match_
            sequence_dict['rouge_score'] = 0.0 #rouge_score_
            # sequence_dict['llama_score'] = llama_score_
            correctness_sequences.append(sequence_dict)
            
            # === Write in a jsonl file 
            importance_scores = [([float(it) for it in item[0]], item[1]) for item in similarities_dict[question_id]['importance_scores']]
            correctness_sequences_jsl = {
                'id': question_id,
                'bem_score': sequence_dict['bem_score'],
                'bert_score': sequence_dict['bert_score'],
                'rouge_score': sequence_dict['rouge_score'],
                'exact_match': sequence_dict['exact_match'],
                # 'llama3_score': sequence_dict['llama_score'],
                'question': question,
                'answers': reference_answers,
                'generated_texts': sample['generated_texts'],
                'importance_scores': importance_scores,
                'most_likely_generation': sample['most_likely_generation'],
                'prompt': prompt_text,
            }
            jl_ofile.write(json.dumps(correctness_sequences_jsl) + '\n')
            
            
    ### === Save the correctness result ============
    with open(correctness_output_file, 'wb') as ofile:
        pickle.dump(correctness_sequences, ofile)
    print(f"Results saved to {correctness_output_file}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--dataset', type=str, default='trivia', choices=[
        'nqgold', 'trivia', 'popqa',
        'webquestions', 'squad1', 'nq',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa',
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--main_prompt_format', type=str, default='q_negative', choices=[
        'only_q', 'q_positive', 'q_negative',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    parser.add_argument('--second_prompt_format', type=str, default='only_q', choices=[
        'only_q', 'q_positive', 'q_negative',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    
    parser.add_argument('--accuracy_metric', type=str, default="exact_match", choices=[
        'bem_score', 'exact_match', 'bert_score', 'rouge_score', 'llama3_score', 'gpt_score'
    ])
    parser.add_argument('--model_llama_eval', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.057)
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
    parser.add_argument('--alpha_generation', type=float, default=0.1)
    parser.add_argument('--alpha_probability', type=float, default=0.1)
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
    get_correctness(args)
    
    # python framework/run/get_correctness.py
