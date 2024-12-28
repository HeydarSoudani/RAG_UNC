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
import torch.nn.functional as F
from typing import Union, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.utils import set_seed
from dataset import single_hop


class CAD:
    def __init__(self, model_name: str, device: Union[int,str] = 0):
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map=device, use_cache=True)
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        if self.tokenizer.__class__.__name__ == 'LlamaTokenizer':
            #eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.']] + [29889]  # seems to be '.' as well
            self.eos_token_id = [self.tokenizer.encode(_)[-1] for _ in ['.', '\n']] + [29889]  # seems to be '.' as well
            if 'mistral' in model_name:
                self.eos_token_id += [28723]
                print('added additional eos token')
            #self.eos_token_id = [tokenizer(_)['input_ids'] for _ in ['\n', ',', '.']]
        elif self.tokenizer.__class__.__name__ == 'GPT2Tokenizer':
            self.eos_token_id = [self.tokenizer.encode(_)[1] for _ in ['.', '\n']]
        elif self.tokenizer.__class__.__name__ == 'PreTrainedTokenizerFast':
            self.eos_token_id = [self.tokenizer.encode(_)[-1] for _ in ['.', '\n']]
            self.eos_token_id += [691]
        elif self.tokenizer.__class__.__name__ == 'CodeGenTokenizer':
            self.eos_token_id = [self.tokenizer.encode(_)[-1] for _ in ['.']]
            #self.eos_token_id += [691]
        else:
            raise NotImplementedError
        
        self.eos_token_id += [self.tokenizer.eos_token_id]
        period_token_id = self.tokenizer('. ')['input_ids'][1]
        eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
        question_framing_ids = [[self.tokenizer(eos_token)['input_ids'][-1]] for eos_token in eos_tokens]

        # special_tokens_dict = {'pad_token': '[PAD]'}
        # self.tokenizer.add_special_tokens(special_tokens_dict)
        # self.model.resize_token_embeddings(len(self.tokenizer))

        if self.tokenizer.pad_token_id is None:
            eos_token = self.tokenizer.decode([self.tokenizer.eos_token_id])
            self.tokenizer.add_special_tokens({"pad_token": eos_token})

    def _top_p_sampling(self, 
                        logits: torch.Tensor, 
                        top_p: float = 0.9, 
                        filter_value: float = -float("Inf"), 
                        min_tokens_to_keep: int = 1
                        ) -> torch.Tensor :

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep - 1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

        return logits

    def _top_k_sampling(self, 
                        logits: torch.Tensor, 
                        top_k: int = 20, 
                        filter_value: float = -float("Inf"), 
                        min_tokens_to_keep: int = 1
                        ) -> torch.Tensor :

        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None] # * logit 값이 Top-k의 토큰 중 가장 작은 값보다 작은 토큰의 인덱스 반환 
        logits[indices_to_remove] = filter_value

        return logits

    def predict_next_token(self, 
                           logits: torch.Tensor, 
                           decoding_strategy: str, 
                           top_p: float, 
                           top_k: int, 
                           use_repetition_penalty: bool, 
                           repetition_penalty_value: float, 
                           generated_tokens: List[set] = None
                           ) -> torch.Tensor :

        # * Repetitin Penalty 참고 코드 : https://huggingface.co/transformers/v2.11.0/_modules/transformers/modeling_utils.html#PreTrainedModel.enforce_repetition_penalty_
        if use_repetition_penalty:
            assert repetition_penalty_value >= 1.0, "Repetition penalty must be >= 1."
            mask = torch.zeros_like(logits)
            for i, token_set in enumerate(generated_tokens):
                mask[i, list(token_set)] = 1.0
            penalty = torch.where(mask == 1.0, repetition_penalty_value, 1.0) # generated_tokens에 있는 토큰들은 penalty를 repetition_penalty_value로, 없는 토큰들은 1.0(현상 유지)으로 설정
            logits *= torch.where(logits < 0, penalty, 1.0/penalty) # if logit is smaller than 0, multiply with penalty, else divide by penalty
        
        if decoding_strategy == 'top_p':
            assert top_p is not None, "top_p must be provided for top_p sampling"
            logits = self._top_p_sampling(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()

        elif decoding_strategy == 'top_k':
            assert top_k is not None, "top_k must be provided for top_k sampling"
            logits = self._top_k_sampling(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()

        elif decoding_strategy == 'greedy':
            next_token = torch.argmax(logits, dim=-1)

        return next_token

    def generate(self, 
                batch_with_context, # : List[str] 
                batch_wo_context,   # : Optional[List[str]] = None 
                alpha: float = 0.5,
                max_length: int = 256,
                decoding_strategy: str = 'top_p',
                top_p_value: float = 0.9,
                top_k_value: int = 20,
                use_repetition_penalty: bool = False, 
                repetition_penalty_value: float = 1.0,
                ) -> List[List[int]]:

        # # Tokenize 'input_texts' and create attention masks
        # tokenized_inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        # input_ids = tokenized_inputs['input_ids']
        # attention_mask = tokenized_inputs['attention_mask']

        # # Tokenize 'contexts' after concatenating with 'input_ids' if 'contexts' is not None
        # if contexts and use_context_aware:
        #     inputs_with_contexts = [context + self.tokenizer.eos_token + input_text for context, input_text in zip(contexts, input_texts)]
        #     tokenized_inputs_with_contexts = self.tokenizer(inputs_with_contexts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        #     input_ids_with_contexts = tokenized_inputs_with_contexts['input_ids']
        #     attention_mask_with_contexts = tokenized_inputs_with_contexts['attention_mask']
        # else:
        #     input_ids_with_contexts = input_ids
        #     attention_mask_with_contexts = attention_mask

        
        input_ids_with_contexts = batch_with_context['input_ids'].to(self.device).reshape(1, -1)
        attention_mask_with_contexts = batch_with_context['attention_mask'].to(self.device).reshape(1, -1)
        input_ids_wo_contexts = batch_wo_context['input_ids'].to(self.device).reshape(1, -1)
        attention_mask_wo_contexts = batch_wo_context['attention_mask'].to(self.device).reshape(1, -1)


        # Initialize variables for generation loop
        cur_len = 0
        batch_size = len(input_ids_wo_contexts)
        unfinished_sents = input_ids_with_contexts.new(batch_size).fill_(1)
        sent_lengths = input_ids_with_contexts.new(batch_size).fill_(max_length)
        generated_tokens = [[] for _ in range(batch_size)] # e.g., [[4132, 102, 29402], [2378, 7893, 23001]]

        # Generate tokens
        with torch.no_grad():
            while cur_len < max_length:
                
                # Generate without context
                outputs = self.model(input_ids_wo_contexts, attention_mask=attention_mask_wo_contexts)
                next_token_logits = outputs.logits[:, -1, :] # (batch_size, vocab_size)
                # Generate with context
                outputs_with_contexts = self.model(input_ids_with_contexts, attention_mask=attention_mask_with_contexts)
                next_token_logits_with_contexts = outputs_with_contexts.logits[:, -1, :]
                # Context-aware decoding
                next_token_logits = (1 + alpha) * next_token_logits_with_contexts - alpha * next_token_logits

                # Predict next token according to decoding strategy
                next_token = self.predict_next_token(logits=next_token_logits, 
                                                    decoding_strategy=decoding_strategy, 
                                                    top_p=top_p_value, 
                                                    top_k=top_k_value, 
                                                    use_repetition_penalty=use_repetition_penalty, 
                                                    repetition_penalty_value=repetition_penalty_value, 
                                                    generated_tokens=[set(tokens) for tokens in generated_tokens])
                
                # Handle EOS token and padding
                if self.tokenizer.eos_token_id is not None:
                    tokens_to_add = next_token * unfinished_sents + (self.tokenizer.pad_token_id) * (1 - unfinished_sents)
                else:
                    tokens_to_add = next_token

                # Update input_ids and attention masks for the next forward pass
                input_ids_wo_contexts = torch.cat([input_ids_wo_contexts, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask_wo_contexts = torch.cat([attention_mask_wo_contexts, unfinished_sents.unsqueeze(-1)], dim=-1)
                input_ids_with_contexts = torch.cat([input_ids_with_contexts, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask_with_contexts = torch.cat([attention_mask_with_contexts, unfinished_sents.unsqueeze(-1)], dim=-1)

                cur_len += 1

                # Update generated tokens and check for completion
                for i, token in enumerate(tokens_to_add.tolist()):
                    if unfinished_sents[i] == 1:
                        generated_tokens[i].append(token)

                # Check for sentences that are finished
                if self.tokenizer.eos_token_id is not None:
                    eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id
                    is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                    sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                    unfinished_sents.mul_((~eos_in_sents).long())

                # Break if all sentences are finished : stop when there is a EOS token in each sentence, or if we exceed the maximul length
                if unfinished_sents.max() == 0:
                    break

                if next_token in self.eos_token_id:
                    break

        # Return the generated tokens
        return generated_tokens


def generation_cad(args):
    
    print("\n--- Step 1: Answers generation (CAD) ...")
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
    sequences_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_generation_cad.pkl'
    cleaned_sequences_output_file = f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.main_prompt_format}/{model}_{args.temperature}_cleaned_generation_cad.pkl'
    os.makedirs(os.path.dirname(sequences_output_file), exist_ok=True)
    
    # === Model definition ======================
    cad_model = CAD(model_name=args.model, device=args.device)
    
    # === Setup dataset ==========================
    # = Main dataset
    Dataset_main = single_hop.RAGDataset(cad_model.tokenizer, args.main_prompt_format, args.dataset)
    dataset_main = Dataset_main.get_dataset()
    if args.fraction_of_data_to_use < 1.0:
        train_dataset_main = dataset_main.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=args.seed)['train']
    else:
        train_dataset_main = dataset_main

    questions = train_dataset_main
    dataloader_main = torch.utils.data.DataLoader(questions, batch_size=1)
    
    # = Secondry dataset
    Dataset_scdry = single_hop.RAGDataset(cad_model.tokenizer, args.second_prompt_format, args.dataset)
    dataset_scdry = Dataset_scdry.get_dataset()
    if args.fraction_of_data_to_use < 1.0:
        train_dataset_scdry = dataset_scdry.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=args.seed)['train']
    else:
        train_dataset_scdry = dataset_scdry
    

    # = Print one sample
    sample_index = 0
    print(f"Dataset example {sample_index}:")
    print(f"Id:               {train_dataset_main[sample_index]['question_id']}")
    print(f"Similarity Score: {train_dataset_main[sample_index]['similarity_score']}")
    print(f"Question:         {train_dataset_main[sample_index]['question']}")
    print(f"Answers:          {train_dataset_main[sample_index]['answers']}")
    print(f"Prompt:         \n{train_dataset_main[sample_index]['prompt']}")
    
    print('-----\n')
    print(f"Id:               {train_dataset_scdry[sample_index]['question_id']}")
    print(f"Similarity Score: {train_dataset_scdry[sample_index]['similarity_score']}")
    print(f"Question:         {train_dataset_scdry[sample_index]['question']}")
    print(f"Answers:          {train_dataset_scdry[sample_index]['answers']}")
    print(f"Prompt:         \n{train_dataset_scdry[sample_index]['prompt']}")
    
    
    ### === Generation loop ====================== 
    with torch.no_grad():
        sequences = []
        for idx, batch in tqdm(enumerate(dataloader_main)):    
            # === Generate multiple time ==========
            generations = torch.ones(
                (args.num_generations_per_prompt, args.max_new_tokens), # input_length + max_length_of_generated_sequence
                dtype=torch.long,
                device=args.device
            )
            
            for i in range(args.num_generations_per_prompt):
                generation = cad_model.generate(
                    batch_with_context=batch,
                    batch_wo_context=train_dataset_scdry[idx],
                    max_length=args.max_new_tokens,
                    alpha=0.5,
                    decoding_strategy='top_p',
                    top_p_value=args.top_p,
                    use_repetition_penalty=True,
                    repetition_penalty_value=1.5,
                )
                generation = torch.tensor(generation[0])
                generated_len = generation.shape[0]
                generations[i, :generated_len] = generation
                # print(cad_model.tokenizer.decode(generation))

            sequence_dict = {
                'id': batch['question_id'][0],
                'question': batch['question'][0], 
                'answers': [ans[0] for ans in batch['answers']],
                'similarity_score': batch['similarity_score'][0],
                'prompt': batch['input_ids'].to(args.device).reshape(1, -1)[0],
                'prompt_text': batch['prompt'][0],
                'generations': generations
            }
            
            generated_texts = []
            for generation in generations:
                generated_texts.append(
                    cad_model.tokenizer.decode(generation, skip_special_tokens=True)
                ) # We already skip special tokens
            sequence_dict['generated_texts'] = generated_texts
    

            # === Generate most likely =============
            most_likely_generation = cad_model.generate(
                batch_with_context=batch,
                batch_wo_context=train_dataset_scdry[idx],
                max_length=args.max_new_tokens,
                alpha=0.5,
                decoding_strategy='greedy',
                use_repetition_penalty=True,
                repetition_penalty_value=1.5,
            )
            most_likely_generation = torch.tensor(most_likely_generation[0])
            
            generated_len = most_likely_generation.shape[0]
            sequence_dict['most_likely_generation_ids'] = most_likely_generation.to('cpu')
            sequence_dict['most_likely_generation'] = cad_model.tokenizer.decode(
                most_likely_generation, skip_special_tokens=True
            )
            sequences.append(sequence_dict)
            
            
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
            if cad_model.tokenizer.__class__.__name__=='PreTrainedTokenizerFast':
                clean_ids = torch.tensor(cad_model.tokenizer(generated_text)['input_ids'][0:], device=args.device)
            else:
                clean_ids = torch.tensor(cad_model.tokenizer(generated_text)['input_ids'][1:], device=args.device)

            sample['cleaned_most_likely_generation'] = generated_text_cleaned
            sample['cleaned_most_likely_generation_ids'] =  clean_ids

            for i, generated_text in enumerate(generated_texts):

                generated_text_cleaned = re.sub(r'[^\x00-\x7f]',r'', generated_text)
                if generated_text_cleaned != generated_text:
                    discard = True
                    break

                cleaned_generated_texts.append(generated_text_cleaned)
                if cad_model.tokenizer.__class__.__name__=='PreTrainedTokenizerFast':
                    clean_ids = torch.tensor(cad_model.tokenizer(generated_text)['input_ids'][0:], device=args.device)
                else:
                    clean_ids = torch.tensor(cad_model.tokenizer(generated_text)['input_ids'][1:], device=args.device)
                cleaned_generations[i, :min(len(clean_ids), max_len_of_generations)] = clean_ids[:max_len_of_generations]

            if not discard:
                sample['cleaned_generated_texts'] = cleaned_generated_texts
                sample['cleaned_generations'] = cleaned_generations
                cleaned_sequences.append(sample)
    
    ### === Save the sequences result ============
    with open(cleaned_sequences_output_file, 'wb') as ofile:
        pickle.dump(cleaned_sequences, ofile)
    print(f"Results saved to {cleaned_sequences_output_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--dataset', type=str, default='webquestions', choices=[
        'trivia', 'nq', 'squad1', 'webquestions',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa_org', 'topicoqa_his', 'topicoqa_rw',
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--main_prompt_format', type=str, default='q_positive', choices=[
        'only_q', 'q_positive', 'q_negative'
    ])
    parser.add_argument('--second_prompt_format', type=str, default='only_q', choices=[
        'only_q', 'q_positive', 'q_negative'
    ])
    
    parser.add_argument('--accuracy_metric', type=str, default="bem_score", choices=[
        'bem_score', 'exact_match', 'bert_score', 'rouge_score', 'llama3_score', 'gpt_score'
    ])
    parser.add_argument('--model_llama_eval', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.05)
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
    
    set_seed(args.seed)
    generation_cad(args)
    
    # python framework/run/answers_generation_cad.py
    
