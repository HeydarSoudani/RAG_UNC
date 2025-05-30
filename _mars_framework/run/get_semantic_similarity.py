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
import evaluate
from tqdm import tqdm
from transformers import BertModel
from transformers import BertTokenizerFast 
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from utils.utils import set_seed


def get_similarity(args):
    
    print("\n--- Step 2: Getting Semantic Similarity ...")
    print(f"""
        Model name:    {args.model}
        Dataset:       {args.dataset} ({args.fraction_of_data_to_use})
        Prompt (1st):  {args.main_prompt_format}
        Prompt (2ed):  {args.second_prompt_format}
        Run id:        {args.run_id}
        Seed:          {args.seed}
    """.replace('   ', ''))
    
    # === Define output files =============
    # === Read the generated data =========
    model_ = args.model.split('/')[-1]
    base_dir = f'{args.output_dir}/{model_}/{args.dataset}/{args.subsec}/{args.run_id}/{args.main_prompt_format}__{args.second_prompt_format}'
    generation_file = f'{base_dir}/cleaned_generation_{args.generation_type}.pkl'
    similarities_output_file = f'{base_dir}/similarities_generation.pkl'
    
    with open(generation_file, 'rb') as infile:
        sequences = pickle.load(infile)

    # === Load model tokenizer ============
    tokenizer = AutoTokenizer.from_pretrained(f"{args.model}", use_fast=False)

    # === Load importance model ===========
    model_importance = torch.load('framework/pretrained_models/model_phrase.pth', map_location=args.device).to(args.device)
    # model_importance = BertModel.from_pretrained('baselines/MARS/models/model_phrase.pth').to(args.device)
    tokenizer_importance = BertTokenizerFast.from_pretrained("bert-base-uncased") 
    
    
    # === Load semantic model =============
    # config = AutoConfig.from_pretrained("microsoft/deberta-large-mnli")
    # print(config.label2id)
    # {'CONTRADICTION': 0, 'NEUTRAL': 1, 'ENTAILMENT': 2}
    semantic_model_name = "microsoft/deberta-large-mnli"
    semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
    semantic_model = AutoModelForSequenceClassification.from_pretrained(semantic_model_name).to(args.device)

    # === Functions =======================
    def inference(model, tokenizer, question, answer):

        words = re.findall(r'\w+|[^\w\s]', answer)
        tokenized_input = tokenizer.encode_plus(
            [question],
            words,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_token_type_ids=True,
            is_split_into_words=True,
            truncation=True,
            max_length=512,           # Pad & truncate all sentences.
        )
        attention_mask = torch.tensor(tokenized_input['attention_mask']).reshape(1,-1).to(args.device)
        input_ids = torch.tensor(tokenized_input['input_ids']).reshape(1,-1).to(args.device)
        token_type_ids = torch.tensor(tokenized_input['token_type_ids']).reshape(1,-1).to(args.device)
        word_ids = tokenized_input.word_ids() 

        logits = model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids).logits[0].cpu()
        classes = logits[:,0:2]
        scores = torch.nn.functional.sigmoid(logits[:,2])

        phrases = []
        importance_scores = []
        i = 0
        while(i<len(scores)):
            if word_ids[i] == None or token_type_ids[0][i] == 0:
                i += 1 
                continue
            
            cl = torch.argmax(classes[i,:])
            if word_ids[i] == 0 or cl == 0: # we handle the edge case as well (beginning of the sentence)
                for j in range(i+1, len(scores)):
                    cl = torch.argmax(classes[j,:])
                    continue_word = False
                    for k in range(i,j):
                        if word_ids[k] == word_ids[j]:
                            continue_word = True
                    if (cl == 0 or  word_ids[j] == None) and continue_word == False:
                        break
                
                #find corresponding words by using word_ids
                min_word_id = word_ids[i]
                max_word_id = word_ids[j-1]
                phrases.append(''.join(words[min_word_id:max_word_id+1]))
                importance_scores.append(scores[i].item())
                i = j 

        #maybe modify phrase with actual sentence
        real_phrases = []
        phrase_ind  = 0
        i = 0
        answer = answer.strip()
        while(i < len(answer)):
            last_token_place  = -1
            for j in range(i+1, len(answer)+1):

                if phrase_ind < len(phrases) and phrases[phrase_ind].strip().replace(" ", "") == answer[i:j].strip().replace(" ", ""):
                # if phrases[phrase_ind].strip().replace(" ", "") == answer[i:j].strip().replace(" ", ""):
                    last_token_place = j

            # I added this
            if last_token_place == -1:
                break  # Avoid infinite loop if no match is found

            real_phrases.append(answer[i:last_token_place].strip())
            i = last_token_place
            phrase_ind += 1
            
        return real_phrases, importance_scores

    def get_importance_vector(cleaned_sequence):
        importance_vector = []
        # answer_ids = cleaned_sequence['cleaned_most_likely_generation_ids'][len(cleaned_sequence['prompt']):]
        answer_ids = cleaned_sequence['cleaned_most_likely_generation_ids']
        #answer_ids = answer_ids[0:100]#normally it shouldn't be longer than 256
        answer = tokenizer.decode(answer_ids)
        question = cleaned_sequence['question']
        # print(answer)
        phrases, importance_vector = inference(model_importance, tokenizer_importance, question, answer)
        
        return torch.tensor(importance_vector), phrases

    # === Main loop, on sequence ==========
    result_dict = {}
    deberta_predictions = []
    multiple_answer = 0
    clusterable_semantic = 0 
    
    for idx, sample in tqdm(enumerate(sequences)):
        id_ = sample['id']
        question = sample['question']
        generated_texts = sample['cleaned_generated_texts']
        
        unique_generated_texts = list(set(generated_texts))
        answer_list_1 = []
        answer_list_2 = []
        has_semantically_different_answers = False
        inputs = []
        syntactic_similarities = {}
        
        # # Src: https://github.com/jinhaoduan/SAR/blob/main/src/get_semantic_clusters.py
        # rouge_types = ['rouge1', 'rouge2', 'rougeL']
        # for rouge_type in rouge_types:
        #     syntactic_similarities[rouge_type] = 0.0

        semantic_set_ids = {}
        for index, answer in enumerate(unique_generated_texts):
            semantic_set_ids[answer] = index
        
        importance_vector = get_importance_vector(sample)
        importance_scores = []
        generations = sample['cleaned_generations'].to(args.device)
        prompt = sample['prompt']
        
        # print(idx)
        # print(sample['cleaned_generated_texts'])
        # print('\n')
        
        for generation_index in range(generations.shape[0]):
            sequence = {}
            prompt = prompt[prompt != 1]
            generation = generations[generation_index][generations[generation_index] != 1]
            sequence['cleaned_most_likely_generation_ids'] = generation
            sequence['prompt'] = prompt
            sequence['question'] = sample['question']
            importance_scores.append(get_importance_vector(sequence))

        has_different_answers = False
        encoded_meanings = []
        encoded_meanings_only_answer = []
        unique_answers_indices = []

        if len(unique_generated_texts) > 1:
            has_different_answers = True
            multiple_answer += 1 
            # Evalauate semantic similarity
            clusterable = False

            for i, reference_answer in enumerate(unique_generated_texts):
                q_a = question + ' ' + unique_generated_texts[i]
                a = unique_generated_texts[i]
                unique_answers_indices.append(generated_texts.index(a))
                
            for i, reference_answer in enumerate(unique_generated_texts):
                for j in range(i + 1, len(unique_generated_texts)):
                    answer_list_1.append(unique_generated_texts[i])
                    answer_list_2.append(unique_generated_texts[j])
                    qa_1 = question + ' ' + unique_generated_texts[i]
                    qa_2 = question + ' ' + unique_generated_texts[j]

                    input = qa_1 + ' [SEP] ' + qa_2
                    inputs.append(input)
                    encoded_input = semantic_tokenizer.encode(input, padding=True)
                    prediction = semantic_model(torch.tensor(torch.tensor([encoded_input]), device=args.device))['logits']
                    predicted_label = torch.argmax(prediction, dim=1)

                    reverse_input = qa_2 + ' [SEP] ' + qa_1
                    encoded_reverse_input = semantic_tokenizer.encode(reverse_input, padding=True)
                    reverse_prediction = semantic_model(torch.tensor(torch.tensor([encoded_reverse_input]), device=args.device))['logits']
                    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                    # Main code:
                    deberta_prediction = 1
                    if 0 in predicted_label or 0 in reverse_predicted_label:
                        has_semantically_different_answers = True
                        deberta_prediction = 0
                    else:
                        semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]
                        clusterable = True

                    # My code:
                    # deberta_prediction = 0
                    # if 2 in predicted_label or 2 in reverse_predicted_label:
                    #     semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]
                    #     deberta_prediction = 1
                    #     clusterable = True
                    # else:
                    #     has_semantically_different_answers = True
                        
                    deberta_predictions.append([unique_generated_texts[i], unique_generated_texts[j], deberta_prediction])

            if clusterable == True: 
                clusterable_semantic += 1
            
            # Evalauate syntactic similarity
            answer_list_1 = []
            answer_list_2 = []
            for i in generated_texts:
                for j in generated_texts:
                    if i != j:
                        answer_list_1.append(i)
                        answer_list_2.append(j)
        
        # Src: https://github.com/jinhaoduan/SAR/blob/main/src/get_semantic_clusters.py
        # rouge = evaluate.load('rouge')
        # results = rouge.compute(predictions=answer_list_1, references=answer_list_2)
        # for rouge_type in rouge_types:
        #     syntactic_similarities[rouge_type] = results[rouge_type]


        result_dict[id_] = {
            'syntactic_similarities': syntactic_similarities,
            'has_semantically_different_answers': has_semantically_different_answers,
            'has_different_answers': has_different_answers
        }
        result_dict[id_]['importance_vector'] = importance_vector
        result_dict[id_]['importance_scores'] = importance_scores
        list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
        result_dict[id_]['semantic_set_ids'] = list_of_semantic_set_ids
        result_dict[id_]['unique_answers_indices'] = unique_answers_indices


    ### === Save the sequences result ======
    with open(similarities_output_file, 'wb') as ofile:
        pickle.dump(result_dict, ofile)
    print(f"Results saved to {similarities_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='trivia', choices=[
        'nqgold', 'trivia', 'popqa',
        'webquestions', 'squad1', 'nq', 'nqswap',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa',
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--main_prompt_format', type=str, default='q_positive', choices=[
        'only_q', 'q_positive', 'q_negative', 'q_conflict',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'contriever_retriever_top1', 'contriever_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    parser.add_argument('--second_prompt_format', type=str, default='only_q', choices=[
        'only_q', 'q_positive', 'q_negative', 'q_conflict',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'contriever_retriever_top1', 'contriever_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
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
    get_similarity(args)
    
    
    # python framework/run/get_semantic_similarity.py
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #     # == *** ==== for run_3: Eva's idea, combine two generation
    # # 'framework/run_output/webquestions/run_3/q_positive/Llama-2-7b-chat-hf_1.0_cleaned_generation_only_q.pkl'
    # with open(f'{args.output_dir}/{args.dataset}/{args.run_id}/only_q/{model}_{args.temperature}_cleaned_generation.pkl', 'rb') as infile:
    #     sequences_only_q = pickle.load(infile)
    # with open(f'{args.output_dir}/{args.dataset}/{args.run_id}/{args.prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl', 'rb') as infile:
    #     sequences_current = pickle.load(infile)

    # if args.prompt_format == 'only_q':
    #     sequences = sequences_only_q
    # else:
    #     if args.mode == 'seperated':
    #         sequences = sequences_current
    #     elif args.mode == 'combined':
    #         sequences_only_q_obj = {}
    #         for item in sequences_only_q:
    #             sequences_only_q_obj[item["id"]] = item
            
    #         sequences = []
    #         for idx, item in enumerate(sequences_current):
    #             qid = item['id']
    #             if qid in sequences_only_q_obj:
    #                 new_item = {
    #                     'id': qid,
    #                     'question': item['question'], 
    #                     'answers': item['answers'],
    #                     'prompt': item['prompt'],
                        
    #                     'generations': torch.cat((item['generations'], sequences_only_q_obj[qid]['generations']), dim=0),
    #                     'generated_texts': item['generated_texts'] + sequences_only_q_obj[qid]['generated_texts'],
    #                     'cleaned_generations': torch.cat((item['cleaned_generations'], sequences_only_q_obj[qid]['cleaned_generations']), dim=0),
    #                     'cleaned_generated_texts': item['cleaned_generated_texts'] + sequences_only_q_obj[qid]['cleaned_generated_texts'],
                        
    #                     'most_likely_generation_ids': item['most_likely_generation_ids'],
    #                     'most_likely_generation': item['most_likely_generation'],
    #                     'cleaned_most_likely_generation_ids': item['cleaned_most_likely_generation_ids'],
    #                     'cleaned_most_likely_generation': item['cleaned_most_likely_generation'],
    #                 }
    #                 sequences.append(new_item)
            
    #         with open(combined_sequences_output_file, 'wb') as ofile:
    #             pickle.dump(sequences, ofile)
    #         print(f"Results saved to {combined_sequences_output_file}")
    #     else:
    #         print('mode is not defined')