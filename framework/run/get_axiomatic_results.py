#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import random
import pickle
import sklearn
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import set_seed, uncertainty_to_confidence_min_max
from metrics.calibration import ECE_estimate

def get_axiomatic_results(args):
    print("\n--- Step 6: Get Axiomatic Results ...")
    print(f"""
        Model name:   {args.model}
        Dataset:      {args.dataset}
        Prompt (1st): {args.main_prompt_format}
        Prompt (2ed): {args.second_prompt_format}
        Run id:       {args.run_id}
        Seed:         {args.seed}
    """.replace('        ', ''))
    
    # === Define output files ===================
    model = args.model.split('/')[-1]
    base_dir_output = f'{args.output_dir}/{args.dataset}/{args.run_id}/'
    similarity_output_jsonl_file = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_similarity_output__sec_{args.second_prompt_format}.jsonl'
    relation_output_jsonl_file = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_relation_output__sec_{args.second_prompt_format}.json'
    
    sequence_input_main = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
    sequence_input_secondry = f'{base_dir_output}/{args.second_prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
    with open(sequence_input_main, 'rb') as infile:
        sequences_main = pickle.load(infile)
    with open(sequence_input_secondry, 'rb') as infile:
        sequences_secondry = pickle.load(infile)
        
    # === Load semantic model ===================
    # 0: Contradiction
    # 1: Neutral
    # 2: Entailment
    if not os.path.isfile(relation_output_jsonl_file):
        # semantic_model_name = "microsoft/deberta-v2-xxlarge-mnli"
        # semantic_model_name = "microsoft/deberta-large-mnli"
        # semantic_model_name = "tals/albert-xlarge-vitaminc-mnli"
        semantic_model_name = "tals/albert-xlarge-vitaminc-mnli"
        semantic_model = AutoModelForSequenceClassification.from_pretrained(semantic_model_name).to(args.device)
        semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
        semantic_model.eval()
    
    
    # === Functions =============================
    ece_estimate = ECE_estimate()
    
    def create_result_df(prompt_format):
        
        similarities_input_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_similarities_generation.pkl'
        likelihoods_input_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_uncertainty_generation.pkl'
        correctness_input_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_correctness.pkl'
        generation_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
        # groundedness_input_file = f'{base_dir_output}/{prompt_format}/{model}_{args.temperature}_groundedness_generation__sec_{args.second_prompt_format}.pkl'
        
        with open(generation_file, 'rb') as infile:
            cleaned_sequences = pickle.load(infile)
        with open(similarities_input_file, 'rb') as f:
            similarities_dict = pickle.load(f)
        with open(likelihoods_input_file, 'rb') as f:
            likelihoods_results  = pickle.load(f)
        with open(correctness_input_file, 'rb') as f:
            correctness_results  = pickle.load(f)
        # with open(groundedness_input_file, 'rb') as f:
        #     groundedness_results  = pickle.load(f)
        
        # === Read data ============================
        # 
        similarities_df = pd.DataFrame.from_dict(similarities_dict, orient='index')
        similarities_df['id'] = similarities_df.index
        similarities_df['has_semantically_different_answers'] = similarities_df['has_semantically_different_answers'].astype('int')

        # 
        generations_df = pd.DataFrame(cleaned_sequences)
        generations_df['length_of_most_likely_generation'] = generations_df['most_likely_generation'].apply(
            lambda x: len(str(x).split(' ')))
        generations_df['variance_of_length_of_generations'] = generations_df['generated_texts'].apply(
            lambda x: np.var([len(str(y).split(' ')) for y in x]))
        
        # 
        correctness_df = pd.DataFrame(correctness_results)
        correctness_keys_to_use = ('id', 'bem_score', 'bert_score', 'exact_match')
        correctness_small = dict((k, correctness_df[k]) for k in correctness_keys_to_use)
        correctness_df = pd.DataFrame.from_dict(correctness_small)
        
        # 
        keys_to_use = (
            'ids',
            'average_predictive_entropy_main_prompt', 'predictive_entropy_over_concepts_main_prompt',
            'average_predictive_entropy_importance_max_main_prompt', 'predictive_entropy_over_concepts_importance_max_main_prompt',
            
            'average_predictive_entropy_second_prompt', 'predictive_entropy_over_concepts_second_prompt',
            'average_predictive_entropy_importance_max_second_prompt', 'predictive_entropy_over_concepts_importance_max_second_prompt',
            
            # 'average_predictive_entropy_third_prompt', 'predictive_entropy_over_concepts_third_prompt',
            # 'average_predictive_entropy_importance_max_third_prompt', 'predictive_entropy_over_concepts_importance_max_third_prompt',
            
        )
        likelihoods = likelihoods_results
        likelihoods_small = dict((k, likelihoods[k]) for k in keys_to_use)
        for key in likelihoods_small:
            if key == 'average_predictive_entropy_on_subsets':
                likelihoods_small[key].shape
            if type(likelihoods_small[key]) is torch.Tensor:
                likelihoods_small[key] = torch.squeeze(likelihoods_small[key].cpu())
        likelihoods_df = pd.DataFrame.from_dict(likelihoods_small)
        likelihoods_df.rename(columns={'ids': 'id'}, inplace=True) 
        
        # 
        # groundedness_df = pd.DataFrame(groundedness_results)

        # 
        result_df = generations_df.merge(similarities_df, on='id').merge(likelihoods_df, on='id').merge(correctness_df, on='id') # .merge(groundedness_df, on='id')
        result_df['len_most_likely_generation_length'] = result_df['most_likely_generation'].apply(lambda x: len(x.split()))
        return result_df
    
    def get_correctness(results):
        correctness_results = {}
        
        if args.accuracy_metric in ['bem_score', 'rouge_score', 'gpt_score', 'exact_match']:
            correctness_bin = (results[args.accuracy_metric] > args.roc_auc_threshold).astype('int') 
        elif args.accuracy_metric == 'bert_score':
            correctness_bin = (results[args.accuracy_metric].apply(lambda x: x['F1']) > args.roc_auc_threshold).astype('int') 
        correctness_results['accuracy'] = correctness_bin.mean()
        
        # non-binarized accuracy
        correctness_results['exact_match_mean'] = results['exact_match'].mean()
        correctness_results['bem_score_mean'] = results['bem_score'].mean()
        correctness_results['bert_score_mean'] = results['bert_score'].apply(lambda x: x['F1']).mean()
        if args.accuracy_metric in ['bem_score', 'rouge_score', 'gpt_score']:
            one_minus_correctness = 1 - results[args.accuracy_metric]
        elif args.accuracy_metric == 'bert_score':
            one_minus_correctness = 1 - results[args.accuracy_metric].apply(lambda x: x['F1'])
        elif args.accuracy_metric == 'exact_match':
            one_minus_correctness = 1 - results[args.accuracy_metric].astype('int') 
        
        return correctness_results, correctness_bin, one_minus_correctness
    
    def get_aggreement(sequence_1, sequence_2, threshold=0.5):
        
        sequence_2_ = {}
        for sample in sequence_2:
            sequence_2_[sample['id']] = sample
        
        agree_list = []
        non_agree_list = []
        with open(similarity_output_jsonl_file, 'w') as jl_ofile:
            
            for i, sample in tqdm(enumerate(sequence_1)):
                id_ = sample['id']
                
                if id_ in sequence_2_:
                    generation_most_likely_seq1 = sample['cleaned_most_likely_generation']
                    generation_most_likely_seq2 = sequence_2_[id_]['cleaned_most_likely_generation']
                    
                    # print(generation_most_likely_seq1)
                    # print(generation_most_likely_seq2)
                    
                    similarity = similarity_model.predict([generation_most_likely_seq1, generation_most_likely_seq2])
                    
                    if similarity > threshold:
                        agree_list.append(id_)
                    else:
                        non_agree_list.append(id_)
                    
                    # print(similarity)
                    result_item = {
                        'id': id_,
                        'question': sample['question'],
                        'generation_seq_1': generation_most_likely_seq1,
                        'generation_seq_2': generation_most_likely_seq2,
                        'sim_score': float(similarity)
                    }
                    jl_ofile.write(json.dumps(result_item) + '\n')
                    
                
                else:
                    print(f"\nQuery {id_} is not common between two sequences !!!")
            
        return agree_list, non_agree_list
        
    def in_doc_existence(sequences, ids):
        samples = [item for item in sequences if item['id'] in ids]
        
        doc_exist, doc_not_exist = [], []
        for idx, sample in enumerate(samples):
            answer = sample['cleaned_most_likely_generation']
            prompt_text = sample['prompt_text']
            doc_text = prompt_text.split('Document:')[-1].split('Question:')[0]
            
            # def is_answer_in_doc(answer, doc, threshold=0.8):
            #     return SequenceMatcher(None, answer, doc).ratio() > threshold
            
            # if answer in doc_text:
            #     print('1')
            if answer.lower() in doc_text.lower():
                doc_exist.append(sample['id'])
            else:
                doc_not_exist.append(sample['id'])
            # if re.search(r'\b' + re.escape(answer) + r'\b', doc_text, re.IGNORECASE):
            #     print('3')
            # if is_answer_in_doc(answer, doc_text):
            #     print('4')
            # else:
            #     print('5')
        return doc_exist, doc_not_exist
    
    
    # ======
    keys_mapping = {
        'main_prompt': {
            'PE': 'average_predictive_entropy_main_prompt',
            'SE': 'predictive_entropy_over_concepts_main_prompt',
            'PE_MARS': 'average_predictive_entropy_importance_max_main_prompt',
            'SE_MARS': 'predictive_entropy_over_concepts_importance_max_main_prompt'
        },
        'second_prompt': {
            'PE': 'average_predictive_entropy_second_prompt',
            'SE': 'predictive_entropy_over_concepts_second_prompt',
            'PE_MARS': 'average_predictive_entropy_importance_max_second_prompt',
            'SE_MARS': 'predictive_entropy_over_concepts_importance_max_second_prompt'
        },
        'third_prompt': {
            'PE': 'average_predictive_entropy_third_prompt',
            'SE': 'predictive_entropy_over_concepts_third_prompt',
            'PE_MARS': 'average_predictive_entropy_importance_max_third_prompt',
            'SE_MARS': 'predictive_entropy_over_concepts_importance_max_third_prompt'
        } 
    }
    
    result_df_main = create_result_df(args.main_prompt_format)
    result_df_main_filtered_pe = result_df_main[result_df_main['average_predictive_entropy_main_prompt'] <= 100]
    result_df_main_filtered_se = result_df_main[result_df_main['predictive_entropy_over_concepts_main_prompt'] <= 100]
    
    result_df_second_prompt = create_result_df(args.second_prompt_format)
    result_df_second_prompt_filtered_pe = result_df_second_prompt[result_df_second_prompt['average_predictive_entropy_main_prompt'] <= 100]
    result_df_second_prompt_filtered_se = result_df_second_prompt[result_df_second_prompt['predictive_entropy_over_concepts_main_prompt'] <= 100]
    
    
    # First check: answer1 is equal to answer2
    if os.path.isfile(similarity_output_jsonl_file):
        print(f"{similarity_output_jsonl_file} exists.")
        threshold = 0.5
        agree_list, non_agree_list = [], []
        with open(similarity_output_jsonl_file, 'r') as file:
            ids = []
            for line in file:
                sample = json.loads(line.strip())
                ids.append(sample['id'])
                if sample['sim_score'] > threshold:
                    agree_list.append(sample['id'])
                else:
                    non_agree_list.append(sample['id'])
    else:
        print("Computing similarity ...")
        similarity_model_name = "cross-encoder/stsb-roberta-large"
        similarity_model = CrossEncoder(model_name=similarity_model_name, num_labels=1)
        similarity_model.model.to(args.device)
        agree_list, non_agree_list = get_aggreement(sequences_main, sequences_secondry)
    # selected_samples = agree_list    # Axiom 1, 2, 3
    # selected_list = non_agree_list
    
    
    if os.path.isfile(relation_output_jsonl_file):
        print(f"{relation_output_jsonl_file} exists.")
        with open(relation_output_jsonl_file, 'r') as file:
            axiom1 = json.load(file)
        
    else:
        axiom1 = {
            'entailment': [],
            'neutral': [],
            'contradiction': []
        }
        with torch.no_grad():
            for idx, sample in tqdm(enumerate(sequences_main)):
                id_ = sample['id']
                question = sample['question']
                generated_texts = sample['cleaned_generated_texts']
                generated_text_most_likely = sample['most_likely_generation']
                prompt_text = sample['prompt_text']
                doc_text = prompt_text.split('Document:')[-1].split('Question:')[0]
                answer_ = f"{question} {generated_text_most_likely}"
                
                if id_ in agree_list: # Axioms 1, 2, 3
                    encoded_input = semantic_tokenizer.encode_plus(
                        doc_text,
                        answer_,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                        truncation_strategy="only_first"
                    )
                    encoded_input = {key: val.to(args.device) for key, val in encoded_input.items()}
                    prediction = semantic_model(**encoded_input).logits
                    predicted_label = prediction.argmax(dim=1).item()
                    
                    if predicted_label==0: # entailment
                        axiom1['entailment'].append(id_)
                    elif predicted_label==1: # neutral
                        axiom1['neutral'].append(id_)
                    elif predicted_label==2: # contradiction
                        axiom1['contradiction'].append(id_)
                    else:
                        axiom1['neutral'].append(id_)
        
        with open(relation_output_jsonl_file, 'w') as file:
            json.dump(axiom1, file, indent=4)
        
    print(f"Entailment: {axiom1['entailment']}")
    print(f"Neutral: {axiom1['neutral']}")
    print(f"Contradiction: {axiom1['contradiction']}")
    
    for uncertainty_model in ['PE', 'SE']: # , 'PE_MARS', 'SE_MARS' 
        
        if uncertainty_model in ['PE', 'PE_MARS']:
            result_df_main_prompt = result_df_main_filtered_pe
            result_df_second_prompt = result_df_second_prompt_filtered_pe
        elif uncertainty_model in ['SE', 'SE_MARS']:
            result_df_main_prompt = result_df_main_filtered_se
            result_df_second_prompt = result_df_second_prompt_filtered_se
        
        unc_model_key_main_prompt = keys_mapping['main_prompt'][uncertainty_model]
        unc_model_key_second_prompt = keys_mapping['second_prompt'][uncertainty_model]
    
        for relation_key in ['entailment', 'contradiction']: # , 'neutral'
            selected_list = axiom1[relation_key]
            
            if len(selected_list) > 0:
                agree_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list)]
                agree_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_list)]
        
                _, correctness_main_prompt_bin, one_minus_correctness_main_prompt = get_correctness(agree_main_prompt_df)
                _, correctness_second_prompt_bin, one_minus_correctness_second_prompt = get_correctness(agree_second_prompt_df)
                correctness_main_prompt = 1 - np.array(one_minus_correctness_main_prompt)
                correctness_second_prompt = 1 - np.array(one_minus_correctness_second_prompt)
            
                uncertainty_main_prompt_values =  agree_main_prompt_df[unc_model_key_main_prompt] # 0.5*
                uncertainty_second_prompt_values = agree_second_prompt_df[unc_model_key_main_prompt] # Axioms: 1, 2, 3
                confidence_main_prompt_values = uncertainty_to_confidence_min_max(uncertainty_main_prompt_values)
                confidence_second_prompt_values = uncertainty_to_confidence_min_max(uncertainty_second_prompt_values)
                
                # auroc_main_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_main_prompt_values)
                # auroc_second_prompt = sklearn.metrics.roc_auc_score(1 - correctness_second_prompt_bin, uncertainty_second_prompt_values)
                # ece_main_prompt = ece_estimate(correctness_main_prompt, confidence_main_prompt_values)
                # ece_second_prompt = ece_estimate(correctness_second_prompt, confidence_second_prompt_values)
                
                print(f"{uncertainty_model}, Axiom1: {relation_key}")
                print(f"Acc. ({args.accuracy_metric}):  {round(correctness_second_prompt.mean(), 3)} -> {round(correctness_main_prompt.mean(), 3)}")
                print(f"Uncertainty: {uncertainty_second_prompt_values.mean():.3f} -> {uncertainty_main_prompt_values.mean():.3f}")
                print(f"Confidence:  {confidence_second_prompt_values.mean():.3f} -> {confidence_main_prompt_values.mean():.3f}")
                # print(f"AUROC:       {round(auroc_second_prompt, 3)} -> {round(auroc_main_prompt, 3)}")
                # print(f"ECE:         {round(ece_second_prompt, 3)} -> {round(ece_main_prompt, 3)}")  
                print('\n')             
                
            else: 
                print(f"{relation_key} does not contain data!!!")
                print('\n')

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
    parser.add_argument('--main_prompt_format', type=str, default='bm25_retriever_top5', choices=[
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
    get_axiomatic_results(args)
    
    # python framework/run/get_axiomatic_results.py
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # print(id_)
    
    # input =  answer_ + ' [SEP] ' + doc_text
    # print(input)
    # encoded_input = semantic_tokenizer.encode(input, padding=True)
    # prediction = semantic_model(torch.tensor(torch.tensor([encoded_input]), device=args.device))['logits']
    # predicted_label = torch.argmax(prediction, dim=1)
    # print(predicted_label)

    # reverse_input = doc_text + ' [SEP] ' + answer_
    # encoded_reverse_input = semantic_tokenizer.encode(reverse_input, padding=True)
    # reverse_prediction = semantic_model(torch.tensor(torch.tensor([encoded_reverse_input]), device=args.device))['logits']
    # reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
    # print(reverse_predicted_label)
    
    # print(doc_text)
    # print(generated_text_most_likely)
    # encoded_input = semantic_tokenizer(
    #     doc_text,
    #     generated_text_most_likely,
    #     return_tensors="pt",
    #     padding="max_length",
    #     truncation=True,
    #     max_length=1024  # Adjust based on your needs
    # )

    
    ### === For random testing =================
    # print(ids)
    # num_samples = 40
    # random_selection = random.sample(ids, num_samples)
    # selected_list = random_selection
    
    ### === Second check: answer2 exist in doc =
    # agree_list_doc_exist, agree_list_doc_not_exist = in_doc_existence(sequences_main, agree_list)
    # selected_list = agree_list_doc_exist # Axiom 1 (positive passage)
    # selected_list = agree_list_doc_not_exist # Axiom 2 (positive passage)
    
    # Axiom 4
    # non_agree_list_doc_exist, non_agree_list_doc_not_exist = in_doc_existence(sequences_main, non_agree_list)
    # selected_list = non_agree_list_doc_exist
    
    # print(f"# samples: {len(selected_list)} ({round((len(selected_list)/len(sequences_main))*100 , 2)}%)")
    
    # for uncertainty_model in ['PE', 'SE']: # , 'PE_MARS', 'SE_MARS' 
    #     # unc_model_key = keys_mapping[uncertainty_model]
    #     unc_model_key_main_prompt = keys_mapping['main_prompt'][uncertainty_model]
    #     unc_model_key_second_prompt = keys_mapping['second_prompt'][uncertainty_model]
        
    #     if uncertainty_model in ['PE', 'PE_MARS']:
    #         result_df_main_prompt = result_df_main_filtered_pe
    #         result_df_second_prompt = result_df_second_prompt_filtered_pe
    #     elif uncertainty_model in ['SE', 'SE_MARS']:
    #         result_df_main_prompt = result_df_main_filtered_se
    #         result_df_second_prompt = result_df_second_prompt_filtered_se
    
    #     ### For whole samples (main prompt)
    #     _, correctness_main_prompt_bin_, one_minus_correctness_main_prompt_ = get_correctness(result_df_main_prompt)
    #     correctness_main_prompt_ = 1 - np.array(one_minus_correctness_main_prompt_)
    #     uncertainty_main_prompt_values_ = result_df_main_prompt[unc_model_key_main_prompt]
        
    #     # test2
    #     # uncertainty_main_prompt_values_ = np.where(
    #     #     result_df_main_prompt['id'].isin(selected_list),
    #     #     result_df_main_prompt[unc_model_key_main_prompt] * 1.2,
    #     #     result_df_main_prompt[unc_model_key_main_prompt]
    #     # )
    #     # test3: combine
    #     # uncertainty_main_prompt_values_ = np.where(
    #     #     result_df_main_prompt['id'].isin(selected_list),
    #     #     result_df_main_prompt[unc_model_key_main_prompt] * 0.5,
    #     #     abs(result_df_main_prompt[unc_model_key_main_prompt] - result_df_main_prompt[unc_model_key_second_prompt])
    #     # )
    #     confidence_main_prompt_values_ = uncertainty_to_confidence_min_max(uncertainty_main_prompt_values_)
        
    #     ### For Axiom1 samples (second prompt)
    #     agree_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list)]
    #     agree_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_list)]
        
    #     _, correctness_main_prompt_bin, one_minus_correctness_main_prompt = get_correctness(agree_main_prompt_df)
    #     _, correctness_second_prompt_bin, one_minus_correctness_second_prompt = get_correctness(agree_second_prompt_df)
    #     correctness_main_prompt = 1 - np.array(one_minus_correctness_main_prompt)
    #     correctness_second_prompt = 1 - np.array(one_minus_correctness_second_prompt)
        
    #     uncertainty_main_prompt_values =  agree_main_prompt_df[unc_model_key_main_prompt] # 0.5*
    #     uncertainty_second_prompt_values = agree_second_prompt_df[unc_model_key_main_prompt] # Axioms: 1, 2, 3
        
    #     confidence_main_prompt_values = uncertainty_to_confidence_min_max(uncertainty_main_prompt_values)
    #     confidence_second_prompt_values = uncertainty_to_confidence_min_max(uncertainty_second_prompt_values)
        
    #     auroc_main_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_main_prompt_values)
    #     auroc_second_prompt = sklearn.metrics.roc_auc_score(1 - correctness_second_prompt_bin, uncertainty_second_prompt_values)
    #     ece_main_prompt = ece_estimate(correctness_main_prompt, confidence_main_prompt_values)
    #     ece_second_prompt = ece_estimate(correctness_second_prompt, confidence_second_prompt_values)
    #     print(f"{uncertainty_model}, Axioms:")
    #     print(f"Acc. (bem):  {correctness_second_prompt.mean()} -> {correctness_main_prompt.mean()}")
    #     print(f"Uncertainty: {uncertainty_second_prompt_values.mean()} -> {uncertainty_main_prompt_values.mean()}")
    #     print(f"Confidence:  {confidence_second_prompt_values.mean()} -> {confidence_main_prompt_values.mean()}")
    #     print(f"AUROC:       {auroc_second_prompt} -> {auroc_main_prompt}")
    #     print(f"ECE:         {ece_second_prompt} -> {ece_main_prompt}")
        
        # Axiom 4-1
        # uncertainty_second_prompt_values = agree_main_prompt_df[unc_model_key_second_prompt] 
        # confidence_second_prompt_values = uncertainty_to_confidence_min_max(uncertainty_second_prompt_values)
        # auroc_second_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_second_prompt_values)
        # ece_second_prompt = ece_estimate(correctness_main_prompt, confidence_second_prompt_values)
        # print(f"{uncertainty_model}, Axioms:")
        # print(f"Acc. (bem):  {correctness_main_prompt.mean()}")
        # print(f"Uncertainty: {uncertainty_second_prompt_values.mean()} -> {uncertainty_main_prompt_values.mean()}")
        # print(f"Confidence:  {confidence_second_prompt_values.mean()} -> {confidence_main_prompt_values.mean()}")
        # print(f"AUROC:       {auroc_second_prompt} -> {auroc_main_prompt}")
        # print(f"ECE:         {ece_second_prompt} -> {ece_main_prompt}")
        
        ### === For test one: decrease the uncertainty of samples with same answer with/wo docs
        # auroc_test1 = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin_, uncertainty_main_prompt_values_)
        # ece_test1 = ece_estimate(correctness_main_prompt_, confidence_main_prompt_values_)
        # plot_correctness_vs_uncertainty_for_axioms(
        #     correctness_main_prompt_, uncertainty_main_prompt_values_,
        #     correctness_main_prompt, uncertainty_main_prompt_values,
        #     f'AUROC: {round(auroc_test1, 4)}\nECE: {round(ece_test1, 4)}',
        #     f'{uncertainty_model}_axiom1', num_bins=40
        # )
        # print(f"AUROC:       {auroc_test1}")
        # print(f"ECE:         {ece_test1}")

