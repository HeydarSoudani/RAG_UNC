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
from scipy.stats import pearsonr, spearmanr
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
# from minicheck.minicheck import MiniCheck

from utils.utils import set_seed, uncertainty_to_confidence_min_max
from metrics.calibration import ECE_estimate

UNC_THERESHOLD = 1000


def get_axiomatic_results(args):
    print("\n--- Step 6: Get Axiomatic Results V3 ...")
    print(f"""
        Model name:   {args.model}
        Dataset:      {args.dataset} / {args.subsec}
        Prompt (1st): {args.main_prompt_format}
        Prompt (2ed): {args.second_prompt_format}
        Run id:       {args.run_id}
        Seed:         {args.seed}
    """.replace('        ', ''))
    
    # === Define output files ===================
    # archive_500samples
    model = args.model.split('/')[-1]
    base_dir_output = f'{args.output_dir}/{args.dataset}/{args.run_id}/'
    generation_type = f"prob_alpha_{str(args.alpha_probability)}"
    
    sequence_input_main = f'{base_dir_output}/{args.main_prompt_format}__{args.second_prompt_format}/{model}_cleaned_generation_{args.generation_type}.pkl'
    with open(sequence_input_main, 'rb') as infile:
        sequences_main = pickle.load(infile)
    
    base_dir_second_output = f'{args.output_dir}/{args.dataset}/run_0/'
    if os.path.isdir(f'{base_dir_second_output}/{args.second_prompt_format}__{args.main_prompt_format}'):
        sequence_input_secondry = f'{base_dir_second_output}/{args.second_prompt_format}__{args.main_prompt_format}/{model}_cleaned_generation_normal.pkl'
    else:
        sequence_input_secondry = f'{base_dir_second_output}/{args.second_prompt_format}__q_positive/{model}_cleaned_generation_normal.pkl'
    with open(sequence_input_secondry, 'rb') as infile:
        sequences_secondry = pickle.load(infile)
   
   
    # === Load semantic model ===================
    semantic_model_name = "microsoft/deberta-large-mnli"
    semantic_model = AutoModelForSequenceClassification.from_pretrained(semantic_model_name).to(args.device)
    semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
    semantic_model.eval()
    
    
    # === Functions =============================
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
        },
        'forth_prompt': {
            'PE': 'average_predictive_entropy_forth_prompt',
            'SE': 'predictive_entropy_over_concepts_forth_prompt',
            'PE_MARS': 'average_predictive_entropy_importance_max_forth_prompt',
            'SE_MARS': 'predictive_entropy_over_concepts_importance_max_forth_prompt'
        },
        # 'fifth_prompt': {
        #     'PE': 'average_predictive_entropy_forth_prompt',
        #     'SE': 'predictive_entropy_over_concepts_forth_prompt',
        #     'PE_MARS': 'average_predictive_entropy_importance_max_forth_prompt',
        #     'SE_MARS': 'predictive_entropy_over_concepts_importance_max_forth_prompt'
        # }
         
    }
    
    def create_result_df(main_prompt_format, second_prompt_format):
        
        if main_prompt_format == 'only_q':
            generation_file = f'{base_dir_output}/{main_prompt_format}__q_positive/{model}_cleaned_generation_{args.generation_type}.pkl'
            similarities_input_file = f'{base_dir_output}/{main_prompt_format}__q_positive/{model}_similarities_generation.pkl'
            likelihoods_input_file = f'{base_dir_output}/{main_prompt_format}__q_positive/{generation_type}/{model}_uncertainty_mars_generation.pkl'
            correctness_input_file = f'{base_dir_output}/{main_prompt_format}__q_positive/{model}_correctness.pkl'
        else:
            generation_file = f'{base_dir_output}/{main_prompt_format}__{second_prompt_format}/{model}_cleaned_generation_{args.generation_type}.pkl'
            similarities_input_file = f'{base_dir_output}/{main_prompt_format}__{second_prompt_format}/{model}_similarities_generation.pkl'
            likelihoods_input_file = f'{base_dir_output}/{main_prompt_format}__{second_prompt_format}/{generation_type}/{model}_uncertainty_mars_generation.pkl'
            correctness_input_file = f'{base_dir_output}/{main_prompt_format}__{second_prompt_format}/{model}_correctness.pkl'
        
        with open(generation_file, 'rb') as infile:
            cleaned_sequences = pickle.load(infile)
        with open(similarities_input_file, 'rb') as f:
            similarities_dict = pickle.load(f)
        with open(likelihoods_input_file, 'rb') as f:
            likelihoods_results  = pickle.load(f)
        with open(correctness_input_file, 'rb') as f:
            correctness_results  = pickle.load(f)
        
        
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
        correctness_keys_to_use = ('id', 'bem_score', 'bert_score', 'exact_match', 'rouge_score')
        correctness_small = dict((k, correctness_df[k]) for k in correctness_keys_to_use)
        correctness_df = pd.DataFrame.from_dict(correctness_small)
        
        # 
        keys_to_use = (
            'ids',
            'average_predictive_entropy_main_prompt', 'predictive_entropy_over_concepts_main_prompt',
            'average_predictive_entropy_importance_max_main_prompt', 'predictive_entropy_over_concepts_importance_max_main_prompt',
            'average_predictive_entropy_second_prompt', 'predictive_entropy_over_concepts_second_prompt',
            'average_predictive_entropy_importance_max_second_prompt', 'predictive_entropy_over_concepts_importance_max_second_prompt',
            'average_predictive_entropy_third_prompt', 'predictive_entropy_over_concepts_third_prompt',
            'average_predictive_entropy_importance_max_third_prompt', 'predictive_entropy_over_concepts_importance_max_third_prompt',
            'average_predictive_entropy_forth_prompt', 'predictive_entropy_over_concepts_forth_prompt',
            'average_predictive_entropy_importance_max_forth_prompt', 'predictive_entropy_over_concepts_importance_max_forth_prompt',
            # 'average_predictive_entropy_fifth_prompt', 'predictive_entropy_over_concepts_fifth_prompt',
            # 'average_predictive_entropy_importance_max_fifth_prompt', 'predictive_entropy_over_concepts_importance_max_fifth_prompt',
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
        
        if args.accuracy_metric in ['bem_score', 'gpt_score', 'exact_match']:
            correctness_bin = (results[args.accuracy_metric] > args.roc_auc_threshold).astype('int') 
        elif args.accuracy_metric == 'bert_score':
            correctness_bin = (results[args.accuracy_metric].apply(lambda x: x['F1']) > args.roc_auc_threshold).astype('int') 
        elif args.accuracy_metric == 'rouge_score':
            correctness_bin = (results[args.accuracy_metric].apply(lambda x: x['rougeL']) > args.roc_auc_threshold).astype('int') 
        correctness_results['accuracy'] = correctness_bin.mean()
        
        # non-binarized accuracy
        correctness_results['exact_match_mean'] = results['exact_match'].mean()
        correctness_results['bem_score_mean'] = results['bem_score'].mean()
        correctness_results['bert_score_mean'] = results['bert_score'].apply(lambda x: x['F1']).mean()
        # correctness_results['rougeL_score_mean'] = results['rouge_score'].apply(lambda x: x['rougeL']).mean()
        if args.accuracy_metric in ['bem_score', 'gpt_score']:
            one_minus_correctness = 1 - results[args.accuracy_metric]
        elif args.accuracy_metric == 'rouge_score':
            one_minus_correctness = 1 - results[args.accuracy_metric].apply(lambda x: x['rougeL'])
        elif args.accuracy_metric == 'bert_score':
            one_minus_correctness = 1 - results[args.accuracy_metric].apply(lambda x: x['F1'])
        elif args.accuracy_metric == 'exact_match':
            one_minus_correctness = 1 - results[args.accuracy_metric].astype('int') 
        
        return correctness_results, correctness_bin, one_minus_correctness
    
    def compute_answer_equality_em(sequence_1, sequence_2):
        sequence_2_ = {}
        for sample in sequence_2:
            sequence_2_[sample['id']] = sample
        
        semantically_similar_list = []
        semantically_not_similar_list = []
        with open(answers_equality_output_jsonl_file, 'w') as jl_ofile:
            for i, sample in tqdm(enumerate(sequence_1)):
                id_ = sample['id']
                is_equal = False
                seq1 = sample['cleaned_most_likely_generation'].strip()
                
                if id_ in sequence_2_:
                    seq2 = sequence_2_[id_]['cleaned_most_likely_generation'].strip()
        
                    # if seq1=='\n' or seq2=='\n':
                    #     is_equal = False
                    # elif seq1=='' or seq2=='':
                    #     is_equal = False
                    # else:
                    if seq1 == seq2 or seq1.lower() == seq2 or seq1.capitalize() == seq2:
                        is_equal = True
                    if seq2 == seq1 or seq2.lower() == seq1 or seq2.capitalize() == seq1:
                        is_equal = True
            
                    if is_equal:
                        semantically_similar_list.append(id_)
                    else:
                        semantically_not_similar_list.append(id_)

                else:
                    print(f"\nQuery {id_} is not common between two sequences !!!")
                    seq2 = ""
                    semantically_not_similar_list.append(id_)

                result_item = {
                    'id': id_,
                    'question': sample['question'],
                    'answers': sample['answers'],
                    'generation_seq_1': seq1,
                    'generation_seq_2': seq2,
                    'is_equal': is_equal
                }
                jl_ofile.write(json.dumps(result_item) + '\n')

        return semantically_similar_list, semantically_not_similar_list

    def get_entail_contradict_relations_nli(axioms, queries_list):
        # Src: https://huggingface.co/vectara/hallucination_evaluation_model
        # Input: a list of pairs of (premise, hypothesis)
        # It returns a score between 0 and 1 for each pair where
        # 0 means that the hypothesis is not evidenced at all by the premise and
        # 1 means the hypothesis is fully supported by the premise.
        
        if axioms == "12":
            results_file = axioms12_output_jsonl_file
            sequences = sequences_main
        elif axioms == "4":
            results_file = axiom4_output_jsonl_file
            sequences = sequences_main
        elif axioms == "5":
            results_file = axiom5_output_jsonl_file
            sequences = sequences_secondry
            
            sequences_main_ = {}
            for item in sequences_main:
                sequences_main_[item['id']] = item
            
        else:
            print(f"No valid axiom number !!!")
        
        if os.path.isfile(results_file):
            print(f"{results_file} exists.")
            with open(results_file, 'r') as file:
                relation_queries = json.load(file) 
        else:
            relation_queries = {
                'entailment': [],
                'neutral': [],
                'contradiction': []
            }
        
            with torch.no_grad():
                for idx, sample in tqdm(enumerate(sequences)):
                    id_ = sample['id']
                    
                    if id_ in queries_list:
                        
                        # Get and prepare variables
                        question = sample['question']
                        generated_text_most_likely = sample['most_likely_generation']
                        
                        if axioms in ["12", "4"]:
                            prompt_text = sample['prompt_text']
                        elif axioms == "5":
                            prompt_text = sequences_main_[id_]['prompt_text']
                        
                        doc_text = prompt_text.split('Document:')[-1].split('Question:')[0]
                        answer_ = f"{question} {generated_text_most_likely}"

                        
                        # Method 1) Common NLI: based on label
                        # encoded_input = semantic_tokenizer.encode_plus(
                        #     doc_text,
                        #     answer_,
                        #     padding=True,
                        #     truncation=True,
                        #     max_length=512,
                        #     return_tensors="pt",
                        #     truncation_strategy="only_first"
                        # )
                        # encoded_input = {key: val.to(args.device) for key, val in encoded_input.items()}
                        # prediction = semantic_model(**encoded_input).logits
                        # predicted_label = prediction.argmax(dim=1).item()

                        # item = (id_, predicted_label)
                        # if predicted_label==0: # entailment
                        #     relation_queries['entailment'].append(item)
                        # elif predicted_label==1: # neutral
                        #     relation_queries['neutral'].append(item)
                        # elif predicted_label==2: # contradiction
                        #     relation_queries['contradiction'].append(item)
                        # else:
                        #     relation_queries['neutral'].append(item)
                        ### === For BART  
                        # if predicted_label==0: # entailment
                        #     relation_queries['contradiction'].append(item)
                        # elif predicted_label==1: # neutral
                        #     relation_queries['neutral'].append(item)
                        # elif predicted_label==2: # contradiction
                        #     relation_queries['entailment'].append(item)
                        # else:
                        #     relation_queries['neutral'].append(item)
                    
            
                        # Method 2) MiniCheck
                        # pred_label, predicted_score_, _, _ = minicheck_factual_scorer.score(docs=[doc_text], claims=[answer_])
                        # predicted_score = predicted_score_[0]
                        
                        
                        # Method 3) 
                        # predicted_score_ = hallucination_detector.predict([(doc_text, answer_)])
                        # predicted_score = predicted_score_.item()
                        
                        # item = (id_, predicted_score)
                        # if predicted_score > 0.50: # entailment
                        #     relation_queries['entailment'].append(item)
                        # elif predicted_score < 0.50: # contradiction
                        #     relation_queries['contradiction'].append(item)
                        # else:
                        #     relation_queries['neutral'].append(item)
            
            
                        # Method 4) Long NLI 
                        # predicted_score_ = long_nli_model([dict(text=doc_text, text_pair=answer_)])
                        # item = (id_, predicted_score_[0]['score'])
                        # relation_queries[predicted_score_[0]['label']].append(item)
                        
                        
                        # Method 5) Common NLI: Similar to semantic semilarity
                        input = doc_text + ' [SEP] ' + answer_
                        encoded_input = semantic_tokenizer.encode(input, padding=True)
                        prediction = semantic_model(torch.tensor(torch.tensor([encoded_input]), device=args.device))['logits']
                        predicted_label = torch.argmax(prediction, dim=1)
                        
                        reverse_input = answer_ + ' [SEP] ' + doc_text
                        encoded_reverse_input = semantic_tokenizer.encode(reverse_input, padding=True)
                        reverse_prediction = semantic_model(torch.tensor(torch.tensor([encoded_reverse_input]), device=args.device))['logits']
                        reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
                        
                        item = (id_, torch.softmax(prediction, dim=1).tolist()[0], torch.softmax(reverse_prediction, dim=1).tolist()[0])
                        if 0 in predicted_label or 0 in reverse_predicted_label:
                            relation_queries['contradiction'].append(item)
                        else:
                            relation_queries['entailment'].append(item)
                        
            # Write to file 
            with open(results_file, 'w') as file:
                json.dump(relation_queries, file, indent=4)
        
        return relation_queries

    def run_axiomatic_metrics(prompt_order):
        
        # Check if answer1 is equal to answer2
        if os.path.isfile(answers_equality_output_jsonl_file):
            print(f"{answers_equality_output_jsonl_file} exists.")
            answer_equal_list, answer_not_equal_list = [], []
            with open(answers_equality_output_jsonl_file, 'r') as file:
                for line in file:
                    if line.strip():
                        item = json.loads(line)
                        if item['is_equal']:
                            answer_equal_list.append(item['id'])
                        else:
                            answer_not_equal_list.append(item['id'])
        else:
            print("Computing similarity ...")
            answer_equal_list, answer_not_equal_list = compute_answer_equality_em(sequences_main, sequences_secondry)

        print(f"Answer equal: {len(answer_equal_list)}")
        print(f"Answer not equal: {len(answer_not_equal_list)}")

        
        for axiom_num in ['1', '2', '4', '5']: #  
            print(f"= Axiom: {axiom_num} =======")
            
            if axiom_num == '1':
                selected_list_ = answer_equal_list
                answer_equal_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_)]
                selected_main_prompt_df = answer_equal_main_prompt_df[(answer_equal_main_prompt_df["exact_match"] == True)]
                print(f'Axiom 1: {len(selected_main_prompt_df)}')
                selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_main_prompt_df['id'].tolist())]
                
            elif axiom_num == '2':
                selected_list_ = answer_equal_list
                answer_equal_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_)]
                selected_main_prompt_df = answer_equal_main_prompt_df[(answer_equal_main_prompt_df["exact_match"] == False)]
                print(f'Axiom 2: {len(selected_main_prompt_df)}')
                selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_main_prompt_df['id'].tolist())]
            
            elif axiom_num == '4':
                selected_list_ = answer_not_equal_list
                answer_not_equal_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_)]
                selected_main_prompt_df = answer_not_equal_main_prompt_df[(answer_not_equal_main_prompt_df["exact_match"] == True)]
                
                selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_main_prompt_df['id'].tolist())]
                selected_second_prompt_correct_df = selected_second_prompt_df[selected_second_prompt_df["exact_match"] == True]
                selected_second_prompt_notcorrect_df = selected_second_prompt_df[selected_second_prompt_df["exact_match"] == False]
                
                print(f'Axiom 4 -> Main {len(selected_main_prompt_df)}')
                print(f'Axiom 4 -> 2ed, c: {len(selected_second_prompt_correct_df)}')
                print(f'Axiom 4 -> 2ed n: {len(selected_second_prompt_notcorrect_df)}')
                
            
            elif axiom_num == '5':
                selected_list_ = answer_not_equal_list
                answer_not_equal_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_)]
                selected_main_prompt_df = answer_not_equal_main_prompt_df[(answer_not_equal_main_prompt_df["exact_match"] == False)]
            
                selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_main_prompt_df['id'].tolist())]
                selected_second_prompt_correct_df = selected_second_prompt_df[selected_second_prompt_df["exact_match"] == True]
                selected_second_prompt_notcorrect_df = selected_second_prompt_df[selected_second_prompt_df["exact_match"] == False]
                
                print(f'Axiom 5 -> Main {len(selected_main_prompt_df)}')
                print(f'Axiom 5 -> 2ed, c: {len(selected_second_prompt_correct_df)}')
                print(f'Axiom 5 -> 2ed, n: {len(selected_second_prompt_notcorrect_df)}')
                
            for uncertainty_model in ['PE', ]: # 'PE', 'SE', 'PE_MARS', 'SE_MARS'
                unc_model_key_main_prompt = keys_mapping[f'{prompt_order}_prompt'][uncertainty_model]
                unc_model_key_second_prompt = keys_mapping['main_prompt'][uncertainty_model]
                
                # Get Uncertainty
                if axiom_num in ['1', '2']:
                    uncertainty_values_main_prompt =  selected_main_prompt_df[unc_model_key_main_prompt]
                    uncertainty_values_second_prompt = selected_second_prompt_df[unc_model_key_second_prompt]
                    uncertainty_values_main_prompt_filtered =  uncertainty_values_main_prompt[uncertainty_values_main_prompt<UNC_THERESHOLD]
                    uncertainty_values_second_prompt_filtered = uncertainty_values_second_prompt[uncertainty_values_second_prompt<UNC_THERESHOLD]
                
                    print(f"Axiom {axiom_num}, {uncertainty_model}")
                    print(f"Unc. : {uncertainty_values_second_prompt_filtered.mean():.3f} -> {uncertainty_values_main_prompt_filtered.mean():.3f}")
                
                elif axiom_num in ['4', '5']:
                    uncertainty_values_main_prompt =  selected_main_prompt_df[unc_model_key_main_prompt]
                    uncertainty_values_second_prompt_correct = selected_second_prompt_correct_df[unc_model_key_second_prompt]
                    uncertainty_values_second_prompt_notcorrect = selected_second_prompt_notcorrect_df[unc_model_key_second_prompt]
                    
                    uncertainty_values_main_prompt_filtered =  uncertainty_values_main_prompt[uncertainty_values_main_prompt<UNC_THERESHOLD]
                    uncertainty_values_second_prompt_correct_filtered = uncertainty_values_second_prompt_correct[uncertainty_values_second_prompt_correct<UNC_THERESHOLD] 
                    uncertainty_values_second_prompt_notcorrect_filtered = uncertainty_values_second_prompt_notcorrect[uncertainty_values_second_prompt_notcorrect<UNC_THERESHOLD] 

                    print(f"Axiom {axiom_num}, {uncertainty_model}")
                    print(f"Unc. corr : {uncertainty_values_second_prompt_correct_filtered.mean():.3f} -> {uncertainty_values_main_prompt_filtered.mean():.3f}")
                    print(f"Unc. ncor : {uncertainty_values_second_prompt_notcorrect_filtered.mean():.3f} -> {uncertainty_values_main_prompt_filtered.mean():.3f}")


                # # Get Confidence
                # uncertainty_values_all_main_prompt = result_df_main_prompt[unc_model_key_main_prompt]
                # uncertainty_values_all_second_prompt = result_df_second_prompt[unc_model_key_second_prompt]
                # uncertainty_values_all_main_prompt_filtered = np.array(uncertainty_values_all_main_prompt[uncertainty_values_all_main_prompt<UNC_THERESHOLD])
                # uncertainty_values_all_second_prompt_filtered = np.array(uncertainty_values_all_second_prompt[uncertainty_values_all_second_prompt<UNC_THERESHOLD])
                # min_all_main, max_all_main = np.min(uncertainty_values_all_main_prompt_filtered), np.max(uncertainty_values_all_main_prompt_filtered)
                # min_all_second, max_all_second = np.min(uncertainty_values_all_second_prompt_filtered), np.max(uncertainty_values_all_second_prompt_filtered)
                # confidence_values_main_prompt = uncertainty_to_confidence_min_max(uncertainty_values_main_prompt_filtered, min_val=min_all_main, max_val=max_all_main)
                # confidence_values_second_prompt = uncertainty_to_confidence_min_max(uncertainty_values_second_prompt_filtered, min_val=min_all_second, max_val=max_all_second)
                
                # Get correctness
                # _, correctness_main_prompt_bin, one_minus_correctness_main_prompt = get_correctness(selected_main_prompt_df)
                # _, correctness_second_prompt_bin, one_minus_correctness_second_prompt = get_correctness(selected_second_prompt_df)
                # correctness_main_prompt = 1 - np.array(one_minus_correctness_main_prompt)
                # correctness_second_prompt = 1 - np.array(one_minus_correctness_second_prompt)
                
                # Print results
                # print(f"Axiom {axiom_num}, {uncertainty_model}")
                # print(f"Unc. : {uncertainty_values_second_prompt_filtered.mean():.3f} -> {uncertainty_values_main_prompt_filtered.mean():.3f}")
                # print(f"Conf.: {confidence_values_second_prompt.mean():.3f} -> {confidence_values_main_prompt.mean():.3f}")
                # print(f"Acc. ({args.accuracy_metric}):  {round(correctness_second_prompt.mean()*100, 2)} -> {round(correctness_main_prompt.mean()*100, 2)}")
                print('\n')             

    # ======
    result_df_main_prompt = create_result_df(args.main_prompt_format, args.second_prompt_format)    
    result_df_second_prompt = create_result_df(args.second_prompt_format, args.main_prompt_format)
        
    for prompt_order in ['main', 'second', 'forth']: # 'third', 
        print(f"=== {prompt_order} ====================================")
        axioms12_output_jsonl_file = f'{base_dir_output}/{args.main_prompt_format}__{args.second_prompt_format}/{generation_type}/axiomatic_results_{prompt_order}/{model}_{args.temperature}_axioms12_output.json'
        axiom4_output_jsonl_file = f'{base_dir_output}/{args.main_prompt_format}__{args.second_prompt_format}/{generation_type}/axiomatic_results_{prompt_order}/{model}_{args.temperature}_axiom4_output.json'
        axiom5_output_jsonl_file = f'{base_dir_output}/{args.main_prompt_format}__{args.second_prompt_format}/{generation_type}/axiomatic_results_{prompt_order}/{model}_{args.temperature}_axiom5_output.json'
        answers_equality_output_jsonl_file = f'{base_dir_output}/{args.main_prompt_format}__{args.second_prompt_format}/{generation_type}/axiomatic_results_{prompt_order}/{model}_{args.temperature}_answers_equality_output.jsonl'
        answers_equality_output_dir = os.path.dirname(answers_equality_output_jsonl_file)
        os.makedirs(answers_equality_output_dir, exist_ok=True)
        
        run_axiomatic_metrics(prompt_order)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--model_llama_eval', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='nqgold', choices=[
        'nqgold', 'nqswap', 'trivia', 'popqa',
        'webquestions', 'squad1', 'nq',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa',
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--main_prompt_format', type=str, default='bm25_retriever_top1', choices=[
        'only_q', 'q_positive', 'q_negative', 'q_negative',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    parser.add_argument('--second_prompt_format', type=str, default='only_q', choices=[
        'only_q', 'q_positive', 'q_negative', 'q_conflict',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    parser.add_argument('--accuracy_metric', type=str, default="exact_match", choices=[
        'exact_match', 'rouge_score', 'bert_score', 'bem_score', 'llama3_score', 'gpt_score'
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
    
    if args.main_prompt_format != 'only_q':
        args.second_prompt_format == 'only_q'
    
    set_seed(args.seed)
    get_axiomatic_results(args)
    
    # python framework/run/get_axiomatic_results_v3.py
    
    
    