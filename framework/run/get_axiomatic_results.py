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
from transformers import pipeline

from minicheck.minicheck import MiniCheck


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
    axioms123_output_jsonl_file = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_axioms123_output__sec_{args.second_prompt_format}.json'
    axiom4_output_jsonl_file = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_axiom4_output__sec_{args.second_prompt_format}.json'
    axiom5_output_jsonl_file = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_axiom5_output__sec_{args.second_prompt_format}.json'
    
    sequence_input_main = f'{base_dir_output}/{args.main_prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
    sequence_input_secondry = f'{base_dir_output}/{args.second_prompt_format}/{model}_{args.temperature}_cleaned_generation.pkl'
    with open(sequence_input_main, 'rb') as infile:
        sequences_main = pickle.load(infile)
    with open(sequence_input_secondry, 'rb') as infile:
        sequences_secondry = pickle.load(infile)
        
    # === Load semantic model ===================
    
    # 1) Common NLI models
    # - Labels: {0: Contradiction, 1: Neutral, 2: Entailment}
    # semantic_model_name = "microsoft/deberta-v2-xxlarge-mnli"
    # semantic_model_name = "microsoft/deberta-large-mnli"
    # semantic_model_name = 'facebook/bart-large-mnli'
    semantic_model_name = "tals/albert-xlarge-vitaminc-mnli"
    semantic_model = AutoModelForSequenceClassification.from_pretrained(semantic_model_name).to(args.device)
    semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
    semantic_model.eval()
    
    # (2), (3), (4) -> Src: https://arxiv.org/pdf/2410.03461
    
    # 2) MiniCheck (EMNLP24)
    # - MiniCheck: https://huggingface.co/lytang/MiniCheck-Flan-T5-Large
    # - pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
    minicheck_factual_scorer = MiniCheck(model_name='flan-t5-large', cache_dir='./ckpts')
    
    # 2) Hallucination Detector
    # - Vectara: https://huggingface.co/vectara/hallucination_evaluation_model
    # - It needs updated version of transformer: pip install --upgrade transformers
    hallucination_detector = AutoModelForSequenceClassification.from_pretrained(
        'vectara/hallucination_evaluation_model', trust_remote_code=True)

    # 3) Long NLI: 
    # - Tasksource: https://huggingface.co/tasksource/deberta-base-long-nli
    long_nli_model = pipeline("text-classification", model="tasksource/deberta-base-long-nli", device=args.device)

    
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
        correctness_results['rougeL_score_mean'] = results['rouge_score'].apply(lambda x: x['rougeL']).mean()
        if args.accuracy_metric in ['bem_score', 'gpt_score']:
            one_minus_correctness = 1 - results[args.accuracy_metric]
        elif args.accuracy_metric == 'rouge_score':
            one_minus_correctness = 1 - results[args.accuracy_metric].apply(lambda x: x['rougeL'])
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
    
    def get_nli_relations(axioms, queries_list):
        # Src: https://huggingface.co/vectara/hallucination_evaluation_model
        # Input: a list of pairs of (premise, hypothesis)
        # It returns a score between 0 and 1 for each pair where
        # 0 means that the hypothesis is not evidenced at all by the premise and
        # 1 means the hypothesis is fully supported by the premise.
        
        if axioms == "123":
            results_file = axioms123_output_jsonl_file
            sequences = sequences_main
        elif axioms == "4":
            results_file = axiom4_output_jsonl_file
            sequences = sequences_main
        elif axioms == "5":
            results_file = axiom5_output_jsonl_file
            sequences = sequences_secondry
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
                        prompt_text = sample['prompt_text']
                        doc_text = prompt_text.split('Document:')[-1].split('Question:')[0]
                        answer_ = f"{question} {generated_text_most_likely}"

                        
                        # Method 1) Common NLI
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
                        predicted_score_ = long_nli_model([dict(text=doc_text, text_pair=answer_)])
                        item = (id_, predicted_score_[0]['score'])
                        relation_queries[predicted_score_[0]['label']].append(item)
            
            
            # Write to file 
            with open(results_file, 'w') as file:
                json.dump(relation_queries, file, indent=4)
        
        return relation_queries

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
    
    
    print("================= Axioms: 1, 2, 3 =================")
    axioms_num = '123'
    axioms_123 = get_nli_relations(axioms_num, agree_list)
    print(f"Entailment:    {len(axioms_123['entailment'])} ({(len(axioms_123['entailment']) / len(sequences_main))*100:.2f}%)")
    print(f"Contradiction: {len(axioms_123['contradiction'])} ({len(axioms_123['contradiction']) / len(sequences_main)*100:.2f}%)")
    print(f"Neutral:       {len(axioms_123['neutral'])} ({len(axioms_123['neutral']) / len(sequences_main)*100:.2f}%)")
    print('\n')
    
    for uncertainty_model in ['PE', 'SE']: # , 'PE_MARS', 'SE_MARS' 
        
        if uncertainty_model in ['PE', 'PE_MARS']:
            result_df_main_prompt = result_df_main_filtered_pe
            result_df_second_prompt = result_df_second_prompt_filtered_pe
        elif uncertainty_model in ['SE', 'SE_MARS']:
            result_df_main_prompt = result_df_main_filtered_se
            result_df_second_prompt = result_df_second_prompt_filtered_se
        
        unc_model_key_main_prompt = keys_mapping['main_prompt'][uncertainty_model]
        unc_model_key_second_prompt = keys_mapping['second_prompt'][uncertainty_model]
    
        for relation_key in ['entailment', 'contradiction', 'neutral']: #  'neutral'
            selected_list = axioms_123[relation_key]
            
            if len(selected_list) > 0:
                selected_list_ = [tup[0] for tup in selected_list]
                agree_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_)]
                agree_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_list_)]
        
                _, correctness_main_prompt_bin, one_minus_correctness_main_prompt = get_correctness(agree_main_prompt_df)
                _, correctness_second_prompt_bin, one_minus_correctness_second_prompt = get_correctness(agree_second_prompt_df)
                correctness_main_prompt = 1 - np.array(one_minus_correctness_main_prompt)
                correctness_second_prompt = 1 - np.array(one_minus_correctness_second_prompt)
            
                uncertainty_main_prompt_values =  agree_main_prompt_df[unc_model_key_main_prompt] # 0.5*
                uncertainty_second_prompt_values = agree_second_prompt_df[unc_model_key_main_prompt] # Axioms: 1, 2, 3
                confidence_main_prompt_values = uncertainty_to_confidence_min_max(uncertainty_main_prompt_values)
                confidence_second_prompt_values = uncertainty_to_confidence_min_max(uncertainty_second_prompt_values)
                
                auroc_main_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_main_prompt_values)
                auroc_second_prompt = sklearn.metrics.roc_auc_score(1 - correctness_second_prompt_bin, uncertainty_second_prompt_values)
                # ece_main_prompt = ece_estimate(correctness_main_prompt, confidence_main_prompt_values)
                # ece_second_prompt = ece_estimate(correctness_second_prompt, confidence_second_prompt_values)
                
                print(f"{uncertainty_model}, Axiom1: {relation_key}")
                print(f"Uncertainty: {uncertainty_second_prompt_values.mean():.3f} -> {uncertainty_main_prompt_values.mean():.3f}")
                print(f"Acc. ({args.accuracy_metric}):  {round(correctness_second_prompt.mean()*100, 2)} -> {round(correctness_main_prompt.mean()*100, 2)}")
                print(f"AUROC:       {round(auroc_second_prompt, 3)} -> {round(auroc_main_prompt, 3)}")
                # print(f"Confidence:  {confidence_second_prompt_values.mean():.3f} -> {confidence_main_prompt_values.mean():.3f}")
                # print(f"ECE:         {round(ece_second_prompt, 3)} -> {round(ece_main_prompt, 3)}")  
                print('\n')             
                
            else: 
                print(f"{relation_key} does not contain data!!!")
                print('\n')


    print("================= Axiom: 4 =================")
    axiom_num = '4'
    axiom_4 = get_nli_relations(axiom_num, non_agree_list)
    print(f"Entailment:    {len(axiom_4['entailment'])} ({(len(axiom_4['entailment']) / len(sequences_main))*100:.2f}%)")
    # print(f"Contradiction: {len(axiom_4['contradiction'])} ({len(axiom_4['contradiction']) / len(sequences_main)*100:.2f}%)")
    # print(f"Neutral:       {len(axiom_4['neutral'])} ({len(axiom_4['neutral']) / len(sequences_main)*100:.2f}%)")
    print('\n')
    relation_key = 'entailment'
    selected_list = axiom_4[relation_key]
    selected_list_ = [tup[0] for tup in selected_list]
    
    if len(selected_list) > 0:
        for uncertainty_model in ['PE', 'SE']: # , 'PE_MARS', 'SE_MARS' 
            
            if uncertainty_model in ['PE', 'PE_MARS']:
                result_df_main_prompt = result_df_main_filtered_pe
                result_df_second_prompt = result_df_second_prompt_filtered_pe
            elif uncertainty_model in ['SE', 'SE_MARS']:
                result_df_main_prompt = result_df_main_filtered_se
                result_df_second_prompt = result_df_second_prompt_filtered_se
            
            unc_model_key_main_prompt = keys_mapping['main_prompt'][uncertainty_model]
            unc_model_key_second_prompt = keys_mapping['second_prompt'][uncertainty_model]

            selected_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_)]
            selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_list_)]
            
            _, correctness_main_prompt_bin, one_minus_correctness_main_prompt = get_correctness(selected_main_prompt_df)
            correctness_main_prompt = 1 - np.array(one_minus_correctness_main_prompt)
            _, correctness_second_prompt_bin, one_minus_correctness_second_prompt = get_correctness(selected_second_prompt_df)
            correctness_second_prompt = 1 - np.array(one_minus_correctness_second_prompt)
            
            uncertainty_main_prompt_values =  selected_main_prompt_df[unc_model_key_main_prompt]
            uncertainty_second_prompt_values = selected_main_prompt_df[unc_model_key_second_prompt]
            auroc_main_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_main_prompt_values)
            auroc_second_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_second_prompt_values)
            print(f"{uncertainty_model}, Axiom 4: {relation_key}")
            print(f"Uncertainty: {uncertainty_second_prompt_values.mean():.3f} -> {uncertainty_main_prompt_values.mean():.3f}")
            print(f"Acc. ({args.accuracy_metric}): {round(correctness_main_prompt.mean()*100, 2)}")
            print(f"AUROC:       {round(auroc_second_prompt, 3)} -> {round(auroc_main_prompt, 3)}")
            print('\n')
                
    else: 
        print(f"{relation_key} does not contain data!!!")
        print('\n')


    print("================= Axiom: 5 =================")
    axiom_num = '5'
    axiom_5 = get_nli_relations(axiom_num, non_agree_list)
    print(f"Contradiction: {len(axiom_5['contradiction'])} ({len(axiom_5['contradiction']) / len(sequences_main)*100:.2f}%)")
    # print(f"Entailment:    {len(axiom_5['entailment'])} ({(len(axiom_5['entailment']) / len(sequences_main))*100:.2f}%)")
    # print(f"Neutral:       {len(axiom_4['neutral'])} ({len(axiom_4['neutral']) / len(sequences_main)*100:.2f}%)")
    
    print('\n')
    relation_key = 'contradiction'
    selected_list = axiom_5[relation_key]
    selected_list_ = [tup[0] for tup in selected_list]
    
    if len(selected_list) > 0:
        
        for uncertainty_model in ['PE', 'SE']: # , 'PE_MARS', 'SE_MARS' 
        
            if uncertainty_model in ['PE', 'PE_MARS']:
                result_df_main_prompt = result_df_main_filtered_pe
                result_df_second_prompt = result_df_second_prompt_filtered_pe
            elif uncertainty_model in ['SE', 'SE_MARS']:
                result_df_main_prompt = result_df_main_filtered_se
                result_df_second_prompt = result_df_second_prompt_filtered_se
            
            unc_model_key_main_prompt = keys_mapping['main_prompt'][uncertainty_model]
            unc_model_key_second_prompt = keys_mapping['second_prompt'][uncertainty_model]

            selected_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_)]
            selected_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_list_)]
            
            _, correctness_main_prompt_bin, one_minus_correctness_main_prompt = get_correctness(selected_main_prompt_df)
            correctness_main_prompt = 1 - np.array(one_minus_correctness_main_prompt)
            _, correctness_second_prompt_bin, one_minus_correctness_second_prompt = get_correctness(selected_second_prompt_df)
            correctness_second_prompt = 1 - np.array(one_minus_correctness_second_prompt)
            
            uncertainty_main_prompt_values = selected_second_prompt_df[unc_model_key_second_prompt]
            uncertainty_second_prompt_values =  selected_second_prompt_df[unc_model_key_main_prompt]
            auroc_main_prompt = sklearn.metrics.roc_auc_score(1 - correctness_second_prompt_bin, uncertainty_main_prompt_values)
            auroc_second_prompt = sklearn.metrics.roc_auc_score(1 - correctness_second_prompt_bin, uncertainty_second_prompt_values)
            print(f"{uncertainty_model}, Axiom 5: {relation_key}")
            print(f"Uncertainty: {uncertainty_second_prompt_values.mean():.3f} -> {uncertainty_main_prompt_values.mean():.3f}")
            print(f"Acc. ({args.accuracy_metric}): {round(correctness_second_prompt.mean()*100, 2)}")
            print(f"AUROC:       {round(auroc_second_prompt, 3)} -> {round(auroc_main_prompt, 3)}")
            print('\n')

    else: 
        print(f"{relation_key} does not contain data!!!")
        print('\n')

    # Axiom 6
    # relation_key = 'contradiction'
    # selected_list = axioms_456[relation_key]
    # selected_list_ = [tup[0] for tup in selected_list]
    
    # if len(selected_list) > 0:
    #     axiom6_main_prompt_df = result_df_main_prompt[result_df_main_prompt['id'].isin(selected_list_)]
    #     axiom6_second_prompt_df = result_df_second_prompt[result_df_second_prompt['id'].isin(selected_list_)]
        
    #     _, correctness_main_prompt_bin, one_minus_correctness_main_prompt = get_correctness(axiom6_main_prompt_df)
    #     correctness_main_prompt = 1 - np.array(one_minus_correctness_main_prompt)
    #     _, correctness_second_prompt_bin, one_minus_correctness_second_prompt = get_correctness(axiom6_second_prompt_df)
    #     correctness_second_prompt = 1 - np.array(one_minus_correctness_second_prompt)
        
    #     uncertainty_main_prompt_values =  axiom6_main_prompt_df[unc_model_key_main_prompt]
    #     uncertainty_second_prompt_values = axiom6_second_prompt_df[unc_model_key_main_prompt]
    #     auroc_main_prompt = sklearn.metrics.roc_auc_score(1 - correctness_main_prompt_bin, uncertainty_main_prompt_values)
    #     auroc_second_prompt = sklearn.metrics.roc_auc_score(1 - correctness_second_prompt_bin, uncertainty_second_prompt_values)
            
    #     print(f"{uncertainty_model}, Axiom 6: {relation_key}")
    #     print(f"Uncertainty: {uncertainty_second_prompt_values.mean():.3f} -> {uncertainty_main_prompt_values.mean():.3f}")
    #     print(f"Acc. ({args.accuracy_metric}):  {round(correctness_second_prompt.mean()*100, 2)} -> {round(correctness_main_prompt.mean()*100, 2)}")
    #     print(f"AUROC:       {round(auroc_second_prompt, 3)} -> {round(auroc_main_prompt, 3)}")
    #     print('\n')

    # else: 
    #     print(f"{relation_key} does not contain data!!!")
    #     print('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--model_llama_eval', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='webquestions', choices=[
        'webquestions', 'nq', 'trivia', 'squad1',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa_org', 'topicoqa_his', 'topicoqa_rw',
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--main_prompt_format', type=str, default='q_positive', choices=[
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

