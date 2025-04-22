#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn.metrics
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from minicheck.minicheck import MiniCheck


from utils.utils import set_seed
UNC_THERESHOLD = 1000

def get_axiomatic_variables(args):
    print("\n--- Step 6: Get Axiomatic Variables ...")
    print(f"""
        Model name:   {args.model}
        Dataset:      {args.dataset} / {args.subsec}
        Prompt (1st): {args.main_prompt_format}
        Prompt (2ed): {args.second_prompt_format}
        Run id:       {args.run_id}
        Seed:         {args.seed}
    """.replace('        ', ''))
    
    # === Define output files ===================
    model_ = args.model.split('/')[-1]
    generation_type = f"prob_alpha_{str(args.alpha_probability)}"
    base_dir = f'{args.output_dir}/{model_}/{args.dataset}/{args.subsec}/{args.run_id}'
    
    # inputs
    sequence_input = f'{base_dir}/{args.main_prompt_format}__{args.second_prompt_format}/cleaned_generation_{args.generation_type}.pkl'
    # outputs
    axiomatic_variables_oe_file = f'{base_dir}/{args.main_prompt_format}__{args.second_prompt_format}/{generation_type}/axiomatic_variables_oe.pkl'
    axiomatic_variables_gn_file = f'{base_dir}/{args.main_prompt_format}__{args.second_prompt_format}/{generation_type}/axiomatic_variables_gn.pkl'
    axiomatic_variables_gk_file = f'{base_dir}/{args.main_prompt_format}__{args.second_prompt_format}/{generation_type}/axiomatic_variables_gk.pkl'
    axiomatic_variables_gm_file = f'{base_dir}/{args.main_prompt_format}__{args.second_prompt_format}/{generation_type}/axiomatic_variables_gm.pkl'
    
    with open(sequence_input, 'rb') as infile:
        sequences = pickle.load(infile)
    
    
    # === Define functions =======================
    def create_result_df(main_prompt_format, second_prompt_format):
        
        # For only query case
        results_dir = f'{base_dir}/{main_prompt_format}__{second_prompt_format}'
        if not os.path.isdir(results_dir):
            temp = 'bm25_retriever_top1' if args.dataset == 'popqa' else 'q_positive'
            results_dir = f'{base_dir}/{main_prompt_format}__{temp}'
        
        generation_file = f'{results_dir}/cleaned_generation_{args.generation_type}.pkl'
        similarities_input_file = f'{results_dir}/similarities_generation.pkl'
        correctness_input_file = f'{results_dir}/correctness.pkl'
        uncertainty_mars_input_file = f'{results_dir}/{generation_type}/uncertainty_mars_generation.pkl'
        
        with open(generation_file, 'rb') as infile:
            cleaned_sequences = pickle.load(infile)
        with open(similarities_input_file, 'rb') as f:
            similarities_dict = pickle.load(f)
        with open(uncertainty_mars_input_file, 'rb') as f:
            uncertainty_mars_results  = pickle.load(f)
        with open(correctness_input_file, 'rb') as f:
            correctness_results  = pickle.load(f)
        
        if os.path.exists(axiomatic_variables_oe_file):
            with open(axiomatic_variables_oe_file, 'rb') as infile:
                av_output_equality = pickle.load(infile)
        
        
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
        correctness_keys_to_use = ('id', 'bem_score', 'bert_score', 'exact_match') # , 'rouge_score'
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
            # 'average_predictive_entropy_forth_prompt', 'predictive_entropy_over_concepts_forth_prompt',
            # 'average_predictive_entropy_importance_max_forth_prompt', 'predictive_entropy_over_concepts_importance_max_forth_prompt',
            # 'average_predictive_entropy_fifth_prompt', 'predictive_entropy_over_concepts_fifth_prompt',
            # 'average_predictive_entropy_importance_max_fifth_prompt', 'predictive_entropy_over_concepts_importance_max_fifth_prompt',
        )
        uncertainty_mars = uncertainty_mars_results
        uncertainty_mars_small = dict((k, uncertainty_mars[k]) for k in keys_to_use)
        for key in uncertainty_mars_small:
            if key == 'average_predictive_entropy_on_subsets':
                uncertainty_mars_small[key].shape
            if type(uncertainty_mars_small[key]) is torch.Tensor:
                uncertainty_mars_small[key] = torch.squeeze(uncertainty_mars_small[key].cpu())
        uncertainty_mars_df = pd.DataFrame.from_dict(uncertainty_mars_small)
        uncertainty_mars_df.rename(columns={'ids': 'id'}, inplace=True) 
        # 
        # uncertainty_bb_df = pd.DataFrame(uncertainty_bb_results)
        # uncertainty_bb_keys_to_use = ('id', 'degree_u', 'ecc_u', 'spectral_u')
        # uncertainty_bb_small = dict((k, uncertainty_bb_df[k]) for k in uncertainty_bb_keys_to_use)
        # uncertainty_bb_df = pd.DataFrame.from_dict(uncertainty_bb_small)
        # 
        
        if os.path.exists(axiomatic_variables_oe_file):
            av_output_equality_df = pd.DataFrame(av_output_equality)
            av_output_equality_keys_to_use = ('id', 'output_equality_em', 'output_equality_nli', 'axiom_num_correctness')
            av_output_equality_small = dict((k, av_output_equality_df[k]) for k in av_output_equality_keys_to_use)
            av_output_equality_df = pd.DataFrame.from_dict(av_output_equality_small)
            
            result_df = generations_df.merge(similarities_df, on='id').merge(uncertainty_mars_df, on='id').merge(correctness_df, on='id').merge(av_output_equality_df, on='id')
        else:
            result_df = generations_df.merge(similarities_df, on='id').merge(uncertainty_mars_df, on='id').merge(correctness_df, on='id')
        
        
        return result_df
        
    # === 
    def get_axiomatic_variables_output_equality():
        
        # === Load semantic model ===================
        # - Labels: {0: Contradiction, 1: Neutral, 2: Entailment}
        semantic_model_name = "microsoft/deberta-large-mnli"
        semantic_model = AutoModelForSequenceClassification.from_pretrained(semantic_model_name).to(args.device)
        semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
        semantic_model.eval()
        
        # === Define functions ======================
        def get_output_equality_em(seq1, seq2):
            seq1 = seq1.strip()
            seq2 = seq2.strip()
            
            if seq1 == seq2 or seq1.lower() == seq2 or seq1.capitalize() == seq2:
                return True
            if seq2 == seq1 or seq2.lower() == seq1 or seq2.capitalize() == seq1:
                return True
            return False
    
        def get_output_equality_nli(question, output_text1, output_text2):
            answer_1 = f"{question} {output_text1}"
            answer_2 = f"{question} {output_text2}"
            
            # === Common NLI: Similar to semantic semilarity
            input = answer_1 + ' [SEP] ' + answer_2
            encoded_input = semantic_tokenizer.encode(input, padding=True)
            prediction = semantic_model(torch.tensor(torch.tensor([encoded_input]), device=args.device))['logits']
            predicted_label = torch.argmax(prediction, dim=1)
            
            reverse_input = answer_2 + ' [SEP] ' + answer_1
            encoded_reverse_input = semantic_tokenizer.encode(reverse_input, padding=True)
            reverse_prediction = semantic_model(torch.tensor(torch.tensor([encoded_reverse_input]), device=args.device))['logits']
            reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
            
            prediction_dist = torch.softmax(prediction, dim=1).tolist()[0]
            reverse_prediction_dist = torch.softmax(reverse_prediction, dim=1).tolist()[0]
            is_equal = False if (0 in predicted_label or 0 in reverse_predicted_label) else True
            entail_score = max(prediction_dist[2], reverse_prediction_dist[2])
            
            return (is_equal, entail_score)
        
            # === Get output equality ====================
        
        def get_axiom_number_correctness(answer_equality, correctness_main, correctness_sec):
            axiom_num = 'others'
            if answer_equality and correctness_main:
                axiom_num = '1'
            if answer_equality and not correctness_main:
                axiom_num = '2'
            if not answer_equality and correctness_main and not correctness_sec:
                axiom_num = '4'
            if not answer_equality and not correctness_main and correctness_sec:
                axiom_num = '5'
            if not answer_equality and correctness_main and correctness_sec:
                axiom_num = '1'
            
            return axiom_num
    
        
        # ===
        result_df_main_prompt_filtered['output_equality_em'] = [
            get_output_equality_em(seq1, seq2)
            for seq1, seq2 in tqdm(zip(
                result_df_main_prompt_filtered['cleaned_most_likely_generation'], 
                result_df_second_prompt_filtered['cleaned_most_likely_generation']
            ), desc='Getting output equality (EM) ...')
        ]
        em_counts = result_df_main_prompt_filtered['output_equality_em'].value_counts()
        print(f"AE-EM (equal): {em_counts.get(True, 0)}")
        print(f"AE-EM (not equal): {em_counts.get(False, 0)}")
        
        result_df_main_prompt_filtered['output_equality_nli'] = [
            get_output_equality_nli(question, seq1, seq2)
            for question, seq1, seq2 in tqdm(zip(
                result_df_main_prompt_filtered['question'],
                result_df_main_prompt_filtered['cleaned_most_likely_generation'], 
                result_df_second_prompt_filtered['cleaned_most_likely_generation']
            ), desc='Getting output equality (NLI) ...')
        ]
        nli_counts = result_df_main_prompt_filtered['output_equality_nli'].apply(lambda x: x[0]).value_counts()
        print(f"AE-NLI (equal): {nli_counts.get(True, 0)}")
        print(f"AE-NLI (not equal): {nli_counts.get(False, 0)}")


        # ==== Correctness ========================
        result_df_main_prompt_filtered['axiom_num_correctness'] = [
            get_axiom_number_correctness(answer_equality, correctness_main, correctness_sec)
            for answer_equality, correctness_main, correctness_sec in tqdm(zip(
                result_df_main_prompt_filtered['output_equality_em'],
                result_df_main_prompt_filtered['exact_match'],
                result_df_second_prompt_filtered['exact_match']
            ), desc='Getting axiom number (EM) ...')
        ]

        
        # === Write to file =====================
        variables_sequences = []
        for idx, sample in tqdm(enumerate(sequences)):
            question_id = sample['id']
            question = sample['question']
            reference_answers = sample['answers']
            sequence_dict = {
                'id': question_id,
                'question': question,
                'answers': reference_answers
            }
            if question_id in result_df_main_prompt_filtered['id'].values:
                sequence_dict['output_equality_em'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'output_equality_em'].iloc[0]
                sequence_dict['output_equality_nli'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'output_equality_nli'].iloc[0]
                sequence_dict['axiom_num_correctness'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'axiom_num_correctness'].iloc[0]
            else:
                sequence_dict['output_equality_em'] = False
                sequence_dict['output_equality_nli'] = (False, 0.0)
                sequence_dict['axiom_num_correctness'] = "not_common"

            variables_sequences.append(sequence_dict)
        
        
        # === Save the file =====================
        with open(axiomatic_variables_oe_file, 'wb') as ofile:
            pickle.dump(variables_sequences, ofile)
        print(f"Results saved to {axiomatic_variables_oe_file}")

    
    def get_axiomatic_variables_groundedness_nli():
        
        # === Load semantic model ===================
        # - Labels: {0: Contradiction, 1: Neutral, 2: Entailment}
        semantic_model_name = "microsoft/deberta-large-mnli"
        semantic_model = AutoModelForSequenceClassification.from_pretrained(semantic_model_name).to(args.device)
        semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
        semantic_model.eval()
        
        
        # === Define functions ======================
        def get_groundedness_nli(prompt_text, question, output_text):
        
            # === Prapare inputs
            doc_text = prompt_text.split('Document:')[-1].split('Question:')[0]
            answer_ = f"{question} {output_text}"
            
            # === Common NLI: Similar to semantic semilarity
            input = doc_text + ' [SEP] ' + answer_
            reverse_input = answer_ + ' [SEP] ' + doc_text
            
            with torch.no_grad():
                encoded_input = semantic_tokenizer.encode(input, padding=True)
                encoded_reverse_input = semantic_tokenizer.encode(reverse_input, padding=True)
                
                prediction = semantic_model(torch.tensor(torch.tensor([encoded_input]), device=args.device))['logits']
                predicted_label = torch.argmax(prediction, dim=1)
                
                reverse_prediction = semantic_model(torch.tensor(torch.tensor([encoded_reverse_input]), device=args.device))['logits']
                reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
            
            # === Get label
            nli_label = 0 if (0 in predicted_label or 0 in reverse_predicted_label) else 2
            prediction_dist = torch.softmax(prediction, dim=1).tolist()[0]
            reverse_prediction_dist = torch.softmax(reverse_prediction, dim=1).tolist()[0]
            # entail_score = max(prediction_dist[1], prediction_dist[2], reverse_prediction_dist[1], reverse_prediction_dist[2])
            entail_score = max(prediction_dist[2], reverse_prediction_dist[2])
            contradict_score = max(prediction_dist[0], reverse_prediction_dist[0])
            
            return (nli_label, entail_score)
        
        def get_axiom_number_nli(answer_equality, nli_main, nli_sec):
            axiom_num = 'others'
            if answer_equality and nli_main[0] == 2:
                axiom_num = '1'
            if answer_equality and nli_main[0] == 0:
                axiom_num = '2'
            if not answer_equality and nli_main[0] == 2 and nli_sec[0] == 0:
                axiom_num = '4'
            if not answer_equality and nli_main[0] == 0 and nli_sec[0] == 2:
                axiom_num = '5'
            
            return axiom_num
    
    
        # ==== NLI ==================================
        result_df_main_prompt_filtered['groundedness_nli_main'] = [
            get_groundedness_nli(prompt_text, question, output)
            for prompt_text, question, output in tqdm(zip(
                result_df_main_prompt_filtered['prompt_text'],
                result_df_main_prompt_filtered['question'],
                result_df_main_prompt_filtered['cleaned_most_likely_generation']
            ), desc='Getting groundedness NLI (main) ...')
        ]
        
        result_df_main_prompt_filtered['groundedness_nli_second'] = [
            get_groundedness_nli(prompt_text, question, output)
            for prompt_text, question, output in tqdm(zip(
                result_df_main_prompt_filtered['prompt_text'],
                result_df_main_prompt_filtered['question'],
                result_df_second_prompt_filtered['cleaned_most_likely_generation']
            ), desc='Getting groundedness NLI (second) ...')
        ]
        
        result_df_main_prompt_filtered['axiom_num_nli'] = [
            get_axiom_number_nli(answer_equality, nli_main, nli_sec)
            for answer_equality, nli_main, nli_sec in tqdm(zip(
                result_df_main_prompt_filtered['output_equality_em'],
                result_df_main_prompt_filtered['groundedness_nli_main'],
                result_df_main_prompt_filtered['groundedness_nli_second']
            ), desc='Getting axiom number (NLI) ...')
        ]

        # ======
        print("==== NLI ========================================")
        cm = confusion_matrix(
            result_df_main_prompt_filtered["axiom_num_correctness"],
            result_df_main_prompt_filtered["axiom_num_nli"],
            labels=["1", "2", "4", "5", "others"])
        # Compute classification report (Precision, Recall, F1-score)
        report = classification_report(
            result_df_main_prompt_filtered["axiom_num_correctness"],
            result_df_main_prompt_filtered["axiom_num_nli"],
            labels=["1", "2", "4", "5", "others"],
            digits=4
        )
        print(f"Confusion Matrix:\n {cm}\n")
        print(f"Classification Report:\n {report}\n")
        
        
        # === Write to file =====================
        variables_sequences = []
        for idx, sample in tqdm(enumerate(sequences)):
            question_id = sample['id']
            question = sample['question']
            reference_answers = sample['answers']
            sequence_dict = {
                'id': question_id,
                'question': question,
                'answers': reference_answers
            }    
            
            if question_id in result_df_main_prompt_filtered['id'].values:  
                sequence_dict['groundedness_nli_main'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'groundedness_nli_main'].iloc[0]
                sequence_dict['groundedness_nli_second'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'groundedness_nli_second'].iloc[0]
                sequence_dict['axiom_num_nli'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'axiom_num_nli'].iloc[0]
            else:
                sequence_dict['groundedness_nli_main'] = (0, 0.0)
                sequence_dict['groundedness_nli_second'] = (0, 0.0)
                sequence_dict['axiom_num_nli'] = "not_common"
            
            variables_sequences.append(sequence_dict)
    
        # === Save the correctness result ============
        with open(axiomatic_variables_gn_file, 'wb') as ofile:
            pickle.dump(variables_sequences, ofile)
        print(f"Results saved to {axiomatic_variables_gn_file}")
    
    
    def get_axiomatic_variables_groundedness_kldiv():
        
        # === Load model ===================
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        tokenizer.pad_token_id = 1 # Very crucial don't forget
        
        
        # === Define functions =============
        def get_groundedness_kldiv(prompt_main, prompt_second, output):
            _generation = output[output != tokenizer.pad_token_id]
            if len(_generation) == 0:
                _generation = torch.tensor([tokenizer.encode('\n')[-1]]).to(args.device)
            
            prompt_main = prompt_main[prompt_main != tokenizer.pad_token_id]
            len_prompt_main = len(prompt_main)
            prompt_second = prompt_second[prompt_second != tokenizer.pad_token_id]
            len_prompt_second = len(prompt_second)
            
            # Main prompt
            p1_generation = torch.cat((prompt_main, _generation), dim=0)
            p1_target_ids = p1_generation.clone()
            p1_target_ids[:len_prompt_main] = -100            
            p1_model_output = model(torch.reshape(p1_generation, (1, -1)), labels=p1_target_ids, output_hidden_states=False)
            _p1_logits = p1_model_output['logits'][0, len_prompt_main-1:-1]
            _p1_logits = _p1_logits.float()
            p1_probs = torch.nn.functional.softmax(_p1_logits, dim=1)
            
            # Second prompt
            p2_generation = torch.cat((prompt_second, _generation), dim=0)
            p2_target_ids = p2_generation.clone()
            p2_target_ids[:len_prompt_second] = -100
            p2_model_output = model(torch.reshape(p2_generation, (1, -1)), labels=p2_target_ids, output_hidden_states=False)
            _p2_logits = p2_model_output['logits'][0, len_prompt_second-1:-1]
            _p2_logits = _p2_logits.float()
            p2_probs = torch.nn.functional.softmax(_p2_logits, dim=1)
            
            # calculate kl
            p1_probs = p1_probs / p1_probs.sum(dim=1, keepdim=True)
            p2_probs = p2_probs / p2_probs.sum(dim=1, keepdim=True)
            kl_divergence_values = torch.sum(p1_probs * torch.log(p1_probs / p2_probs), dim=1, keepdim=True)
            kl_divergence_values = kl_divergence_values.squeeze(dim=1) #(num_token, )
            kl_divergence_values = kl_divergence_values.cpu().detach().numpy()
            # 
            
            threshold = np.mean(kl_divergence_values) + np.std(kl_divergence_values)
            binarized_kl_divergence_values = [1 if x >= threshold else 0 for x in kl_divergence_values]
            proportion_of_ones = sum(binarized_kl_divergence_values) / len(binarized_kl_divergence_values) if len(binarized_kl_divergence_values) > 0 else 0
            pred_label = 1 if proportion_of_ones >= 0.20 else 0
            
            return (pred_label, proportion_of_ones)
        
        def get_axiom_number_kldiv(answer_equality, kldiv_main, kldiv_sec):
            pass


        # ==== KL-Div ======================
        result_df_main_prompt_filtered['groundedness_kldiv_main'] = [
            get_groundedness_kldiv(prompt_main, prompt_second, output)
            for prompt_main, prompt_second, output in tqdm(zip(
                result_df_main_prompt_filtered['prompt'],
                result_df_second_prompt_filtered['prompt'],
                result_df_main_prompt_filtered['cleaned_most_likely_generation_ids']
            ), desc='Getting groundedness KL-div (main) ...')
        ]
        result_df_main_prompt_filtered['groundedness_kldiv_second'] = [
            get_groundedness_kldiv(prompt_main, prompt_second, output)
            for prompt_main, prompt_second, output in tqdm(zip(
                result_df_main_prompt_filtered['prompt'],
                result_df_second_prompt_filtered['prompt'],
                result_df_second_prompt_filtered['cleaned_most_likely_generation_ids']
            ), desc='Getting groundedness KL-div (second) ...')
        ]
        
        # result_df_second_prompt_filtered = result_df_second_prompt_filtered.rename(columns={'exact_match': 'exact_match_second'})
        # result_df_second_prompt_filtered = result_df_second_prompt_filtered.rename(columns={'cleaned_most_likely_generation': 'cleaned_most_likely_generation_second'})
        # merged_df = pd.merge(result_df_main_prompt_filtered, result_df_second_prompt_filtered[['id', 'exact_match_second', 'cleaned_most_likely_generation_second']], on='id', how='left')

        # selected_columns = merged_df[['id', 'question', 'exact_match', 'exact_match_second', 'cleaned_most_likely_generation', 'cleaned_most_likely_generation_second', 'groundedness_kldiv_main', 'groundedness_kldiv_second']]
        # selected_columns.to_json('output.jsonl', orient='records', lines=True)
        
        # result_df_main_prompt_filtered['axiom_num_kldiv'] = [
        #     get_axiom_number_kldiv(answer_equality, nli_main, nli_sec)
        #     for answer_equality, nli_main, nli_sec in tqdm(zip(
        #         result_df_main_prompt_filtered['answer_equality_em'],
        #         result_df_main_prompt_filtered['groundedness_kldiv_main'],
        #         result_df_main_prompt_filtered['groundedness_kldiv_second']
        #     ), desc='Getting axiom number (KL-div) ...')
        # ]

        # === Write to file =====================
        variables_sequences = []
        for idx, sample in tqdm(enumerate(sequences)):
            question_id = sample['id']
            question = sample['question']
            reference_answers = sample['answers']
            sequence_dict = {
                'id': question_id,
                'question': question,
                'answers': reference_answers
            }
            
            if question_id in result_df_main_prompt_filtered['id'].values:
                sequence_dict['groundedness_kldiv_main'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'groundedness_kldiv_main'].iloc[0]
                sequence_dict['groundedness_kldiv_second'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'groundedness_kldiv_second'].iloc[0]
            else:
                sequence_dict['groundedness_kldiv_main'] = (0, 0.0)
                sequence_dict['groundedness_kldiv_second'] = (0, 0.0)
                
            variables_sequences.append(sequence_dict)
        
        
        # === Save the correctness result ============
        with open(axiomatic_variables_gk_file, 'wb') as ofile:
            pickle.dump(variables_sequences, ofile)
        print(f"Results saved to {axiomatic_variables_gk_file}")
        

    def get_axiomatic_variables_groundedness_minicheck():
        
        ### flan-t5-large  | Bespoke-MiniCheck-7B
        groundedness_scorer = MiniCheck(model_name='Bespoke-MiniCheck-7B', enable_prefix_caching=False)


        # === Define functions =======================
        def get_groundedness_minicheck(prompt_text, question, output_text):
            # === Prapare inputs
            doc_text = prompt_text.split('Document:')[-1].split('Question:')[0]
            answer_ = f"{question} {output_text}"
            
            pred_label, raw_prob, _, _ = groundedness_scorer.score(
                docs=[doc_text], claims=[answer_])

            return (pred_label[0], raw_prob[0])
        
        def get_axiom_number_minicheck(answer_equality, minicheck_main, minicheck_sec):
            axiom_num = 'others'
            if answer_equality and minicheck_main[0] == 1:
                axiom_num = '1'
            if answer_equality and minicheck_main[0] == 0:
                axiom_num = '2'
            if not answer_equality and minicheck_main[0] == 1 and minicheck_sec[0] == 0:
                axiom_num = '4'
            if not answer_equality and minicheck_main[0] == 0 and minicheck_sec[0] == 1:
                axiom_num = '5'
            
            return axiom_num


        # ===
        result_df_main_prompt_filtered['groundedness_minicheck_main'] = [
            get_groundedness_minicheck(prompt_text, question, output)
            for prompt_text, question, output in tqdm(zip(
                result_df_main_prompt_filtered['prompt_text'],
                result_df_main_prompt_filtered['question'],
                result_df_main_prompt_filtered['cleaned_most_likely_generation']
            ), desc='Getting groundedness Minicheck (main) ...')
        ]
        
        result_df_main_prompt_filtered['groundedness_minicheck_second'] = [
            get_groundedness_minicheck(prompt_text, question, output)
            for prompt_text, question, output in tqdm(zip(
                result_df_main_prompt_filtered['prompt_text'],
                result_df_main_prompt_filtered['question'],
                result_df_second_prompt_filtered['cleaned_most_likely_generation']
            ), desc='Getting groundedness Minicheck (second) ...')
        ]
        
        result_df_main_prompt_filtered['axiom_num_minicheck'] = [
            get_axiom_number_minicheck(answer_equality, minicheck_main, minicheck_sec)
            for answer_equality, minicheck_main, minicheck_sec in tqdm(zip(
                result_df_main_prompt_filtered['output_equality_em'],
                result_df_main_prompt_filtered['groundedness_minicheck_main'],
                result_df_main_prompt_filtered['groundedness_minicheck_second']
            ), desc='Getting axiom number (Minicheck) ...')
        ]
   
   
        # ===
        print("==== Minicheck ===================================")
        cm = confusion_matrix(
            result_df_main_prompt_filtered["axiom_num_correctness"],
            result_df_main_prompt_filtered["axiom_num_minicheck"],
            labels=["1", "2", "4", "5", "others"])
        # Compute classification report (Precision, Recall, F1-score)
        report = classification_report(
            result_df_main_prompt_filtered["axiom_num_correctness"],
            result_df_main_prompt_filtered["axiom_num_minicheck"],
            labels=["1", "2", "4", "5", "others"],
            digits=4
        )
        print(f"Confusion Matrix:\n {cm}\n")
        print(f"Classification Report:\n {report}\n")

        
        # === Write to file =====================
        variables_sequences = []
        for idx, sample in tqdm(enumerate(sequences)):
            question_id = sample['id']
            question = sample['question']
            reference_answers = sample['answers']
            sequence_dict = {
                'id': question_id,
                'question': question,
                'answers': reference_answers
            }
        
            if question_id in result_df_main_prompt_filtered['id'].values:
                sequence_dict['groundedness_minicheck_main'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'groundedness_minicheck_main'].iloc[0]
                sequence_dict['groundedness_minicheck_second'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'groundedness_minicheck_second'].iloc[0]
                sequence_dict['axiom_num_minicheck'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'axiom_num_minicheck'].iloc[0]
            else:
                sequence_dict['groundedness_minicheck_main'] = (0, 0.0)
                sequence_dict['groundedness_minicheck_second'] = (0, 0.0)
                sequence_dict['axiom_num_minicheck'] = "not_common"
                
            variables_sequences.append(sequence_dict)
        
        
        # === Save the correctness result ============
        with open(axiomatic_variables_gm_file, 'wb') as ofile:
            pickle.dump(variables_sequences, ofile)
        print(f"Results saved to {axiomatic_variables_gm_file}")
        
    
    # === Main process ===========================
    result_df_main_prompt = create_result_df(args.main_prompt_format, args.second_prompt_format)
    result_df_second_prompt = create_result_df(args.second_prompt_format, args.main_prompt_format)
    
    common_ids = pd.merge(result_df_main_prompt, result_df_second_prompt, on='id')['id']
    result_df_main_prompt_filtered = result_df_main_prompt[result_df_main_prompt['id'].isin(common_ids)]
    result_df_second_prompt_filtered = result_df_second_prompt[result_df_second_prompt['id'].isin(common_ids)]
    
    get_axiomatic_variables_output_equality()
    torch.cuda.empty_cache()
    get_axiomatic_variables_groundedness_nli()
    torch.cuda.empty_cache()
    get_axiomatic_variables_groundedness_kldiv()
    torch.cuda.empty_cache()
    get_axiomatic_variables_groundedness_minicheck()
    torch.cuda.empty_cache()
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.3')
    parser.add_argument('--dataset', type=str, default='nqgold', choices=[
        'nqgold', 'trivia', 'popqa', 'nqswap',
        'webquestions', 'squad1', 'nq',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa',
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--main_prompt_format', type=str, default='q_negative', choices=[
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
        'exact_match', 'rouge_score', 'bert_score', 'bem_score', 'llama3_score', 'gpt_score'
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
    
    if args.main_prompt_format != 'only_q':
        args.second_prompt_format == 'only_q'
    
    set_seed(args.seed)
    get_axiomatic_variables(args)
    
    
    # python framework/run/get_axiomatic_variables.py
    