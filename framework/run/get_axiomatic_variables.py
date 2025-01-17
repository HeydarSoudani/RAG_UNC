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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    model = args.model.split('/')[-1]
    generation_type = f"prob_alpha_{str(args.alpha_probability)}"
    base_dir = f'{args.output_dir}/{args.dataset}/{args.subsec}/{args.run_id}'
    
    # inputs
    sequence_input = f'{base_dir}/{args.main_prompt_format}__{args.second_prompt_format}/{model}_cleaned_generation_{args.generation_type}.pkl'
    # outputs
    axiomatic_variables_file = f'{base_dir}/{args.main_prompt_format}__{args.second_prompt_format}/{generation_type}/{model}_axiomatic_variables.pkl'
    
    with open(sequence_input, 'rb') as infile:
        sequences = pickle.load(infile)
    
    # === Load semantic model ===================
    # - Labels: {0: Contradiction, 1: Neutral, 2: Entailment}
    semantic_model_name = "microsoft/deberta-large-mnli"
    semantic_model = AutoModelForSequenceClassification.from_pretrained(semantic_model_name).to(args.device)
    semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
    semantic_model.eval()
    
    # === Define functions =======================
    def create_result_df(main_prompt_format, second_prompt_format):
        
        # For only query case
        results_dir = f'{base_dir}/{main_prompt_format}__{second_prompt_format}'
        if not os.path.isdir(results_dir):
            temp = 'bm25_retriever_top1' if args.dataset == 'popqa' else 'q_positive'
            results_dir = f'{base_dir}/{main_prompt_format}__{temp}'
        
        generation_file = f'{results_dir}/{model}_cleaned_generation_{args.generation_type}.pkl'
        similarities_input_file = f'{results_dir}/{model}_similarities_generation.pkl'
        correctness_input_file = f'{results_dir}/{model}_correctness.pkl'
        uncertainty_mars_input_file = f'{results_dir}/{generation_type}/{model}_uncertainty_mars_generation.pkl'
        
        with open(generation_file, 'rb') as infile:
            cleaned_sequences = pickle.load(infile)
        with open(similarities_input_file, 'rb') as f:
            similarities_dict = pickle.load(f)
        with open(uncertainty_mars_input_file, 'rb') as f:
            uncertainty_mars_results  = pickle.load(f)
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
            'average_predictive_entropy_third_prompt', 'predictive_entropy_over_concepts_third_prompt',
            'average_predictive_entropy_importance_max_third_prompt', 'predictive_entropy_over_concepts_importance_max_third_prompt',
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
        result_df = generations_df.merge(similarities_df, on='id').merge(uncertainty_mars_df, on='id').merge(correctness_df, on='id')
        result_df['len_most_likely_generation_length'] = result_df['most_likely_generation'].apply(lambda x: len(x.split()))
        return result_df
        
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
        
    def get_nli_relation(prompt_text, question, output_text):
        
        # === Prapare inputs
        doc_text = prompt_text.split('Document:')[-1].split('Question:')[0]
        answer_ = f"{question} {output_text}"
        
        # === Common NLI: Similar to semantic semilarity
        input = doc_text + ' [SEP] ' + answer_
        encoded_input = semantic_tokenizer.encode(input, padding=True)
        prediction = semantic_model(torch.tensor(torch.tensor([encoded_input]), device=args.device))['logits']
        predicted_label = torch.argmax(prediction, dim=1)
        
        reverse_input = answer_ + ' [SEP] ' + doc_text
        encoded_reverse_input = semantic_tokenizer.encode(reverse_input, padding=True)
        reverse_prediction = semantic_model(torch.tensor(torch.tensor([encoded_reverse_input]), device=args.device))['logits']
        reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
        
        # === Get label
        nli_label = 0 if (0 in predicted_label or 0 in reverse_predicted_label) else 2
        prediction_dist = torch.softmax(prediction, dim=1).tolist()[0]
        reverse_prediction_dist = torch.softmax(reverse_prediction, dim=1).tolist()[0]
        entail_score = max(prediction_dist[1], prediction_dist[2], reverse_prediction_dist[1], reverse_prediction_dist[2])
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
    
    def get_axiomatic_coef(answer_equality_nli, nli_main, nli_sec, coefs=(0.33, 0.33, 0.33)):
        C1, C2, C3 = coefs[0], coefs[1], coefs[2]
        # first_part = 1.0 if answer_equality else 0.0
        # second_part = 1.0 if nli_main[0]==2 else 0.0
        return C1*answer_equality_nli[1] + C2*nli_main[1] + C3*nli_sec[1]
    
    # === Main process ===========================
    result_df_main_prompt = create_result_df(args.main_prompt_format, args.second_prompt_format)
    result_df_second_prompt = create_result_df(args.second_prompt_format, args.main_prompt_format)
    
    common_ids = pd.merge(result_df_main_prompt, result_df_second_prompt, on='id')['id']
    result_df_main_prompt_filtered = result_df_main_prompt[result_df_main_prompt['id'].isin(common_ids)]
    result_df_second_prompt_filtered = result_df_second_prompt[result_df_second_prompt['id'].isin(common_ids)]
    
    result_df_main_prompt_filtered['answer_equality_em'] = [
        get_output_equality_em(seq1, seq2)
        for seq1, seq2 in tqdm(zip(
            result_df_main_prompt_filtered['cleaned_most_likely_generation'], 
            result_df_second_prompt_filtered['cleaned_most_likely_generation']
        ), desc='Getting output equality (EM) ...')
    ]
    em_counts = result_df_main_prompt_filtered['answer_equality_em'].value_counts()
    print(f"AE-EM (equal): {em_counts.get(True, 0)}")
    print(f"AE-EM (not equal): {em_counts.get(False, 0)}")
    
    result_df_main_prompt_filtered['answer_equality_nli'] = [
        get_output_equality_nli(question, seq1, seq2)
        for question, seq1, seq2 in tqdm(zip(
            result_df_main_prompt_filtered['question'],
            result_df_main_prompt_filtered['cleaned_most_likely_generation'], 
            result_df_second_prompt_filtered['cleaned_most_likely_generation']
        ), desc='Getting output equality (NLI) ...')
    ]
    nli_counts = result_df_main_prompt_filtered['answer_equality_nli'].apply(lambda x: x[0]).value_counts()
    print(f"AE-NLI (equal): {nli_counts.get(True, 0)}")
    print(f"AE-NLI (not equal): {nli_counts.get(False, 0)}")

    result_df_main_prompt_filtered['nli_relation_main'] = [
        get_nli_relation(prompt_text, question, output)
        for prompt_text, question, output in tqdm(zip(
            result_df_main_prompt_filtered['prompt_text'],
            result_df_main_prompt_filtered['question'],
            result_df_main_prompt_filtered['cleaned_most_likely_generation']
        ), desc='Getting NLI relations (main) ...')
    ]
    
    result_df_main_prompt_filtered['nli_relation_second'] = [
        get_nli_relation(prompt_text, question, output)
        for prompt_text, question, output in tqdm(zip(
            result_df_main_prompt_filtered['prompt_text'],
            result_df_main_prompt_filtered['question'],
            result_df_second_prompt_filtered['cleaned_most_likely_generation']
        ), desc='Getting NLI relations (second) ...')
    ]
    
    result_df_main_prompt_filtered['axiom_num_nli'] = [
        get_axiom_number_nli(answer_equality, nli_main, nli_sec)
        for answer_equality, nli_main, nli_sec in tqdm(zip(
            result_df_main_prompt_filtered['answer_equality_em'],
            result_df_main_prompt_filtered['nli_relation_main'],
            result_df_main_prompt_filtered['nli_relation_second']
        ), desc='Getting axiom number (NLI) ...')
    ]
    
    result_df_main_prompt_filtered['axiom_num_correctness'] = [
        get_axiom_number_correctness(answer_equality, correctness_main, correctness_sec)
        for answer_equality, correctness_main, correctness_sec in tqdm(zip(
            result_df_main_prompt_filtered['answer_equality_em'],
            result_df_main_prompt_filtered['exact_match'],
            result_df_second_prompt_filtered['exact_match']
        ), desc='Getting axiom number (EM) ...')
    ]
    
    result_df_main_prompt_filtered['axiomatic_coef'] = [
        get_axiomatic_coef(answer_equality_nli, nli_main, nli_sec)
        for answer_equality_nli, nli_main, nli_sec in tqdm(zip(
            result_df_main_prompt_filtered['answer_equality_nli'],
            result_df_main_prompt_filtered['nli_relation_main'],
            result_df_main_prompt_filtered['nli_relation_second']
        ), desc='Getting axiomatic coef. ...')
    ]
    
    # Grid search for C1, C2, C3
    # TODO
    
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

    # Display results
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    
    
    # print(result_df_main_prompt_filtered[['id', 'exact_match', 'cleaned_most_likely_generation', 'answer_equality', 'nli_relation_main', 'nli_relation_second', 'axiom_num', 'axiomatic_coef']])
    # print('\n')
    # print(result_df_second_prompt_filtered[['id', 'exact_match', 'cleaned_most_likely_generation']])

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
            sequence_dict['answer_equality_em'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'answer_equality_em'].iloc[0]
            sequence_dict['answer_equality_nli'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'answer_equality_nli'].iloc[0]
            sequence_dict['nli_relation_main'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'nli_relation_main'].iloc[0]
            sequence_dict['nli_relation_second'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'nli_relation_second'].iloc[0]
            sequence_dict['axiom_num_nli'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'axiom_num_nli'].iloc[0]
            sequence_dict['axiom_num_correctness'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'axiom_num_correctness'].iloc[0]
            sequence_dict['axiomatic_coef'] = result_df_main_prompt_filtered.loc[result_df_main_prompt_filtered['id'] == question_id, 'axiomatic_coef'].iloc[0]
        else:
            sequence_dict['answer_equality_em'] = False
            sequence_dict['answer_equality_nli'] = (False, 0.0)
            sequence_dict['nli_relation_main'] = (0, 0.0)
            sequence_dict['nli_relation_second'] = (0, 0.0)
            sequence_dict['axiom_num_nli'] = "not_common"
            sequence_dict['axiom_num_correctness'] = "not_common"
            sequence_dict['axiomatic_coef'] = 0.0
    
        variables_sequences.append(sequence_dict)

    ### === Save the correctness result ============
    with open(axiomatic_variables_file, 'wb') as ofile:
        pickle.dump(variables_sequences, ofile)
    print(f"Results saved to {axiomatic_variables_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--dataset', type=str, default='popqa', choices=[
        'nqgold', 'trivia', 'popqa', 'nqswap',
        'webquestions', 'squad1', 'nq',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa',
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--main_prompt_format', type=str, default='rerank_retriever_top1', choices=[
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
    