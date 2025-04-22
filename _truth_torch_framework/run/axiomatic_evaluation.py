#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from minicheck.minicheck import MiniCheck
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

import TruthTorchLM as ttlm
from utils.utils import set_seed

import nltk
nltk.download('punkt_tab')

def axiomatic_evaluation(args):
    print("\n== Generation with truthfulness ...")
    print(f"""
        Model name:  {args.model}
        Dataset:     {args.dataset}/{args.subsec} ({args.fraction_of_data_to_use})
        Prompt:      {args.prompt_format}
        Correctness: {args.accuracy_metric}
        Seed:        {args.seed}
    """.replace('        ', ''))
    
    # === Input/Output files =====================
    model_ = args.model.split('/')[-1]
    
    generations_rag_input_file = f'{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.prompt_format}/generations.jsonl'
    uncertainties_rag_input_file = f'{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.prompt_format}/uncertainties.jsonl'
    generations_llm_input_file = f'{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/only_q/generations.jsonl'
    uncertainties_llm_input_file = f'{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/only_q/uncertainties.jsonl'
    axiomatic_results_output_file = f'{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.prompt_format}/axiomatic_results.jsonl'

    # === Define functions =======================
    def create_result_df(generations_input_file, uncertainties_input_file):
        
        with open(generations_input_file, 'r', encoding='utf-8') as f:
            generations_df = pd.read_json(f, lines=True)
        with open(uncertainties_input_file, 'r', encoding='utf-8') as f:
            uncertainties_df = pd.read_json(f, lines=True)
        
        generations_keys_to_use = ('qid', 'question', 'ground_truths', 'generation_text_most_likely', 'correctness', 'prompt_text', 'samples_generation_text')
        uncertainties_keys_to_use = ('qid', 'Pt', 'Conf', 'PE', 'SE', 'EigV')
        generations_small = dict((k, generations_df[k]) for k in generations_keys_to_use)
        generations_df = pd.DataFrame.from_dict(generations_small)
        uncertainties_small = dict((k, uncertainties_df[k]) for k in uncertainties_keys_to_use)
        uncertainties_df = pd.DataFrame.from_dict(uncertainties_small)
        result_df = generations_df.merge(uncertainties_df, on='qid')
        result_df.rename(columns={'generation_text_most_likely': 'prediction'}, inplace=True)
        
        sorted_result_df = result_df.sort_values(by='qid', key=lambda x: x.str.extract(r'q_(\d+)', expand=False).astype(int))
        
        return sorted_result_df

    def get_axiomatic_variables_output_equality():
        
        # === Load semantic model ===================
        # - Labels: {0: Contradiction, 1: Neutral, 2: Entailment}
        semantic_model_name = "microsoft/deberta-large-mnli"
        semantic_model = AutoModelForSequenceClassification.from_pretrained(semantic_model_name).to(args.device)
        semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
        semantic_model.eval()
        
        
        # === Define functions ======================
        def get_output_equality_spectral_distance(question, rag_generations, llm_generations):
            W_rag = ttlm.utils.calculate_affinity_matrix(rag_generations, question, model_for_entailment=semantic_model, tokenizer_for_entailment=semantic_tokenizer)
            W_llm = ttlm.utils.calculate_affinity_matrix(llm_generations, question, model_for_entailment=semantic_model, tokenizer_for_entailment=semantic_tokenizer)
            L_rag = ttlm.utils.get_L_mat(W_rag)
            L_llm = ttlm.utils.get_L_mat(W_llm)
            eigvals_rag = np.linalg.eigvalsh(L_rag)
            eigvals_llm = np.linalg.eigvalsh(L_llm)
            distance = np.linalg.norm(eigvals_llm - eigvals_rag)
            return distance
            
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
                axiom_num = '3'
            if not answer_equality and not correctness_main and correctness_sec:
                axiom_num = '4'
            if not answer_equality and correctness_main and correctness_sec:
                axiom_num = '1'
            
            return axiom_num
    
        # ===
        
        rag_results_df['output_equality_sd'] = [
            get_output_equality_spectral_distance(question, generations_rag, generations_llm)
            for question, generations_rag, generations_llm in tqdm(zip(
                rag_results_df['question'],
                rag_results_df['samples_generation_text'],
                llm_results_df['samples_generation_text'],
            ), desc='Getting output equality (SD) ...')
        ]
        # sd_counts = rag_results_df['output_equality_sd'].value_counts()
        # print(f"OE-SD (equal): {sd_counts.get(True, 0)}")
        # print(f"OE-SD (not equal): {sd_counts.get(False, 0)}")
        
        
        rag_results_df['output_equality_em'] = [
            get_output_equality_em(seq1, seq2)
            for seq1, seq2 in tqdm(zip(
                rag_results_df['prediction'], 
                llm_results_df['prediction']
            ), desc='Getting output equality (EM) ...')
        ]
        em_counts = rag_results_df['output_equality_em'].value_counts()
        print(f"OE-EM (equal): {em_counts.get(True, 0)}")
        print(f"OE-EM (not equal): {em_counts.get(False, 0)}")
        
        rag_results_df['output_equality_nli'] = [
            get_output_equality_nli(question, seq1, seq2)
            for question, seq1, seq2 in tqdm(zip(
                rag_results_df['question'],
                rag_results_df['prediction'], 
                llm_results_df['prediction']
            ), desc='Getting output equality (NLI) ...')
        ]
        nli_counts = rag_results_df['output_equality_nli'].apply(lambda x: x[0]).value_counts()
        print(f"OE-NLI (equal): {nli_counts.get(True, 0)}")
        print(f"OE-NLI (not equal): {nli_counts.get(False, 0)}")

        # ==== Correctness ========================
        rag_results_df['axiom_num_correctness'] = [
            get_axiom_number_correctness(answer_equality, correctness_main, correctness_sec)
            for answer_equality, correctness_main, correctness_sec in tqdm(zip(
                rag_results_df['output_equality_em'],
                rag_results_df['correctness'],
                llm_results_df['correctness']
            ), desc='Getting axiom number (EM) ...')
        ]

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
        rag_results_df['groundedness_kldiv_rag'] = [
            get_groundedness_kldiv(prompt_main, prompt_second, output)
            for prompt_main, prompt_second, output in tqdm(zip(
                rag_results_df['prompt_text'],
                llm_results_df['prompt_text'],
                rag_results_df['prediction']
            ), desc='Getting groundedness KL-div (RAG) ...')
        ]
        rag_results_df['groundedness_kldiv_llm'] = [
            get_groundedness_kldiv(prompt_main, prompt_second, output)
            for prompt_main, prompt_second, output in tqdm(zip(
                rag_results_df['prompt_text'],
                llm_results_df['prompt_text'],
                llm_results_df['prediction']
            ), desc='Getting groundedness KL-div (LLM) ...')
        ]

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
                axiom_num = '3'
            if not answer_equality and nli_main[0] == 0 and nli_sec[0] == 2:
                axiom_num = '4'
            
            return axiom_num
    
        # ==== NLI ==================================
        rag_results_df['groundedness_nli_rag'] = [
            get_groundedness_nli(prompt_text, question, output)
            for prompt_text, question, output in tqdm(zip(
                rag_results_df['prompt_text'],
                rag_results_df['question'],
                rag_results_df['prediction']
            ), desc='Getting groundedness NLI (RAG) ...')
        ]
        
        rag_results_df['groundedness_nli_llm'] = [
            get_groundedness_nli(prompt_text, question, output)
            for prompt_text, question, output in tqdm(zip(
                rag_results_df['prompt_text'],
                rag_results_df['question'],
                llm_results_df['prediction']
            ), desc='Getting groundedness NLI (LLM) ...')
        ]
        
        rag_results_df['axiom_num_nli'] = [
            get_axiom_number_nli(answer_equality, nli_main, nli_sec)
            for answer_equality, nli_main, nli_sec in tqdm(zip(
                rag_results_df['output_equality_em'],
                rag_results_df['groundedness_nli_rag'],
                rag_results_df['groundedness_nli_llm']
            ), desc='Getting axiom number (NLI) ...')
        ]

        # # ======
        # print("==== NLI ========================================")
        # cm = confusion_matrix(
        #     rag_results_df["axiom_num_correctness"],
        #     rag_results_df["axiom_num_nli"],
        #     labels=["1", "2", "3", "4", "others"])
        # # Compute classification report (Precision, Recall, F1-score)
        # report = classification_report(
        #     rag_results_df["axiom_num_correctness"],
        #     rag_results_df["axiom_num_nli"],
        #     labels=["1", "2", "3", "4", "others"],
        #     digits=4
        # )
        # print(f"Confusion Matrix:\n {cm}\n")
        # print(f"Classification Report:\n {report}\n")

    def get_axiomatic_variables_groundedness_minicheck():
        
        ### flan-t5-large  | Bespoke-MiniCheck-7B
        groundedness_scorer = MiniCheck(model_name='Bespoke-MiniCheck-7B', enable_prefix_caching=False)

        # === Define functions =======================
        def get_groundedness_minicheck(prompt_text, question, output_text):
            doc_text = prompt_text.split('Document:')[-1].split('Question:')[0]
            answer_ = f"{question} {output_text}"
            
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
                axiom_num = '3'
            if not answer_equality and minicheck_main[0] == 0 and minicheck_sec[0] == 1:
                axiom_num = '4'
            
            return axiom_num


        # ===
        rag_results_df['groundedness_minicheck_rag'] = [
            get_groundedness_minicheck(prompt_text, question, output)
            for prompt_text, question, output in tqdm(zip(
                rag_results_df['prompt_text'],
                rag_results_df['question'],
                rag_results_df['prediction']
            ), desc='Getting groundedness Minicheck (RAG) ...')
        ]
        
        rag_results_df['groundedness_minicheck_llm'] = [
            get_groundedness_minicheck(prompt_text, question, output)
            for prompt_text, question, output in tqdm(zip(
                rag_results_df['prompt_text'],
                rag_results_df['question'],
                llm_results_df['prediction']
            ), desc='Getting groundedness Minicheck (LLM) ...')
        ]
        
        # rag_results_df['axiom_num_minicheck'] = [
        #     get_axiom_number_minicheck(answer_equality, minicheck_main, minicheck_sec)
        #     for answer_equality, minicheck_main, minicheck_sec in tqdm(zip(
        #         rag_results_df['output_equality_em'],
        #         rag_results_df['groundedness_minicheck_rag'],
        #         rag_results_df['groundedness_minicheck_llm']
        #     ), desc='Getting axiom number (Minicheck) ...')
        # ]
   
        # ===
        # print("==== Minicheck ===================================")
        # cm = confusion_matrix(
        #     rag_results_df["axiom_num_correctness"],
        #     rag_results_df["axiom_num_minicheck"],
        #     labels=["1", "2", "3", "4", "others"])
        # # Compute classification report (Precision, Recall, F1-score)
        # report = classification_report(
        #     rag_results_df["axiom_num_correctness"],
        #     rag_results_df["axiom_num_minicheck"],
        #     labels=["1", "2", "3", "4", "others"],
        #     digits=4
        # )
        # print(f"Confusion Matrix:\n {cm}\n")
        # print(f"Classification Report:\n {report}\n")
    # ============================================ 

    if os.path.isfile(axiomatic_results_output_file):
        print(f"The file '{axiomatic_results_output_file}' exists.")
        llm_results_df = create_result_df(generations_llm_input_file, uncertainties_llm_input_file)
        rag_results_df_ = create_result_df(generations_rag_input_file, uncertainties_rag_input_file)
        
        with open(axiomatic_results_output_file, 'r', encoding='utf-8') as f:
            rag_results_df = pd.read_json(f, lines=True)
        
        rag_results_df['PE'] = rag_results_df_['PE']
        rag_results_df['SE'] = rag_results_df_['SE']
        rag_results_df['EigV'] = rag_results_df_['EigV']
        rag_results_df['correctness'] = rag_results_df_['correctness']
        
    else:
        print(f"The file '{axiomatic_results_output_file}' does not exist.")
        
        rag_results_df = create_result_df(generations_rag_input_file, uncertainties_rag_input_file)
        llm_results_df = create_result_df(generations_llm_input_file, uncertainties_llm_input_file)
        
        get_axiomatic_variables_output_equality()
        # get_axiomatic_variables_groundedness_kldiv()
        get_axiomatic_variables_groundedness_nli()
        get_axiomatic_variables_groundedness_minicheck()
        
        columns_to_save = [
            'qid', 'question', 'ground_truths', 'prediction',
            'output_equality_em', 'output_equality_nli', 'output_equality_sd',
            # 'groundedness_kldiv_rag', 'groundedness_kldiv_llm',
            'groundedness_nli_rag', 'groundedness_nli_llm', 'axiom_num_nli',
            'groundedness_minicheck_rag', 'groundedness_minicheck_llm'
        ]
        rag_results_df_selected = rag_results_df[columns_to_save]
        rag_results_df_selected.to_json(axiomatic_results_output_file, orient='records', lines=True)


    
    
    # rag_results_df['samples_generation_text_llm'] = llm_results_df['samples_generation_text']
    rag_results_df['PE_llm'] = llm_results_df['PE']
    
    groundedness_method = 'nli'
    total_samples = len(rag_results_df)
    stats = rag_results_df.groupby(f'axiom_num_{groundedness_method}').agg(
        p_samples=(f'axiom_num_{groundedness_method}', lambda x: (len(x) / total_samples) * 100),
        correctness_mean=('correctness', 'mean'),
        pe_llm_mean=('PE_llm', 'mean'),
        pe_mean=('PE', 'mean'),
        se_mean=('SE', 'mean'),
        EigV_mean=('EigV', 'mean'),
        OE_EM=('output_equality_em', 'mean'),
        # OE_mean=('output_equality_nli', lambda x: sum(val[1] for val in x) / len(x)),
        rag_putility=(f'groundedness_{groundedness_method}_rag', lambda x: sum(val[1] for val in x) / len(x)),
        llm_putility=(f'groundedness_{groundedness_method}_llm', lambda x: sum(val[1] for val in x) / len(x)),
        OE_SD=('output_equality_sd', 'mean')
    ).reset_index()
    print(stats)
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--dataset', type=str, default='popqa', choices=[
        'nqgold', 'trivia', 'popqa',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'webquestions', 'squad1', 'nq', 'nqswap',
        'topicoqa',
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--prompt_format', type=str, default='bm25_retriever_top1', choices=[
        'only_q', 'q_positive', 'q_negative', 'q_conflict',
        'bm25_retriever_top1', 'bm25_retriever_top5',
        'contriever_retriever_top1', 'contriever_retriever_top5',
        'rerank_retriever_top1', 'rerank_retriever_top5'
    ])
    parser.add_argument('--accuracy_metric', type=str, default="exact_match", choices=[
        'exact_match', 'model_judge', 'bem_score', 'bert_score', 'rouge_score'
    ])
    parser.add_argument('--model_eval', type=str, default='gpt-3.5-turbo') # meta-llama/Llama-3.1-8B-Instruct
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    parser.add_argument("--roc_auc_threshold", type=float, default=0.8)
    parser.add_argument('--num_generations', type=int, default=10)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--decoding_method', type=str, default='beam_search')
    parser.add_argument('--temperature', type=float, default='1.0')
    parser.add_argument('--num_beams', type=int, default='1')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_2 (500s-EM)')
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    
    ### === Define CUDA device =================== 
    args.output_dir = f"_truth_torch_framework/run_output/{args.run}"
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected.")
        
    
    ### === Run Steps ============================
    set_seed(args.seed)
    axiomatic_evaluation(args)
    
    
    # python _truth_torch_framework/run/axiomatic_evaluation.py
