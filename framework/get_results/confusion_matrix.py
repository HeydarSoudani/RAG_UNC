
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from utils.utils import set_seed
UNC_THERESHOLD = 1000
FOLDER_PATH = 'framework/plots/imgs'

def main(args):
    model = args.model.split('/')[-1]
    generation_type = f"prob_alpha_{str(args.alpha_probability)}"
    
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
        
        if main_prompt_format != 'only_q':
            axiomatic_variables_input_file = f'{results_dir}/{generation_type}/{model}_axiomatic_variables.pkl'
            with open(axiomatic_variables_input_file, 'rb') as f:
                axiomatic_variables_results  = pickle.load(f)
            axiomatic_variables_df = pd.DataFrame(axiomatic_variables_results)
            result_df = generations_df.merge(similarities_df, on='id').merge(uncertainty_mars_df, on='id').merge(correctness_df, on='id').merge(axiomatic_variables_df, on='id')
        else:
            result_df = generations_df.merge(similarities_df, on='id').merge(uncertainty_mars_df, on='id').merge(correctness_df, on='id')
        # result_df = generations_df.merge(similarities_df, on='id').merge(uncertainty_mars_df, on='id').merge(correctness_df, on='id')
        
        result_df['len_most_likely_generation_length'] = result_df['most_likely_generation'].apply(lambda x: len(x.split()))
        return result_df
    
    labels=["1", "2", "4", "5", "others"]
    datasets = [('nqgold', 'test'), ('trivia', 'dev'), ('popqa', 'test')]
    retrieval_models = ['q_negative', 'bm25_retriever_top1', 'contriever_retriever_top1', 'rerank_retriever_top1', 'q_positive']

    fig, axes = plt.subplots(len(datasets), len(retrieval_models), figsize=(15, 9))
    
    for i, (dataset_name, subsec) in enumerate(datasets):
        print(f"Processing {dataset_name} dataset ...")
        base_dir = f'{args.output_dir}/{dataset_name}/{subsec}/{args.run_id}'
        
        ret_list = retrieval_models[:-1] if dataset_name == 'popqa' else retrieval_models
        for j, model_name in enumerate(ret_list):
            result_df_main_prompt = create_result_df(model_name, args.second_prompt_format)
            
            y_true = result_df_main_prompt["axiom_num_correctness"]
            y_pred = result_df_main_prompt["axiom_num_nli"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            
            ax = axes[i, j] if len(datasets) > 1 else axes[j]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)

            if i == 0:
                ax.set_title(f'{model_name}')
        
        ax[i, 0].set_ylabel(f'{dataset_name}')
        # ax.set_ylabel('True')

    plt.tight_layout()
    plt.savefig(f'{FOLDER_PATH}/cm.png')
    plt.savefig(f'{FOLDER_PATH}/cm.pdf', format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--dataset', type=str, default='nqgold', choices=[
        'nqgold', 'trivia', 'popqa', 'nqswap',
        'webquestions', 'squad1', 'nq',
        '2wikimultihopqa', 'hotpotqa', 'musique',
        'topicoqa',
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--main_prompt_format', type=str, default='bm25_retriever_top1', choices=[
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
    main(args)
  
    # python framework/plots/confusion_matrix.py
