
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


UNC_THERESHOLD = 1000
FOLDER_INPUT_PATH = 'framework/plots/results'
FOLDER_OUTPUT_PATH = 'framework/plots/imgs'

def main(args):
    print("\n--- Plot ...")
    print(f"""
        Model name:   {args.model}
        Run id:       {args.run_id}
        Seed:         {args.seed}
    """.replace('        ', ''))
    
    # === Define output files ===================
    model_ = args.model.split('/')[-1]
    generation_type = f"prob_alpha_{str(args.alpha_probability)}"
    
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
        # 'third_prompt': {
        #     'PE': 'average_predictive_entropy_third_prompt',
        #     'SE': 'predictive_entropy_over_concepts_third_prompt',
        #     'PE_MARS': 'average_predictive_entropy_importance_max_third_prompt',
        #     'SE_MARS': 'predictive_entropy_over_concepts_importance_max_third_prompt'
        # },
        # 'forth_prompt': {
        #     'PE': 'average_predictive_entropy_forth_prompt',
        #     'SE': 'predictive_entropy_over_concepts_forth_prompt',
        #     'PE_MARS': 'average_predictive_entropy_importance_max_forth_prompt',
        #     'SE_MARS': 'predictive_entropy_over_concepts_importance_max_forth_prompt'
        # },
        # 'fifth_prompt': {
        #     'PE': 'average_predictive_entropy_forth_prompt',
        #     'SE': 'predictive_entropy_over_concepts_forth_prompt',
        #     'PE_MARS': 'average_predictive_entropy_importance_max_forth_prompt',
        #     'SE_MARS': 'predictive_entropy_over_concepts_importance_max_forth_prompt'
        # }
         
    }
    
    def create_result_df(main_prompt_format, second_prompt_format):
        
        results_dir = f'{base_dir}/{main_prompt_format}__{second_prompt_format}'
        if not os.path.isdir(results_dir):
            temp = 'bm25_retriever_top1' if args.dataset == 'popqa' else 'q_positive'
            results_dir = f'{base_dir}/{main_prompt_format}__{temp}'
        
        generation_file = f'{results_dir}/cleaned_generation_{args.generation_type}.pkl'
        similarities_input_file = f'{results_dir}/similarities_generation.pkl'
        correctness_input_file = f'{results_dir}/correctness.pkl'
        uncertainty_mars_input_file = f'{results_dir}/{generation_type}/uncertainty_mars_generation.pkl'
        uncertainty_bb_input_file = f'{results_dir}/uncertainty_bb_generation.pkl'
        axiomatic_variables_input_file = f'{results_dir}/{generation_type}/axiomatic_variables.pkl'
        
        with open(generation_file, 'rb') as infile:
            cleaned_sequences = pickle.load(infile)
        with open(similarities_input_file, 'rb') as f:
            similarities_dict = pickle.load(f)
        with open(correctness_input_file, 'rb') as f:
            correctness_results  = pickle.load(f)
        with open(uncertainty_mars_input_file, 'rb') as f:
            uncertainty_mars_results  = pickle.load(f)
        with open(uncertainty_bb_input_file, 'rb') as f:
            uncertainty_bb_results  = pickle.load(f)
        
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
        uncertainty_bb_df = pd.DataFrame(uncertainty_bb_results)
        uncertainty_bb_keys_to_use = ('id', 'degree_u', 'ecc_u', 'spectral_u')
        uncertainty_bb_small = dict((k, uncertainty_bb_df[k]) for k in uncertainty_bb_keys_to_use)
        uncertainty_bb_df = pd.DataFrame.from_dict(uncertainty_bb_small)
        
        # 
        if main_prompt_format != 'only_q':
            with open(axiomatic_variables_input_file, 'rb') as f:
                axiomatic_variables_results  = pickle.load(f)
            axiomatic_variables_df = pd.DataFrame(axiomatic_variables_results)
            result_df = generations_df.merge(similarities_df, on='id').merge(uncertainty_mars_df, on='id').merge(uncertainty_bb_df, on='id').merge(correctness_df, on='id').merge(axiomatic_variables_df, on='id')
        else:
            result_df = generations_df.merge(similarities_df, on='id').merge(uncertainty_mars_df, on='id').merge(uncertainty_bb_df, on='id').merge(correctness_df, on='id')
        
        # 
        result_df['len_most_likely_generation_length'] = result_df['most_likely_generation'].apply(lambda x: len(x.split()))
        return result_df
    
    def get_axiomatic_coef(answer_equality_nli, nli_main, nli_sec, coefs=(0.33, 0.33, 0.33)):
        C1, C2, C3 = coefs[0], coefs[1], coefs[2]
        return C1 * answer_equality_nli[1] + C2 * nli_main[1] + C3 * nli_sec[1]

    # === Plot ==================================    
    prompt_order = 'main'
    uncertainty_model = 'SE' # 'PE', 'SE', 'PE_MARS', 'SE_MARS', 'degree_u', 'ecc_u', 'spectral_u'
    if uncertainty_model in ['PE', 'SE', 'PE_MARS', 'SE_MARS']:
        unc_model_key_main_prompt = keys_mapping[f'{prompt_order}_prompt'][uncertainty_model]
    elif uncertainty_model in ['degree_u', 'ecc_u', 'spectral_u']:
        unc_model_key_main_prompt = uncertainty_model
    
    datasets = [
        {"dataset": 'nqgold', "subsec": 'test'},
        {"dataset": 'trivia', "subsec": 'dev'},
        {"dataset": 'popqa', "subsec": 'test'}
    ]
    datasets2title = {
        'nqgold': 'NQ-open',
        'trivia': 'TriviaQA',
        'popqa': 'PopQA'
    }
    
    retrievers = [ 
        'bm25_retriever_top1',
        'contriever_retriever_top1',
        'rerank_retriever_top1',
        'q_positive'
    ]
    retrievers2title = {
        'bm25_retriever_top1': "BM25",
        'contriever_retriever_top1': "Contriever",
        'rerank_retriever_top1': "Re-rank",
        'q_positive': "Doc+"
    }
    
    labels = ['A1', 'A2', 'A3', 'A4', 'Other'] # 'A1', 'A2', 'A4', 'A5', 'others'
    metrics = ['coef_mean', 'accuracy', 'p_samples']
    metrics2title = {
        'coef_mean': 'Calibration Coef.',
        'accuracy': 'Accuracy',
        'p_samples': '% Samples'
    }
    
    colors = ['cornflowerblue', 'gold', 'hotpink'] # 'mediumseagreen',
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, axes = plt.subplots(len(datasets), len(retrievers), subplot_kw={'projection': 'polar'}, figsize=(len(retrievers)*3.1, len(datasets)*3))
    if len(datasets) == 1:
        axes = np.atleast_2d(axes)

    
    for i, dataset in enumerate(datasets):
        base_dir = f"{args.output_dir}/{model_}/{dataset['dataset']}/{dataset['subsec']}/{args.run_id}/"
        
        for j, retriever in enumerate(retrievers):
            print(f"=== {dataset['dataset']}, {retriever} =================")
            
            # Skip positive for popQA
            if i == 2 and j == 3:  # Skip last dataset's missing model
                axes[i, j].axis('off')  # Hide subplot
                continue
            
            
            # =============================================
            result_df_main_prompt = create_result_df(retriever, args.second_prompt_format)    
            result_df_main_prompt['axiomatic_coef'] = [
                get_axiomatic_coef(answer_equality_nli, nli_main, nli_sec, coefs=(0.4, 0.6, 0.0))
                for answer_equality_nli, nli_main, nli_sec in tqdm(zip(
                    result_df_main_prompt['answer_equality_nli'],
                    result_df_main_prompt['nli_relation_main'],
                    result_df_main_prompt['nli_relation_second']
                ), desc='Getting axiomatic coef. ...')
            ]
            result_df_main_prompt_filtered = result_df_main_prompt[result_df_main_prompt[unc_model_key_main_prompt] < UNC_THERESHOLD]
            total_rows = len(result_df_main_prompt_filtered)
            print(f"= Len (filtered): {total_rows}")
            
            # =============================================
            result = result_df_main_prompt_filtered.groupby('axiom_num_nli').agg(
                accuracy=('exact_match', lambda x: x.sum() / len(x)),
                average_uncertainty=(unc_model_key_main_prompt, 'mean'),
                n_samples=(unc_model_key_main_prompt, 'count'),
                p_samples=(unc_model_key_main_prompt, lambda x: len(x) / total_rows),
                coef_mean=('axiomatic_coef', 'mean'),
            ).reset_index()
            print(f'{result}\n')
            result = result[result['axiom_num_nli'] != 'not_common']
            
            # =============================================
            ax = axes[i, j]
            for k, metric_name in enumerate(metrics):
                metric_value = np.array(result[metric_name])
                metric_value = np.concatenate((metric_value, [metric_value[0]]))
                ax.plot(angles, metric_value, color=colors[k], linewidth=2, label=metrics2title[metric_name], marker='o')
                ax.fill(angles, metric_value, color=colors[k], alpha=0.2)

            ax.set_ylim(0, 1)
            ax.set_yticks([0.5, 1.0])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=10)

            if i == 0:
                ax.set_title(retrievers2title[retriever], fontsize=11, fontweight='bold')

    # Add row labels manually
    for i, dataset in enumerate(datasets):
        fig.text(0.02, 0.83 - i * 0.31, datasets2title[dataset['dataset']], fontsize=11, fontweight='bold', rotation=90, va='center')

    
    plt.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1)

    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
    unique_handles_labels = dict(zip(labels, handles))
    fig.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01))

    # Show plot
    plt.savefig(f'{FOLDER_OUTPUT_PATH}/axioms_stat.png')
    plt.savefig(f'{FOLDER_OUTPUT_PATH}/axioms_stat.pdf', format="pdf", bbox_inches="tight")
    # plt.show()




  # # Define values for each model (ensure they are in the same order as labels)
    # acd_values = [78, 0, 52, 68, 72]  # ACD
    # fsb_values = [78, 52, 52, 68, 70]  # FSB
    # cda_m_values = [78, 52, 58, 72, 74]  # CDA-m

    # # Normalize values (optional, based on range)
    # acd = np.array(acd_values)
    # fsb = np.array(fsb_values)
    # cda_m = np.array(cda_m_values)

    # # Repeat first value at the end to close the circular shape
    # acd = np.concatenate((acd, [acd[0]]))
    # fsb = np.concatenate((fsb, [fsb[0]]))
    # cda_m = np.concatenate((cda_m, [cda_m[0]]))


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
    parser.add_argument('--main_prompt_format', type=str, default='contriever_retriever_top1', choices=[
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
    
    main(args)
    
    
    # python framework/plots/axioms_stat.py
    
    