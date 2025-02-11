import json
import numpy as np
import matplotlib.pyplot as plt


FOLDER_INPUT_PATH = 'framework/get_results/files'
FOLDER_OUTPUT_PATH = 'framework/get_results/imgs'

llm2title = {
    'llama2': 'Llama-2-7b-chat-hf',
    'mistral': 'Mistral-7B-Instruct-v0.3'
}
ret2title = {
    "BM25": "bm25_retriever_top1",
    "Contriever": "contriever_retriever_top1",
    "Rerank": "rerank_retriever_top1",
    r'$\mathrm{Doc^{+}}$': "q_positive"
}
cal2title = {
    'normal': 'Normal',
    'kldiv': 'KLDiv',
    'nli': 'NLI',
    'minicheck': 'MiniCheck'   
}

unc2name = {
    'PE': 'PE',
    'SE': 'SE',
    'PE+M': 'PE_MARS',
    'SE+M': 'SE_MARS',
    'EigV': 'spectral_u',
    'ECC': 'ecc_u',
    'Deg': 'degree_u'
}
dataset2title = {
    'nqgold': 'NQ-open',
    'trivia': 'TriviaQA',
    'popqa': 'PopQA',
}



llm_list = ['llama2'] # llama2, mistral
datasets = [
    {"dataset": 'nqgold', "subsec": 'test'},
    # {"dataset": 'trivia', "subsec": 'dev'},
    {"dataset": 'popqa', "subsec": 'test'}
]

# 
retrieval_models = ["BM25", "Contriever", "Rerank", r'$\mathrm{Doc^{+}}$']
uncertainty_methods = ["PE", "SE", "EigV", "ECC"]
calibration_methods = ['normal', 'kldiv', 'nli', 'minicheck']
colors = ['darkgrey', 'skyblue', 'cornflowerblue', 'steelblue']


fig, axes = plt.subplots(len(datasets), len(uncertainty_methods), figsize=(len(uncertainty_methods)*4, len(datasets)*3.6), sharey=True)
fig.subplots_adjust(right=0.8)  # Adjust spacing to avoid overlap

if len(llm_list) == 1:
    axes = np.atleast_2d(axes)

bar_width = 0.18
x = np.arange(1, len(retrieval_models)+1)


llm_name = llm_list[0]
for i, dataset in enumerate(datasets):
# for i, llm_name in enumerate(llm_list):
    for j, unc_method in enumerate(uncertainty_methods):
        ax = axes[i, j]
    
        ### === Only Doc =====================
        second_prompt = 'bm25_retriever_top1' if dataset['dataset'] == 'popqa' else 'q_positive'
        filename = f"framework/run_output/{llm2title[llm_name]}/{dataset['dataset']}/{dataset['subsec']}/run_0/only_q__{second_prompt}/prob_alpha_0.5/calibration_results_main_prompt/calibration_results.jsonl"
        with open(filename, 'r') as file:
            no_doc_value = json.load(file)[f"{unc2name[unc_method]}"]["AUROC (Unc.)"]*100
        ax.bar(0, no_doc_value, width=bar_width, label="No Doc", color="orange")
        
        
        ### === Retrievers =================== 
        results_data = []
        for l, ret_name in enumerate(retrieval_models):
            filename = f"framework/run_output/{llm2title[llm_name]}/{dataset['dataset']}/{dataset['subsec']}/run_0/{ret2title[ret_name]}__only_q/prob_alpha_0.5/calibration_results_main_prompt/calibration_results.jsonl"
            with open(filename, 'r') as file:
                data = json.load(file)
                results_data.append(data)

        for k, cal_method in enumerate(calibration_methods):
            auroc_values = []
            for item in results_data:
                if cal_method == 'normal':
                    auroc_values.append(item[f"{unc2name[unc_method]}"]["AUROC (Unc.)"]*100)
                else:
                    auroc_values.append(item[f"{unc2name[unc_method]}_{cal_method}"]["AUROC (Unc.)"]*100)
            auroc_values = np.array(auroc_values, dtype=float)
            
            ax.bar((x -1.5*bar_width) + (k * bar_width), auroc_values, width=bar_width, label=cal2title[cal_method], color=colors[k])
    
        ax.set_ylim([55, 85])
        
        if i == 0:
            ax.set_title(f"{unc_method}", color='black', fontsize=13, fontweight='bold')
    

        ax.set_xticks(np.arange(len(retrieval_models)+1))
        if i == len(datasets)-1:
            ax.set_xticklabels(["No Doc"]+retrieval_models, rotation=20, fontsize=11)
        else:
            ax.set_xticklabels([])
    
    axes[i, 0].set_ylabel(dataset2title[dataset['dataset']], color='black', fontsize=14)

# fig.text(0.06, 0.5, 'Dataset 2', va='center', ha='center', rotation='vertical', color='b')



plt.tight_layout()
# fig.subplots_adjust(right=0.91)
fig.subplots_adjust(top=0.9)


handles, labels = [], []
for ax_row in axes:
    for ax in ax_row:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        
unique_handles_labels = dict(zip(labels, handles))
# fig.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc="center right", bbox_to_anchor=(1.0, 0.5))
fig.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc="upper center", ncol=len(calibration_methods)+1, bbox_to_anchor=(0.5, 1.0), fontsize=14)

plt.savefig(f'{FOLDER_OUTPUT_PATH}/auroc_cal.png')
plt.savefig(f'{FOLDER_OUTPUT_PATH}/auroc_cal.pdf', format="pdf", bbox_inches="tight", dpi=500)


# python framework/get_results/auroc_calibrated.py