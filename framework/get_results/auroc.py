import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


# === AUROC file guide:
#  Dataset order:          NQ, TQA, PQA
#  Unc order (rows):       "PE", "PE+M", "SE", "SE+M", "Deg", "ECC", "EigV"
#  Retriever order (cols): "No doc", "-doc", "BM25", "Contriever", "Rerank", "+doc"


FOLDER_INPUT_PATH = 'framework/get_results/files'
FOLDER_OUTPUT_PATH = 'framework/get_results/imgs'
auroc_file = f"{FOLDER_INPUT_PATH}/auroc.json"
auroc_cal_file = f"{FOLDER_INPUT_PATH}/auroc_calibrated.json"
accuracy_file = f"{FOLDER_INPUT_PATH}/accuracy.json"

with open(auroc_file, "r") as file:
    auroc_data = json.load(file)
with open(auroc_cal_file, "r") as file:
    auroc_cal_data = json.load(file)
with open(accuracy_file, "r") as file:
    accuracy_data = json.load(file)

llm_list = ['llama2'] # 'llama2', 'mistral', 'vicuna'

datasets = ["NQ-open", "TriviaQA", "PopQA"]
uncertainty_methods = ["PE", "PE+M", "SE", "SE+M", "Deg", "ECC", "EigV"]
retrieval_models = ["No Doc", r'$\mathrm{Doc^{-}}$', "BM25", "Contriever", "Rerank", r'$\mathrm{Doc^{+}}$']
colors = [
    'darkorange', 'sandybrown',
    'deepskyblue', 'lightskyblue',
    'darkorchid', 'mediumorchid', 'orchid' 
]

# Random AUROC data for demonstration (shape: 3 datasets x 4 methods x 5 models)
# auroc_values = np.random.rand(3, len(uncertainty_methods), len(retrieval_models))

fig, axes = plt.subplots(len(llm_list), len(datasets), figsize=(15, len(llm_list)*3.8), sharey=True)
fig.subplots_adjust(right=0.8)  # Adjust spacing to avoid overlap


if len(llm_list) == 1:
    axes = np.atleast_2d(axes)

bar_width = 0.1
x = np.arange(len(retrieval_models))


for i, llm_name in enumerate(llm_list):
    auroc_values = np.array(auroc_data[llm_name], dtype=float)
    accuracy_values = np.array(accuracy_data[llm_name], dtype=float)
    
    for j in range(len(datasets)):
        ax = axes[i, j]
        for k, method in enumerate(uncertainty_methods):
            # ax.plot(retrieval_models, auroc_values[i, j], marker='o', label=method)
            ax.bar(x + k * bar_width, auroc_values[j, k, :], width=bar_width, label=method, color=colors[k])

        # ax.axhline(y=np.max(auroc_values[j, :, 0]), color='gray', linestyle='--', linewidth=1.3, label=f"No doc")
        
        ax_right = ax.twinx()
        accuracy_line, = ax_right.plot(x + 3*bar_width, accuracy_values[j, :], marker='o', linestyle='-', color='forestgreen', label='Accuracy')
        
        if i == 0:
            ax.set_title(datasets[j], color='black', fontsize=13, fontweight='bold')
            
        ax.set_ylim([40, 85])
        ax_right.set_ylim([0, 90])

        ax.set_xticks(x + 3*bar_width)
        if i == len(llm_list)-1:
            ax.set_xticklabels(retrieval_models, rotation=20, fontsize=11)
        
        if j != len(datasets)-1:
            ax_right.set_yticks([])
        
    ax_right.set_ylabel("Accuracy", color='black', fontsize=13)
    axes[i, 0].set_ylabel("AUROC", color='black', fontsize=13)
    
# axes[-1, 1].set_xlabel("Retrievers", color='black', fontsize=13)



plt.tight_layout()
fig.subplots_adjust(right=0.875)


handles, labels = [], []
for ax_row in axes:
    for ax in ax_row:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        handles.append(accuracy_line)  # Add the accuracy line's handle
        labels.append('Accuracy')      # Add the accuracy label
unique_handles_labels = dict(zip(labels, handles))
fig.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc="center right", bbox_to_anchor=(1.0, 0.5))


plt.savefig(f'{FOLDER_OUTPUT_PATH}/auroc.png')
plt.savefig(f'{FOLDER_OUTPUT_PATH}/auroc.pdf', format="pdf", bbox_inches="tight", dpi=300)
# plt.show()


# python framework/get_results/auroc.py

