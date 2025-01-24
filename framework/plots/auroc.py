import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


# === AUROC file guide:
#  Dataset order:          NQ, TQA, PQA
#  Unc order (rows):       "PE", "PE+M", "SE", "SE+M", "Deg", "ECC", "EigV"
#  Retriever order (cols): "No doc", "-doc", "BM25", "Contriever", "Rerank", "+doc"


FOLDER_INPUT_PATH = 'framework/plots/results'
FOLDER_OUTPUT_PATH = 'framework/plots/imgs'
auroc_file = f"{FOLDER_INPUT_PATH}/auroc.json"
auroc_cal_file = f"{FOLDER_INPUT_PATH}/auroc_calibrated.json"
accuracy_file = f"{FOLDER_INPUT_PATH}/accuracy.json"

with open(auroc_file, "r") as file:
    auroc_data = json.load(file)
with open(auroc_cal_file, "r") as file:
    auroc_cal_data = json.load(file)
with open(accuracy_file, "r") as file:
    accuracy_data = json.load(file)

llm_list = ['llama2'] # 'mistral', 'vicuna'

datasets = ["NQ-open", "TriviaQA", "PopQA"]
uncertainty_methods = ["PE", "PE+M", "SE", "SE+M", "Deg", "ECC", "EigV"]
retrieval_models = ["No doc", "-doc", "BM25", "Contriever", "Rerank", "+doc"]
colors = [
    'darkorange', 'sandybrown',
    'deepskyblue', 'lightskyblue',
    'darkorchid', 'mediumorchid', 'orchid' 
]

# Random AUROC data for demonstration (shape: 3 datasets x 4 methods x 5 models)
# auroc_values = np.random.rand(3, len(uncertainty_methods), len(retrieval_models))

fig, axes = plt.subplots(len(llm_list), len(datasets), figsize=(15, len(llm_list)*3.3), sharey=True)
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

        ax.axhline(y=np.max(auroc_values[j, :, 0]), color='gray', linestyle='--', linewidth=1.3, label=f"No doc")
        
        ax_right = ax.twinx()
        ax_right.plot(x + 3*bar_width, accuracy_values[j, :], marker='o', linestyle='-', color='seagreen', label='Accuracy')
        
        if i == 0:
            ax.set_title(datasets[j])
            
        ax.set_ylim([45, 85])
        ax_right.set_ylim([0, 80])

        ax.set_xticks(x + 3*bar_width)
        if i == len(llm_list)-1:
            ax.set_xticklabels(retrieval_models)
        
        if j != len(datasets)-1:
            ax_right.set_yticks([])
        
    ax_right.set_ylabel("Accuracy", color='black')
    axes[i, 0].set_ylabel("AUROC", color='black')
    
axes[-1, 1].set_xlabel("Retrieval Models")

plt.tight_layout()
fig.subplots_adjust(right=0.89)

handles, labels = [], []
for ax_row in axes:
    for ax in ax_row:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
unique_handles_labels = dict(zip(labels, handles))
fig.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc="center right", bbox_to_anchor=(1.0, 0.5))


plt.savefig(f'{FOLDER_OUTPUT_PATH}/auroc.png')
plt.savefig(f'{FOLDER_OUTPUT_PATH}/auroc.pdf', format="pdf", bbox_inches="tight")
# plt.show()


# python framework/plots/auroc.py
















# Llama2 After calibration: with 2 coef
# auroc_values = np.array([
#     [   # NQ
#         [64.03, 62.08, 68.34, 66.75, 66.93, 68.93], # PE
#         [65.69, 67.70, 69.61, 69.32, 66.53, 67.14], # SE
#         [59.21, 59.14, 66.63, 66.91, 66.91, 69.41], # PE+M
#         [60.70, 61.14, 67.10, 68.04, 65.35, 67.46]  # SE+M
#     ], 
#     [   # TQA
#         [73.49, 69.14, 71.30, 70.10, 69.40, 72.37], # PE
#         [76.69, 77.86, 81.53, 77.71, 77.92, 77.10], # SE
#         [75.92, 70.44, 72.71, 71.70, 70.88, 74.73], # PE+M
#         [75.87, 75.83, 80.79, 77.32, 77.66, 77.69]  # SE+M
#     ], 
#     [   # PQA
#         [72.44, 56.98, 64.09, 62.07, 57.71, np.nan], # PE
#         [78.11, 70.90, 70.65, 66.20, 63.95, np.nan], # SE
#         [76.10, 58.39, 64.78, 62.93, 58.52, np.nan], # PE+M
#         [76.65, 70.79, 71.13, 66.43, 64.37, np.nan]  # SE+M
#     ]
# ])