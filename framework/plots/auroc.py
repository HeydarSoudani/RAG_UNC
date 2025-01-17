import matplotlib.pyplot as plt
import numpy as np

FOLDER_PATH = 'framework/plots/imgs'

datasets = ["NQ-open", "TriviaQA", "PopQA"]
uncertainty_methods = ["PE", "SE", "PE+M", "SE+M"]
retrieval_models = ["No doc", "-doc", "BM25", "Contriever", "Rerank", "+doc"]
colors = ['deepskyblue', 'darkorange', 'mediumseagreen', 'orchid']

# Random AUROC data for demonstration (shape: 3 datasets x 4 methods x 5 models)
# auroc_values = np.random.rand(3, len(uncertainty_methods), len(retrieval_models))
auroc_values = np.array([
    [   # NQ
        [64.03, 52.82, 66.48, 64.87, 65.19, 68.59], # PE
        [65.69, 51.06, 67.12, 67.48, 65.24, 69.37], # SE
        [59.21, 51.63, 64.56, 64.87, 65.05, 69.13], # PE+M
        [60.70, 49.55, 65.18, 66.57, 64.52, 69.10]  # SE+M
    ], 
    [   # TQA
        [73.49, 64.88, 68.35, 68.18, 66.97, 71.65], # PE
        [76.69, 64.90, 74.30, 73.44, 72.59, 79.64], # SE
        [75.92, 66.32, 69.79, 69.66, 68.31, 74.13], # PE+M
        [75.87, 64.72, 74.16, 73.54, 72.49, 79.64]  # SE+M
    ], 
    [   # PQA
        [72.44, 52.96, 62.17, 61.59, 56.54, np.nan], # PE
        [78.11, 54.57, 64.63, 63.78, 58.85, np.nan], # SE
        [76.10, 54.21, 62.72, 61.72, 57.25, np.nan], # PE+M
        [76.65, 55.59, 65.32, 63.96, 59.70, np.nan]  # SE+M
    ]
])


fig, axes = plt.subplots(1, len(datasets), figsize=(15, 3.3), sharey=True)
bar_width = 0.15
x = np.arange(len(retrieval_models))

for i, ax in enumerate(axes):
    for j, method in enumerate(uncertainty_methods):
        # ax.plot(retrieval_models, auroc_values[i, j], marker='o', label=method)
        ax.bar(x + j * bar_width, auroc_values[i, j, :], width=bar_width, label=method, color=colors[j])

    ax.axhline(y=np.max(auroc_values[i, :, 0]), color='gray', linestyle='--', linewidth=1.3, label=f"No doc")
    
    ax.set_xticks(x + 1.5*bar_width)
    ax.set_title(datasets[i])
    # ax.set_xticks(range(len(retrieval_models)))
    ax.set_ylim([50, 85])
    ax.set_xticklabels(retrieval_models)
    # ax.grid(True, linestyle='--', alpha=0.6)

axes[0].set_ylabel("AUROC")
axes[1].set_xlabel("Retrieval Models")
axes[-1].legend(title="UE Methods", loc='upper right')


plt.tight_layout()
plt.show()
plt.savefig(f'{FOLDER_PATH}/auroc.png')
plt.savefig(f'{FOLDER_PATH}/auroc.pdf', format="pdf", bbox_inches="tight")



# python framework/plots/auroc.py
