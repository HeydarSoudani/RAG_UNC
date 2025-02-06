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
        [1.98, 1.92, 1.53, 1.41, 1.31, 1.19], # PE
        [5.40, 5.09, 4.29, 4.20, 3.99, 3.88], # SE
        [3.90, 3.89, 3.33, 3.26, 3.12, 2.97], # PE+M
        [7.41, 6.93, 5.97, 5.88, 5.62, 5.49]  # SE+M
    ], 
    [   # TQA
        [1.14, 1.42, 1.05, 1.03, 0.90, 0.96], # PE
        [4.39, 4.48, 3.89, 3.85, 3.66, 3.73], # SE
        [1.74, 2.06, 1.64, 1.64, 1.46, 1.51], # PE+M
        [5.16, 5.21, 4.51, 4.50, 4.22, 4.30]  # SE+M
    ], 
    [   # PQA
        [1.29, 1.11, 0.54, 0.46, 0.35, np.nan], # PE
        [4.86, 4.37, 3.45, 3.30, 3.13, np.nan], # SE
        [1.59, 1.34, 0.65, 0.55, 0.44, np.nan], # PE+M
        [5.38, 4.71, 3.62, 3.43, 3.23 , np.nan]  # SE+M
    ]
])

# (shape: 3 datasets x 1 x 5 models)
accuracy_values = np.array([
    [
        [0.216, 0.076, 0.195, 0.282, 0.353, 0.634]
    ],
    [
        [0.504, 0.309, 0.515, 0.531, 0.590, 0.680]
    ],
    [
        [0.202, 0.073, 0.245, 0.374, 0.367, np.nan]
    ]
])


fig, axes = plt.subplots(1, len(datasets), figsize=(15, 3.3), sharey=True)
bar_width = 0.15
x = np.arange(len(retrieval_models))

for i, ax in enumerate(axes):
    for j, method in enumerate(uncertainty_methods):
        # ax.plot(retrieval_models, auroc_values[i, j], marker='o', label=method)
        bars = ax.bar(x + j * bar_width, auroc_values[i, j, :], width=bar_width, label=method, color=colors[j])
        
    max_value = np.max(auroc_values[i, :, 0])
    ax.axhline(y=max_value, color=colors[-1], linestyle='--', linewidth=1.3) # label=f"No doc"
    
    height = max_value + 0.2
    for idx, model in enumerate(retrieval_models):
        acc = accuracy_values[i, 0, idx]
        if not np.isnan(acc):
            ax.text(x[idx] + 1.5 * bar_width, height, f'{accuracy_values[i, 0, idx]}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x + 1.5*bar_width)
    ax.set_title(datasets[i])
    # ax.set_xticks(range(len(retrieval_models)))
    ax.set_ylim([0, 9.1])
    ax.set_xticklabels(retrieval_models)
    # ax.grid(True, linestyle='--', alpha=0.6)

axes[0].set_ylabel("Uncertainty Value")
axes[1].set_xlabel("Retrieval Models")
axes[-1].legend(title="UE Methods", loc='upper right')

plt.tight_layout()
plt.show()
plt.savefig(f'{FOLDER_PATH}/uncertainty_value.png')
plt.savefig(f'{FOLDER_PATH}/uncertainty_value.pdf', format="pdf", bbox_inches="tight")



# python framework/plots/uncertainty_value.py
