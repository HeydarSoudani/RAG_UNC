import matplotlib.pyplot as plt
import numpy as np

FOLDER_PATH = 'framework/plots/imgs'

datasets = ["NQ-open", "TriviaQA", "PopQA"]
uncertainty_methods = ["PE", "SE", "PE+M", "SE+M"]
retrieval_models = ["No doc", "-doc", "BM25", "Contriever", "Rerank", "+doc"]
colors = ['deepskyblue', 'darkorange', 'mediumseagreen', 'orchid']

# Random AUROC data for demonstration (shape: 3 datasets x 4 methods x 5 models)
# auroc_values = np.random.rand(3, len(uncertainty_methods), len(retrieval_models))
# auroc_values = np.array([
#     [   # NQ
#         [64.03, 52.82, 66.48, 64.87, 65.19, 68.59], # PE
#         [65.69, 51.06, 67.12, 67.48, 65.24, 69.37], # SE
#         [59.21, 51.63, 64.56, 64.87, 65.05, 69.13], # PE+M
#         [60.70, 49.55, 65.18, 66.57, 64.52, 69.10]  # SE+M
#     ], 
#     [   # TQA
#         [73.49, 64.88, 68.35, 68.18, 66.97, 71.65], # PE
#         [76.69, 64.90, 74.30, 73.44, 72.59, 79.64], # SE
#         [75.92, 66.32, 69.79, 69.66, 68.31, 74.13], # PE+M
#         [75.87, 64.72, 74.16, 73.54, 72.49, 79.64]  # SE+M
#     ], 
#     [   # PQA
#         [72.44, 52.96, 62.17, 61.59, 56.54, np.nan], # PE
#         [78.11, 54.57, 64.63, 63.78, 58.85, np.nan], # SE
#         [76.10, 54.21, 62.72, 61.72, 57.25, np.nan], # PE+M
#         [76.65, 55.59, 65.32, 63.96, 59.70, np.nan]  # SE+M
#     ]
# ])

# After calibration
auroc_values = np.array([
    [   # NQ
        [64.03, 62.08, 68.34, 66.75, 66.93, 68.93], # PE
        [65.69, 67.70, 69.61, 69.32, 66.53, 67.14], # SE
        [59.21, 59.14, 66.63, 66.91, 66.91, 69.41], # PE+M
        [60.70, 61.14, 67.10, 68.04, 65.35, 67.46]  # SE+M
    ], 
    [   # TQA
        [73.49, 69.14, 71.30, 70.10, 69.40, 72.37], # PE
        [76.69, 77.86, 81.53, 77.71, 77.92, 77.10], # SE
        [75.92, 70.44, 72.71, 71.70, 70.88, 74.73], # PE+M
        [75.87, 75.83, 80.79, 77.32, 77.66, 77.69]  # SE+M
    ], 
    [   # PQA
        [72.44, 56.98, 64.09, 62.07, 57.71, np.nan], # PE
        [78.11, 70.90, 70.65, 66.20, 63.95, np.nan], # SE
        [76.10, 58.39, 64.78, 62.93, 58.52, np.nan], # PE+M
        [76.65, 70.79, 71.13, 66.43, 64.37, np.nan]  # SE+M
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
plt.savefig(f'{FOLDER_PATH}/auroc_cal.png')
plt.savefig(f'{FOLDER_PATH}/auroc_cal.pdf', format="pdf", bbox_inches="tight")



# python framework/plots/auroc.py
