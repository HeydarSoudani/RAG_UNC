import matplotlib.pyplot as plt
import numpy as np

FOLDER_PATH = 'framework/plots/imgs'


datasets = ["NQ-open", "TriviaQA", "PopQA"]
uncertainty_methods = ["PE", "SE", "PE+M", "SE+M"]
retrieval_models = ["No doc", "-doc", "BM25", "Contriever", "Rerank", "+doc"]

# Random AUROC data for demonstration (shape: 3 datasets x 4 methods x 5 models)
auroc_values = np.random.rand(3, len(uncertainty_methods), len(retrieval_models))
auroc_values = np.array([
    [   # NQ
        [0.200, 0.026, 0.226, 0.231, 0.251, 0.310], # PE
        [0.223, 0.009, 0.235, 0.272, 0.252, 0.323], # SE
        [0.131, 0.015, 0.199, 0.231, 0.249, 0.319], # PE+M
        [0.152, 0.004, 0.208, 0.258, 0.240, 0.318]  # SE+M
    ], 
    [   # TQA
        [0.406, 0.238, 0.317, 0.314, 0.289, 0.349], # PE
        [0.462, 0.238, 0.420, 0.405, 0.385, 0.479], # SE
        [0.449, 0.261, 0.342, 0.339, 0.312, 0.389], # PE+M
        [0.448, 0.235, 0.418, 0.407, 0.383, 0.479]  # SE+M
    ], 
    [   # PQA
        [0.312, 0.026, 0.181, 0.194, 0.109, np.nan], # PE
        [0.391, 0.041, 0.218, 0.231, 0.147, np.nan], # SE
        [0.363, 0.037, 0.189, 0.196, 0.121, np.nan], # PE+M
        [0.370, 0.050, 0.228, 0.234, 0.161, np.nan]  # SE+M
    ]
])

fig, axes = plt.subplots(1, len(datasets), figsize=(15, 3.3), sharey=True)
for i, ax in enumerate(axes):
    for j, method in enumerate(uncertainty_methods):
        ax.plot(retrieval_models, auroc_values[i, j], marker='o', label=method)
    
    ax.set_title(datasets[i])
    ax.set_xticks(range(len(retrieval_models)))
    ax.set_xticklabels(retrieval_models)
    ax.grid(True, linestyle='--', alpha=0.6)

axes[0].set_ylabel("Spearman")
axes[1].set_xlabel("Retrieval Models")
axes[-1].legend(title="UQ Methods", loc='upper right')

plt.tight_layout()
plt.show()
plt.savefig(f'{FOLDER_PATH}/spearman.png')
plt.savefig(f'{FOLDER_PATH}/spearman.pdf', format="pdf", bbox_inches="tight")


# python framework/plots/spearman.py
