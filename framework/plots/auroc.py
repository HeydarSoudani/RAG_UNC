import matplotlib.pyplot as plt
import numpy as np

FOLDER_PATH = 'framework/plots/imgs'

datasets = ["NQ-open", "TriviaQA", "PopQA"]
uncertainty_methods = ["PE", "SE", "PE+M", "SE+M"]
retrieval_models = ["No doc", "-doc", "BM25", "Contriever", "Rerank", "+doc"]
colors = ['deepskyblue', 'darkorange', 'mediumseagreen', 'orchid']

# Random AUROC data for demonstration (shape: 3 datasets x 4 methods x 5 models)
# auroc_values = np.random.rand(3, len(uncertainty_methods), len(retrieval_models))

# LLama2
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
acc_values = np.array([
    [   # NQ
        []
    ],
    [   # TQA
        
    ],
    [   # PQA
        
    ],
])

# Llama2 After calibration: with 2 coef
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


# Qwen2.5
auroc_values = np.array([
    [   # NQ
        [51.21, 35.08, 41.87, 45.28, 45.99, 47.04], # PE
        [56.03, 28.59, 39.83, 43.31, 44.01, 43.43], # SE
        [50.44, 30.19, 40.23, 44.06, 45.48, 46.86], # PE+M
        [54.07, 25.19, 38.89, 42.59, 43.82, 43.62]  # SE+M
    ], 
    [   # TQA
        [43.17, 30.25, 41.54, 40.86, 41.84, 47.76], # PE
        [52.39, 31.50, 37.48, 37.68, 38.68, 47.00], # SE
        [42.33, 30.76, 41.87, 40.57, 42.05, 46.11], # PE+M
        [51.02, 32.86, 38.44, 37.89, 39.24, 44.79]  # SE+M
    ], 
    [   # PQA
        [46.25, 53.55, 32.03, 41.97, 38.39, np.nan], # PE
        [65.87, 59.81, 33.02, 39.37, 38.36, np.nan], # SE
        [43.06, 48.56, 30.17, 40.32, 36.91, np.nan], # PE+M
        [62.25, 52.49, 31.83, 38.63, 37.70, np.nan]  # SE+M
    ]
])
# Qwen2.5: calibrated
auroc_values = np.array([
    [   # NQ
        [ ], # PE
        [ ], # SE
        [ ], # PE+M
        [ ]  # SE+M
    ], 
    [   # TQA
        [ ], # PE
        [ ], # SE
        [ ], # PE+M
        [ ]  # SE+M
    ], 
    [   # PQA
        [ ], # PE
        [ ], # SE
        [ ], # PE+M
        [ ]  # SE+M
    ]
])



# llama3.1
auroc_values = np.array([
    [   # NQ
        [44.99, 0.0, 0.0, 0.0, 0.0, 46.35], # PE
        [52.78, 0.0, 0.0, 0.0, 0.0, 48.43], # SE
        [44.57, 0.0, 0.0, 0.0, 0.0, 47.63], # PE+M
        [52.37, 0.0, 0.0, 0.0, 0.0, 47.44]  # SE+M
    ], 
    [   # TQA
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # SE
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE+M
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # SE+M
    ], 
    [   # PQA
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # SE
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE+M
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # SE+M
    ]
])


# llama3.1: calibrated
auroc_values = np.array([
    [   # NQ
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # SE
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE+M
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # SE+M
    ], 
    [   # TQA
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # SE
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE+M
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # SE+M
    ], 
    [   # PQA
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # SE
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE+M
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # SE+M
    ]
])



fig, axes = plt.subplots(1, len(datasets), figsize=(15, 3.3), sharey=True)
bar_width = 0.15
x = np.arange(len(retrieval_models))

for i, ax in enumerate(axes):
    ax_right = ax.twinx()
    
    for j, method in enumerate(uncertainty_methods):
        # ax.plot(retrieval_models, auroc_values[i, j], marker='o', label=method)
        ax.bar(x + j * bar_width, auroc_values[i, j, :], width=bar_width, label=method, color=colors[j])

    ax.axhline(y=np.max(auroc_values[i, :, 0]), color='gray', linestyle='--', linewidth=1.3, label=f"No doc")
    ax_right.plot(x + 1.5 * bar_width, acc_values[i, :], marker='o', linestyle='-', color='black', label='Accuracy')

    ax.set_xticks(x + 1.5*bar_width)
    ax.set_title(datasets[i])
    ax.set_ylim([50, 85])
    ax.set_xticklabels(retrieval_models)
    
    ax_right.set_ylabel("Accuracy", color='black')
    ax_right.tick_params(axis='y', labelcolor='black')
    ax_right.set_ylim([0, 70])

axes[0].set_ylabel("AUROC")
axes[1].set_xlabel("Retrieval Models")
axes[-1].legend(title="UE Methods", loc='upper right')


plt.tight_layout()
plt.show()
plt.savefig(f'{FOLDER_PATH}/auroc_cal.png')
plt.savefig(f'{FOLDER_PATH}/auroc_cal.pdf', format="pdf", bbox_inches="tight")



# python framework/plots/auroc.py
