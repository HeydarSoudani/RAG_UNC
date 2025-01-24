import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


FOLDER_PATH = 'framework/plots/imgs'

datasets = ["NQ-open", "TriviaQA", "PopQA"]
uncertainty_methods = ["PE", "SE", "PE+M", "SE+M"]
retrieval_models = ["No doc", "-doc", "BM25", "Contriever", "Rerank", "+doc"]
colors = ['deepskyblue', 'darkorange', 'mediumseagreen', 'orchid']

# Random AUROC data for demonstration (shape: 3 datasets x 4 methods x 5 models)
# auroc_values = np.random.rand(3, len(uncertainty_methods), len(retrieval_models))

# LLama2
auroc_values = np.array([
    [   # NQ
        [64.03, 52.82, 66.48, 64.87, 65.19, 68.59], # PE
        [65.69, 51.06, 67.12, 67.48, 65.24, 69.37], # SE
        [59.21, 51.63, 64.56, 64.87, 65.05, 69.13], # PE+M
        [60.70, 49.55, 65.18, 66.57, 64.52, 69.10], # SE+M
        [69.80, 54.88, 64.87, 64.00, 59.80, 59.80],
        [59.66, 49.72, 63.69, 63.58, 59.57, 59.41],
        [67.83, 52.88, 64.53, 63.94, 59.74, 59.62]
    ], 
    [   # TQA
        [73.49, 64.88, 68.35, 68.18, 66.97, 71.65], # PE
        [76.69, 64.90, 74.30, 73.44, 72.59, 79.64], # SE
        [75.92, 66.32, 69.79, 69.66, 68.31, 74.13], # PE+M
        [75.87, 64.72, 74.16, 73.54, 72.49, 79.64], # SE+M
        [79.24, 65.90, 70.81, 70.71, 67.49, 79.21], # Deg
        [71.50, 61.16, 67.85, 68.22, 65.74, 75.77], # ECC
        [77.86, 64.58, 70.16, 70.00, 66.95, 78.56], # EigV
    ], 
    [   # PQA
        [72.44, 52.96, 62.17, 61.59, 56.54, np.nan], # PE
        [78.11, 54.57, 64.63, 63.78, 58.85, np.nan], # SE
        [76.10, 54.21, 62.72, 61.72, 57.25, np.nan], # PE+M
        [76.65, 55.59, 65.32, 63.96, 59.70, np.nan], # SE+M
        [80.75, 57.54, 65.56, 62.57, 59.45, np.nan],
        [70.96, 54.24, 64.94, 62.05, 59.27, np.nan],
        [80.71, 57.13, 65.41, 62.41, 59.37, np.nan], 
    ]
])
acc_values = np.array([
    [21.6, 7.66, 19.5, 28.2, 35.3, 63.4], # NQ
    [50.4, 30.9, 51.5, 53.1, 59.0, 68.0], # TQA
    [20.2, 7.3, 24.5, 37.4, 36.7, np.nan], # PQA
])

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



# Mistral
auroc_values = np.array([
    [   # NQ
        [66.91, 0.0, 62.89, 64.40, 66.84, 68.30], # PE
        [67.76, 0.0, 63.22, 65.66, 67.72, 68.40], # SE
        [64.77, 0.0, 62.50, 64.09, 66.52, 68.50], # PE+M
        [65.06, 0.0, 62.44, 65.23, 67.46, 69.00],  # SE+M
        [72.10, 0.0, 63.67, 63.10, 62.47, 58.29], # Deg
        [64.38, 0.0, 63.08, 62.87, 62.23, 58.13], # ECC
        [70.21, 0.0, 63.31, 63.03, 62.36, 58.18]  # EigV
    ], 
    [   # TQA
        [81.99, 74.93, 81.04, 79.13, 71.50, 82.15], # PE
        [82.19, 68.13, 80.85, 78.83, 77.54, 82.82], # SE
        [81.88, 74.89, 80.86, 78.84, 77.66, 81.83], # PE+M
        [81.90, 68.02, 80.80, 78.68, 77.70, 82.57], # SE+M
        [80.56, 64.10, 72.90, 72.62, 68.96, 74.95], # Deg
        [77.32, 60.89, 72.02, 71.72, 68.51, 73.99], # ECC
        [80.80, 62.79, 72.55, 72.25, 68.78, 74.88]  # EigV
    ], 
    [   # PQA
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # SE
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE+M
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # SE+M
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Deg
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # ECC
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # EigV
    ]
])
acc_values = np.array([
    [[30.71, 0.0, 24.71, 32.85, 38.93, 67.63]], # NQ
    [[62.59, 37.66, 57.92, 60.58, 64.32, 72.70]], # TQA
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], # PQA
])


# acc_values = np.array([
#     [[23.9, 3.8, 20.9, 30.2, 36.9, 67.1]],  # NQ
#     [[54.6, 18.7, 51.6, 52.6, 60.3, 71.0]], # TQA
#     [[17.9, 3.0, 23.35, 38.1, 38.7]],       # PQA
# ])

# Qwen2.5: calibrated
# auroc_values = np.array([
#     [   # NQ
#         [ ], # PE
#         [ ], # SE
#         [ ], # PE+M
#         [ ]  # SE+M
#     ], 
#     [   # TQA
#         [ ], # PE
#         [ ], # SE
#         [ ], # PE+M
#         [ ]  # SE+M
#     ], 
#     [   # PQA
#         [ ], # PE
#         [ ], # SE
#         [ ], # PE+M
#         [ ]  # SE+M
#     ]
# ])



# llama3.1
# auroc_values = np.array([
#     [   # NQ
#         [44.99, 0.0, 0.0, 0.0, 0.0, 46.35], # PE
#         [52.78, 0.0, 0.0, 0.0, 0.0, 48.43], # SE
#         [44.57, 0.0, 0.0, 0.0, 0.0, 47.63], # PE+M
#         [52.37, 0.0, 0.0, 0.0, 0.0, 47.44]  # SE+M
#     ], 
#     [   # TQA
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # SE
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE+M
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # SE+M
#     ], 
#     [   # PQA
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # SE
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE+M
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # SE+M
#     ]
# ])
# acc_values = np.array([
#     [[30.76, 0.0, 21.2, 31.5, 37.6, 67.1]], # NQ
#     [[]], # TQA
#     [[]], # PQA
# ])

# llama3.1: calibrated
# auroc_values = np.array([
#     [   # NQ
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # SE
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE+M
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # SE+M
#     ], 
#     [   # TQA
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # SE
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE+M
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # SE+M
#     ], 
#     [   # PQA
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # SE
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # PE+M
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # SE+M
#     ]
# ])



fig, axes = plt.subplots(1, len(datasets), figsize=(15, 3.3), sharey=True)
bar_width = 0.15
x = np.arange(len(retrieval_models))

for i, ax in enumerate(axes):
    ax_right = ax.twinx()
    
    for j, method in enumerate(uncertainty_methods):
        # ax.plot(retrieval_models, auroc_values[i, j], marker='o', label=method)
        ax.bar(x + j * bar_width, auroc_values[i, j, :], width=bar_width, label=method, color=colors[j])

    ax.axhline(y=np.max(auroc_values[i, :, 0]), color='gray', linestyle='--', linewidth=1.3, label=f"No doc")
    
    
    # Create new smooth x values
    # x_cur = x + 1.5 * bar_width
    # x_smooth = np.linspace(x_cur.min(), x_cur.max(), 300)  # More points for smoothness
    # y_cur = acc_values[i, :]
    # spline = make_interp_spline(x, y_cur, k=3)
    # y_smooth = spline(x_smooth)
    # ax_right.plot(x_smooth, y_smooth, marker='', linestyle='-', color='black', label='Accuracy')
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
plt.savefig(f'{FOLDER_PATH}/auroc_cal_.png')
plt.savefig(f'{FOLDER_PATH}/auroc_cal_.pdf', format="pdf", bbox_inches="tight")



# python framework/plots/auroc.py
