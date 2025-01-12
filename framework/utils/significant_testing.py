
# from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon
import json


def wilcoxon_test(sequence1, sequence2):
    is_significant  = False
    stat, p_value = wilcoxon(sequence1, sequence2)
    
    # Output the test statistic and p-value
    print(f"Wilcoxon test statistic: {stat}, p-value: {p_value}")
    
    # Interpretation
    if p_value < 0.01:
        print("The difference in performance is statistically significant.")
        is_significant = True
    else:
        print("The difference in performance is not statistically significant.")
    
    return stat, p_value, is_significant
    
    
    


# TODO: Need to be cleaned
# def mcnemar_test(file1, file2):
      
#     def load_jsonl(file_path):
#         with open(file_path, 'r') as f:
#             return [json.loads(line) for line in f]
      
#     model_a_results = load_jsonl(file1)
#     model_b_results = load_jsonl(file2)
#     assert len(model_a_results) == len(model_b_results)
    
#     # Extract is_correct values
#     model_a_correct = [entry['is_correct'] for entry in model_a_results]
#     model_b_correct = [entry['is_correct'] for entry in model_b_results]

#     # Create a contingency table
#     a = sum(1 for a, b in zip(model_a_correct, model_b_correct) if a == 0 and b == 0)
#     b = sum(1 for a, b in zip(model_a_correct, model_b_correct) if a == 0 and b == 1)
#     c = sum(1 for a, b in zip(model_a_correct, model_b_correct) if a == 1 and b == 0)
#     d = sum(1 for a, b in zip(model_a_correct, model_b_correct) if a == 1 and b == 1)

#     contingency_table = [[a, b], [c, d]]

#     # Perform McNemar's test
#     result = mcnemar(contingency_table, exact=True)

#     print("Statistic:", result.statistic)
#     print("p-value:", result.pvalue)

#     # Interpretation
#     if result.pvalue < 0.05:
#         print("The difference in performance is statistically significant.")
#     else:
#         print("The difference in performance is not statistically significant.")


