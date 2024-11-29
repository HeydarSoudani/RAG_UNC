import torch

def flatten_likelihoods(distribution, smoothing=0.01):
    smoothed = distribution + smoothing
    flattened = smoothed / smoothed.sum()
    return flattened

def entropy(likelihoods):
    entropy = -sum([item*torch.log(item) for item in likelihoods]) / len(likelihoods)
    return entropy

# Define the samples
sample1 = torch.tensor([0.5, 0.4, 0.1], dtype=torch.float32)
sample2 = torch.tensor([0.9, 0.1], dtype=torch.float32)
sample3 = flatten_likelihoods(sample2, smoothing=0.05)


# Calculate and print the entropy
print(entropy(sample1))
print(entropy(sample2))
print(entropy(sample3))

# python test/entropy_calculation.py
