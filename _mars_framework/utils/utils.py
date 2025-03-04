import torch
import random, os
import numpy as np
from scipy.stats import norm

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def uncertainty_to_confidence_min_max(uncertainty, min_val=None, max_val=None):
    uncertainty = np.array(uncertainty)
    
    if min_val==None and max_val==None:
        min_val = np.min(uncertainty)
        max_val = np.max(uncertainty)
    
    # if max(uncertainty) - min(uncertainty) > 1000:
    #     uncertainty = np.log10(uncertainty)
    #     min_val = np.log10(min_val)
    #     max_val = np.log10(max_val)
    
    normalized_uncertainty = np.array([1 - ((unc - min_val)/(max_val - min_val)) for unc in uncertainty])
    
    return normalized_uncertainty
        

def uncertainty_to_confidence_gaussian(uncertainty, mean=None, std=None):
    uncertainty = np.array(uncertainty)
    
    if np.max(uncertainty) - np.min(uncertainty) > 1000:
        uncertainty = np.log10(uncertainty)
    
    if mean==None and std==None:
        mean = np.mean(uncertainty)
        std = np.std(uncertainty)
    
    standardized_uncertainty = (uncertainty - mean) / std
    normalized_uncertainty = norm.cdf(standardized_uncertainty)
    
    return 1 - normalized_uncertainty
     
     
def uncertainty_to_confidence_sigmoid(uncertainty, alpha=1.0):
    uncertainty = np.array(uncertainty)
    
    if np.max(uncertainty) - np.min(uncertainty) > 1000:
        uncertainty = np.log10(uncertainty)
    
    mean = np.mean(uncertainty)
    confidence = 1 / (1 + np.exp(-alpha * (mean - uncertainty)))  # Adjusted for correct inversion
    return confidence   
   
   
def uncertainty_to_confidence_tanh(uncertainty, alpha=1.0):
    uncertainty = np.array(uncertainty)
    
    if np.max(uncertainty) - np.min(uncertainty) > 1000:
        uncertainty = np.log10(uncertainty)
    
    mean = np.mean(uncertainty)
    std = np.std(uncertainty)
    confidence = 0.5 * (1 + np.tanh(-alpha * (uncertainty - mean) / std))  # Adjusted for correct inversion
    return confidence 
