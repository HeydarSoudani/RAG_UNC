
import numpy as np
from scipy.stats import norm
from scipy.stats import rankdata
import matplotlib.pyplot as plt


def reflected_Gaussian_kernel(x, y, sigma = 0.05):
    '''
    Compute the reflected Gaussian kernel between two numbers in [0,1].
    Input:
        x \in [0, 1]
        y \in [0, 1]
        sigma: the kernel width
    Output:
        the paired kernel values K_\sigma(x, y) \in R_+
    '''
    return np.sum([norm.pdf(x-y+2*k, 0, sigma)+norm.pdf(y-x+2*k, 0, sigma) for k in range(-4, 4)]) 

def regressed_correctness_vs_uncertainty_cdf(correctness, uncertainties, num_bins = 20, use_kernel_regress = False, sigma = 0.1):
    '''
    Compute the regressed correctness levels with binning or kernel smoothing.
    Input:
        correctness: (a_1, ..., a_n) \in R^n 
        uncertainties: (u_1, ..., u_n ) \in R^n
    Output:
        uncertainty_cdfs: (p_1=0, ..., p_n=1) \in R^n the CDF estimates of sorted uncertainties
        regressed_correctness: (\bar{a}_1, ..., \bar{a}_n) \in R^n the regressed correctness 
            listed in the sorted order
    '''
    n = len(uncertainties)
    sorted_indices = np.argsort(uncertainties)
    sorted_correctness = correctness[sorted_indices]
    uncertainty_cdfs =  np.arange(0, n)/ (n-1)
    regressed_correctness = np.zeros(n)
    if not use_kernel_regress:
        bin_endpoints = [round(ele) for ele in np.linspace(0, n, num_bins+1)]
        for idx_bin in range(1, num_bins+1):
            lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
            if hi > lo:
                a_hat = np.mean(sorted_correctness[lo:hi])
                for i in range(lo, hi):
                    regressed_correctness[i] = a_hat
        return regressed_correctness, uncertainty_cdfs
    elif use_kernel_regress:
        kernel_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                kernel_mat[i][j] = reflected_Gaussian_kernel(uncertainty_cdfs[i], uncertainty_cdfs[j], sigma)
                kernel_mat[j][i] = kernel_mat[i][j]
        regressed_correctness = kernel_mat@sorted_correctness/kernel_mat.sum(axis=1)
        return regressed_correctness, uncertainty_cdfs
    
def indication_diagram(correctness, uncertainties, fig, ax, num_bins=20, use_kernel = False, sigma=0.1, **kwargs):
    '''
        Draw the indication diagram.
        Option I: using equal-width binning.
        Option II: using kernel smoothing.
    '''
    n = len(correctness)
    regressed_correctness, uncertainty_cdfs = regressed_correctness_vs_uncertainty_cdf(correctness=correctness, uncertainties=uncertainties,\
                                                                                        num_bins=num_bins, use_kernel_regress=use_kernel, sigma=sigma)
    
    regressed_correctness_cdfs = np.array([(np.sum([regressed_correctness[i] >= regressed_correctness])-1)/(n-1) for i in range(n)])
    if not use_kernel:
        bin_endpoints = [round(ele) for ele in np.linspace(0, n, num_bins+1)]
        # compute binned a_hat, u_hat for histogram ploting 
        a_hats, u_hats = [], []
        for idx_bin in range(1, num_bins+1):
            # breakpoint()
            lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
            if hi > lo:
                a_hat = np.mean(regressed_correctness_cdfs[lo:hi]) - (hi-lo-1)/(2*(n-1))
                u_hat = np.mean(uncertainty_cdfs[lo:hi])
                a_hats.append(a_hat)
                u_hats.append(u_hat)
        # sns.barplot(x=[round(u_hat*100) for u_hat in u_hats], y=[a_hat*100 for a_hat in a_hats], ax=ax, **kwargs)
        ucc, acc, B = np.array(u_hats), np.array(a_hats), num_bins
        ax.bar(np.arange(B)/B*100, np.minimum(1-ucc, acc)*100, width=100/B, color='crimson', align='edge', edgecolor='crimson', label='CDF($\mathbb{E}[A|U]$)')
        ax.bar(np.arange(B)/B*100, (1-ucc-np.minimum(1-ucc, acc))*100, width=100/B, bottom=np.minimum(1-ucc, acc)*100, color='dodgerblue', align='edge', edgecolor='dodgerblue')
        # ax.bar(np.arange(B)/B*100, (1-ucc-np.minimum(1-ucc, acc))*100, width=100/B, bottom=np.minimum(1-ucc, acc)*100, color='dodgerblue', align='edge', label='CDF($U$)')
        ax.bar(np.arange(B)/B*100, (acc-np.minimum(1-ucc, acc))*100, width=100/B, bottom=np.minimum(1-ucc, acc)*100, color='salmon', align='edge', edgecolor='salmon')
        # Plot the anti-diagonal line
        ax.plot([100, 0], [0, 100], linestyle='--', color='black', linewidth=2)
        # Add legend
        # ax.legend(loc='upper right', frameon=False, fontsize=15)
        # ax.set_xlabel('Percentage of Unertainty (%)', fontsize=15)
        # ax.set_ylabel('Percentage of Regressed Correctness (%)', fontsize=15)
        plt.xlim(0, 100)
        ax.grid()
        fig.tight_layout()
        return ax
    else:
        plt.plot(uncertainty_cdfs*100, regressed_correctness_cdfs*100, color='r', linewidth=2)
        # ax.set_xlabel('Percentage of Unertainty (%)', fontsize=15)
        # ax.set_ylabel('Percentage of Regressed Correctness (%)', fontsize=15)
        ax.grid()
        fig.tight_layout()
        return ax

def plugin_RCE_est(correctness, uncertainties, num_bins=20, p=1, use_kernel = False, sigma=0.1, **kwargs):
    '''
    Input:
        uncertainties: (U_1, ... , U_n) \in R^n
        correctness: (A_1, ..., A_n) \in [0, 1]^n
        num_bins: B
    Output:
        Plug-in estimator of l_p-ERCE(f)^p w.r.t. B equal-mass bins
    '''

    n = len(correctness)
    regressed_correctness, uncertainty_cdfs = regressed_correctness_vs_uncertainty_cdf(correctness=correctness, uncertainties=uncertainties,\
                                                                                        num_bins=num_bins, use_kernel_regress=use_kernel, sigma=sigma)
    regressed_correctness_inv_cdfs = np.array([(np.sum([regressed_correctness[i] <= regressed_correctness])-1)/(n-1) for i in range(n)])
    if not use_kernel:
        # compute the detied (due to binning) inverse cdf of regressed correctness
        regressed_correctness_detied_inv_cdfs = np.zeros(n)
        bin_endpoints = [round(ele) for ele in np.linspace(0, n, num_bins+1)]
        for idx_bin in range(1, num_bins+1):
            lo, hi = bin_endpoints[idx_bin-1], bin_endpoints[idx_bin]
            if hi > lo:
                regressed_correctness_detied_inv_cdfs[lo:hi] = np.mean(regressed_correctness_inv_cdfs[lo:hi]) - (hi-lo-1)/(2*(n-1))
    if use_kernel:
        regressed_correctness_detied_inv_cdfs = regressed_correctness_inv_cdfs

    if p == 1:
        return np.sum(np.abs(regressed_correctness_detied_inv_cdfs - uncertainty_cdfs))/n
    elif p == 2:
        return np.sum((regressed_correctness_detied_inv_cdfs - uncertainty_cdfs)**2)/n
    else:
        raise ValueError("Please specify a valid order p!")


### === ECE 
def plugin_ece_est(correctness, confidences, num_bins, p=2, debias=True):
    '''
    Input:
        confidences: (C_1, ... , C_n) \in [0, 1]^n
        correctness: (A_1, ..., A_n) \in [0, 1]^n
        num_bins: B
        debias: If True, debias the plug-in estimator (only for p = 2)
    Output:
        Plug-in estimator of l_p-ECE(f)^p w.r.t. B equal-width bins
    '''
    # reindex to [0, min(num_bins, len(scores))]
    indexes = np.floor(num_bins * confidences).astype(int) 
    indexes = rankdata(indexes, method='dense') - 1
    counts = np.bincount(indexes)

    if p == 2 and debias:
        counts[counts < 2] = 2
        error = ((np.bincount(indexes, weights=confidences-correctness)**2
              - np.bincount(indexes, weights=(confidences-correctness)**2)) / (counts-1)).sum()
    else:
        counts[counts == 0] = 1
        error = (np.abs(np.bincount(indexes, weights=confidences-correctness))**p / counts**(p - 1)).sum()

    return error / len(confidences)

def adaptive_ece_est(correctness, confidences):
    '''
    Input:
        confidences: (C_1, ... , C_n) \in [0, 1]^n
        correctness: (A_1, ..., A_n) \in [0, 1]^n
    Output:
        Adaptive debiased estimator of l_p-ECE(f)^2 using the dyadic grid of binning numbers
    '''

    num_bins_list = [2**b for b in range(1, np.floor(np.log2(len(confidences))-2).astype(int))]
    return np.max([plugin_ece_est(confidences, correctness, num_bins, p=2, debias=True) for num_bins in num_bins_list])

class ECE_estimate():

    def __init__(self, metric_name='ADPE'):
        self.metric_name = metric_name
        if metric_name not in ['PE', 'PE2', 'DPE', 'ADPE']:
            raise ValueError("Please specify a valid calibration metric!")
        self.metric = {'PE': lambda c, y, B: plugin_ece_est(y, c, B, 1, False),\
                        'PE2': lambda c, y, B: plugin_ece_est(y, c, B, 2, False),\
                        'DPE': lambda c, y, B: plugin_ece_est(y, c, B, 2, True),\
                        'ADPE': lambda c, y: adaptive_ece_est(y, c)}[metric_name]

    def __call__(self, labels, confidences, num_bins=None):
        if self.metric_name == 'ADPE':
            return self.metric(labels, confidences, )
        else:
            return self.metric(labels, confidences, num_bins)

