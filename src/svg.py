import numpy as np
import pandas as pd
import pickle
import statsmodels
import statsmodels.api as sm


def select_commonly_expressed_genes(count, depth, q=0.75, threshold=10, num_pools=None):
    """Select commonly expressed genes for fitting expression function per layer. A gene is selected if it is expressed >= threshold UMI in at least q percentage of pooled spots.
    :param count: UMI count matrix of SRT gene expression, G genes by n spots
    :type count: np.array
    :param depth: Inferred layer depth, vector of n spots
    :type depth: np.array
    :param q: Quantile for gene selection.
    :type q: float
    :return: An array of indices of selected genes.
    :rtype: np.array
    """
    G = count.shape[0]
    if num_pools is None:
        num_pools = int(np.sum(count) / 1e5)
    depth_bounds_in_pool = np.linspace(np.min(depth), np.max(depth), num_pools)
    depth_bounds_in_pool[-1] += 1
    pooled_int_data = np.zeros((G, num_pools))
    for i in range(len(depth_bounds_in_pool)-1):
        idx_spots = np.where(np.logical_and(depth >= depth_bounds_in_pool[i], depth < depth_bounds_in_pool[i+1]))[0]
        pooled_int_data[:,i] = np.sum(count[:, idx_spots], axis=1)
    quantiles = np.quantile(pooled_int_data, q, axis=1)
    selection = np.where(quantiles >= threshold)[0]
    return selection


def segmented_poisson_regression(count, totalumi, dp_labels, depth):
    """ Fit Poisson regression per gene per layer.
    :param count: UMI count matrix of SRT gene expression, G genes by n spots
    :type count: np.array
    :param totalumi: Total UMI count per spot, a vector of n spots.
    :type totalumi: np.array
    :param dp_labels: Layer labels obtained by DP, a vector of n spots.
    :type dp_labels: np.array
    :param depth: Inferred layer depth, vector of n spots
    :type depth: np.array
    :return: A dataframe for the offset and slope of piecewise linear expression function, size of G genes by 2*L layers.
    :rtype: pd.DataFrame
    """
    G = count.shape[0]
    unique_layers = np.sort(np.unique(dp_labels))
    L = len(unique_layers)
    # return matrices initialization
    offset = np.zeros((G, L))
    slope = np.zeros((G, L))
    # Poisson regression
    for g in range(G):
        for l,layername in enumerate(unique_layers):
            idx_spots = np.where(dp_labels == layername)[0]
            if np.sum(count[g,idx_spots] > 0) < 2:
                continue
            X = np.vstack([ np.ones(len(idx_spots)), depth[idx_spots] ]).T
            res = statsmodels.discrete.discrete_model.Poisson(count[g,idx_spots], X, exposure=totalumi[idx_spots]).fit(disp=0, maxiter=100)
            offset[g, l] = res.params[0]
            slope[g, l] = res.params[1]
    combined_params = np.zeros((G, 2*L))
    combined_params[:, np.arange(0, 2*L, 2)] = offset
    combined_params[:, np.arange(1, 2*L, 2)] = slope
    df_gene_func = pd.DataFrame(combined_params, columns=sum([[f"intercept {layer}", f"slope {layer}"] for layer in unique_layers], []))
    return df_gene_func