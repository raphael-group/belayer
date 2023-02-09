import numpy as np
import pandas as pd
import pickle
import statsmodels
import statsmodels.api as sm

from sklearn import linear_model,preprocessing
from scipy.stats import mode
import matplotlib.pyplot as plt
from tqdm import trange


def select_commonly_expressed_genes(count, q=0.75, threshold=10, pooling=False, depth=None, num_pools=None):
    """Select commonly expressed genes for fitting expression function per layer. A gene is selected if it is expressed >= threshold UMI in at least q percentage of (pooled) spots.
    :param count: UMI count matrix of SRT gene expression, G genes by n spots
    :type count: np.array
    :param q: Quantile for gene selection.
    :type q: float
    :param pooling: whether thresholding based on UMI in original spots or pooled spots
    :type pooling: boolean
    :param depth: Inferred layer depth, vector of n spots (optional, only needed if pooling=True)
    :type depth: np.array
    :return: An array of indices of selected genes.
    :rtype: np.array
    """
    G = count.shape[0]
    if not pooling:
        quantiles = np.quantile(count.T, q, axis=0)
        selection = np.where(quantiles > threshold)[0]
        return selection



def poisson_regression(y, xcoords=None, exposure=None, alpha=0):
    # run poisson fit on pooled data and return slope, intercept
    clf = linear_model.PoissonRegressor(fit_intercept=True,alpha=alpha,max_iter=500,tol=1e-10)
    clf.fit(np.reshape(xcoords,(-1,1)),y/exposure, sample_weight=exposure)

    return [clf.coef_[0], clf.intercept_ ]

def segmented_poisson_regression(count, totalumi, dp_labels, depth, opt_function=poisson_regression, alpha=0):
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

    G, N = count.shape
    unique_layers = np.sort(np.unique(dp_labels))
    L = len(unique_layers)

    slope_matrix=np.zeros((G,L))
    intercept_matrix=np.zeros((G,L))

    for g in trange(G):
        for t in np.arange(L):
            pts_t=np.where(dp_labels==t)[0]

            slope, intercept = opt_function(count[g,pts_t], xcoords=depth[pts_t], exposure=totalumi[pts_t], alpha=alpha)

            slope_matrix[g,t]=slope
            intercept_matrix[g,t]=intercept

    combined_params = np.zeros((G, 2*L))
    combined_params[:, np.arange(0, 2*L, 2)] = intercept_matrix
    combined_params[:, np.arange(1, 2*L, 2)] = slope_matrix

    df_gene_func = pd.DataFrame(combined_params, columns=sum([[f"intercept {layer}", f"slope {layer}"] for layer in unique_layers], []))
    return df_gene_func


# DISCONTINUITY
def compute_discontinuity(df_gene_func, dp_labels, depth):
    """ Compute discontinuity of expression function under Poisson regression between each pair of adjacent layers.
    :param df_gene_func: A dataframe for the offset and slope of piecewise linear expression function, size of G genes by 2*L layers.
    :type df_gene_func: pd.DataFrame
    :param dp_labels: Layer labels obtained by DP, a vector of n spots.
    :type dp_labels: np.array
    :param depth: Inferred layer depth, vector of n spots
    :type depth: np.array
    :return: A dataframe for the discontinuity of gene expression function between each pair of adjacent layers, size of G genes by L-1 layer-pairs.
    :rtype: pd.DataFrame
    """
    G = df_gene_func.shape[0]
    unique_layers = np.sort(np.unique(dp_labels))
    L = len(unique_layers)

    discontinuity = np.zeros((G,L-1))

    for g in range(G):
        for t in np.arange(1, L):
            # discontinuity/difference between layer t-1 and layer t
            depth_t_1=depth[np.where(dp_labels==t-1)[0]]
            depth_t=depth[np.where(dp_labels==t)[0]]
            # breakpoint
            mid_depth = (np.max(depth_t_1) + np.min(depth_t)) / 2 if depth_t_1[0] < depth_t[0] else (np.max(depth_t) + np.min(depth_t_1)) / 2
            # evaluate expression function of layer t-1 at breakpoint
            val_t_1 = df_gene_func.iloc[g, 2*(t-1)] + df_gene_func.iloc[g, 2*(t-1) + 1] * mid_depth
            # evaluate expression function of layer t at breakpoint
            val_t = df_gene_func.iloc[g, 2*(t)] + df_gene_func.iloc[g, 2*(t) + 1] * mid_depth
            discontinuity[g, t-1] = val_t - val_t_1

    discontinuity = pd.DataFrame(discontinuity, columns=[f"discontinuity between {t-1} and {t}" for t in range(1,L)])
    return discontinuity


# BINNING
def bin_data(count, dp_labels, depth):
    exposure=np.sum(count,axis=0) # total UMI per spot
    G,N=count.shape

    # BINNING
    binned_depths=np.round(depth) # each bin is centered at an integer
    unique_binned_depths=np.unique(binned_depths)

    N_1d=len(unique_binned_depths)
    binned_count=np.zeros( (G, N_1d) )
    binned_exposure=np.zeros( N_1d )
    binned_labels=np.zeros(N_1d)

    map_1d_bins_to_2d={} # map b -> [list of cells in bin b]
    for ind, b in enumerate(unique_binned_depths):
        bin_pts=np.where(binned_depths==b)[0]

        binned_count[:,ind]=np.sum(count[:,bin_pts],axis=1)
        binned_exposure[ind]=np.sum(exposure[bin_pts])
        binned_labels[ind]= int(mode( dp_labels[bin_pts] ).mode[0])

        map_1d_bins_to_2d[b]=bin_pts

    L=len(np.unique(dp_labels))
    segs=[np.where(binned_labels==i)[0] for i in range(L)]

    to_return={}
    to_return['binned_depths']=binned_depths
    to_return['unique_binned_depths']=unique_binned_depths
    to_return['binned_count']=binned_count
    to_return['binned_exposure']=binned_exposure
    to_return['binned_labels']=binned_labels
    to_return['map_1d_bins_to_2d']=map_1d_bins_to_2d
    to_return['segs']=segs

    return to_return

# VISUALIZATION
def plot_gene_pwlinear(gene_index, idx_kept, count, slope_offsets, depth, binning_output):
    gene_index_idx_kept=np.where(idx_kept==gene_index)[0][0]
    slope_offsets_g=slope_offsets.iloc[gene_index_idx_kept]

    unique_binned_depths=binning_output['unique_binned_depths']
    binned_labels=binning_output['binned_labels']

    binned_count=binning_output['binned_count']
    binned_exposure=binning_output['binned_exposure']

    segs=binning_output['segs']
    L=len(segs)

    fig,ax=plt.subplots(figsize=(7,3))

    for seg in range(L):
        pts_seg=np.where(binned_labels==seg)[0]
        plt.scatter(unique_binned_depths[pts_seg],
                   np.log( (binned_count[gene_index,pts_seg]) / binned_exposure[pts_seg] ))

        slope=slope_offsets.iloc[gene_index_idx_kept][f'slope {float(seg)}']
        offset=slope_offsets.iloc[gene_index_idx_kept][f'intercept {float(seg)}']

        plt.plot( unique_binned_depths[pts_seg], offset + slope*unique_binned_depths[pts_seg], color='grey', alpha=1 )
