# ----------------------------------------------------------------------
# Dynamic program functions: fill dp matrix, find first layer, wrapper
# ----------------------------------------------------------------------

import sys
import numpy as np
from copy import deepcopy 
import pickle
from tqdm import tqdm
from general_helper_funcs import *

def clockwise_range(i, j, n):
    """Returns all values between [i, j) clockwise, given that the highest value is n - 1.

        :param i: Start index
        :type i: int
        :param j: End index, first index not included in range
        :type j: int
        :param n: One more than the highest value
        :type n: int
        :return: List of all values between [i, j) clockwise
        :rtype: list
        """
    if i <= j:
        return list(range(i, j))
    else:
        return list(range(i, n)) + list(range(0, j))

def find_first_layer(dp_mat, n, p, disable_tqdm=False):
    """Finds best layer boundaries for full shape.

    :param dp_mat: DP matrix storing best layer boundaries for different p' < p where each entry is a tuple of (loss, layer boundaries)
    :type dp_mat: ndarray
    :param n: Length of boundary points
    :type n: int
    :param p: Number of layer boundaries
    :type p: int
    :param disable_tqdm: If True, then tqdm output will be surpressed
    :type disable_tqdm: bool, optional
    :return: Tuple of minimum loss, p best layer boundaries for full shape
    :rtype: tuple
    """
    overall_min = np.inf
    overall_lines = []
    # try all pairs of coords as first layer boundary
    for i in tqdm(range(n), disable=disable_tqdm, desc=f'Finding layer boundary {p}'):
        for j in range(n):
            if not i == j:
                a_min, a_lines = dp_mat[0, i, j]
                b_min, b_lines = dp_mat[p - 1, j, i]
                cur_min = a_min + b_min
                # save best first layer boundary found so far
                if cur_min < overall_min:
                    overall_min = cur_min
                    overall_lines = deepcopy(a_lines) + [(i, j)] + deepcopy(b_lines)
    return overall_min, overall_lines

def dp_helper(points, p, i, j, dp_mat, func):
    """DP helper recurrence to fill DP matrix

    :param points: List of boundary coordinates
    :type points: list
    :param p: Number of layer boundaries
    :type p: int
    :param i: Index of first layer boundary endpoint, possible next layer boundary endpoints lie in [i + 1, j - 1]
    :type i: int
    :param j: Index of first layer boundary endpoint
    :type j: int
    :param dp_mat: DP matrix storing best layer boundaries for different p, i, j where each entry is a tuple of (loss, layer boundaries)
    :type dp_mat: ndarray
    :param func: Loss function with parameters (points, i, j, k, m). When k, m is not None, returns loss on region defined by convex_hull(ik), line(km), convex_hull(mj), line(ji) Otherwise, returns loss on region defined by convex_hull(ij), line(ji)
    :type func: function
    """
    n = len(points)
    # we do not need to find layer boundaries when p = 0
    if p == 0:
        dp_mat[p, i, j] = (func(points, i, j, None, None), [])
        return
    # if not enough endpoints between i,j, return infeasible
    elif (j - i) % n < 2 * p + 1:
        dp_mat[p, i, j] = (np.inf, [])
        return
    # otherwise, check all possible endpoints for next layer boundary
    else:
        min_cost = np.inf
        min_k = -1
        min_m = -1
        min_lines = []
        # check all k, m in [i + 1, j - 1] such that k < m
        for k in clockwise_range((i + 1) % n, (j - 1) % n, n):
            for m in clockwise_range((k + 1) % n, j % n, n):
                sub_loss, sub_lines = dp_mat[p -1, k, m]
                cur = sub_loss + func(points, i, j, k, m)
                # if current loss is lower than min, set new min
                if cur < min_cost:
                    min_cost = cur
                    min_k = k
                    min_m = m
                    min_lines = sub_lines
        dp_mat[p, i, j] = (min_cost, deepcopy(min_lines) + [(min_k, min_m)])
        return

def construct_dp_mat(points, p, func, disable_tqdm=False):
    """Constructs DP matrix.

    :param points: List of boundary coordinates
    :type points: list
    :param p: Number of layer boundaries
    :type p: int
    :param func: Loss function with parameters (points, i, j, k, m). When k, m is not None, returns loss on region defined by convex_hull(ik), line(km), convex_hull(mj), line(ji) Otherwise, returns loss on region defined by convex_hull(ij), line(ji)
    :type func: function
    :param disable_tqdm: If True, then tqdm output will be surpressed
    :type disable_tqdm: bool, optional
    :return dp_mat: DP matrix storing best layer boundaries for different p, i, j where each entry is a tuple of (loss, layer boundaries)
    :rtype dp_mat: ndarray
    """
    n = len(points)
    dp_mat = np.empty((p + 1, n, n), dtype=tuple)
    # for p_prime in tqdm(range(p), disable=disable_tqdm, desc='DP level'):
    for p_prime in range(p):
        for i in tqdm(range(n), disable=disable_tqdm, desc=f'{p_prime} level index', file = sys.stdout):
            for j in range(n):
                if i == j:
                    dp_mat[p, i, j] = np.inf
                else:
                    dp_helper(points, p_prime, i, j, dp_mat, func)
    return dp_mat

def dp_outer(points, p, func, disable_tqdm=False):
    """Wrapper for full DP run.

    :param points: List of boundary coordinates
    :type points: list
    :param p: Number of layer boundaries
    :type p: int
    :param func: Loss function with parameters (points, i, j, k, m). When k, m is not None, returns loss on region defined by convex_hull(ik), line(km), convex_hull(mj), line(ji) Otherwise, returns loss on region defined by convex_hull(ij), line(ji)
    :type func: function
    :param disable_tqdm: If True, then tqdm output will be surpressed
    :type disable_tqdm: bool, optional
    :return: Tuple of minimum loss, p best layer boundaries for full shape
    :rtype: tuple
    """
    dp_mat = construct_dp_mat(points, p, func, disable_tqdm)
    overall_min, overall_lines = find_first_layer(dp_mat, len(points), p, disable_tqdm)
    return overall_min, overall_lines


def run_dp_linear_boundary(FOLDER_TO_SAVE_LOSSES_IN, sorted_boundary, max_nlayers, PREFIX_TO_SAVE_DP):
    """Wrapper to run DP with linear layer boundaries using pre-computed loss (likelihood) values.

    :param FOLDER_TO_SAVE_LOSSES_IN: prefix of pre-computed likelihood pickle filename, the input to precompute_class
    :type FOLDER_TO_SAVE_LOSSES_IN: str
    :param sorted_boundary: list of 2D coordinates of tissue boundary points (sorted by clockwise order)
    :type sorted_boundary: list of pairs
    :param max_nlayers: maximum number of layers
    :type max_nlayers: int
    :param PREFIX_TO_SAVE_DP: output folder to save the optimal DP likelihood objective and corresponding layer boundary lines for number of layers from 1 to max_nlayers
    :type PREFIX_TO_SAVE_DP: str

    :return: A pair of optimal DP objective function values and corresponding layer boundaries. Note that for any of number layers <= max_nlayers, the optimal solution can be retrieved from the DP table and will be saved and returned.
    :rtype: (np.array, np.array)
    """
   # load pre-saved losses
    pre_saved_loss = pickle.load(open(f"{FOLDER_TO_SAVE_LOSSES_IN}_pre_saving.pkl", 'rb'))

    # create wrapper for linear DP
    loss_wrapper = gen_loss_wrapper(pre_saved_loss)

    # initialize arrays for endpoints, loss, runtime
    arr_layers, arr_loss = [list(np.zeros(max_nlayers + 1)) for _ in range(2)]

    # create DP matrix
    dp_mat = construct_dp_mat(sorted_boundary, max_nlayers, loss_wrapper)

    # for number of layers from 1 to max_layers, find best layer boundaries
    for num_layers in range(1, max_nlayers + 1):
        res = find_first_layer(dp_mat, len(sorted_boundary), num_layers)
        loss, layers = res
        
        # order layer boundaries using func above
        layers = [[sorted_boundary[ele[0]], sorted_boundary[ele[1]]] for ele in layers]
        ordered_layers = order_layer_boundaries(layers)
        
        # save layer boundaries, loss, runtime
        arr_layers[num_layers] = ordered_layers
        arr_loss[num_layers] = loss
        
    # save loss and ordered layer boundaries 
    arr_loss_np = np.asarray(arr_loss, dtype=object)
    arr_layers_np = np.asarray(arr_layers, dtype=object)

    np.save(f'{PREFIX_TO_SAVE_DP}_dp_loss.npy', arr_loss_np, allow_pickle=True, fix_imports=True)
    np.save(f'{PREFIX_TO_SAVE_DP}_dp_layers.npy', arr_layers_np, allow_pickle=True, fix_imports=True)

    return arr_loss_np, arr_layers_np