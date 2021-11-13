import numpy as np
from glmpca import glmpca

# INPUT:
# Y: matrix of size G x T
# xcoords: list of T x-coordinates

# OUTPUT:
# Fit each row y_g of Y as y_g = xcoords * theta + const using linear regression
# (1) theta, const for each gene g
# (2) Total loss \sum_g ||y_g - yhat_g||^2 
def opt_linear(Y, xcoords):
    G,T = Y.shape

    # if T=1, can always fit
    if T==1:
        return np.array([]),0

    # if T > 1, compute using linear regression
    # theta has size 2 x G
    # each col is the linreg coefficients for gene g

    X=np.ones((T,2))
    X[:,0]=xcoords
    theta=np.linalg.inv(X.T @ X) @ X.T @ Y.T
    error=np.linalg.norm(X @ theta - Y.T)**2
    
    return theta.T,error


# INPUT:
# Y: matrix of size G x T
# xcoords: list of T x-coordinates

# OUTPUT:
# Fit each row y_g of Y as y_g = const
# (1) const for each gene g
# (2) Total loss \sum_g ||y_g - yhat_g||^2 
def opt_const(y, xcoords):
    G,T=y.shape
    centered_y= y - np.outer( y.mean(axis=1), np.ones(T) )
    return y.mean(axis=1), np.linalg.norm(centered_y)


# INPUT
# data: array of size G x T
# L: number of pieces
# xcoords: size T
# opt_function: how to fit each row of data (eg linear, Poisson)
# min_seg_size: minimum piece size

# OUTPUT:
# (1) error_mat: DP table of size T x L
# where error_mat[t,l] = error from using linreg to fit first t+1 times using (l+1)-piece segmented regression
# (2) segment_map: pointers for DP table
# (t,l) -> (t', l-1) where you use l-1 pieces for times 1,...,t' and one piece for t'+1,...,t
def dp_raw(data, L, xcoords, opt_function=opt_linear, min_seg_size=3):
    G=data.shape[0]
    T=data.shape[1]

    # dp table error_mat (T x L)
    # where error_mat[t,l] = error from fitting first t+1 coords using (l+1)-piece segmented regression
    error_mat=np.zeros((T,L))

    # map (t,p) -> (t', p-1) where you use p-1 segments for times 1,...,t' and a segment for t'+1,...,t
    segment_map={}

    # save previous calls to opt_function
    saved_opt_functions=np.zeros((T+1,T+1)) - 1

    # fill out first column of matrix (0th row is just 0)
    for t in range(min_seg_size-1,T):
        xc=xcoords[:t+1]
        _,err=opt_function(data[:,:t+1],xc)
        error_mat[t,0]=err

    # fill out each subsequent column l
    for l in range(1,L):

        # for each column, go from top to bottom [ignoring first row]
        col_l_starting=(l+1)*min_seg_size-1

        for t in range(col_l_starting,T):
            best_tprime=-1
            best_error=np.Inf

            for tprime in range(l*min_seg_size-1,t-min_seg_size+1):
                if saved_opt_functions[tprime+1,t+1] >= 0:
                    tprime_fit=saved_opt_functions[tprime+1,t+1]
                else:
                    xc=xcoords[tprime+1:t+1]
                    tprime_fit=opt_function(data[:,tprime+1:t+1], xc)[1]
                    saved_opt_functions[tprime+1,t+1]=tprime_fit
                cur_error=error_mat[tprime,l-1] + tprime_fit

                if cur_error < best_error:
                    best_error=cur_error
                    best_tprime=tprime
            error_mat[t,l] = best_error
            segment_map[(t,l)] = (best_tprime,l-1)

    return error_mat, segment_map


# INPUT
# pooled_count: pooled expression matrix of size G x N_1d
# L: number of pieces
# pooled_xcoord: size N_1d x 1, x coordinates for pooled spots
# map_1d_bins_to_2d (output of from pool_data): dictionary mapping each pooled spot -> [2d spots pooled together]


# OUTPUT:
# layer_pooled: N_1d x 1 matrix of layer labels for each pooled spot
# layer_pooled: N x 1 matrix of layer labels for each (2D) spot
def dp(pooled_count, pooled_xcoord, L, map_1d_bins_to_2d):
    G = pooled_count.shape[0]
    N_1d = pooled_count.shape[1]
    N = np.sum([len(v) for v in map_1d_bins_to_2d.values()])

    glmpca_res_1d_poisson = glmpca.glmpca(pooled_count, 2*L, fam="poi", penalty=100, verbose=True,ctl = {"maxIter":1000, "eps":1e-3, "optimizeTheta":True})
    F_glmpca_1d_poisson = glmpca_res_1d_poisson['factors']
    error_mat, seg_map = dp_raw(F_glmpca_1d_poisson.T, L, pooled_xcoords)
    
    # get DP labels
    segs = find_segments_from_dp(error_mat, seg_map, L)
    layer_pooled = np.zeros(N_1d, dtype=np.int)
    layer_2d = np.zeros(N, dtype=np.int)

    c=0
    for seg in segs:
        layer_pooled[ np.array(seg) ] = c
        for s in seg:
            layer_2d[ map_1d_bins_to_2d[pooled_xcoords[s]] ] = c
        c += 1
    return layer_pooled, layer_2d

# backtrack through DP to find l segments
def find_segments_from_dp(error_mat, segment_map, l):
    num_times=error_mat.shape[0]
    
    segs=[[] for i in range(l)]
    seg_val=l-1
    time_val=num_times-1

    while seg_val > 0:
        new_time_val,new_seg_val=segment_map[(time_val,seg_val)]
        segs[seg_val]=np.arange(new_time_val+1,time_val+1)
        time_val=new_time_val
        seg_val=new_seg_val
    segs[0]=np.arange(0,time_val+1)
    return segs


def find_rotation_angle(count, pos, L):
    return NotImplemented
