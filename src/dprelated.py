import numpy as np
from glmpca import glmpca
from utils import rotate_by_theta

# INPUT:
# Y: matrix of size G x T
# xcoords: list of T x-coordinates

# OUTPUT:
# Fit each row y_g of Y as y_g = xcoords * theta + const using linear regression
# (1) theta, const for each gene g
# (2) Total loss \sum_g ||y_g - yhat_g||^2 
def opt_linear(y, xcoords):
    g,n = y.shape
    if n==1:
        return np.array([]),0
    
    X=np.ones((n,2))
    X[:,0]=xcoords

    try:
        theta=np.linalg.inv(X.T @ X) @ X.T @ y.T
        error=np.linalg.norm(X @ theta - y.T)**2

        # linreg coefficients theta.T have size g x 2
        # each row is the linreg coefficients for gene g
        return theta.T,error
    except: # in case no linreg fit
        placeholder=np.ones((2,g))-2
        return placeholder, np.Inf
    
# INPUT (TODO: UPDATE)
# data: array of size G x T
# L: number of pieces
# xcoords: size T
# opt_function: how to fit each row of data (eg linear, Poisson)
# min_seg_size: minimum segment size

# OUTPUT:
# (1) error_mat: DP table of size T x L
# where error_mat[t,l] = error from using linreg to fit first t+1 times using (l+1)-piece segmented regression
# (2) segment_map: pointers for DP table
# (t,l) -> (t', l-1) where you use l-1 pieces for times 1,...,t' and one piece for t'+1,...,t
def dp_raw(data, Lmax, xcoords, opt_function=opt_linear):
    G=data.shape[0]
    T=data.shape[1]
    
    sorted_xcoords=np.sort(xcoords)
    sorted_xcoords_inds=np.argsort(xcoords)

    # dp table error_mat (T x L)
    # where error_mat[t,l] = error from fitting first t+1 coords using (l+1)-piece segmented regression
    error_mat=np.zeros((T,Lmax))

    # map (t,p) -> (t', p-1) where you use p-1 segments for times 1,...,t' and a segment for t'+1,...,t
    segment_map={}

    # save previous calls to opt_function
    saved_opt_functions=np.zeros((T+1,T+1)) - 1

    # fill out first column of matrix (0th row is just 0)
    for t in range(T):
        xc=sorted_xcoords[:t+1]
        xc_inds=sorted_xcoords_inds[:t+1]
        
        _,err=opt_function(data[:,xc_inds],xc)
        error_mat[t,0]=err

    # fill out each subsequent column l
    for l in range(1,Lmax):

        # for each column, go from top to bottom [ignoring first row]
        for t in range(T):
            best_tprime=-1
            best_error=np.Inf

            for tprime in range(t):
                if saved_opt_functions[tprime+1,t+1] >= 0:
                    tprime_fit=saved_opt_functions[tprime+1,t+1]
                else:
                    xc=sorted_xcoords[tprime+1:t+1]
                    xc_inds=sorted_xcoords_inds[tprime+1:t+1]
                    
                    tprime_fit=opt_function(data[:,xc_inds], xc)[1]
                    saved_opt_functions[tprime+1,t+1]=tprime_fit
                cur_error=error_mat[tprime,l-1] + tprime_fit

                if cur_error < best_error:
                    best_error=cur_error
                    best_tprime=tprime
            error_mat[t,l] = best_error
            segment_map[(t,l)] = (best_tprime,l-1)

    return error_mat, segment_map

# INPUT (TODO: UPDATE)
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
def dp_bucketized(data, bucket_endpoints, Lmax, xcoords, opt_function=opt_linear):
    G=data.shape[0]
    T=data.shape[1]
    
    B=len(bucket_endpoints)-1
    buckets=np.digitize(xcoords, bucket_endpoints) - 1 # for some reason bucket labels are 1-indexed
    
    # dp on matrix error_mat of size B x Lmax
    
    # where error_mat[b,p] = error from using linreg to fit 
    # first b+1 buckets using (p+1)-part segmented regression
    error_mat=np.zeros((B,Lmax))

    # map (b,p) -> (b', p-1) where you use p-1 segments for buckets 1,...,b' and a segment for b'+1,...,b
    segment_map={}

    # saved opt_function
    saved_opt_functions=np.zeros((B+1,B+1)) - 1

    # fill out first column of matrix (0th row is just 0)
    for b in range(B):
        inds_upto_bucket_b = np.where(buckets <= b)[0]
        
        xc=xcoords[inds_upto_bucket_b]
        _,err=opt_function(data[:,inds_upto_bucket_b],xc)
        error_mat[b,0]=err

    # fill out each subsequent column p
    for p in range(1,Lmax):

        # for each column, go from top to bottom [ignoring first row]
        for b in range(B):
            # fill out entry t,p

            best_bprime=-1
            best_error=np.Inf
            for bprime in range(b):
                
                if saved_opt_functions[bprime+1,b+1] >= 0:
                    bprime_fit=saved_opt_functions[bprime+1,b+1]
                else:
                    inds_between_bprime_b=np.where( (buckets > bprime) & (buckets <= b) )[0]
                    
                    xc=xcoords[inds_between_bprime_b]                    
                    
                    bprime_fit=opt_function(data[:,inds_between_bprime_b], xc)[1]
                    saved_opt_functions[bprime+1,b+1]=bprime_fit
                    
                cur_error=error_mat[bprime,p-1] + bprime_fit
                if cur_error < best_error:
                    best_error=cur_error
                    best_bprime=bprime
            error_mat[b,p] = best_error
            segment_map[(b,p)] = (best_bprime,p-1)
    
    return error_mat, segment_map


# INPUT (TODO: UPDATE)
# data: matrix of size G x N (count matrix or GLM-PC reduced matrix)
# xcoords: array of size N x 1 of xcoords
# L: number of pieces

# OUTPUT:
# layer_pooled: N_1d x 1 matrix of layer labels for each pooled spot
# layer_pooled: N x 1 matrix of layer labels for each (2D) spot
# def dp(count, xcoords, L, use_buckets=True, num_buckets=150):
def rotation_dp(data, coords, Lmax=8, rotation_angle_list=[0,5,10,15,17.5,20], 
                use_buckets=True, num_buckets=150, glmpca_penalty=10,
               opt_function=opt_linear):
    
    G = data.shape[0]
    N = data.shape[1]

    # glmpca_res = glmpca.glmpca(count, 2*Lmax, fam="poi", penalty=glmpca_penalty, verbose=True)
    # F_glmpca = glmpca_res['factors']
    
    # loop over all angles in rotation_angle_list
    theta_list=[deg*np.pi/180 for deg in rotation_angle_list]
    
    # each row is [loss at layers 1, ..., Lmax] for each rotation angle
    losses=np.zeros( (len(rotation_angle_list), Lmax) )
    labels={}
    
    for ind_t,theta in enumerate(theta_list):
        print('\n angle: {}'.format(rotation_angle_list[ind_t]))
        coords_rotated=rotate_by_theta(coords,theta)
        xcoords_rotated=coords_rotated[:,0]
        if use_buckets:
            bin_endpoints=np.linspace(np.min(xcoords_rotated),np.max(xcoords_rotated)+0.01,num_buckets+1)
            print('running DP')
            error_mat,seg_map=dp_bucketized(data, bin_endpoints, Lmax, 
                                            xcoords_rotated, opt_function=opt_linear)

            # get labels, save to res
            for l in range(1,Lmax+1):
                print('finding segments for {} layers'.format(l))
                bin_labels=np.digitize(xcoords_rotated,bin_endpoints)
                segs=find_segments_from_dp(error_mat, seg_map, l)
                dp_labels=np.zeros(N)
                c=0
                for seg in segs:
                    for s in seg:
                        dp_labels[ np.where(bin_labels==s+1)[0] ] = c
                    c+=1
                
                losses[ind_t,l-1] = error_mat[-1,l-1] / N
                labels[(ind_t,l)]=dp_labels
        else:
            print('running DP without buckets')
            error_mat, seg_map = dp_raw(data, Lmax, xcoords_rotated)
            
            # get labels, save to res
            for l in range(1,Lmax+1):
                print('finding segments for {} layers'.format(l))
                segs=find_segments_from_dp(error_mat, seg_map, l, xcoords=xcoords_rotated)
                dp_labels=np.zeros(N)
                c=0
                for seg in segs:
                    dp_labels[seg]=c
                    c+=1
                
                losses[ind_t,l-1] = error_mat[-1,l-1] / N
                labels[(ind_t,l)]=dp_labels
    return losses,labels

# backtrack through DP to find l segments
# TODO: add description!!
def find_segments_from_dp(error_mat, segment_map, l,xcoords=None):
    num_times=error_mat.shape[0]
    
    segs=[[] for i in range(l)]
    seg_val=l-1
    time_val=num_times-1
    
    if xcoords is None:
        xcoords=np.arange(num_times)
    
    sorted_xcoords=np.sort(xcoords)
    sorted_xcoords_inds=np.argsort(xcoords)

    while seg_val > 0:
        new_time_val,new_seg_val=segment_map[(time_val,seg_val)]
        segs[seg_val]=sorted_xcoords_inds[new_time_val+1:time_val+1]
        time_val=new_time_val
        seg_val=new_seg_val
    segs[0]=np.arange(0,time_val+1)
    return segs

# TODO: add description!!
def rotate_by_theta(coords, theta, rotate_about=np.array([0,0])):
    coordsT=coords.T
    
    c,s=np.cos(theta), np.sin(theta)
    rotation_matrix=np.array(((c, -s), (s, c)))
    
    return (rotation_matrix @ coordsT).T