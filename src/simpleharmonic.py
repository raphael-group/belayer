import numpy as np
import scipy
import networkx as nx
from numba import njit
# from spatialcoord import *


def get_signed_distance(coords, endp1, endp2, p_within):
    nvec = np.array([ -(endp2[1] - endp1[1]), (endp2[0] - endp1[0]) ])
    nvec = nvec / np.sqrt(nvec.dot(nvec))
    c = np.array([endp1[0], endp1[1]]).dot(nvec)
    assert np.abs( np.array([endp2[0], endp2[1]]).dot(nvec) - c ) < 1e-4
    # which side of boundary p_within falls into
    side = (np.array([p_within[0], p_within[1]]).dot(nvec) - c > 0)
    # signed distance from all points to boundary
    signed_distance = coords.dot(nvec) - c
    return side, signed_distance


def fractional_hausdorff_distance(pairwise_dist, point_list_A, point_list_B, percentile=100):
    dist = pairwise_dist[np.ix_( np.array(point_list_A), np.array(point_list_B) )]
    from_A_to_B = np.min(dist, axis=1)
    from_A_to_B = from_A_to_B.A.flatten()
    from_B_to_A = np.min(dist, axis=0)
    from_B_to_A = from_B_to_A.A.flatten()
    if len(point_list_B) > len(point_list_A):
        phd = np.percentile(from_A_to_B, percentile)
    else:
        phd = np.percentile(from_B_to_A, percentile)
    return phd


# def check_consistency(b1_i, b1_j, b2_i, b2_j):
#     idx_b1, idx_b2, idx_inside = interpolation.get_spots_index_within_region(coords[b1_i], coords[b1_j], coords[b2_i], coords[b2_j])
#     idx_b1_o, idx_b2_o, idx_inside_o = interpolation_old.get_spots_index_within_region(b1_i, b1_j, b2_i, b2_j)
#     check1 = np.all(idx_b1 == idx_b1_o)
#     check2 = np.all(idx_b2 == idx_b2_o)
#     check3 = np.all(idx_inside == idx_inside_o)
#     return np.all([check1, check2, check3])


# def tmp():
#     check_consistency(2000, 1249, 2150, 3498)
#     check_consistency(10, 2049, 1, 2649)
#     check_consistency(10, 949, 2850, 2899)
#     check_consistency(13, 499, 2900, 2049)
#     check_consistency(450, 2449, 2250, 3471)
#     check_consistency(32, 3493, 1200, 3466)
#     check_consistency(7, 3494, 2450, 3476)
#     check_consistency(1600, 3482, 2500, 3458)
#     check_consistency(28, 1749, 2000, 2699)
#     check_consistency(25, 1799, 2600, 3477)


class simpleinterpolation(object):
    '''
    Harmonic interpolation with two linear boundaries.
    '''
    def __init__(self, coords, adjacency_mat, pairwise_dist, Gamma_0=None, Gamma_L=None, method="harmonic", sigma=0.5):
        self.coords = coords
        self.adjacency_mat = scipy.sparse.csr_matrix(adjacency_mat)
        self.pairwise_dist = scipy.sparse.csr_matrix(pairwise_dist)
        self.method = method
        self.sigma = sigma
        self.Gamma_0 = Gamma_0
        self.Gamma_L = Gamma_L
        # store graph laplacian
        self.Laplacian = np.diag(np.sum(adjacency_mat, axis=0)) - adjacency_mat
        self.Laplacian = scipy.sparse.csr_matrix(self.Laplacian)
    #
    def get_spots_index_within_region(self, b1_endp1, b1_endp2, b2_endp1, b2_endp2, threshold=1):
        eps = 1e-6
        # line for boundary 1: <(x,y), n> = c
        side1, signed_distance1 = get_signed_distance(self.coords, b1_endp1, b1_endp2, b2_endp1)
        same_side1 = np.logical_or( np.abs(signed_distance1) < eps, side1 == (signed_distance1 > 0))
        indicator1 = (np.abs(signed_distance1) < threshold) * same_side1
        # line for boundary 2: <(x,y), n> = c
        side2, signed_distance2 = get_signed_distance(self.coords, b2_endp1, b2_endp2, b1_endp1)
        same_side2 = np.logical_or( np.abs(signed_distance2) < eps, side2 == (signed_distance2 > 0))
        indicator2 = (np.abs(signed_distance2) < threshold) * same_side2
        # remove overlapping between indicator1 and indicator2
        shared = np.where(np.logical_and(indicator1, indicator2))[0]
        moved = shared[np.abs(signed_distance1)[shared] < np.abs(signed_distance2)[shared]]
        indicator1[shared] = False
        indicator1[moved] = True
        indicator2[moved] = False
        # find all spots between the two layer boundaries
        indicator = np.logical_and.reduce((side1 == (signed_distance1 > 0), side2 == (signed_distance2 > 0), \
            np.logical_not(indicator1), np.logical_not(indicator2) ))
        # indicators to indices
        idx_b1 = np.where(indicator1)[0]
        idx_b2 = np.where(indicator2)[0]
        idx_inside = np.where(indicator > 0)[0]
        return idx_b1, idx_b2, idx_inside
    #
    def get_spots_index_within_halfcircle(self, b1_endp1, b1_endp2, bnext, threshold=1):
        # line for boundary 1: <(x,y), n> = c
        side1, signed_distance1 = get_signed_distance(self.coords, b1_endp1, b1_endp2, bnext)
        same_side1 = np.logical_or(signed_distance1 == 0, side1 == (signed_distance1 > 0))
        indicator1 = (np.abs(signed_distance1) < threshold) * same_side1
        indicator = np.logical_and(side1 == (signed_distance1 > 0), np.logical_not(indicator1) )
        # indicators to indices
        idx_b1 = np.where(indicator1)[0]
        idx_inside = np.where(indicator > 0)[0]
        return idx_b1, idx_inside
    #
    def get_spots_index_within_halfcircle_v2(self, b1_endp1, b1_endp2, other_boundary="first", threshold=1):
        eps = 1e-6
        assert (not self.Gamma_0 is None) and (not self.Gamma_L is None), "Gamma_0 or Gamma_L haven't been set!"
        if other_boundary == "first":
            bnext = self.Gamma_0[0]
            idx_b2 = self.Gamma_0
        elif other_boundary == "last":
            bnext = self.Gamma_L[0]
            idx_b2 = self.Gamma_L
        else:
            raise Exception("other_boundary must of one of the following: \"first\", \"last\"")
        # line for boundary 1: <(x,y), n> = c
        side1, signed_distance1 = get_signed_distance(self.coords, b1_endp1, b1_endp2, self.coords[bnext])
        same_side1 = np.logical_or( np.abs(signed_distance1) < eps, side1 == (signed_distance1 > 0))
        indicator1 = (np.abs(signed_distance1) < threshold) * same_side1
        # find all spots between the two layer boundaries
        indicator2 = np.zeros(len(self.coords[:,0]), dtype=bool)
        indicator2[idx_b2] = True
        indicator = np.logical_and.reduce((side1 == (signed_distance1 > 0), \
            np.logical_not(indicator1), np.logical_not(indicator2) ))
        # remove from indicator1 the shared ones with indicator2
        shared = np.where(np.logical_and(indicator1, indicator2))[0]
        indicator1[shared] = False
        # indicators to indices
        idx_b1 = np.where(indicator1)[0]
        idx_inside = np.where(indicator > 0)[0]
        return idx_b1, idx_b2, idx_inside
    #
    def interpolate_harmonic(self, idx_b1, idx_b2, idx_inside):
        phd = fractional_hausdorff_distance(self.pairwise_dist, idx_b1, idx_b2, percentile=50)
        if len(idx_inside) == 0:
            return np.array([0]*len(idx_b1)), np.array([phd]*len(idx_b2)), np.array([])
        else:
            heats = [0] * len(idx_b1) + [phd] * len(idx_b2)
            idx_all = np.concatenate( (idx_b1, idx_b2, idx_inside) )
            adjacency_mat = self.adjacency_mat[idx_all,:][:,idx_all]
            # L = np.diag(np.sum(adjacency_mat, axis=0)) - adjacency_mat
            L = self.Laplacian[idx_all, :][:, idx_all]
            L_I = L[(len(idx_b1)+len(idx_b2)):,:][:,(len(idx_b1)+len(idx_b2)):]
            D = L[:(len(idx_b1)+len(idx_b2)),:][:,(len(idx_b1)+len(idx_b2)):]
            heats_rest, info = scipy.sparse.linalg.cg(L_I, -D.T.dot(heats))
            # heats_rest = scipy.sparse.linalg.spsolve(L_I, -D.T.dot(heats))
            return np.array([0]*len(idx_b1)), np.array([phd]*len(idx_b2)), heats_rest
    #
    def diffusion_harmonic(self, idx_b1, idx_inside, max_iter=2000):
        assert len(idx_b1) > 0 and len(idx_inside) > 0
        idx_all = np.concatenate( (idx_b1, idx_inside) )
        adjacency_mat = self.adjacency_mat[idx_all,:][:,idx_all]
        pairwise_dist = self.pairwise_dist[idx_all,:][:,idx_all] #
        d = np.sum(adjacency_mat, axis=0).A.flatten()
        d[np.where(d == 0)[0]] = 1
        idx_connected = np.where(d > 2)[0]
        R = 1.0 * np.diag(d) + adjacency_mat.A
        R = R / np.sum(R, axis=0, keepdims=True)
        # initialization
        x = np.zeros(adjacency_mat.shape[0])
        phd = np.percentile(np.min(pairwise_dist[:len(idx_b1), :][:, len(idx_b1):], axis=0).A.flatten(), 50)
        x[:len(idx_b1)] = phd # old version is 1
        for r in range(max_iter):
            newx = x.dot(R)
            newx[:len(idx_b1)] = phd
            x = newx
            if np.sum(x[idx_connected] < phd * 1e-2) == 0:
                break
        return x[:len(idx_b1)], x[len(idx_b1):]
    #
    def interpolate_tps(self, idx_b1, idx_b2, idx_inside):
        phd = fractional_hausdorff_distance(self.pairwise_dist, idx_b1, idx_b2, percentile=50)
        if len(idx_inside) == 0:
            return np.array([0]*len(idx_b1)), np.array([phd]*len(idx_b2)), np.array([])
        else:
            heats = [0] * len(idx_b1) + [phd] * len(idx_b2)
            idx_both_b = np.concatenate( [idx_b1, idx_b2] )
            fun = scipy.interpolate.Rbf(self.coords[idx_both_b,0], self.coords[idx_both_b,1], heats, function="thin-plate", smooth=0.5)
            heats_b1 = np.array([ fun(self.coords[i,0], self.coords[i,1]) for i in idx_b1 ])
            heats_b2 = np.array([ fun(self.coords[i,0], self.coords[i,1]) for i in idx_b2 ])
            heats_rest = np.array([ fun(self.coords[i,0], self.coords[i,1]) for i in idx_inside ])
            return heats_b1, heats_b2, heats_rest
    #
    def interpolate(self, idx_b1, idx_b2, idx_inside):
        if self.method == "harmonic":
            return self.interpolate_harmonic(idx_b1, idx_b2, idx_inside)
        else:
            return self.interpolate_tps(idx_b1, idx_b2, idx_inside)