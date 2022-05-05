import time
import numpy as np
import anndata
import scanpy as sc
from spatialcoord import *
from simpleharmonic import *
from glmpca import glmpca
import networkx as nx
from sklearn import linear_model,preprocessing


def equal_size_bins(interpolation, number_bins):
    min_v = np.min(interpolation)
    max_v = np.max(interpolation)
    bin_endpoints = np.linspace(min_v, max_v, num=number_bins+1)
    bin_endpoints = bin_endpoints[1:]
    bin_endpoints[-1] = bin_endpoints[-1] + 1e-4
    contours = np.array([ np.where(x < bin_endpoints)[0][0] for x in interpolation ])
    return contours


def bin_to_reach_UMI(interpolation, umi_counts, number_bins):
    path = np.argsort( interpolation )
    s = np.min(umi_counts)
    t = np.sum(umi_counts) / number_bins
    step = max(1, int((2*t - s) / 20))
    best_target = s
    best_distance = np.inf
    for target_umi in range(s, int(2*t), step):
        tmpmap_to_bin = bin_spots_to_targetUMI(umi_counts, target_umi=target_umi)
        N_1d = np.max(list(tmpmap_to_bin.values())) + 1
        if N_1d == number_bins:
            best_target = target_umi
            break
        elif np.abs(N_1d - number_bins) < best_distance:
            best_distance = np.abs(N_1d - number_bins)
            best_target = target_umi
    tmpmap_to_bin = bin_spots_to_targetUMI(umi_counts, target_umi=best_target)
    map_to_bin = { path[i]:tmpmap_to_bin[i] for i in range(len(interpolation)) }
    contours = np.array([map_to_bin[i] for i in range(len(interpolation))])
    return contours


def equal_numspots_bins(interpolation, number_bins):
    path = np.argsort( interpolation )
    map_original = {path[i]:i for i in range(len(path))}
    bin_endpoints = np.linspace(0, len(interpolation), num=number_bins+1)
    bin_endpoints = bin_endpoints[1:]
    contours = np.array([ np.where(map_original[i] < bin_endpoints)[0][0] for i in range(len(interpolation)) ])
    return contours


def pooling_spots(count, interpolation, number_bins, pool_method):
    if pool_method == "equal_size":
        contours = equal_size_bins(interpolation, number_bins)
    elif pool_method == "equal_spots":
        contours = equal_numspots_bins(interpolation, number_bins)
    elif pool_method == "reach_umi":
        umi_counts = np.sum(count, axis=0)
        contours = bin_to_reach_UMI(interpolation, umi_counts, number_bins)
    else:
        raise Example("pool_method must be one of the following: equal_size, equal_spots, reach_umi!")
    # from contours to 1D bin to 2D spot map
    unique_contours = np.sort(np.unique(contours))
    G = count.shape[0]
    N_1d = len(unique_contours)
    map_1d_bins_to_2d = {i:[] for i in range(N_1d)}
    for i,x in enumerate(unique_contours):
        v = np.where(contours == x)[0]
        map_1d_bins_to_2d[i] = v
    # summing up counts
    pooled_int_data = np.zeros((G, N_1d))
    for i,v in map_1d_bins_to_2d.items():
        pooled_int_data[:, i] = np.sum(count[:, v], axis=1)
    # take the mean of interpolation as pooled_xcoords
    pooled_xcoords = np.zeros(N_1d)
    for i,v in map_1d_bins_to_2d.items():
        pooled_xcoords[i] = np.mean(interpolation[v])
    return pooled_int_data, pooled_xcoords, map_1d_bins_to_2d


def fill_geometry(points, is_hexagon=True):
    '''
    points: number spots * 2D
    When the geometry is a hexagon, there are even-odd constraints of x and y. E.g. even x values are always paired with odd y values.
    '''
    xrange = [np.min(points[:,0]), np.max(points[:,0])+0.1]
    yrange = [np.min(points[:,1]), np.max(points[:,1])+0.1]
    if is_hexagon:
        even_odd_constraint = {(0,0):False, (0,1):False, (1,0):False, (1,1):False}
        for i in range(points.shape[0]):
            is_even_x = int(points[i,0]) % 2
            is_even_y = int(points[i,1]) % 2
            even_odd_constraint[ (is_even_x, is_even_y) ] = True
    else:
        even_odd_constraint = {(0,0):True, (0,1):True, (1,0):True, (1,1):True}
    # convert points to a set for checking whether a spot in the full geometry is in tissue
    original_set = set([ (x[0],x[1]) for x in points ])
    # fill up the box bounded by xrange * yrange
    newpoints = [x for x in points]
    in_tissue = [True for x in points]
    for x in np.arange(xrange[0], xrange[1]):
        for y in np.arange(yrange[0], yrange[1]):
            if (x,y) in original_set:
                continue
            is_even_x = int(x) % 2
            is_even_y = int(y) % 2
            if is_hexagon and (not even_odd_constraint[ (is_even_x, is_even_y) ]):
                continue
            newpoints.append( np.array([x,y]) )
            in_tissue.append( False )
    newpoints = np.array(newpoints)
    in_tissue = np.array(in_tissue)
    return newpoints, in_tissue


'''
Example usage of the following loss function object:
    from region_cost_fun import *
    llf = lossfunction(count_matrix, x_coord, y_coord, total_num_cluster=5, Gamma_0=Gamma_0, Gamma_L=Gamma_L, forbidden=forbidden, platform="Visium")
    ##### Evaluate loss of a region bounded by two layer boundaries #####
    # b1_i and b1_j are the index of endpoints for layer boundary 1, and b2_i and b2_j are the index of endpoints for layer boundary 2
    loss = llf.eval_loss(b1_i, b1_j, b2_i, b2_j) # this is the l2 loss of points between layer boundary 1 and layer boundary 2
    ##### Evaluate loss of a region bounded by one layer boundary and tissue boundary arcs #####
    # b1_i and b1_j are the index of endpoints for layer boundary 1, and bnext is the index of an arbitrary point in the region to be evaluated.
    # bnext is needed to determine which side of line b1_ib1_j is the region of interest.
    loss = llf.eval_loss(b1_i, b1_j, other_boundary="first")
    loss = llf.eval_loss(b1_i, b1_j, other_boundary="last")
'''
class lossfunction(object):
    def __init__(self, count, x, y, total_num_cluster, Gamma_0=None, Gamma_L=None, forbidden=[], consider_points=None, interpolation_type="harmonic", platform="ST"):
        '''
        Input:
            count: UMI count matrix of size number_genes by number_points. It includes both tissue boundary points and interior points.
            x: x coordinate vector of the points (in the same order as in count)
            y: y coordinate vectoor of the points (in the same order as in count)
            total_num_cluster: total number of layers. This parameter is needed for dimensionality reduction, and the low dimension space has 2 * total_num_cluster dimensions.
            Gamma_0: indices of points of the tissue boundary in the 1st layer.
            Gamma_L: indices of points of the tissue boundary in the last layer.
            forbidden: indices of points to be ignored. Their will not be included in regression or loss evaluation. When Gamma_0 and Gamma_L are used, forbidden can be set as the points beyond Gamma_0 and Gamma_L.
            interpolation_type: either "harmonic" or "tps".
            platform: either "ST" or "Visium". This specifies the grid layout of the points, ST is a square grid layout, Visium is a hexagon layout
        '''
        # all attributes
        self.N = count.shape[1]
        self.G = count.shape[0]
        assert len(x) == self.N and len(y) == self.N
        self.count = count
        self.x = x
        self.y = y
        self.total_num_cluster = total_num_cluster
        self.interpolation_type = interpolation_type
        self.platform = platform
        spos = spatialcoord(x=self.x, y=self.y, platform=self.platform)
        self.interpolation = simpleinterpolation(np.vstack([x,y]).T, spos.adjacency_mat, np.sqrt(spos.pairwise_squared_dist), \
            Gamma_0=Gamma_0, Gamma_L=Gamma_L, method=self.interpolation_type)
        # for filling up the geoometry, some points will not be in tissue, add a boolean vector to indicate which points are considered for regression.
        self.consider_points = np.array([True] * self.N)
        if not (consider_points is None):
            assert len(consider_points) == self.N
            self.consider_points = consider_points
        # glmPCA on 2D, don't need to compute it unless loss="Gaussian" and pool=False
        self.F_glmpca_2d_poisson = None
    #
    def compute_2d_glmPCA(self):
        np.random.seed(0)
        print("Running 2D glmPCA")
        glmpca_res_2d_poisson = glmpca.glmpca(self.count, 2*self.total_num_cluster, fam="poi", penalty=10, verbose=True)
        print("Done!")
        self.F_glmpca_2d_poisson = glmpca_res_2d_poisson['factors']
    
    ###############################################################
    
    # helpers for poisson
    def poisson_loss(self, params,y,exposure=None):
        t=y.shape[0]
        x=np.arange(t)

        features=np.zeros((t,2))
        features[:,0]=np.arange(t)
        features[:,1]=1

        logyhat=features @ params

        if exposure is not None:
            logyhat=logyhat + np.log(exposure)

        return np.sum( np.exp(logyhat) - y*logyhat )
    
    # y has shape t x 1
    def opt_poisson_sklearn_singlegene(self, y, xcoords=None, exposure=None):
        clf = linear_model.PoissonRegressor(fit_intercept=True,alpha=0,max_iter=500)

        if xcoords is None:
            x=np.arange(len(y))
        else:
            x=xcoords

        if exposure is None:
            clf.fit(np.reshape(x,(-1,1)),y)
            loss=self.poisson_loss(np.array([clf.coef_[0], clf.intercept_ ]),y)
        else:
            clf.fit(np.reshape(x,(-1,1)),y, sample_weight=exposure)
            loss=self.poisson_loss(np.array([clf.coef_[0], clf.intercept_ ]),y, exposure=exposure)

        return loss
    #
    def fit_poisson_singlegene(self, y, xcoords, exposure=None):
        clf = linear_model.PoissonRegressor(fit_intercept=True,alpha=0,max_iter=500)
        if exposure is None:
            clf.fit(np.reshape(xcoords,(-1,1)),y)
        else:
            clf.fit(np.reshape(xcoords,(-1,1)),y, sample_weight=exposure)
        return np.array([clf.coef_[0], clf.intercept_ ])
    #
    def poisson_loss_singlegene(self, y, xcoords, params, exposure=None):
        features = np.ones((len(y), 2))
        features[:,0] = xcoords
        logyhat=features @ params
        if exposure is not None:
            logyhat=logyhat + np.log(exposure)
        return np.sum( np.exp(logyhat) - y*logyhat )
    
    ###############################################################
    
    
    # def eval_loss_old(self, b1_i, b1_j, b2_i, b2_j=None, loss='Gaussian', pool=True):
    #     if b2_j is None:
    #         idx_b1, idx_b2, idx_inside = self.interpolation.get_spots_index_within_halfcircle(b1_i, b1_j, b2_i)
    #     else:
    #         idx_b1, idx_b2, idx_inside = self.interpolation.get_spots_index_within_region(b1_i, b1_j, b2_i, b2_j)
    #     if len(idx_b1) == 0:
    #         # print("The first boundary contains zero points for eval_loss({},{},{},{}).! Return -inf because it is not possible to interpolate.".format(b1_i, b1_j, b2_i, b2_j))
    #         return np.inf
    #     elif len(idx_b2) == 0:
    #         # print("The second boundary contains zero points for eval_loss({},{},{},{}).! Return -inf because it is not possible to interpolate.".format(b1_i, b1_j, b2_i, b2_j))
    #         return np.inf
    #     proj_b1, proj_b2, proj_inside = self.interpolation.interpolate(idx_b1, idx_b2, idx_inside)
    #     # pooling
    #     # ????? do we need pooling at all???? or maybe pooling is wrong here?????
    #     index = np.concatenate( [idx_b1, idx_b2, idx_inside] )
    #     xcoords = np.round(np.concatenate( [proj_b1, proj_b2, proj_inside] ))
    #     xcoords_int = xcoords.astype(int)
    #     pooled_xcoords = np.sort(np.unique(xcoords_int))
    #     N_1d = pooled_xcoords.shape[0]
    #     pooled_int_data = np.zeros( (self.G, N_1d) )
    #     map_1d_bins_to_2d = {}
    #     for ind, b in enumerate(pooled_xcoords):
    #         bin_pts = np.where(xcoords_int == b)[0]
    #         pooled_int_data[:,ind] = np.sum(self.count[:,index[bin_pts]], axis=1)
    #         map_1d_bins_to_2d[b] = bin_pts

    #     if loss=='Gaussian':
    #         if pool:
    #             # glmPCA of pooled data
    #             np.random.seed(0)
    #             glmpca_res_1d_poisson = glmpca.glmpca(pooled_int_data, 2*self.total_num_cluster, fam="poi", penalty=100, verbose=False)
    #             F_glmpca_1d_poisson = glmpca_res_1d_poisson['factors']
    #             # regression
    #             X = np.ones((N_1d, 2))
    #             X[:,0] = pooled_xcoords
    #             theta = np.linalg.inv(X.T @ X) @ X.T @ F_glmpca_1d_poisson
    #             error = np.linalg.norm(X @ theta - F_glmpca_1d_poisson)**2
    #             return error
    #         else:
    #             # check whether 2D glmPCA exists
    #             if self.F_glmpca_2d_poisson is None:
    #                 self.compute_2d_glmPCA()
    #             # regression
    #             X = np.ones(( len(xcoords), 2))
    #             X[:,0] = xcoords
    #             theta = np.linalg.inv(X.T @ X) @ X.T @ self.F_glmpca_2d_poisson[index, :]
    #             error = np.linalg.norm(X @ theta - self.F_glmpca_2d_poisson[index, :])**2
    #             return error
    #     elif loss=='Poisson':
    #         error=0
    #         if pool:
    #             pooled_exposures=np.sum( pooled_int_data,axis=0)
    #             for g in range(self.G):
    #                 error += self.opt_poisson_sklearn_singlegene(pooled_int_data[g,:] + 1, xcoords=pooled_xcoords, exposure=pooled_exposures)
    #         else:
    #             exposures = np.sum(self.count[:, index], axis=0)
    #             for g in range(self.G):
    #                 params = self.fit_poisson_singlegene(self.count[g,index] + 1, xcoords=xcoords, exposure=exposures)
    #                 error += self.poisson_loss_singlegene(self.count[g,index] + 1, xcoords=xcoords, params=params, exposure=exposures)
    #         return error
    #     else:
    #         raise Exception('Loss function not yet implemented')
    #
    def eval_loss(self, b1_endp1, b1_endp2, b2_endp1=None, b2_endp2=None, other_boundary=None, loss='Gaussian'):
        # find spots on the layer boundaries and inside
        if b2_endp2 is None:
            idx_b1, idx_inside = self.interpolation.get_spots_index_within_halfcircle(b1_endp1, b1_endp2, b2_endp1)
            if len(idx_b1) == 0 or len(idx_inside) == 0:
                return np.inf
            # it's meaningless if the half circle contains all points, so return inf if that happens
            if len(np.unique(np.concatenate([idx_b1, idx_inside]))) == self.N:
                return np.inf
            # regression
            proj_b1, proj_inside = self.interpolation.diffusion_harmonic(idx_b1, idx_inside)
            index = np.concatenate( [idx_b1, idx_inside] )
            proj = np.concatenate( [proj_b1, proj_inside] )
        else:
            idx_b1, idx_b2, idx_inside = self.interpolation.get_spots_index_within_region(b1_endp1, b1_endp2, b2_endp1, b2_endp2)
            if len(idx_b1) == 0 or len(idx_b2) == 0 or len(idx_inside) == 0:
                return np.inf
            # interpolation
            proj_b1, proj_b2, proj_inside = self.interpolation.interpolate(idx_b1, idx_b2, idx_inside)
            # regression
            index = np.concatenate( [idx_b1, idx_b2, idx_inside] )
            proj = np.concatenate( [proj_b1, proj_b2, proj_inside] )
        # remove points that are not in tissue/shouldn't considered
        whether_keep = self.consider_points[index]
        index = index[whether_keep]
        proj = proj[whether_keep]
        xcoords = proj
        if len(np.unique( np.round(xcoords, decimals=4) )) < 3:
            return np.inf
        ##### compute loss #####
        if loss=='Gaussian':
            # check whether 2D glmPCA exists
            if self.F_glmpca_2d_poisson is None:
                self.compute_2d_glmPCA()
            # regression
            X = np.ones(( len(xcoords), 2))
            X[:,0] = xcoords
            theta = np.linalg.inv(X.T @ X) @ X.T @ self.F_glmpca_2d_poisson[index, :]
            error = np.linalg.norm(X @ theta - self.F_glmpca_2d_poisson[index, :])**2
            return error
        elif loss=='Poisson':
            error=0
            exposures = np.sum(self.count[:, index], axis=0)
            for g in range(self.G):
                params = self.fit_poisson_singlegene(self.count[g,index], xcoords=xcoords, exposure=exposures)
                error += self.poisson_loss_singlegene(self.count[g,index], xcoords=xcoords, params=params, exposure=exposures)
            return error
        else:
            raise Exception('Loss function not yet implemented')
            