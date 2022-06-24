import sys
import numpy as np
from tqdm import trange
import pickle
from general_helper_funcs import *
from region_cost_fun import *


def precompute_batch(F_glmpca, coords, max_nlayers, FOLDER_TO_SAVE_LOSSES_IN, batch_id=None, num_batches=None):
    # fill in the full hexagon geometry
    fullpoints, in_tissue = fill_geometry(coords, is_hexagon=True)
    F_full_glmpca = np.ones((fullpoints.shape[0], F_glmpca.shape[1])) * np.nan
    F_full_glmpca[in_tissue] = F_glmpca

    # construct loss function/likelihood computing object
    llf = lossfunction(F_full_glmpca.T, fullpoints[:,0], fullpoints[:,1], total_num_cluster=max_nlayers,  platform="Visium")
    llf.F_glmpca_2d_poisson = F_full_glmpca

    # get points on layer boundaries by Delaunay triangulatioon
    sorted_boundary = get_sorted_boundary_points(fullpoints)

    # enumerate (1) 4-tuple of boundary points to define an interior region and (2) triple of boundary points to define the first/last layer region.
    boundary_tuples = get_full_boundary_tuples_from_sorted(sorted_boundary)
    logger.info("There are {} tuples of boundary end points.".format( len(boundary_tuples) ))
    boundary_triples = get_full_boundary_triples_from_sorted(sorted_boundary)
    logger.info("There are {} triples of boundary end points for half circle regions.".format(len(boundary_triples)))

    # output dictionary
    pre_saving = {}

    # batch for region bounded by two lines
    if num_batches is None or batch_id is None:
        s = 0
        t = len(boundary_tuples)
    else:
        assert num_batches > 0 and batch_id >= 0 and batch_id < num_batches
        s = int( np.round(len(boundary_tuples) / num_batches * batch_id) )
        t = int( np.round(len(boundary_tuples) / num_batches * (batch_id+1)) )

    for r in trange(s, t):
        i, j, k, m = boundary_tuples[r]
        this_loss = llf.eval_loss(sorted_boundary[i], sorted_boundary[j], b2_endp1=sorted_boundary[k],  
                                b2_endp2=sorted_boundary[m], loss='Gaussian')
        pre_saving[ boundary_tuples[r] ] = this_loss
        
    # batch for region bounded by one line and tissue boundary
    if num_batches is None or batch_id is None:
        s = 0
        t = len(boundary_triples)
    else:
        assert num_batches > 0 and batch_id >= 0 and batch_id < num_batches
        s = int( np.round(len(boundary_triples) / num_batches * batch_id) )
        t = int( np.round(len(boundary_triples) / num_batches * (batch_id+1)) )

    for r in trange(s, t):
        i, j, k = boundary_triples[r]
        this_loss = llf.eval_loss(sorted_boundary[i], sorted_boundary[j], b2_endp1=sorted_boundary[k], loss='Gaussian')
        if k == i+1:
            pre_saving[ (i,j,np.inf,np.inf) ] = this_loss
        else:
            pre_saving[ (np.inf,np.inf, i,j) ] = this_loss

    # save precomputed likelihoods
    if num_batches is None or batch_id is None:
        pickle.dump(pre_saving, open(f"{FOLDER_TO_SAVE_LOSSES_IN}_pre_saving.pkl", 'wb'))
    else:
        assert num_batches > 0 and batch_id >= 0 and batch_id < num_batches
        pickle.dump(pre_saving, open(f"{FOLDER_TO_SAVE_LOSSES_IN}_pre_saving_{batch_id}_{num_batches}.pkl", 'wb'))


class precompute_class(object):
    def __init__(self, F_glmpca, coords, max_nlayers, FOLDER_TO_SAVE_LOSSES_IN, num_batches=None):
        # fill in the full hexagon geometry
        fullpoints, in_tissue = fill_geometry(coords, is_hexagon=True)
        F_full_glmpca = np.ones((fullpoints.shape[0], F_glmpca.shape[1])) * np.nan
        F_full_glmpca[in_tissue] = F_glmpca

        # construct loss function/likelihood computing object
        llf = lossfunction(F_full_glmpca.T, fullpoints[:,0], fullpoints[:,1], total_num_cluster=max_nlayers,  platform="Visium")
        llf.F_glmpca_2d_poisson = F_full_glmpca

        # get points on layer boundaries by Delaunay triangulatioon
        sorted_boundary = get_sorted_boundary_points(fullpoints)

        # enumerate (1) 4-tuple of boundary points to define an interior region and (2) triple of boundary points to define the first/last layer region.
        boundary_tuples = get_full_boundary_tuples_from_sorted(sorted_boundary)
        logger.info("There are {} tuples of boundary end points.".format( len(boundary_tuples) ))
        boundary_triples = get_full_boundary_triples_from_sorted(sorted_boundary)
        logger.info("There are {} triples of boundary end points for half circle regions.".format(len(boundary_triples)))

        # class attribute
        self.llf = llf
        self.sorted_boundary = sorted_boundary
        self.boundary_tuples = boundary_tuples
        self.boundary_triples = boundary_triples
        self.num_batches = num_batches
        self.FOLDER_TO_SAVE_LOSSES_IN = FOLDER_TO_SAVE_LOSSES_IN

    def precompute_likelihood(self, batch_id = None):
        # storing the precomputed regional likelihoods
        pre_saving = {}

        # batch for region bounded by two lines
        if self.num_batches is None or batch_id is None:
            s = 0
            t = len(self.boundary_tuples)
        else:
            assert self.num_batches > 0 and batch_id >= 0 and batch_id < self.num_batches
            s = int( np.round(len(self.boundary_tuples) / self.num_batches * batch_id) )
            t = int( np.round(len(self.boundary_tuples) / self.num_batches * (batch_id+1)) )

        for r in trange(s, t):
            i, j, k, m = self.boundary_tuples[r]
            this_loss = self.llf.eval_loss(self.sorted_boundary[i], self.sorted_boundary[j], b2_endp1=self.sorted_boundary[k],  
                                    b2_endp2=self.sorted_boundary[m], loss='Gaussian')
            pre_saving[ self.boundary_tuples[r] ] = this_loss
            
        # batch for region bounded by one line and tissue boundary
        if self.num_batches is None or batch_id is None:
            s = 0
            t = len(self.boundary_triples)
        else:
            assert self.num_batches > 0 and batch_id >= 0 and batch_id < self.num_batches
            s = int( np.round(len(self.boundary_triples) / self.num_batches * batch_id) )
            t = int( np.round(len(self.boundary_triples) / self.num_batches * (batch_id+1)) )

        for r in trange(s, t):
            i, j, k = self.boundary_triples[r]
            this_loss = self.llf.eval_loss(self.sorted_boundary[i], self.sorted_boundary[j], b2_endp1=self.sorted_boundary[k], loss='Gaussian')
            if k == i+1:
                pre_saving[ (i,j,np.inf,np.inf) ] = this_loss
            else:
                pre_saving[ (np.inf,np.inf, i,j) ] = this_loss
        
        # save precomputed likelihoods
        if self.num_batches is None or batch_id is None:
            pickle.dump(pre_saving, open(f"{self.FOLDER_TO_SAVE_LOSSES_IN}_pre_saving.pkl", 'wb'))
        else:
            assert self.num_batches > 0 and batch_id >= 0 and batch_id < self.num_batches
            pickle.dump(pre_saving, open(f"{self.FOLDER_TO_SAVE_LOSSES_IN}_pre_saving_{batch_id}_{self.num_batches}.pkl", 'wb'))

    def combine_precomputed_likelihoods(self):
        assert not (self.num_batches is None)
        pre_saving = {}
        for batch_id in range(self.num_batches):
            tmp = pickle.load(f"{self.FOLDER_TO_SAVE_LOSSES_IN}_pre_saving_{batch_id}_{self.num_batches}.pkl")
            pre_saving = pre_saving.update(tmp)
        pickle.dump(pre_saving, open(f"{self.FOLDER_TO_SAVE_LOSSES_IN}_pre_saving.pkl", 'wb'))