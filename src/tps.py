


import numpy as np
import scipy
import networkx as nx
# from spatialcoord import *


class tps(object):
    '''
    A module to perform thin-plate spline and identify contours from
    (1) either a list of boundary points (in topological order), where points are represented as their indices in the same order as spatial coordinate;
    (2) or a vector of cluster/layer annotations, where the length of annotation should be the same as the number of points, and in the same order.
    The output is a vector of contour IDs.
    The intermediate information learned is the TPS spline function that maps 2D points to 1D elevation, and the binning function (a step function) where each bin represents a contour.
    '''
    def __init__(self, spos, list_cluster_boundary=None, cluster_annotation=None, smoothing_param=0.5):
        ##### a list of all attributes #####
        self.spos = None
        self.smoothing_param = None        
        self.list_boundary = None
        self.list_edge_beginning = None
        self.list_edge_ending = None
        self.spline_fun = None
        self.interpolation = None
        self.binning_ranges = None
        # auxilliary
        self.cleaned = None # cleaned cluster annotation
        ##### initialize basic info #####
        assert isinstance(spos, spatialcoord), "Must provide an spatialcoord object!"
        self.spos = spos
        self.smoothing_param = smoothing_param
        assert not ((list_cluster_boundary is None) and (cluster_annotation is None)), "Must provide at least one of list_cluster_boundary and cluster_annotation!"
        if not (list_cluster_boundary is None):
            assert len(np.concatenate(list_cluster_boundary)) > 0, "list_cluster_boundary is of zero length!"
            self.list_boundary = list_cluster_boundary
        # if list of boundary is not directly provided, preprocess to get thit from cluster_annotation
        if not (cluster_annotation is None):
            assert len(cluster_annotation) == len(self.spos.x), "cluster_annotation has a different length from the unmber of spots in spos!"
            if not isinstance(cluster_annotation, np.ndarray):
                cluster_annotation = np.array(cluster_annotation)
            cleaned = self.clean_annotation(cluster_annotation)
            self.cleaned = cleaned
            self.list_boundary = self.preprocess_get_boundary(cleaned)
        # after we have a list of boundaries (equivalent to change points), add tissue edges (equivalent to initial point and last point)
        self.list_edge_beginning, self.list_edge_ending = self.add_tissue_edges()
        ##### learn tps function and binning ranges #####
        self.spline_fun = self.learn_spline_function()
        self.interpolation = np.array([ self.spline_fun(self.spos.x[i], self.spos.y[i]) for i in range(len(self.spos.x)) ])
        self.binning_ranges = self.learn_binning_ranges()
    #
    def clean_annotation(self, cluster_annotation):
        '''
        This function identifies and modify clusters such that (1) the cluster is not itself a connected component (2) the cluster connects to 3 or more other clusters
        For the above clusters, only keep the largest connected component (CC) as the original cluster label, and modify other CCs to be a cluster label that their spots are nearest neighbors.
        '''
        unique_clusters = np.unique(cluster_annotation)
        unique_cluster_map = {unique_clusters[i]:i for i in range(len(unique_clusters))}
        cleaned = [None] * len(cluster_annotation)
        # cluster adjacency matrix
        cluster_adjacency = np.zeros( (len(unique_clusters), len(unique_clusters)) )
        for i in range(len(self.spos.x)):
            idx_adj = np.where(self.spos.adjacency_mat[i,:] > 0)[0]
            for j in idx_adj:
                if cluster_annotation[i] != cluster_annotation[j]:
                    cluster_adjacency[unique_cluster_map[cluster_annotation[i]], unique_cluster_map[cluster_annotation[j]]] = 1
        cluster_adjacency += cluster_adjacency.T
        # loop over cluster names, identify the clusters with >= 3 adjacent clusters, and identify the largest CC
        for k, cname in enumerate(unique_clusters):
            if np.sum(cluster_adjacency[k,:] > 0) >= 3:
                components = self.spos.connected_components( np.where(cluster_annotation == cname)[0] )
                # now components[0] is the largest CC
                for i in components[0]:
                    cleaned[i] = cname
            else:
                for i in np.where(cluster_annotation == cname)[0]:
                    cleaned[i] = cname
        # identify the spots for which the cleaned cluster is None
        ind_unassigned = np.array([i for i in range(len(cleaned)) if cleaned[i] is None])
        if len(ind_unassigned) > 0:
            components = self.spos.connected_components( ind_unassigned )
            # for each CC without an assigned cluster, assign the cluster based on the cluster of its nearest neighbors
            for CC in components:
                tmpind = np.argsort( self.spos.pairwise_squared_dist[:, CC], axis=0 )
                # the first row is CC themselves because they always have 0 distance to themselves. So remove the first row
                tmpind = tmpind[1:, :]
                is_assigned = False
                for i in range(tmpind.shape[0]):
                    for j in range(tmpind.shape[1]):
                        if not (cleaned[tmpind[i,j]] is None):
                            for k in CC:
                                cleaned[k] = cleaned[tmpind[i,j]]
                            is_assigned = True
                            break
                    if is_assigned:
                        break
        assert len([i for i in range(len(cleaned)) if cleaned[i] is None]) == 0, "Still have spots without a cleaned cluster!"
        return cleaned
    #
    def preprocess_get_boundary(self, cleaned):
        '''
        This function seek to get a llist of cluster boundaries in topological order.
        '''
        if not isinstance(cleaned, np.ndarray):
            cleaned = np.array(cleaned)
        unique_clusters = np.unique(cleaned)
        unique_cluster_map = {unique_clusters[i]:i for i in range(len(unique_clusters))}
        # topologically order the cluster names by constructing a graph and find eulerian path
        # construct graph by adjacency between clusters
        cluster_adjacency = np.zeros( (len(unique_clusters), len(unique_clusters)) )
        for i in range(len(self.spos.x)):
            idx_adj = np.where(self.spos.adjacency_mat[i,:] > 0)[0]
            for j in idx_adj:
                if cleaned[i] != cleaned[j]:
                    cluster_adjacency[unique_cluster_map[cleaned[i]], unique_cluster_map[cleaned[j]]] = 1
        cluster_adjacency += cluster_adjacency.T
        G = nx.Graph()
        G.add_edges_from( [(i, j) for i in np.arange(cluster_adjacency.shape[0]) for j in np.arange(i+1, cluster_adjacency.shape[1]) if cluster_adjacency[i,j] > 0] )
        # eulerian path
        assert nx.algorithms.euler.has_eulerian_path(G), "Cluster annotation doesn't have layered structure!"
        iter_edges = nx.algorithms.euler.eulerian_path(G)
        eulerian_path = []
        for e in iter_edges:
            if len(eulerian_path) == 0:
                eulerian_path += list(e)
            else:
                eulerian_path.append( e[1] )
        if eulerian_path[-1] == eulerian_path[0]:
            eulerian_path = eulerian_path[:-1]
        ordered_unique_clusters = [unique_clusters[i] for i in eulerian_path]
        # now the clusters are topologically ordered, find the cluster boundaries based on this order
        # the first list is the boundary points between ordered_unique_clusters[0] and ordered_unique_clusters[1], the second list is the boundary between ordered_unique_clusters[1] and ordered_unique_clusters[2], and so on
        tmp_list_boundary = [[] for i in range(len(ordered_unique_clusters) - 1)]
        # loop over each spot, check whether its adjacent spots have a different cluster ID.
        # If so, add the spot to the corresponding boundary point list
        for i in range(len(self.spos.x)):
            idx_adj = np.where(self.spos.adjacency_mat[i,:] > 0)[0]
            if len(idx_adj) > 0 and np.any(cleaned[idx_adj] != cleaned[i]):
                this_idx_cluster = ordered_unique_clusters.index( cleaned[i] )
                this_adj_clusters = np.array(cleaned[idx_adj])
                this_adj_clusters = np.unique(this_adj_clusters)
                if this_idx_cluster != 0 and len(this_adj_clusters) == 2 and np.any(this_adj_clusters == ordered_unique_clusters[this_idx_cluster - 1]):
                    tmp_list_boundary[this_idx_cluster - 1].append( i )
        assert np.all([len(x) > 0 for x in tmp_list_boundary]), "Some cluster boundaries have zero boundary points!"
        tmp_list_boundary = [np.array(x) for x in tmp_list_boundary]
        # for ordered_unique_clusters[0], there is no point of this cluster in tmp_list_boundary. We find the spot from ordered_unique_clusters[0] that is the farthest to tmp_list_boundary[0] as an addition layer boundary (which is the edge of tissue)
        indexes = np.where(cleaned == ordered_unique_clusters[0])[0]
        pdist = self.spos.pairwise_squared_dist[indexes,:][:,tmp_list_boundary[0]]
        assert np.min(pdist, axis=1).shape[0] == len(indexes)
        edge_point = indexes[ np.argmax(np.min(pdist, axis=1)) ]
        tmp_list_edge = [ np.array([edge_point]) ]
        return tmp_list_boundary
    #
    def add_tissue_edges(self):
        assert not (self.list_boundary is None), "List of boundary points hasn't been provided or processed!"
        assert len(self.list_boundary) > 1, "There must be at least two layer boundaries!"
        # beginning of tissue (a point in front of self.list_boundary[0] and is the farthest from self.list_boundary[0])
        indexes = np.array([i for i in range(len(self.spos.x)) if np.min(self.spos.pairwise_squared_dist[i,self.list_boundary[0]]) < np.min(self.spos.pairwise_squared_dist[i,self.list_boundary[1]]) ])
        pdist = self.spos.pairwise_squared_dist[indexes,:][:,self.list_boundary[0]]
        edge_start = indexes[ np.argmax(np.min(pdist, axis=1)) ]
        # ending of tissue (a point after self.list_boundary[-1] and is the farthest from self.list_boundary[-1])
        indexes = np.array([i for i in range(len(self.spos.x)) if np.min(self.spos.pairwise_squared_dist[i,self.list_boundary[-1]]) < np.min(self.spos.pairwise_squared_dist[i,self.list_boundary[-2]]) ])
        pdist = self.spos.pairwise_squared_dist[indexes,:][:,self.list_boundary[-1]]
        edge_end = indexes[ np.argmax(np.min(pdist, axis=1)) ]
        return [ np.array([edge_start]) ], [ np.array([edge_end]) ]
    #
    def learn_spline_function(self):
        assert not (self.list_boundary is None), "List of boundary points hasn't been provided or processed!"
        appended_list_boundary = self.list_boundary
        if not (self.list_edge_beginning is None):
            appended_list_boundary = self.list_edge_beginning + appended_list_boundary
        if not (self.list_edge_ending is None):
            appended_list_boundary = appended_list_boundary + self.list_edge_ending
        # assign elevations of list of boundary points by their partial hausdorff distance
        elevations = [0] * len(appended_list_boundary[0])
        for i in range(len(appended_list_boundary) - 1):
            phd = self.spos.fractional_hausdorff_distance(appended_list_boundary[i], appended_list_boundary[i+1], percentile=95)
            last_elevation = elevations[-1]
            elevations += [last_elevation + phd] * len(appended_list_boundary[i+1])
        elevations = np.array(elevations)
        # fit spline function using the boundary points and the assigned elevations
        ind_boundary = np.concatenate(appended_list_boundary)
        assert len(ind_boundary) == len(elevations)
        fun = scipy.interpolate.Rbf(self.spos.x[ind_boundary], self.spos.y[ind_boundary], elevations, function="thin-plate", smooth=self.smoothing_param)
        return fun
    #
    def learn_binning_ranges(self):
        assert not (self.list_boundary is None), "List of boundary points hasn't been provided or processed!"
        assert not (self.spline_fun is None), "TPS function hasn't been fitted!"
        assert not (self.interpolation is None), "Interpolated values of each spot haven't been computed!"
        # start from list_boundary[0], and extend to the larger interpolated values
        ranges = [ (np.min(self.interpolation[self.list_boundary[0]]), np.max(self.interpolation[self.list_boundary[0]])) ]
        max_elevation = np.max(self.interpolation)
        while ranges[-1][1] < max_elevation:
            # indexes of previous points
            idx_prev = np.where(np.logical_and(self.interpolation >= ranges[-1][0], self.interpolation < ranges[-1][1]))[0]
            # indexes of points adjacent to the previous points and have a larger interpolated value
            idx_adj = []
            for i in idx_prev:
                tmpind = np.where(self.spos.adjacency_mat[i,:] > 0)[0]
                idx_adj += [j for j in tmpind if self.interpolation[j] >= ranges[-1][1]]
            idx_adj = np.array(idx_adj)
            # define the next contour/next range to be the range of interpolated values of idx_adj
            if len(idx_adj) > 0:
                ranges.append( (np.min(self.interpolation[idx_adj]), np.max(self.interpolation[idx_adj])) )
            else:
                break
        ranges[-1] = (ranges[-1][0], max_elevation+1e-2)
        # extend to the smaller side of interpolated values
        min_elevation = np.min(self.interpolation)
        while ranges[0][0] > min_elevation:
            # indexes of previous points
            idx_prev = np.where(np.logical_and(self.interpolation >= ranges[0][0], self.interpolation < ranges[0][1]))[0]
            # indexes of points adjacent to the previous points and have a smaller interpolated value
            idx_adj = []
            for i in idx_prev:
                tmpind = np.where(self.spos.adjacency_mat[i,:] > 0)[0]
                idx_adj += [j for j in tmpind if self.interpolation[j] < ranges[0][0]]
            idx_adj = np.array(idx_adj)
            # define the next contour/next range to be the range of interpolated values of idx_adj
            if len(idx_adj) > 0:
                ranges = [ (np.min(self.interpolation[idx_adj]), np.max(self.interpolation[idx_adj])) ] + ranges
            else:
                break
        ranges[0] = (min_elevation-1e-2, ranges[0][1])
        # adjust the ranges to cover the full interval from (min_elevation-1e-2, max_elevation+1e-2)
        for i in range(len(ranges)-1):
            if ranges[i][1] < ranges[i+1][0]:
                ranges[i] = (ranges[i][0], ranges[i+1][0])
        return ranges
    # 
    def get_contour_ids(self):
        contour = np.ones( len(self.spos.x) ) * np.nan
        for i in range(len(self.binning_ranges)):
            contour[ np.logical_and(self.interpolation >= self.binning_ranges[i][0], self.interpolation < self.binning_ranges[i][1]) ] = i
        assert np.sum(np.isnan(contour)) == 0, "Binning ranges still don't cover the whole range of interpolation!"
        contour = contour.astype(np.int)
        return contour


def pool_adata_based_tps(count, interpolation, contour, clusters=None, n_part=2, seed=0):
    '''
    Pooling ocunt matrix based on TPS interpolated valuesas well as the binned contours.
    Input
        count is the UMI count matrix of size n_spots-by-n_genes.
        n_part is the number of count matrices after pooling. 
            n_part=2 facilitate partition the data into 2 parts, so that cluster boundary selection and statistical testing can be performed on different parts to avoid double dipping issue.
        clusters is the clusters IDs corresponding to the original 2D spots. If provided, the cluster IDs will be transferred to the pooled dataset.
    '''
    np.random.seed(seed)
    assert len(interpolation) == len(contour)
    assert count.shape[0] == len(interpolation)
    assert issubclass(contour.dtype.type, np.integer)
    if not (clusters is None):
        unique_clusters = np.unique(clusters)
    # get the unique values of contours, these are also the x axis of the pooled 1D data
    id_contours = np.sort(np.unique(contour))
    pooled_counts = [ np.zeros((len(id_contours), count.shape[1]), dtype=np.int) for i in range(n_part )]
    pooled_indices = [ [] for i in range(n_part) ]
    cluster_label_contours = [None] * len(id_contours)
    for i, x in enumerate(id_contours):
        # find the indices of spots with contour id = x
        indexes = np.where(contour == x)[0]
        ## sort the indices by the actual interpolation values
        indexes = indexes[np.argsort( interpolation[indexes] )]
        #np.random.shuffle(indexes)
        if len(indexes) > 1:
            # assigning indices to each part
            for p in np.arange(n_part):
                sub_indexes = indexes[p:len(indexes):n_part]
                pooled_counts[p][i, :] = np.sum(count[sub_indexes, :], axis=0)
                pooled_indices[p] += list(sub_indexes)
            # if cluster labels are provide, transfer the majority of cluster labels of indexes to the pooled data
            if not (clusters is None):
                cluster_count = [ (la, np.sum(clusters[indexes] == la)) for la in unique_clusters ]
                cluster_count.sort(key = lambda x:x[1], reverse=True)
                cluster_label_contours[i] = cluster_count[0][0]
    pooled_indices = [np.array(x) for x in pooled_indices]
    # remove pooled spots for the edge case: only one original spot belonging to the contour ID
    part_umi_counts = np.vstack( [np.sum(pooled_counts[i],axis=1) for i in range(n_part)] ) # this is a mtrix of n_part by n_contour_ID matrix
    indexes = np.array([j for j in range(len(id_contours)) if not np.any(part_umi_counts[:, j] == 0) ])
    pooled_counts = [x[indexes,:] for x in pooled_counts]
    id_contours = id_contours[indexes]
    if not (clusters is None):
        cluster_label_contours = np.array(cluster_label_contours)[indexes]
        assert np.all([ (not x is None) for x in cluster_label_contours ])
        return pooled_counts, id_contours, pooled_indices, cluster_label_contours
    else:
        return pooled_counts, id_contours, pooled_indices, None


##### remap TPS interpolation by arc length of gradient curve #####
def identify_middle_point(index_points, adjacency_mat):
    '''
    Given a list of points (presumably connected to a boundary curve), find the middle one in the curve.
    This is done by preorder traversal of graph constructed for the list of points based on the adjacency matrix.
    The middle point is the middle one in the preorder traversal result.
    '''
    import networkx as nx
    sub_adjacency_mat = adjacency_mat[index_points,:][:, index_points]
    G = nx.Graph()
    G.add_nodes_from( np.arange(len(index_points)) )
    G.add_edges_from( [(i,j) for i in range(len(index_points)) for j in range(i+1, len(index_points)) if sub_adjacency_mat[i,j] > 0] )
    mid = int(len(index_points) / 2)
    for source in np.argsort(np.sum(sub_adjacency_mat, axis=0)):
        traversal = np.array(list(nx.dfs_preorder_nodes(G, source)))
        print("source = {}, traversal = {}".format(source, traversal))
        if len(traversal) > mid:
            break
    return index_points[traversal][mid]


def find_gradient_line(spline_fun, starting_point, fun_min, fun_max, norm_step=0.1):
    '''
    Starting from starting_point, get a sequence of points along the gradient direction, each with a norm step size of norm_step.
    Starting from starting_point, get a sequence of points along the negative gradient direction, each with a norm step size of norm_step.
    Concatenate these two sequences, we get the gradient curve, the arc length between each adjacent point is norm_step.
    The output contains:
        (1) the concatenated sequence of points.
        (2) the spline function value at these points.
    '''
    xstep = norm_step
    ystep = norm_step
    # along the gradient direction
    grad_increase = [starting_point]
    while spline_fun(grad_increase[-1][0], grad_increase[-1][1]) < fun_max + 1:
        this_p = grad_increase[-1]
        this_value = spline_fun(this_p[0], this_p[1])
        diff_x = spline_fun(this_p[0]+xstep, this_p[1]) - this_value
        diff_y = spline_fun(this_p[0], this_p[1]+ystep) - this_value
        this_grad = np.array([diff_x, diff_y])
        this_grad = this_grad / np.sqrt(np.sum(np.square(this_grad))) * norm_step
        # check whether the last point becomes local maximum, and if so, jump out of local maximum by multipling norm_step to a large size (only consider the horizontal and vertical direction in this case)
        multiplier = 1
        while spline_fun(this_p[0] + this_grad[0], this_p[1] + this_grad[1]) < this_value:
            multiplier += 1
            tmp_grads = [ np.array([0, multiplier * norm_step]), np.array([0, -multiplier * norm_step]), np.array([multiplier * norm_step, 0]), np.array([-multiplier * norm_step,0]) ]
            tmp_diff = np.array([ spline_fun(this_p[0]+tmp[0], this_p[1]+tmp[1]) - this_value for tmp in tmp_grads])
            this_grad = tmp_grads[ np.argmax(tmp_diff) ]
        if multiplier > 1:
            print("Warning: jumping out of local maximum along the increasing curve at {} point using multiplier {}".format( len(grad_increase), multiplier ))
        grad_increase.append( this_p + this_grad )
        if len(grad_increase) > 800:
            break
    # along the negative gradient direction
    grad_decrease = [starting_point]
    while spline_fun(grad_decrease[-1][0], grad_decrease[-1][1]) > fun_min - 1:
        this_p = grad_decrease[-1]
        this_value = spline_fun(this_p[0], this_p[1])
        diff_x = spline_fun(this_p[0]+xstep, this_p[1]) - this_value
        diff_y = spline_fun(this_p[0], this_p[1]+ystep) - this_value
        this_grad = np.array([diff_x, diff_y])
        this_grad = this_grad / np.sqrt(np.sum(np.square(this_grad))) * norm_step
        # check whether the last point becomes local minimum. If so, jump out of local minimum by multiplying norm_step to a large size (only consider the horizontal and vertical step in this case)
        multiplier = 1
        while spline_fun(this_p[0] - this_grad[0], this_p[1] - this_grad[1]) > this_value:
            multiplier += 1
            angles = np.arange(0, np.pi, 0.2)
            tmp_diff_clockwise = np.array([ spline_fun(this_p[0] + multiplier*norm_step*np.cos(a), this_p[1] + multiplier*norm_step*np.sin(a)) - this_value for a in angles])
            tmp_diff_counter = np.array([ spline_fun(this_p[0] + multiplier*norm_step*np.cos(a), this_p[1] - multiplier*norm_step*np.sin(a)) - this_value for a in angles])
            if np.min(tmp_diff_clockwise) < np.min(tmp_diff_counter):
                a = angles[np.argmin(tmp_diff_clockwise)]
                this_grad = -np.array([multiplier*norm_step*np.cos(a), multiplier*norm_step*np.sin(a)])
            else:
                a = angles[np.argmin(tmp_diff_counter)]
                this_grad = -np.array([multiplier*norm_step*np.cos(a), -multiplier*norm_step*np.sin(a)])
        if multiplier > 1:
            print("Warning: jumping out of local minimum along the decreasing curve at {} point using multiplier {}".format( len(grad_decrease), multiplier))
        grad_decrease.append( this_p - this_grad )
        if len(grad_decrease) > 800:
            break
    # combining both directions
    sequence_points = np.array(grad_decrease[::-1] + grad_increase[1:])
    sequence_values = np.array([spline_fun(x[0],x[1]) for x in sequence_points])
    return sequence_points, sequence_values


def remap_spline_interpolation(t):
    assert isinstance(t, tps), "Must provide a tps  object!"
    idx_starting_point = identify_middle_point(t.list_boundary[1], t.spos.adjacency_mat)
    norm_step = 0.1
    sequence_points, sequence_values = find_gradient_line(t.spline_fun, np.array([t.spos.x[idx_starting_point], t.spos.y[idx_starting_point]]), \
                                                          fun_min=np.min(t.interpolation), fun_max=np.max(t.interpolation), norm_step=norm_step)
    initial_interpolation = t.interpolation
    sequence_arc_length = np.zeros( sequence_points.shape[0] )
    for i in range(1, sequence_points.shape[0]):
        diff = sequence_points[i] - sequence_points[i-1]
        sequence_arc_length[i] = sequence_arc_length[i-1] + np.sqrt(np.sum(np.square( diff )))
    remap_interpolation = np.zeros(len(initial_interpolation))
    for i in range(len(initial_interpolation)):
        idx = np.argmin(np.abs( initial_interpolation[i] - sequence_values ))
        # remap_interpolation[i] = sequence_arc_length[idx]
        remap_interpolation[i] = idx * norm_step
    return remap_interpolation
