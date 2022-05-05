import numpy as np
import scipy
import networkx as nx
from spatialcoord import *
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()


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


def connected_components(A):
    G = nx.Graph()
    G.add_nodes_from( np.arange(A.shape[0]) )
    G.add_edges_from( [(i,j) for i in range(A.shape[0]) for j in range(i+1, A.shape[0]) if A[i,j] > 0] )
    components = [ list(g) for g in nx.connected_components(G)]
    components.sort(key = lambda x:len(x), reverse=True)
    return components


def clean_annotation(cluster_annotation, adjacency_mat, pairwise_dist):
    '''
    This function identifies and modify clusters such that (1) the cluster is not itself a connected component (2) the cluster connects to 3 or more other clusters
    For the above clusters, only keep the largest connected component (CC) as the original cluster label, and modify other CCs to be a cluster label that their spots are nearest neighbors.
    '''
    unique_clusters = np.unique(cluster_annotation)
    unique_cluster_map = {unique_clusters[i]:i for i in range(len(unique_clusters))}
    cleaned = [None] * len(cluster_annotation)
    # cluster adjacency matrix
    cluster_adjacency = np.zeros( (len(unique_clusters), len(unique_clusters)) )
    for i in range(len(cluster_annotation)):
        idx_adj = np.nonzero(adjacency_mat[i,:])[1]
        for j in idx_adj:
            if cluster_annotation[i] != cluster_annotation[j]:
                cluster_adjacency[unique_cluster_map[cluster_annotation[i]], unique_cluster_map[cluster_annotation[j]]] = 1
    cluster_adjacency += cluster_adjacency.T
    # loop over cluster names, identify the clusters with >= 3 adjacent clusters, and identify the largest CC
    for k, cname in enumerate(unique_clusters):
        if np.sum(cluster_adjacency[k,:] > 0) >= 3:
            idx = np.where(cluster_annotation == cname)[0]
            components = connected_components( adjacency_mat[np.ix_(idx,idx)] )
            # now components[0] is the largest CC
            for i in components[0]:
                cleaned[idx[i]] = cname
        else:
            for i in np.where(cluster_annotation == cname)[0]:
                cleaned[i] = cname
    # identify the spots for which the cleaned cluster is None
    ind_unassigned = np.array([i for i in range(len(cleaned)) if cleaned[i] is None])
    if len(ind_unassigned) > 0:
        components = connected_components( adjacency_mat[np.ix_(ind_unassigned,ind_unassigned)] )
        # for each CC without an assigned cluster, assign the cluster based on the cluster of its nearest neighbors
        for CC in components:
            tmpind = np.argsort( pairwise_dist[:, CC], axis=0 )
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


def preprocess_get_boundary(cleaned, adjacency_mat):
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
    for i in range(len(cleaned)):
        idx_adj = np.nonzero(adjacency_mat[i,:])[1]
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
    list_boundary = [[] for i in range(len(ordered_unique_clusters) - 1)]
    # loop over each spot, check whether its adjacent spots have a different cluster ID.
    # If so, add the spot to the corresponding boundary point list
    for i in range(len(cleaned)):
        idx_adj = np.nonzero(adjacency_mat[i,:])[1]
        if len(idx_adj) > 0 and np.any(cleaned[idx_adj] != cleaned[i]):
            this_idx_cluster = ordered_unique_clusters.index( cleaned[i] )
            this_adj_clusters = np.array(cleaned[idx_adj])
            this_adj_clusters = np.unique(this_adj_clusters)
            if this_idx_cluster != 0 and len(this_adj_clusters) == 2 and np.any(this_adj_clusters == ordered_unique_clusters[this_idx_cluster - 1]):
                list_boundary[this_idx_cluster - 1].append( i )
    assert np.all([len(x) > 0 for x in list_boundary]), "Some cluster boundaries have zero boundary points!"
    list_boundary = [np.array(x) for x in list_boundary]
    return list_boundary, ordered_unique_clusters


def order_boundary_points_by_graph(li, coords, pairwise_dist, source_type="min y"):
    set_li = set(li)
    # find start and end of boundary curve by minimum spanning tree
    dist_matrix = pairwise_dist[np.ix_(li,li)]
    G = nx.Graph()
    G.add_weighted_edges_from( [(a, b, dist_matrix[a,b]) for a in range(dist_matrix.shape[0]) for b in range(a+1, dist_matrix.shape[0])] )
    MST = nx.minimum_spanning_tree(G)
    if source_type == "min y":
        source = np.argmin(coords[li, 1])
    elif source_type == "max y":
        source = np.argmax(coords[li, 1])
    elif source_type == "min x":
        source = np.argmin(coords[li, 0])
    elif source_type == "max x":
        source = np.argmax(coords[li, 0])
    else:
        raise ValueError("Invalid source_type argument!")
    edges = nx.bfs_edges(MST, source)
    nodes = np.array([source] + [v for u,v in edges])
    nodes = li[nodes]
    return nodes


def extend_boundary_to_full_geometry(nodes, coords, pairwise_dist, in_tissue, source_type="min y"):
    # for each connected components, extend to padded spots
    extension = []
    CCs = [nodes[:7][::-1], nodes[-7:]]
    normvecs = []
    constants = []
    sep_extensions = []
    for CC in CCs:
        # principle of extension: multiple endpoints are assumed to form a line, we extend the line to padded region
        # fit the line and extend: separate into 3 cases (horizontal line, vertical line, general line)
        if np.max(coords[CC, 0]) - np.min(coords[CC,0]) <= 1: # vertical line
            normvecs.append( np.array([1,0]) )
            constants.append( np.mean(coords[CC,0]) )
            distance = np.abs(coords[:,0] - np.mean(coords[CC,0]))
            relative_proximity = np.abs(coords[:,1]-coords[CC[0],1]) > np.abs(coords[:,1]-coords[CC[-1],1])
            sep_extensions.append( list( np.where(np.logical_and.reduce( [distance <= 1, relative_proximity, np.logical_not(in_tissue)] ))[0] ) )
            extension += sep_extensions[-1]
        elif np.max(coords[CC,1]) - np.min(coords[CC,1]) <= 1: # horizontal line
            normvecs.append( np.array([0,1]) )
            constants.append( np.mean(coords[CC,1]) )
            distance = np.abs(coords[:,1] - np.mean(coords[CC,1]))
            relative_proximity = np.abs(coords[:,0]-coords[CC[0],0]) > np.abs(coords[:,0]-coords[CC[-1],0])
            sep_extensions.append( list( np.where(np.logical_and.reduce( [distance <= 1, relative_proximity, np.logical_not(in_tissue)] ))[0] ) )
            extension += sep_extensions[-1]
        else:
            this_y = coords[CC,1]
            this_x = np.vstack([np.ones(len(CC)), coords[CC,0]]).T
            beta = np.linalg.inv(this_x.T @ this_x) @ (this_y.dot(this_x)) # line coefficient
            normvecs.append( np.array([-beta[1], 1]) )
            constants.append( beta[0] )
            threshold = np.max(np.abs( this_y - this_x.dot(beta) )) # distance from y to fitted line
            distance = np.abs(coords[:,1] - np.vstack([np.ones(coords.shape[0]), coords[:,0]]).T.dot(beta))
            relative_proximity = np.abs(pairwise_dist[CC[0],:].A.flatten()) > np.abs(pairwise_dist[CC[-1],:].A.flatten())
            sep_extensions.append( list( np.where(np.logical_and.reduce( [distance <= threshold, relative_proximity, np.logical_not(in_tissue)] ))[0] ) )
            extension += sep_extensions[-1]
    return np.array(list(nodes) + extension), CCs, normvecs, constants, sep_extensions


def label_oneside_boundary_extension(coords, adjacency_mat, pairwise_dist, CC, normvec, constant, extension, current_labels, side_name):
    '''
    current_labels is of size n_samples with filled-in geometry.
    side_name is from 0 to the number of clusters, indicate which side of the extension to add labels, and the label is the same as side_name.
    '''
    if len(extension) == 0:
        return current_labels
    idx_unknown = np.where(np.isnan(current_labels))[0]
    components = connected_components(adjacency_mat[idx_unknown,:][:,idx_unknown])
    components = [ [idx_unknown[i] for i in x] for x in components ]
    # check which components contains extension
    components = [np.array(x) for x in components if len(set(x) & set(extension)) > 0]
    # separate the components by extended layer boundary
    for com in components:
        signed_distances = coords[com,:].dot(normvec) - constant
        com1 = com[signed_distances >= 0]
        com2 = com[signed_distances <= 0]
        # select between com1 and com2 by side_name
        window = 7
        idx_side_name = np.where(np.logical_and.reduce((coords[:,0] >= coords[CC[-1],0]-window, coords[:,0] <= coords[CC[-1],0]+window, \
                                                        coords[:,1] >= coords[CC[-1],1]-window, coords[:,1] <= coords[CC[-1],1]+window, \
                                                        current_labels == side_name)))[0]
        while len(idx_side_name) == 0:
            window += 2
            idx_side_name = np.where(np.logical_and.reduce((coords[:,0] >= coords[CC[-1],0]-window, coords[:,0] <= coords[CC[-1],0]+window, \
                                                        coords[:,1] >= coords[CC[-1],1]-window, coords[:,1] <= coords[CC[-1],1]+window, \
                                                        current_labels == side_name)))[0]
        ref_signed_distances = coords[idx_side_name,:].dot(normvec) - constant
        if 1.0 * np.sum(ref_signed_distances >= 0) / len(ref_signed_distances) > 0.5:
            # print(len(com1), len(com2), 1.0 * np.sum(ref_signed_distances >= 0) / len(ref_signed_distances), "com1")
            current_labels[com1] = side_name
        else:
            # print(len(com1), len(com2), 1.0 * np.sum(ref_signed_distances >= 0) / len(ref_signed_distances), "com1")
            current_labels[com2] = side_name
        # if np.percentile(np.min(pairwise_dist[idx_side_name, :][:, com1].A, axis=1),1) < np.percentile(np.min(pairwise_dist[idx_side_name, :][:, com2].A, axis=1),1):
        #     print(len(com1), len(com2), "com1")
        #     current_labels[com1] = side_name
        # else:
        #     print(len(com1), len(com2), "com2")
        #     current_labels[com2] = side_name
    return current_labels


def label_by_closest(coords, adjacency_mat, pairwise_dist, current_labels):
    # find connected components of spots with unknown labels
    idx_unknown = np.where(np.isnan(current_labels))[0]
    components = connected_components(adjacency_mat[idx_unknown,:][:,idx_unknown])
    components = [ [idx_unknown[i] for i in x] for x in components ]
    # unique known labels
    unique_clusters = np.unique(current_labels[np.logical_not(np.isnan(current_labels))])
    for com in components:
        distances_to_each_cluster = np.zeros(len(unique_clusters))
        for i,la in enumerate(unique_clusters):
            idx_la = np.where(current_labels == la)[0]
            distances_to_each_cluster[i] = np.mean(np.min(pairwise_dist[idx_la, :][:, com].A, axis=0))
        la = unique_clusters[np.argmin(distances_to_each_cluster)]
        current_labels[com] = la
    return current_labels


class harmonic(object):
    '''
    A module to perform thin-plate spline and identify contours from
    (1) either a list of boundary points (in topological order), where points are represented as their indices in the same order as spatial coordinate;
    (2) or a vector of cluster/layer annotations, where the length of annotation should be the same as the number of points, and in the same order.
    The output is the harmonic interpolation given the layer boundaries.
    '''
    def __init__(self, coords, adjacency_mat, pairwise_dist):
        self.coords = coords
        self.adjacency_mat = scipy.sparse.csr_matrix(adjacency_mat)
        self.pairwise_dist = scipy.sparse.csr_matrix(pairwise_dist)
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
    def interpolate_two_boundaries(self, idx_b1, idx_b2, idx_inside):
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
    def interpolation_using_list(self, list_boundary):
        heats = [1] * len(list_boundary[0])
        for i in range(len(list_boundary) - 1):
            phd = fractional_hausdorff_distance(self.pairwise_dist, list_boundary[i], list_boundary[i+1], percentile=50)
            last_heat = heats[-1]
            heats += [last_heat + phd] * len(list_boundary[i+1])
        heats = np.array(heats)
        ##### harmonic interpolation by solving linear system derived from graph Laplacian matrix for labeled and unlabeled data #####
        ##### separete graph Laplacian into nodes with known heat (K) and nodes with unknown heat (U): L = (L_K & B \\ B^T & L_U). Separate heat into K and U: x = (x_K & x_U). Solve linear system L_U x_U = -B x_K #####
        ind_boundary = np.concatenate(list_boundary)
        ind_rest = np.array([i for i in range(self.coords.shape[0]) if not (i in ind_boundary)])
        assert len(ind_boundary) == len(heats)
        # split graph Laplacian
        L_U = self.Laplacian[ind_rest,:][:,ind_rest]
        B = self.Laplacian[ind_boundary,:][:,ind_rest]
        heats_rest, info = scipy.sparse.linalg.cg(L_U, -B.T.dot(heats))
        results = np.zeros( self.coords.shape[0] )
        results[ind_boundary] = heats
        results[ind_rest] = heats_rest
        return results
    #
    def interpolation_using_annotation(self, cluster_annotation, in_tissue):
        '''
        The code below assumes that all in_tissue points are in from of out of tissue points in coords AND in the same order as cluster_annotation.
        '''
        logger.info("Start computing harmonic interpolation.")
        idx_in_tissue = np.where(in_tissue)[0]
        cleaned = clean_annotation(cluster_annotation, self.adjacency_mat[idx_in_tissue,:][:,idx_in_tissue], \
                                   self.pairwise_dist[idx_in_tissue,:][:,idx_in_tissue])
        list_boundary, ordered_unique_clusters = preprocess_get_boundary(cleaned, self.adjacency_mat[idx_in_tissue,:][:,idx_in_tissue])
        # label the filled geometry
        cleaned = np.array(cleaned)
        current_labels = np.ones(self.coords.shape[0]) * np.nan
        for i,la in enumerate(ordered_unique_clusters):
            current_labels[ np.where(cleaned == la)[0] ] =  i
        # order the list of boundary point by graph traversal order
        for i,li in enumerate(list_boundary):
            li = order_boundary_points_by_graph(li, self.coords, self.pairwise_dist)
            list_boundary[i], CCs, normvecs, constants, sep_extensions = extend_boundary_to_full_geometry(li, self.coords, self.pairwise_dist, in_tissue)
            # expand the labels by extended layer boundaries
            for j in range(len(CCs)):
                if len(sep_extensions[j]) > 0:
                    current_labels = label_oneside_boundary_extension(self.coords, self.adjacency_mat, self.pairwise_dist, \
                                                                      CCs[j], normvecs[j], constants[j], sep_extensions[j], current_labels, i)
            if i + 1 == len(list_boundary):
                for j in range(len(CCs)):
                    if len(sep_extensions[j]) > 0:
                        current_labels = label_oneside_boundary_extension(self.coords, self.adjacency_mat, self.pairwise_dist, \
                                                                          CCs[j], normvecs[j], constants[j], sep_extensions[j], current_labels, i+1)
        # expand labels of the filled-in geometry by the closest labels
        current_labels = label_by_closest(self.coords, self.adjacency_mat, self.pairwise_dist, current_labels)
        ##### harmonic interpolation by solving linear system derived from graph Laplacian matrix for labeled and unlabeled data #####
        set_boundary_points = set(list(np.concatenate(list_boundary)))
        is_out_boundary = np.array([not j in set_boundary_points for j in range(self.coords.shape[0])])
        heat_offset = 0
        heats = np.zeros(self.coords.shape[0])
        for i in range(len(list_boundary)+1):
            if i == 0:
                idx_b1 = list_boundary[i]
                idx_inside = np.where(np.logical_and(current_labels==i, is_out_boundary))[0]
                if len(idx_inside) > 0:
                    heats_b1, heats_inside = self.diffusion_harmonic(idx_b1, idx_inside)
                    heats[np.concatenate([idx_b1, idx_inside])] = heat_offset + np.concatenate([heats_b1, heats_inside])
                    heat_offset += np.max(np.concatenate([heats_b1, heats_inside]))
                else:
                    heats[idx_b1] = heat_offset
            elif i == len(list_boundary):
                idx_b1 = list_boundary[i-1]
                idx_inside = np.where(np.logical_and(current_labels==i, is_out_boundary))[0]
                if len(idx_inside) > 0:
                    heats_b1, heats_inside = self.diffusion_harmonic(idx_b1, idx_inside)
                    heats[np.concatenate([idx_b1, idx_inside])] = heat_offset + np.max([np.max(heats_b1), np.max(heats_inside)]) - np.concatenate([heats_b1, heats_inside])
                    heat_offset += np.max(np.concatenate([heats_b1, heats_inside]))
                else:
                    heats[idx_b1] = heat_offset
            else:
                idx_b1 = list_boundary[i-1]
                idx_b2 = list_boundary[i]
                idx_inside = np.where(np.logical_and(current_labels==i, is_out_boundary))[0]
                heats_b1, heats_b2, heats_inside = self.interpolate_two_boundaries(idx_b1, idx_b2, idx_inside)
                heats[np.concatenate([idx_b1, idx_b2, idx_inside])] = heat_offset + np.concatenate([heats_b1, heats_b2, heats_inside])
                heat_offset += np.max(np.concatenate([heats_b1, heats_b2, heats_inside]))
        logger.info("Finish computing harmonic interpolation.")
        return heats, list_boundary, current_labels