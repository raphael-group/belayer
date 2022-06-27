import numpy as np
import sklearn.neighbors
import networkx as nx


class spatialcoord(object):
    '''
    A module to store spatial position and its derived information.
    '''
    def __init__(self, x, y, platform="Visium", n_neighbors=10):
        ##### a list of all attributes #####
        self.platform = None
        self.adjacency_type = None
        self.n_neighbors = None
        self.x = None
        self.y = None
        self.unit_xsquared = None
        self.unit_ysquared = None
        self.pairwise_squared_dist = None
        self.adjacency_mat = None
        ##### initialize basic information #####
        self.platform = platform
        self.n_neighbors = n_neighbors
        assert( len(x) == len(y) )
        self.x = x
        self.y = y
        # adjacency type is the way to construct adjacency matrix, it is  KNN graph in general
        # but for certain platforms if the spatial coordinate is integer valued, directly use the grid information
        self.adjacency_type = "knn"
        if (self.platform == "ST" or self.platform == "Visium"):
            self.adjacency_type = "grid"
            self.x = np.round(x).astype(np.int)
            self.y = np.round(y).astype(np.int)
        # for certain platforms, specify the unit distance of x^2 and y^2
        self.unit_xsquared = 1
        self.unit_ysquared = 1
        if self.platform == "Visium":
            self.unit_xsquared = 9
            self.unit_ysquared = 3
        ##### initialize extended information #####
        self.pairwise_squared_dist = self.compute_pairwise_distance()
        self.adjacency_mat = self.compute_adjacency_matrix()
    #
    def compute_pairwise_distance(self):
        x_dist = self.x[None,:] - self.x[:,None]
        y_dist = self.y[None,:] - self.y[:,None]
        squared_dist = x_dist**2 * self.unit_xsquared + y_dist**2 * self.unit_ysquared
        return squared_dist
    # 
    def compute_adjacency_matrix(self):
        assert(self.adjacency_type == "knn" or self.adjacency_type == "grid")
        assert( not (self.pairwise_squared_dist is None) )
        A = np.zeros( (len(self.x), len(self.y)), dtype=np.int8 )
        if self.adjacency_type == "grid":
            assert( self.platform == "Visium" or self.platform == "ST" )
            if self.platform == "Visium":
                for i in range(len(self.x)):
                    indexes = np.where(self.pairwise_squared_dist[i,:] <= self.unit_xsquared + self.unit_ysquared)[0]
                    indexes = np.array([j for j in indexes if j != i])
                    if len(indexes) > 0:
                        A[i, indexes] = 1
            elif self.platform == "ST":
                for i in range(len(self.x)):
                    indexes = np.where(self.pairwise_squared_dist[i,:] <= 1)[0]
                    indexes = np.array([j for j in indexes if j != i])
                    if len(indexes) > 0:
                        A[i, indexes] = 1
            return A
        else:
            NN = sklearn.neighbors.NearestNeighbors(n_neighbors=self.n_neighbors, metric="precomputed").fit(self.pairwise_squared_dist)
            knn_graph = NN.kneighbors_graph()
            # convert from CSR sparse matrix to dense matrix
            return np.array(knn_graph.todense()).astype(np.int8)
    #
    def gaussian_kernel(self, sigma):
        # normalize squared distance if Visium
        dist_pos = self.pairwise_squared_dist if not (self.platform == "Visium") \
            else self.pairwise_squared_dist / (self.unit_xsquared + self.unit_ysquared)
        K = 1.0 / sigma / np.sqrt(2 * np.pi) * np.exp(- dist_pos / 2.0 / sigma / sigma)
        return K
    #
    def graph_laplacian(self, sigma):
        K = self.gaussian_kernel(sigma)
        L = np.diag(np.sum(K, axis=0)) - K
        return L
    #
    def fractional_hausdorff_distance(self, point_list_A, point_list_B, percentile=100):
        dist = np.sqrt( self.pairwise_squared_dist[np.ix_( np.array(point_list_A), np.array(point_list_B) )] )
        from_A_to_B = np.min(dist, axis=0)
        from_B_to_A = np.min(dist, axis=1)
        if len(from_A_to_B) > len(from_B_to_A):
            phd = np.percentile(from_B_to_A, percentile)
        else:
            phd = np.percentile(from_A_to_B, percentile)
        return phd
    #
    def connected_components(self, point_list):
        # create a subgraph with point_list
        sub_adj_mat = self.adjacency_mat[np.ix_( np.array(point_list), np.array(point_list) )]
        # map the node index of the subgraph to the original node index
        map_index = {i:point_list[i] for i in range(len(point_list))}
        G = nx.Graph()
        G.add_nodes_from( np.arange(len(point_list)) )
        G.add_edges_from( [(i,j) for i in range(sub_adj_mat.shape[0]) for j in range(i+1, sub_adj_mat.shape[0]) if sub_adj_mat[i,j] > 0] )
        tmpcomponents = [g for g in nx.connected_components(G)]
        # convert the index from the subgraph to the original points
        components = []
        for g in tmpcomponents:
            components.append( [map_index[x] for x in g] )
        components.sort(key = lambda x:len(x), reverse=True)
        return components