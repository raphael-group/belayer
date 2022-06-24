# import statements
import numpy as np
from scipy.spatial import Delaunay
from functools import reduce
import operator
import math
from tqdm import trange

def get_loss_index(coords, coord):
    """Get index of coord in coords.

    :param coords: List of tuples
    :type coords: list
    :param coord: Tuple in coords
    :type coord: tuple
    :return: Index of coord in coords
    :rtype: int
    """
    return int(np.where(np.all(coords == coord, axis = 1))[0])

def sort_indices(i, j, k, m):
    """Sort indices by the following rule. If k is none, then return (i, j, inf, inf) if i < j and (inf, inf, j, i) otherwise. Else, sort so that a < b, c < d, a < c in the return order of (a, b, c, d).

    :param i: Index of first layer boundary endpoint
    :type i: int
    :param j: Index of first layer boundary endpoint
    :type j: int
    :param k: Index of second layer boundary endpoint
    :type k: int
    :param m: Index of second layer boundary endpoint
    :type m: int
    :return: Tuple of sorted indices
    :rtype: tuple
    """
    tup_1 = (i, j)
    tup_2 = (k, m)
    if k is None:
        if i < j:
            return (i, j, np.infty, np.infty)
        else:
            return (np.infty, np.infty, j, i)
    if j < i:
        tup_1 = (j, i)
    if m < k:
        tup_2 = (m, k)
    if tup_1 < tup_2:
        return sum((tup_1, tup_2), ())
    else:
        return sum((tup_2, tup_1), ())

def gen_loss_wrapper(pre_saved_loss):
    """Wrapper for generating loss function.

    :param pre_saved_loss: Matrix of pre-saved losses, sorted as in sort_indices
    :type pre_saved_loss: ndarray
    :return: Loss function
    :rtype: function
    """
    def loss_wrapper(points, i, j, k, m):
        tup = sort_indices(i,j,k,m)
        return pre_saved_loss[tup]
    return loss_wrapper

def sort_endpoints_by_x_coord(line_1):
    """Sort two endpoints of a line by increasing x coordinate.

    :param line_1: Line, specified by [(x_1, y_1), (x_2, y_2)]
    :type line_1: list
    :return: Line with sorted x coordinates
    :rtype: list
    """
    x_coord_1 = line_1[0][0]
    x_coord_2 = line_1[1][0]
    if x_coord_1 < x_coord_2:
        return line_1
    else:
        return [line_1[1], line_1[0]]

def is_line_below(line_1, line_2):
    """Check whether a line_2 is to the right of line_1, given that the two lines do not intersect. 

    :param line_1: Line, specified by [(x_1, y_1), (x_2, y_2)]
    :type line_1: list
    :param line_2: Line, specified by [(x_1, y_1), (x_2, y_2)]
    :type line_1: list
    :return: Return True if line_2 is to the right of line_1, False otherwise
    :rtype: bool
    """
    prev_1, prev_2 = sort_endpoints_by_x_coord(line_1)
    cur_1, cur_2 = sort_endpoints_by_x_coord(line_2)

    v1 = (prev_2[0] - prev_1[0], prev_2[1] - prev_1[1])
    v2 = (prev_2[0] - cur_1[0], prev_2[1] - cur_1[1])
    xp = v1[0]*v2[1] - v1[1]*v2[0]
    
    if xp > 0:
        return True
    else:
        return False

def order_layer_boundaries(lines):
    """Order layer boundaries from left to right.

    :param lines: List of lines, where one line is specified by [(x_1, y_1), (x_2, y_2)]
    :type lines: list
    :return: List of sorted lines from left to right
    :rtype: list
    """
    for i, line in enumerate(lines):
        lines[i] = sort_endpoints_by_x_coord(line)
    
    sorted_lines = [lines[0]]
    
    for i in range(1, len(lines)):
        cur = lines[i]
        for j in range(len(sorted_lines)):
            if not is_line_below(sorted_lines[j], cur):
                sorted_lines.insert(j, cur)
                break
            if j == len(sorted_lines) - 1:
                sorted_lines.append(cur)
    
    return sorted_lines


# ARI calculation functions
# ----------------------------------------------------------------------

def get_spots_within_region(fullpoints, b1_i, b1_j, b2_i, b2_j):
    """Get all spots between two lines.

    :param fullpoints: Matrix of coordinates
    :type fullpoints: ndarray
    :param b1_i: Index of first layer boundary endpoint
    :type b1_i: int
    :param b1_j: Index of first layer boundary endpoint
    :type b1_j: int
    :param b2_i: Index of second layer boundary endpoint
    :type b2_i: int
    :param b2_j: Index of second layer boundary endpoint
    :type b2_j: int

    :return: Tuple of coordinates on first line, coordinates on second line, and coordinates between first and second line
    :rtype: tuple
    """
    x_coords = fullpoints[:,0]
    y_coords = fullpoints[:,1]
    
    # line for boundary 1: <(x,y), n> = c
    nvec_1 = np.array([ -(y_coords[b1_j] - y_coords[b1_i]), (x_coords[b1_j] - x_coords[b1_i]) ])
    nvec_1 = nvec_1 / np.sqrt(nvec_1.dot(nvec_1))
    c1 = np.array([x_coords[b1_i], y_coords[b1_i]]).dot(nvec_1)
    assert np.abs( np.array([x_coords[b1_j], y_coords[b1_j]]).dot(nvec_1) - c1 ) < 1e-4
    # which side of boundary 1
    side1 = (np.array([x_coords[b2_i], y_coords[b2_i]]).dot(nvec_1) - c1 > 0)
    # find all spots on boundary 1: on the same side as b2, but distance(spot, b1) < threshold
    signed_distance1 = np.vstack([x_coords, y_coords]).T.dot(nvec_1) - c1
    same_side1 = np.logical_or(signed_distance1 == 0, side1 == (signed_distance1 > 0))
    indicator1 = (np.abs(signed_distance1) < 1) * same_side1
    indicator1[b1_i] = 1; indicator1[b1_j] = 1;
    idx_b1 = np.where(indicator1)[0]
    coords_b1 = fullpoints[idx_b1,:]
    
    # line for boundary 2: <(x,y), n> = c
    nvec_2 = np.array([ -(y_coords[b2_j] - y_coords[b2_i]), (x_coords[b2_j] - x_coords[b2_i]) ])
    nvec_2 = nvec_2 / np.sqrt(nvec_2.dot(nvec_2))
    c2 = np.array([x_coords[b2_i], y_coords[b2_i]]).dot(nvec_2)
    assert np.abs( np.array([x_coords[b2_j], y_coords[b2_j]]).dot(nvec_2) - c2 ) < 1e-4
    # which side of boundary 2
    side2 = (np.array([x_coords[b1_i], y_coords[b1_i]]).dot(nvec_2) - c2 > 0)
    # find all spots on boundary 2: on the same side as b1, but distance(spot, b2) < threshold
    signed_distance2 = np.vstack([x_coords, y_coords]).T.dot(nvec_2) - c2
    same_side2 = np.logical_or(signed_distance2 == 0, side2 == (signed_distance2 > 0))
    indicator2 = (np.abs(signed_distance2) < 1) * same_side2
    indicator2[b2_i] = 1; indicator2[b2_j] = 1;
    idx_b2 = np.where(indicator2)[0]
    coords_b2 = fullpoints[idx_b2,:]
    
    # find all spots between the two layer boundaries
    indicator = np.logical_and.reduce((side1 == (signed_distance1 > 0), side2 == (signed_distance2 > 0), \
        np.logical_not(indicator1), np.logical_not(indicator2) ))
    idx_inside = np.where(indicator > 0)[0]
    coords_inside = fullpoints[idx_inside,:]
    
    return coords_b1, coords_b2, coords_inside

# label points by layer boundaries
def create_pred_labels(lines, fullpoints, in_tissue):
    """Label each point by layer boundaries enclosing it.

    :param lines: List of lines, where one line is specified by [(x_1, y_1), (x_2, y_2)]
    :type lines: list
    :param fullpoints: Matrix of coordinates
    :type fullpoints: ndarray
    :param in_tissue: Matrix of booleans corresponding to fullpoints, where True indicates point to label and False indicates point to skip
    :type in_tissue: ndarray
    :return: List of labels for all points marked as to label
    :rtype: list
    """
    # create label dict for {coord: label} per coord
    coord_to_label = {}
    # start label counter at 1
    label_counter = 1
    
    # for each pair of lines (line_1, line_2) , generate set of coords such that coord in [line_1, line_2)
    for i in range(len(lines) - 1):
        b1 = [get_loss_index(fullpoints, ele) for ele in lines[i]]
        b2 = [get_loss_index(fullpoints, ele) for ele in lines[i + 1]]
             
        coords_b1, coords_b2, coords_inside = get_spots_within_region(fullpoints, b1[0], b1[1], b2[0], b2[1])
        
        # for each point in set, add to dict with label = label_counter
        for coord in coords_b1:
            coord_to_label[tuple(coord)] = label_counter
        for coord in coords_inside:
            coord_to_label[tuple(coord)] = label_counter 
        for coord in coords_b2:
            coord_to_label[tuple(coord)] = label_counter 
                
        label_counter += 1
          
    # create empty pred array of size fullpoints
    fullpoints_in_tissue = fullpoints[in_tissue,:]
    pred_labels = np.empty(len(fullpoints_in_tissue))
    
    # fill pred array based on label dict
    for i in range(len(pred_labels)):
        cur_coord = fullpoints_in_tissue[i]
        pred_labels[i] = coord_to_label[tuple(cur_coord)]
    
    return pred_labels
    

def convert_boundary_index_to_loss_index(points, coords, index):
    coord = points[index]
    return get_loss_index(coords, coord)


def get_sorted_boundary_points(coords):
    # take convex hull of subsection
    tri = Delaunay(coords)
    bxy = (coords[tri.convex_hull]).flatten()
    bx = bxy[0:-2:2]
    by = bxy[1:-1:2]

    # get sorted boundary points
    boundary = list(set(zip(bx, by)))
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), boundary), [len(boundary)] * 2))
    sorted_boundary = sorted(boundary, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    
    return sorted_boundary


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it is not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    def sorted_nodes_from_edges(edges, nodes):
        # mapping a node to its precedent/decedent node
        decedent_map = {e[0]:e[1] for e in edges}
        precedent_map = {e[1]:e[0] for e in edges}
        # start from the nodes with smallest index in the edge and trace its children
        nodes.append( np.min([x[0] for x in edges]) )
        for r in range(len(edges)):
            nodes.append( decedent_map[nodes[-1]] )
        assert nodes[-1] == nodes[0]
        nodes = nodes[:-1]

    tri = Delaunay(points)
    edges = set()
    nodes = []
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # print(circum_r)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    # get sorted nodes from edges
    sorted_nodes_from_edges(edges)
    return nodes


def get_boundary_tuples_from_sorted(sorted_boundary, is_boundary_leftbottom, is_boundary_righttop):
    boundary_tuples = []
    # (i,j) are indices in sorted_boundary that define the first layer boundary, wlog we require i < j
    for i in trange(len(sorted_boundary)):
        for j in range(i+1, len(sorted_boundary)):
            # (k,m) are indices in sorted_boundary that define the second layer boundary, wlog we require k < m
            # in addition, we avoid sharing endpoints between the first and the second layer boundary, so i != k; wlog, we require i < k
            for k in range(i+1, len(sorted_boundary)):
                for m in range(k+1, len(sorted_boundary)):
                    # check non-intersection
                    not_intersect = (k < j and m < j) or (k > j and m > j)
                    # check whether one of (i,j) is from left bottom edge, and the other is from top right edge
                    good_edge_ij = (is_boundary_leftbottom[i] and is_boundary_righttop[j]) or (is_boundary_leftbottom[j] and is_boundary_righttop[i])
                    # check whether one of (k,m) is from left bottom edge, and the other is from top right edge
                    good_edge_km = (is_boundary_leftbottom[k] and is_boundary_righttop[m]) or (is_boundary_leftbottom[m] and is_boundary_righttop[k])
                    if j != m and not_intersect and good_edge_ij and good_edge_km:
                        boundary_tuples.append( (i,j,k,m) )
    return boundary_tuples


def get_full_boundary_tuples_from_sorted(sorted_boundary):
    boundary_tuples = []
    # (i,j) are indices in sorted_boundary that define the first layer boundary, wlog we require i < j
    for i in trange(len(sorted_boundary)):
        for j in range(i+1, len(sorted_boundary)):
            # (k,m) are indices in sorted_boundary that define the second layer boundary, wlog we require k < m
            # in addition, we avoid sharing endpoints between the first and the second layer boundary, so i != k; wlog, we require i < k
            for k in range(i+1, len(sorted_boundary)):
                for m in range(k+1, len(sorted_boundary)):
                    # check non-intersection
                    not_intersect = (k < j and m < j) or (k > j and m > j)
                    if j != m and not_intersect:
                        boundary_tuples.append( (i,j,k,m) )
    return boundary_tuples


def get_full_boundary_triples_from_sorted(sorted_boundary):
    boundary_triples = []
    for i in trange(len(sorted_boundary)):
        for j in range(i+1, len(sorted_boundary)):
            boundary_triples.append( (i,j,i+1) )
            boundary_triples.append( (i,j,i-1) )
    return boundary_triples


