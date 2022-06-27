import os
from pprint import pprint
from csv import reader

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, average_precision_score, precision_recall_curve
from sklearn import linear_model,preprocessing

from scipy.stats import mode, poisson, chi2
from scipy.spatial import ConvexHull, convex_hull_plot_2d, Delaunay
from scipy.optimize import minimize, Bounds
from scipy.linalg import orth, eigh
from scipy import sparse, io

import statsmodels.api
import statsmodels as sm

import anndata
import scanpy as sc

from glmpca import glmpca

import alphashape

import networkx as nx

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
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

    tri = Delaunay(points)
    edges = set()
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
    return edges

# find closest point in grid {xlist X ylist} to a single cell=[x,y]
# can filter to only look at a subset select_grid_pts of grid points

# if there is no such grid point you should try increasing the radius
def dist_to_grid(cell, xlist, ylist, select_grid_pts=None, radius=1):
    x,y=cell
    xl_ind_topright  = np.searchsorted( xlist,x)
    yl_ind_topright = np.searchsorted( ylist,y)
    
    cands=[ np.array([xlist[xl_ind_topright],ylist[yl_ind_topright]]) ]
    
    for r in range(1,radius+1):
        cands.append( np.array([xlist[xl_ind_topright], ylist[yl_ind_topright-r]]) )
        cands.append( np.array([xlist[xl_ind_topright-r], ylist[yl_ind_topright]]) )
        cands.append( np.array([xlist[xl_ind_topright-r], ylist[yl_ind_topright-r]]) )
    
    if select_grid_pts is not None:
        cands=[cand for cand in cands if tuple(cand) in select_grid_pts]
        
    if len(cands) > 0:
    
        min_pt=cands[ np.argmin( [np.linalg.norm(cell-cand) for cand in cands] ) ]
        min_dist=np.linalg.norm(min_pt-cell)
        
        # https://stackoverflow.com/questions/25823608/find-matching-rows-in-2-dimensional-numpy-array
        min_ind=np.where((np.array(select_grid_pts) == (min_pt[0],min_pt[1])).all(axis=1))[0][0]

        return min_dist,min_pt,min_ind
    else:
        return -1, -1, -1


    
#########################################################################################################


# solves harmonic equation on a small grid (finite difference method)
# then rounds each cell to nearest neighbor in grid (using radius variable)
# if something does not work, try making grid_spacing smaller and increasing the radius

# Assumes len(boundary_array) = 2
# TODO: extend to >2 approximate layer boundaries
def harmonic_slideseq(coords, boundary_array, grid_spacing=40, radius=1):
    
    # STEP 1: make grid
    minX=np.floor( np.min(coords[:,0]) / grid_spacing ) * grid_spacing
    maxX=np.ceil( np.max(coords[:,0]) / grid_spacing ) * grid_spacing + 1

    minY=np.floor( np.min(coords[:,1]) / grid_spacing ) * grid_spacing
    maxY=np.ceil( np.max(coords[:,1]) / grid_spacing ) * grid_spacing + 1
    
    xlist=np.arange(minX, maxX, grid_spacing)
    ylist=np.arange(minY, maxY, grid_spacing)
    xv,yv=np.meshgrid(xlist,ylist)
    
    ######################################################################
    
    # STEP 2: create alpha-shape of coords
    
    edges = list(alpha_shape(coords, alpha=1000, only_outer=True))
    G = nx.DiGraph(edges)

    edge_pts=list(nx.simple_cycles(G))[0]
    edge_pts=[(coords[e,0],coords[e,1]) for e in edge_pts]

    # STEP 2.5: restrict to grid points lying inside alpha-shape
    
    G=nx.generators.lattice.grid_2d_graph(len(xlist), len(ylist))
    A=nx.adjacency_matrix(G) # adjacency matrix of grid graph

    polygon=Polygon(edge_pts)

    grid_pts_in_polygon=[]
    grid_pts_in_polygon_inds=[]

    for ind,node in enumerate(G.nodes()):
        x_old,y_old=node # graph nodes go from (0,0) to (n,m), need to convert to our coords, eg (2400,2200)
        x=x_old*grid_spacing+minX
        y=y_old*grid_spacing+minY

        point=Point((x,y))
        if polygon.contains(point):
            grid_pts_in_polygon.append((x,y))
            grid_pts_in_polygon_inds.append(ind)

    all_pts=np.array( grid_pts_in_polygon )
    
    A=A[np.ix_(grid_pts_in_polygon_inds,grid_pts_in_polygon_inds)] # restrict to grid points in polygon
    
    ######################################################################
    
    # STEP 3: assign grid points to either left boundary or right boundary (or both/neither)
    
    left_boundary, right_boundary=boundary_array

    left_inds=[]
    right_inds=[]
    lr_inds=[]

    for c in np.arange( left_boundary.shape[0] ):
        cell=left_boundary[c,:]
        _,_,closest_grid_ind=dist_to_grid(cell, xlist, ylist, select_grid_pts=grid_pts_in_polygon)
        left_inds.append(closest_grid_ind)
        lr_inds.append(closest_grid_ind)

    for c in np.arange( right_boundary.shape[0] ):
        cell=right_boundary[c,:]
        _,_,closest_grid_ind=dist_to_grid(cell, xlist, ylist, select_grid_pts=grid_pts_in_polygon)
        right_inds.append(closest_grid_ind)
        lr_inds.append(closest_grid_ind)

    non_lr_inds=[t for t in range(len(grid_pts_in_polygon)) if t not in lr_inds]
    
    ######################################################################
    
    # STEP 4: harmonic interpolation (following http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.69.5962&rep=rep1&type=pdf)
    
    D=np.diag(np.asarray(np.sum(A,0)).flatten())
    L=D-A

    u=-1 * np.ones(A.shape[0])
    u[left_inds]=0
    u[right_inds]=100

    uB=u[lr_inds]
    RT=L[ np.ix_(non_lr_inds, lr_inds) ]
    L_U=L[np.ix_(non_lr_inds,non_lr_inds)]
    
    # solve system of eqns (eqn (7) in above textbook)
    print('starting harmonic interpolation...')
    uothers,_=sparse.linalg.cg(L_U, (-RT@uB).T) 
    u[non_lr_inds]=uothers
    
    ######################################################################
    
    # STEP 5: assign each cell to nearest grid point
    
    N=coords.shape[0]
    depth=np.zeros(N)

    for i in range(N):
        cell=coords[i,:]
        _,_,closest_grid_ind=dist_to_grid(cell, xlist, ylist, select_grid_pts=grid_pts_in_polygon)

        depth[i]=u[closest_grid_ind]
    
    return depth


    
    