import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path


def read_input_10xdirectory(indir):
    """Read count matrix and spatial coordinates from 10X folder.
    :param indir: 10X output directory.
    :type indir: str

    :return: a gene-by-spot UMI count matrix, a cell-by-2 spatial coordinate matrix, an array of barcodes, an array of genes
    :rtype: np.array, np.array, np.array, np.array
    """
    # read barcodes
    file_barcodes = [str(x) for x in Path(indir).rglob("*barcodes.tsv*")]
    if len(file_barcodes) == 0:
        logger.error('There is no barcode.tsv file in the 10X directory.')
    elif len(file_barcodes) > 1:
        file_barcodes = [x for x in file_barcodes if "filtered" in x]
    barcodes = np.asarray(pd.read_csv(file_barcodes[0], header=None)).flatten()
    # read genes
    file_features = [str(x) for x in Path(indir).rglob("*features.tsv*")]
    if len(file_features) == 0:
        logger.error('There is no features.tsv file in the 10X directory.')
    elif len(file_features) > 1:
        file_features = [x for x in file_features if "filtered" in x]
    genes = np.asarray(pd.read_csv(file_features[0], sep='\t', header=None))
    genes = genes[:,1]
    # spatial coordinate file
    file_coords = [str(x) for x in Path(indir).rglob("*tissue_positions_list.csv*")]
    if len(file_coords) == 0:
        logger.error('There is no tissue_positions_list.csv file in the 10X directory.')
    coords = pd.read_csv(file_coords[0], sep=',', header=None)
    coords.columns = ["barcodes", "intissue", "x", "y", "image x", "image y"]
    coords = coords[coords.intissue == 1].iloc[:, np.array([0,2,3])] # in-tissue spots
    coords.barcodes = pd.Categorical(coords.barcodes, categories=barcodes, ordered=True)
    coords.sort_values(by="barcodes", inplace=True)
    coords = np.array(coords.iloc[:, np.array([1,2])])
    # count matrix file
    file_matrix = [str(x) for x in Path(indir).rglob("*matrix.mtx*")]
    if len(file_matrix) == 0:
        logger.error('There is no matrix.mtx file in the 10X directory.')
    elif len(file_matrix) > 1:
        file_matrix = [x for x in file_matrix if "filtered" in x]
    count = scipy.io.mmread(file_matrix[0]).toarray() # count is a gene-by-spot matrix
    return count, coords, barcodes, genes


def read_boundary_list(boundary_npyfile, fullpoints):
    # after loading boundary_npyfile, the data has the following structure.
    # array(list of points in Gamma_i, list of points in Gamma_j, ...)
    # list of points in Gamma_i = [[x1,y1], [x2,y2], ...]
    boundary_coords_list = np.load(boundary_npyfile, allow_pickle=True)
    # convert array of 2D point list to indices list, where indices are within fullpoints
    boundary_list = []
    for l in boundary_coords_list:
        xcoord = np.array([x[0] for x in l])
        ycoord = np.array([x[1] for x in l])
        is_equal = np.logical_and(fullpoints[:,0].reshape(-1,1) == xcoord.reshape(1,-1), \
                                  fullpoints[:,1].reshape(-1,1) == ycoord.reshape(1,-1))
        boundary_list.append( np.where(is_equal)[0] )
    return boundary_list