import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path


def read_input_10xdirectory(indir):
    # read barcodes
    file_barcodes = [str(x) for x in Path(indir).rglob("*barcodes.tsv*")]

    if len(file_barcodes) == 0:
        logger.error('There is no barcode.tsv file in the 10X directory.')

    barcodes = np.asarray(pd.read_csv(file_barcodes[0], header=None)).flatten()
    
    # read genes
    file_features = [str(x) for x in Path(indir).rglob("*features.tsv*")]
    if len(file_features) == 0:
        logger.error('There is no features.tsv file in the 10X directory.')
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
    pos = np.array(coords.iloc[:, np.array([1,2])])

    # count matrix file
    file_matrix = [str(x) for x in Path(indir).rglob("*matrix.mtx*")]
    if len(file_matrix) == 0:
        logger.error('There is no matrix.mtx file in the 10X directory.')
    count = scipy.io.mmread(file_matrix[0]).toarray() # count is a gene-by-spot matrix

    return count, pos, barcodes, genes


def read_input_st_files(stfiles):
    # count matrix file: assume it is a gene-by-spot matrix
    # check the existence of header, check delimiter of count file
    with open(stfiles[0], 'r') as fp:
        line = fp.readline().strip()
        if len(line.split(",")) > 1:
            delim = ","
        elif len(line.split("\t")) > 1:
            delim = "\t"
        else:
            logger.error("Unknown delimiter for file {}".format(stfiles[0]))
        has_header = None
        strs = np.array(line.split(delim))
        try:
            strs = strs[1:].astype(int)
        except:
            has_header = 0
    count = pd.read_csv(stfiles[0], sep=delim, header=has_header)
    has_index = None
    try:
        tmp = count.iloc[:,0].astype(int)
        if np.all(tmp == np.arange(len(tmp))):
            has_index = 0
    except:
        has_index = 0
    if has_index:
        count.index = count[:,0]
        count = count.iloc[:,1:].astype(int)
    # spatial position file
    with open(stfiles[1], 'r') as fp:
        line = fp.readline().strip()
        if len(line.split(",")) > 1:
            delim = ","
        elif len(line.split("\t")) > 1:
            delim = "\t"
        else:
            logger.error("Unknown delimiter for file {}".format(stfiles[0]))
        has_header = None
        strs = np.array(line.split(delim))
        try:
            strs = strs[1:].astype(int)
        except:
            has_header = 0
    pos = pd.read_csv(stfiles[1], sep=delim, header=has_header)
    if pos.shape[1] > 2:
        pos = pos.iloc[:,1:3].astype(int)
    if count.shape[1] != pos.shape[0]:
        logger.error("Count matrix and spatial positions have different number of spots.")
    return np.array(count), np.array(pos), count.columns, count.index



# INPUT:
# count: G x N count data matrix 
# xcoords: N x 1 vector, has x coordinates for each spot

# OUTPUT:
# pooled count: G x N_1d matrix over pooled spots
# pooled_xcoords: N_1d x 1 matrix of xcoords for each pooled spot 
# map_1d_bins_to_2d: dictionary  mapping each pooled spot -> [2d spots pooled together]
def pool_data(count, xcoords):
    G,N = count.shape

    xcoords_int = np.round(xcoords).astype(int)
    pooled_xcoords = np.sort(np.unique(xcoords_int))
    N_1d = pooled_xcoords.shape[0]

    # map b -> [list of cells in bin b]
    map_1d_bins_to_2d={}
    pooled_count = np.zeros( (G, N_1d) )

    for ind, b in enumerate(pooled_xcoords):
        bin_pts = np.where(xcoords_int == b)[0]
        pooled_count[:,ind] = np.sum(count[:,bin_pts], axis=1)
        map_1d_bins_to_2d[b] = bin_pts
        
    return pooled_count, pooled_xcoords, map_1d_bins_to_2d

def rotate_by_theta(coords, theta, rotate_about=np.array([0,0])):
    coordsT=coords.T
    
    c,s=np.cos(theta), np.sin(theta)
    rotation_matrix=np.array(((c, -s), (s, c)))
    
    return (rotation_matrix @ coordsT).T