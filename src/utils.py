import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path


def read_input_10xdirectory(indir):
    # read barcodes
    file_barcodes = [str(x) for x in Path(data).rglob("*barcodes.tsv*")]
    if len(file_barcodes) == 0:
        logger.error('There is no barcode.tsv file in the 10X directory.')
    barcodes = np.asarray(pd.read_csv(file_barcodes[0], header=None)).flatten()
    # read genes
    file_features = [str(x) for x in Path(data).rglob("*features.tsv*")]
    if len(file_features) == 0:
        logger.error('There is no features.tsv file in the 10X directory.')
    genes = np.asarray(pd.read_csv(file_features[0], sep='\t', header=0))
    # spatial coordinate file
    file_coords = [str(x) for x in Path(data).rglob("*tissue_positions_list.csv*")]
    if len(file_coords) == 0:
        logger.error('There is no tissue_positions_list.csv file in the 10X directory.')
    coords = np.asarray(pd.read_csv(file_coords[0], sep=',', header=None))
    # count matrix file
    return NotImplemented


def read_input_st_files(stfiles):
    return NotImplemented


def pool_data(count, T):
    return NotImplemented


def find_rotation_angle(count, pos, L):
    return NotImplemented
