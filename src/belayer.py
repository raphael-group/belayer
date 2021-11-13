#!/bin/python

import numpy as np
import pandas as pd
import argparse
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()
from utils import *
from dprelated import *
from spatialcoord import *
from tps import *


##############################
# PARSING ARGUMENTS
##############################

def construct_parser(parser):
    group = parser.add_mutually_exclusive_group(required=True)

    # input 10x directory
    group.add_argument('-i', '--indir', type=str, help="Input 10X directory for ST data.")

    # input count matrix and spatial coordinates as separate files
    group.add_argument('-s', '--stfiles', nargs=2, type=str, help="Input count matrix file followed by spatial coordinate file for ST data. Count matrix and spatial coordinate must have the same number of spots.")

    # running mode
    parser.add_argument('-m', '--mode', required=True, choices=["A", "R", "S", "L"], help="Running mode:\n\tA: axis-aligned layered tissue" + \
                        "\n\tR:rotated axis-aligned layered tissue" + \
                        "\n\tS:arbitrarily curved tissue supervised by annotated layers" + \
                        "\n\tL:layered tissue with linear layer boundaries")

    # number of layers
    parser.add_argument('-L', '--nlayers', required=True, type=int, help="Number of layers to infer.")

    # layer annotation (supervised mode)
    parser.add_argument('-a', '--annotation', type=str, help="Annotated layers for each spot when using S mode.")

    # output prefix
    parser.add_argument('-o', '--outprefix', type=str, default='belayer', help="Output prefix.")

    # platform
    parser.add_argument('-p', '--platform', choices=["ST","Visium"], default='Visium', help="Platform for spatial transcriptomics data. Only used when running mode is S.")

    return parser
    

##############################
# RUN
##############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = construct_parser(parser)
    args = parser.parse_args()

    # print parsed input info
    if args.mode == 'S' and (args.annotation is None):
        logger.error("Annotated layers file must exist when running mode is S.")
    logger.info("indir={}".format(args.indir))
    logger.info("stfiles={}".format(args.stfiles))
    logger.info("mode={}".format(args.mode))
    logger.info("nlayers={}".format(args.nlayers))
    logger.info("annotation={}".format(args.annotation))
    logger.info("outprefix={}".format(args.outprefix))
    logger.info("platform={}".format(args.platform))

    ############################################################
    # STEP 1: parse ST data input
    ############################################################

    if not (args.indir is None):
        count, pos, barcodes, genes = read_input_10xdirectory(args.indir)
    else:
        count, pos, barcodes, genes = read_input_st_files(args.stfiles)

    ############################################################
    # STEP 2: identify layers by belayer according to input mode
    ############################################################

    # AXIS-ALIGNED
    if args.mode == 'A':
        xcoords = pos[:,0]
        pooled_int_data, pooled_xcoords, map_1d_bins_to_2d = pool_data(count, xcoords)
        layer_pooled, layer_2d = dp(pooled_int_data, pooled_xcoords, args.nlayers, map_1d_bins_to_2d)

    # ROTATED
    elif args.mode == 'R':
        angle, xcoords = find_rotation_angle(count, pos, args.nlayers)
        logger.info("ESTIMATED ROTATION ANGLE={}".format(angle))
        pooled_int_data, pooled_xcoords,  map_1d_bins_to_2d = pool_data(count, xcoords)
        layer_pooled, layer_2d = dp(pooled_int_data, pooled_xcoords, args.nlayers, map_1d_bins_to_2d)

    # SUPERVISED
    elif args.mode == 'S':
        spos = spatialcoord(x=pos[:,0], y=pos[:,1], platform=args.platform)
        annot = pd.read_csv(args.annotation) # this file reading function need to be checked
        tps = tps(spos, cluster_annotation=annot.iloc[:,0])
        xcoords = tps.interpolation
        pooled_int_data, pooled_xcoords, map_1d_bins_to_2d = pool_data(count, xcoords)
        layer_pooled, layer_2d = dp(pooled_count, pooled_xcoords, args.nlayers, map_1d_bins_to_2d)

    # LINEAR BOUNDARIES
    elif args.mode == 'L':
        logger.error("L mode to be implemented.")

    # invalid INPUT MODE
    else:
        logger.error("Invalid input mode.")

    #################################################################################
    # STEP 3: estimating piecewise linear function coefficients for individual genes
    #################################################################################

    # TBD

    #################################################################################   
    # STEP 4: output the inferred layers and expression functions
    #################################################################################

    # OUTPUT inferred layers
    df_layer = pd.DataFrame( {"xcoords":xcoords, "layer":layer_2d}, index=barcodes )
    df_layer.to_csv(args.outprefix + "_layer.csv")

    # OUTPUT expression functions (I.e. piecewise linear function coefficient)
    # TBD
