#!/bin/python

import numpy as np
import pandas as pd
import argparse
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()
from utils import *
from spatialcoord import *
from tps import *


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
    # number layers
    parser.add_argument('-L', '--nlayers', required=True, type=int, help="Number of layers to infer.")
    # layer annotation (supervised mode)
    parser.add_argument('-a', '--annotation', type=str, help="Annotated layers for each spot when using S mode.")
    # output prefix
    parser.add_argument('-o', '--outprefix', type=str, default='belayer', help="Output prefix")
    return parser
    

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
    ##### step 1: parse ST data input #####
    if not (args.indir is None):
        count, pos = read_input_10xdirectory(args.indir)
    else:
        count, pos = read_input_st_files(args.stfiles)
    ##### step 2: identify layers by belayer according to input mode #####
    if args.mode == 'A':
        pooled_xcoord_int, pooled_count = pool_data(count, pos[:,0])
        layer_pooled, layer_2d = dp(pooled_count, pooled_xcoord_int, args.nlayers)
    elif args.mode == 'R':
        angle, rotated_x = find_rotation_angle(count, pos, args.nlayers)
        pooled_xcoord_int, pooled_count = pool_data(count, rotated_x)
        layer_pooled, layer_2d = dp(pooled_count, pooled_xcoord_int, args.nlayers)
    elif args.mode == 'S':
        spos = spatialcoord(x=pos[:,0], y=pos[:,1], platform=args.platform)
        annot = pd.read_csv(args.annotation) # this file reading function need to be checked
        tps = tps(spos, cluster_annotation=annot.iloc[:,0])
        pooled_xcoord_int, pooled_count = pool_data(count, tps.interpolation)
        layer_pooled, layer_2d = dp(pooled_count, pooled_xcoord_int, args.nlayers)
    else:
        logger.error("L mode to be implemented.")
    ##### step 3: estimating piecewise linear function coefficients for individual genes #####
    # TBD
    ##### step 4: output the inferred layer and piecewise linear function info #####
    # TBD
