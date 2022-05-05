#!/bin/python

import numpy as np
import pandas as pd
import argparse
import subprocess
from pathlib import Path
import pickle
from tqdm import trange
from glmpca import glmpca
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()
from utils_IO import *
from harmonic import *
from region_cost_fun import *
from dprelated import *

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
    parser.add_argument('-m', '--mode', required=True, choices=["A", "R", "S", "L"], help="Running mode:\n\tX: axis-aligned layered tissue" + \
                        "\n\tR:rotated axis-aligned layered tissue" + \
                        "\n\tA:approximate layer boundaries" + \
                        "\n\tL:layered tissue with linear layer boundaries")

    # number of layers
    parser.add_argument('-l', '--nlayers', required=True, type=int, help="Number of layers to infer.")

    # layer annotation (supervised mode)
    parser.add_argument('-b', '--boundaries', type=str, help="Coordinates of points on approximate layer boundaries")

    # output prefix
    parser.add_argument('-o', '--outprefix', type=str, default='belayer', help="Output prefix.")

    # platform
    parser.add_argument('-p', '--platform', choices=["ST", "Visium", "Other"], default='Visium', help="Platform for spatial transcriptomics data.")

    # subsample x coordinate in DP table
    parser.add_argument('--subsample_percent', type=float, default=1, help="Percentage of points to fill in DP table.")

    # specific options for linear layer boundary mode
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--precomputing', action='store_true', help="L mode specific option: precompute log-likelihood.")
    group2.add_argument('--inference', action='store_true', help="L mode specific option: infer layer boundaries and expression function using precomputed log-likelihood.")
    parser.add_argument('--num_batches', type=int, default=100, help="L mode specific option: number batches to split the computation.")
    parser.add_argument('--batch_id', type=int, default=0, help="L mode specific option: batch ID to be computed (0 <= batch ID < num_batches).")

    return parser
    

##############################
# RUN
##############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = construct_parser(parser)
    args = parser.parse_args()

    # print parsed input info
    if args.mode == 'A' and (args.boundaries is None):
        logger.error("approximate layer boundaries file must exist when running mode is S.")
    logger.info("indir={}".format(args.indir))
    logger.info("stfiles={}".format(args.stfiles))
    logger.info("mode={}".format(args.mode))
    logger.info("nlayers={}".format(args.nlayers))
    logger.info("approximate boundaries file={}".format(args.boundaries))
    logger.info("outprefix={}".format(args.outprefix))
    logger.info("platform={}".format(args.platform))

    ############################################################
    # STEP 1: parse ST data input
    ############################################################
    # load count
    if not (args.indir is None):
        count, coords, barcodes, genes = read_input_10xdirectory(args.indir)
    else:
        count, coords, barcodes, genes = read_input_st_files(args.stfiles)

    # load glmpca
    penalty = 10
    npc = 2 * args.nlayers
    if not Path( f"{args.outprefix}_glmpca_{npc}PC_{penalty}.npy").exists():
        np.random.seed(1)
        glmpca_res_2d_poisson = glmpca.glmpca(count, npc, fam="poi", penalty=penalty, verbose=False)["factors"]
        np.save( f"{args.outprefix}_glmpca_{npc}PC_{penalty}.npy", glmpca_res_2d_poisson)
    else:
        glmpca_res_2d_poisson = np.load( f"{args.outprefix}_glmpca_{npc}PC_{penalty}.npy" )


    ############################################################
    # STEP 2: identify layers by belayer according to input mode
    ############################################################

    # AXIS-ALIGNED
    if args.mode == 'X':
        raise Exception

    # ROTATED
    elif args.mode == 'R':
        angle, xcoords = find_rotation_angle(count, pos, args.nlayers)
        logger.info("ESTIMATED ROTATION ANGLE={}".format(angle))
        pooled_int_data, pooled_xcoords,  map_1d_bins_to_2d = pool_data(count, xcoords)
        layer_pooled, layer_2d = dp(pooled_int_data, pooled_xcoords, args.nlayers, map_1d_bins_to_2d)

    # SUPERVISED
    elif args.mode == 'A':
        if args.platform == "Visium":
            fullpoints, in_tissue = fill_geometry(coords, is_hexagon=True)
        else:
            fullpoints, in_tissue = fill_geometry(coords, is_hexagon=False)
        boundary_list = read_boundary_list(args.boundaries, fullpoints)
        ##### identifying layer boundaries #####
        # harmonic interpolation
        spos = spatialcoord(x = fullpoints[:,0], y = fullpoints[:,1])
        har = harmonic(fullpoints, spos.adjacency_mat, np.sqrt(spos.pairwise_squared_dist))
        interpolation = har.interpolation_using_list( boundary_list )
        interpolation = interpolation[in_tissue]
        # 1d DP for finding optimal layer boundaries
        error_mat, segment_map, saved_opt_functions, unique_xcoords = dp_raw(glmpca_res_2d_poisson, interpolation, n_seg=args.nlayers, subsample_percent=args.subsample_percent)
        # backtracing in DP table to output optimal layer boundaries
        segs = find_segments_from_dp(error_mat, segment_map, args.nlayers)
        dp_labels = np.zeros(count.shape[1], dtype=int)
        c=1
        for seg in segs:
            dp_labels[ np.where(np.logical_and(interpolation >= unique_xcoords[seg[0]], interpolation <= unique_xcoords[seg[-1]] )) ] = c
            c+=1
        # output labels
        df_dp_labels = pd.DataFrame({"depth":interpolation, "layer":dp_labels}, index=barcodes)
        df_dp_labels.to_csv(args.outprefix + "_layers.txt", sep="\t")
        ##### fitting expression function per gene ##### 
        selected_genes = select_commonly_expressed_genes(count, interpolation, q=0.75)
        totalumi = np.sum(count, axis=0)
        df_gene_func = segmented_poisson_regression(count[selected_genes,:], totalumi, dp_labels, interpolation)
        df_gene_func.index = genes[selected_genes]
        # output per-gene regression parameter
        df_gene_func.to_csv(args.outprefix + "_function_coefficients.txt", sep="\t")

    # LINEAR BOUNDARIES
    elif args.mode == 'L':
        # complete geometry
        if args.platform == "Visium":
            fullpoints, in_tissue = fill_geometry(coords, is_hexagon=True)
        else:
            fullpoints, in_tissue = fill_geometry(coords, is_hexagon=False)
        full_glmpca_res_2d_poisson = np.ones((fullpoints.shape[0], glmpca_res_2d_poisson.shape[1])) * np.nan
        full_glmpca_res_2d_poisson[in_tissue] = glmpca_res_2d_poisson
        ##### get boundary and enumerate pair of boundary lines #####
        sorted_boundary = get_sorted_boundary_points(fullpoints)
        # batch-precomputation of log-likelihood
        if args.precomputing:
            llf = lossfunction(full_glmpca_res_2d_poisson.T, fullpoints[:,0], fullpoints[:,1], total_num_cluster=args.nlayers, consider_points=in_tissue, platform=args.platform)
            llf.F_glmpca_2d_poisson = full_glmpca_res_2d_poisson
            # get boundary tuples
            boundary_tuples = get_full_boundary_tuples_from_sorted(sorted_boundary)
            logger.info("There are {} tuples of boundary end points.".format( len(boundary_tuples) ))
            boundary_triples = get_full_boundary_triples_from_sorted(sorted_boundary)
            logger.info("There are {} triples of boundary end points for half circle regions.".format(len(boundary_triples)))
            # output dictionary
            pre_saving = {}
            # batch for region bounded by two lines
            s = int( np.round(len(boundary_tuples) / args.num_batches * args.batch_id) )
            t = int( np.round(len(boundary_tuples) / args.num_batches * (args.batch_id+1)) )
            for r in trange(s, t):
                i, j, k, m = boundary_tuples[r]
                this_loss = llf.eval_loss(sorted_boundary[i], sorted_boundary[j], b2_endp1=sorted_boundary[k],  b2_endp2=sorted_boundary[m], loss='Gaussian')
                pre_saving[ boundary_tuples[r] ] = this_loss
            # batch for region bounded by one line and tissue boundary
            s = int( np.round(len(boundary_triples) / args.num_batches * args.batch_id) )
            t = int( np.round(len(boundary_triples) / args.num_batches * (args.batch_id+1)) )
            for r in trange(s, t):
                i, j, k = boundary_triples[r]
                this_loss = llf.eval_loss(sorted_boundary[i], sorted_boundary[j], b2_endp1=sorted_boundary[k], loss='Gaussian')
                if k == i+1:
                    pre_saving[ (i,j,np.inf,np.inf) ] = this_loss
                else:
                    pre_saving[ (np.inf,np.inf, i,j) ] = this_loss
            pickle.dump(pre_saving, open(f"{args.outprefix}_pre_saving_{args.batch_id}_{args.num_batches}.pkl", 'wb'))
        # after precomputation, inferring layer boundaries and gene expression function
        else:
            ##### load presaved log likelihood #####
            outdir = "/".join( args.outprefix.split("/")[:-1] )
            if outdir == "":
                outdir = "./"
            example_pre_saving_file = [x for x in Path(outdir).glob("*pre_saving*")][0]
            args.num_batches = int(example_pre_saving_file.split(".")[0].split("_")[-1])
            pre_saved_loss = {}
            for r in range(args.num_batches):
                tmp = pickle.load(open(f"{args.outprefix}_pre_saving_{r}_{args.num_batches}.pkl", 'rb'))
                pre_saved_loss.update(tmp)
            ##### identifying layer boundaries #####
            # create wrapper for linear DP
            loss_wrapper = gen_loss_wrapper(pre_saved_loss)
            # initialize arrays for endpoints, loss, runtime
            arr_layers, arr_loss = [list(np.zeros(args.nlayers + 1)) for _ in range(2)]
            # create DP matrix
            dp_mat = construct_dp_mat(sorted_boundary, args.nlayers, loss_wrapper)
            # for number of layers from 1 to max_layers, find best layer boundaries
            for num_layers in range(1, args.nlayers + 1):
                res = find_first_layer(dp_mat, len(sorted_boundary), args.nlayers)
                loss, layers = res
                # order layer boundaries using func above
                layers = [[sorted_boundary[ele[0]], sorted_boundary[ele[1]]] for ele in layers]
                ordered_layers = order_layer_boundaries(layers)
                # save layer boundaries, loss, runtime
                arr_layers[num_layers] = ordered_layers
                arr_loss[num_layers] = loss
            # save loss and ordered layer boundaries 
            arr_loss_np = np.asarray(arr_loss, dtype=object)
            arr_layers_np = np.asarray(arr_layers, dtype=object)
            np.save(f'{args.outprefix}_non_init_loss.npy', arr_loss_np, allow_pickle=True, fix_imports=True)
            np.save(f'{args.outprefix}_non_init_layers.npy', arr_layers_np, allow_pickle=True, fix_imports=True)
            ##### fitting expression function per gene ##### 
            raise Exception

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
