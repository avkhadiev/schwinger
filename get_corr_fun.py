#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from schwinger_helper import cfg_dir, res_bin_dir, cfg_fname, res_fname_binned
from schwinger import Cfg
from schwinger import alpha
from numba import jit
import os

# type of interpolating operator used (for filenames)
op_string = "smart"
# job settings
job_id = 1

###############################################################################
#                             SMART OPERATOR                                  #
###############################################################################
weights_a = np.array([0.671637489719553726352785361087,
    -0.640828927131357750646145632345,
    0.233282602163959656982328283448,
    0.151574737111272478395918028582,
    0.151574737111272506151493644211,
    0.153253378044338056662709846023,
    -0.0853008570576624658432507430916,
    -0.0773472889600883423133481642253])

weights_b = np.array([-0.704461396541017470518397658452,
    0.528854887280259311488350704167,
    0.337820465792675772576103554456,
    -0.294185447976209768494726404242,
    0.090152056046443884707031202197])

#
#   |IN EVEN PARITY SECTOR
#
def overcnt(nsites, separation):
    L = nsites / 2
    factor = 1
    if (L % separation == 0):
        factor = 2
    return 1./float(np.sqrt(factor))

# (* ground state stays in the ground state*)
def a11(cfg, site, time):
    return 1
def a11d(cfg, site, time):
    return 1

# (* 1 hop *)
def a21(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = 1./float(np.sqrt(nsites))
    a21 = norm * (int(cfg.is_hop(i, i-1, t)) + int(cfg.is_hop(i, i+1, t)))
    return a21
def a21d(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = 1./float(np.sqrt(nsites))
    a21d = norm * (int(cfg.is_hop(i-1, i, t)) + int(cfg.is_hop(i+1, i, t)))
    return a21d

# (* simultaneous hopping on two neighboring  spatial sites in the same direction *)
def a31(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = 1./float(np.sqrt(nsites))
    a31 = norm * (
                int(cfg.is_hop(i, i+1, t)) * int(cfg.is_hop(i+2, i+3, t))
                + int(cfg.is_hop(i, i-1, t)) * int(cfg.is_hop(i-2, i-3, t))
            )
    return a31
def a31d(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = 1./float(np.sqrt(nsites))
    a31d = norm * (
                int(cfg.is_hop(i+3, i+2, t)) * int(cfg.is_hop(i+1, i, t))
                + int(cfg.is_hop(i-3, i-2, t)) * int(cfg.is_hop(i-1, i, t))
            )
    return a31d

# simultaneous hopping on spatial sites, separated by 2 spatial sites, \
# in opposite directions, summed over all spatial sites
def a41(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = 1./float(np.sqrt(nsites/2))
    a41 = norm * (int(cfg.is_hop(i, i+1, t)) * int(cfg.is_hop(i+6, i+5, t)))
    return a41
def a41d(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = 1./float(np.sqrt(nsites/2))
    a41d = norm * (int(cfg.is_hop(i+5, i+6, t)) * int(cfg.is_hop(i+1, i, t)))
    return a41d

# simultaneous hopping on two neighboring even sites, separated by a \
# spatial site, in opposite directions, summed over all spatial sites
def a51(cfg, site, time):
    i = site
    t = time
    norm = 1./float(np.sqrt(nsites/2))
    a51 = norm * (int(cfg.is_hop(i, i+1, t)) * int(cfg.is_hop(i+4, i+3, t)))
    return a51
def a51d(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = 1./float(np.sqrt(nsites/2))
    a51d = norm * (int(cfg.is_hop(i+3, i+4, t)) * int(cfg.is_hop(i+1, i, t)))
    return a51d

# simultaneous hopping on two spatial sites in the same direction, \
# separated by a spatial site, summed over all spatial sites
# extra factor of normalization because each term is counted twice with L = 4
def a61(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = overcnt(nsites, 2) * (1./float(np.sqrt(nsites)))
    a61 = norm * (
            int(cfg.is_hop(i, i+1, t)) * int(cfg.is_hop(i+4, i+5, t))
            + int(cfg.is_hop(i, i-1, t)) * int(cfg.is_hop(i-4, i-5, t))
            )
    return a61
def a61d(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = overcnt(nsites, 2) * (1./float(np.sqrt(nsites)))
    a61d = norm * (
            int(cfg.is_hop(i+5, i+4, t)) * int(cfg.is_hop(i+1, i, t))
            + int(cfg.is_hop(i-5, i-4, t)) * int(cfg.is_hop(i-1, i, t))
            )
    return a61d

# simultaneous hopping on 3 adjacent spatial sites in the same \
# direction, summed over all spatial sites
def a71(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = (1./float(np.sqrt(nsites)))
    a71 = norm * (
                int(cfg.is_hop(i, i+1, t)) * int(cfg.is_hop(i+2, i+3, t)) * int(cfg.is_hop(i+4, i+5, t))
                + int(cfg.is_hop(i, i-1, t)) * int(cfg.is_hop(i-2, i-3, t)) * int(cfg.is_hop(i-4, i-5, t))
            )
    return a71
def a71d(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = (1./float(np.sqrt(nsites)))
    a71d = norm * (
                int(cfg.is_hop(i+5, i+4, t)) * int(cfg.is_hop(i+3, i+2, t)) * int(cfg.is_hop(i+1, i, t))
                + int(cfg.is_hop(i-5, i-4, t)) * int(cfg.is_hop(i-3, i-2, t)) * int(cfg.is_hop(i-1, i, t))
            )
    return a71d

# simultaneous hopping on two adjacent spatial sites in the same \
# direction; and in the opposite direction from the spatial site that \
# is next to nearest to one of the two adjacent sites
def a81(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = (1./float(np.sqrt(nsites)))
    a81 = norm * (
                int(cfg.is_hop(i, i+1, t)) * int(cfg.is_hop(i+2, i+3, t)) * int(cfg.is_hop(i+6, i+5, t))
                + int(cfg.is_hop(i, i+1, t)) * int(cfg.is_hop(i+4, i+3, t)) * int(cfg.is_hop(i+6, i+5, t))
            )
    return a81
def a81d(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = (1./float(np.sqrt(nsites)))
    a81d = norm * (
                int(cfg.is_hop(i+5, i+6, t)) * int(cfg.is_hop(i+3, i+2, t)) * int(cfg.is_hop(i+1, i, t))
                + int(cfg.is_hop(i+5, i+6, t)) * int(cfg.is_hop(i+3, i+4, t)) * int(cfg.is_hop(i+1, i, t))
            )
    return a81d

# cannot readily implement in MC
def a91(cfg, site, time):
    pass
def a91d(cfg, site, time):
    pass

#
#   | 1 > -> | j' > IN ODD PARITY SECTOR
#   | j'> -> | 1  >
#

# 1 hop, summed over all spatial sites,
# hopping to the right = > plus sign, hopping to the left = > minus sign
def b11(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = 1./float(np.sqrt(nsites))
    b11 = norm * ((-1)*int(cfg.is_hop(i, i-1, t)) + int(cfg.is_hop(i, i+1, t)))
    return b11
def b11d(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = 1./float(np.sqrt(nsites))
    b11d = norm * ((-1)*int(cfg.is_hop(i-1, i, t)) + int(cfg.is_hop(i+1, i, t)))
    return b11d

# simultaneous hopping on two neighboring spatial sites in the same \
# direction, summed over all spatial sites,
# jumps to the right = > plus sign; jumps to the left = > minus sign
def b21(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = 1./float(np.sqrt(nsites))
    b21 = norm * (
                int(cfg.is_hop(i, i+1, t)) * int(cfg.is_hop(i+2, i+3, t))
                - int(cfg.is_hop(i, i-1, t)) * int(cfg.is_hop(i-2, i-3, t))
            )
    return b21
def b21d(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = 1./float(np.sqrt(nsites))
    b21d = norm * (
                int(cfg.is_hop(i+3, i+2, t)) * int(cfg.is_hop(i+1, i, t))
                - int(cfg.is_hop(i-3, i-2, t)) * int(cfg.is_hop(i-1, i, t))
            )
    return b21d

# simultaneous hopping on two spatial sites in the same direction, \
# separated by 1 spatial site, summed over all spatial sites,
# jumps to the right => plus sign; jumps to the left => minus sign
# note the normalization factor => overcounting of pairs for the \
# special case of L = 4
def b31(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = overcnt(nsites, 2) * (1./float(np.sqrt(nsites)))
    b31 = norm * (
            int(cfg.is_hop(i, i+1, t)) * int(cfg.is_hop(i+4, i+5, t))
            - int(cfg.is_hop(i, i-1, t)) * int(cfg.is_hop(i-4, i-5, t))
            )
    return b31
def b31d(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = overcnt(nsites, 2) * (1./float(np.sqrt(nsites)))
    b31d = norm * (
            int(cfg.is_hop(i+5, i+4, t)) * int(cfg.is_hop(i+1, i, t))
            - int(cfg.is_hop(i-5, i-4, t)) * int(cfg.is_hop(i-1, i, t))
            )
    return b31d

# simultaneous hopping on three neighboring even sites in the same \
# direction, summed over all even sites,
# jumps to the right = > plus sign,
# jumps to the left = > minus sign
def b41(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = (1./float(np.sqrt(nsites)))
    b41 = norm * (
                int(cfg.is_hop(i, i+1, t)) * int(cfg.is_hop(i+2, i+3, t)) * int(cfg.is_hop(i+4, i+5, t))
                - int(cfg.is_hop(i, i-1, t)) * int(cfg.is_hop(i-2, i-3, t)) * int(cfg.is_hop(i-4, i-5, t))
            )
    return b41
def b41d(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = (1./float(np.sqrt(nsites)))
    b41d = norm * (
                int(cfg.is_hop(i+5, i+4, t)) * int(cfg.is_hop(i+3, i+2, t)) * int(cfg.is_hop(i+1, i, t))
                - int(cfg.is_hop(i-5, i-4, t)) * int(cfg.is_hop(i-3, i-2, t)) * int(cfg.is_hop(i-1, i, t))
            )
    return b41d

# cannot readily implement in MC
def b51(cfg, site, time):
    pass

#
# simultaneous hopping on two adjacent spatial sites in the same \
# direction and in the opposite direction on the spatial site next to \
# nearest to one of the two adjacent sites, summed over all spatial sites
#
# two jumps to the right => minus sign; two jumps to the left => plus \
# sign
def b61(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = (1./float(np.sqrt(nsites)))
    b61 = norm * (
                (-1) * int(cfg.is_hop(i, i+1, t)) * int(cfg.is_hop(i+2, i+3, t)) * int(cfg.is_hop(i+6, i+5, t))
                + int(cfg.is_hop(i, i+1, t)) * int(cfg.is_hop(i+4, i+3, t)) * int(cfg.is_hop(i+6, i+5, t))
            )
    return b61
def b61d(cfg, site, time):
    i = site
    t = time
    nsites = cfg.nsites
    norm = (1./float(np.sqrt(nsites)))
    b61d = norm * (
                (-1) * int(cfg.is_hop(i+5, i+6, t)) * int(cfg.is_hop(i+3, i+2, t)) * int(cfg.is_hop(i+1, i, t))
                + int(cfg.is_hop(i+5, i+6, t)) * int(cfg.is_hop(i+3, i+4, t)) * int(cfg.is_hop(i+1, i, t))
            )
    return b61d

#
# CONSTRUCT FULL OPERATOR
#
# 1
# returns a list of coefficients
def smart_op_1(cfg, src_i, time, dagger = False):
    i = src_i
    t = time
    op1_sequence = ['a11d(cfg, i, t)',
            'a21d(cfg, i, t)',
            'a31d(cfg, i, t)',
            'a41d(cfg, i, t)',
            'a51d(cfg, i, t)',
            'a61d(cfg, i, t)',
            'a71d(cfg, i, t)',
            'a81d(cfg, i, t)'
            ]
    if(dagger):
        op1_sequence = ['a11(cfg, i, t)',
                'a21(cfg, i, t)',
                'a31(cfg, i, t)',
                'a41(cfg, i, t)',
                'a51(cfg, i, t)',
                'a61(cfg, i, t)',
                'a71(cfg, i, t)',
                'a81(cfg, i, t)'
                ]
    op1 = np.fromiter(map(eval, op1_sequence), dtype=np.float64)
    op1 *= weights_a
    return op1
# 2
# returns a list of coefficients
def smart_op_2(cfg, src_i, time, dagger = False):
    i = src_i
    t = time
    op2_sequence = ['b11(cfg, i, t)',
            'b21(cfg, i, t)',
            'b31(cfg, i, t)',
            'b41(cfg, i, t)',
            'b61(cfg, i, t)'
            ]
    if(dagger):
        op2_sequence = ['b11d(cfg, i, t)',
                'b21d(cfg, i, t)',
                'b31d(cfg, i, t)',
                'b41d(cfg, i, t)',
                'b61d(cfg, i, t)'
                ]
    op2 = np.fromiter(map(eval, op2_sequence), dtype=np.float64)
    op2 *= weights_b
    return op2

def smart_op(op1, op2):
    # go through list of a operators
    res = 0.
    op1_list = op1[op1 != 0.]
    if op1_list.size:
        op2_list = op2[op2 != 0.]
        if op2_list.size:
            res = np.sum( np.multiply.outer(op1_list, op2_list) )
    return res

#
# CONSTRUCT CORRELATION FUNCTION
#
def smart_tp_corr(cfg, src_i, src_j, snk_i, snk_j, source_time, step):
    i = src_i
    j = src_j
    k = snk_i
    l = snk_j
    t = source_time
    a = step
    dagger = True
    create = smart_op(cfg, i, j, t, not(dagger))
    destroy = smart_op(cfg, k, l, t + a, dagger)
    corr = create * destroy
    return corr

###############################################################################
#                           CORRELATION FUNCTIONS                             #
###############################################################################
#
# n(sink_site, (source_time + step) mod ntimes) * n(source_site, source_time)
#
def tp_corr(cfg, source_site, sink_site, source_time, step):
    i = source_site
    j = sink_site
    t = source_time
    # how to compute dis?
    # inner_prod = cfg.dot(t, t + step)
    # correct for 0-indexed lattice sites
    corr = (float( ((-1)**(j+1) * cfg.n(j, t + step)) * ((-1)**(i+1) * cfg.n(i, t))))
    return corr
#
# vacuum expectation value
#
def vac_expect(cfg, site, time):
    i = site
    t = time
    # correct for 0-indexed lattice sites
    return float((-1)**(i+1) * cfg.n(i, t))

#
# mesonic operator \chi^\dagger_i 1/2 * (\chi_i+1 + \chi_i-1)
#
def m01(cfg, site, time):
    i = site
    t = time
    return 0.5 * (int(cfg.is_hop(i, i-1, t)) - int(cfg.is_hop(i, i+1, t)))

# mesonic operator \chi^\dagger_i 1/2 * (\chi_i+1 + \chi_i-1)
#
def m01_dagger(cfg, site, time):
    i = site
    t = time
    return 0.5 * (int(cfg.is_hop(i-1, i, t)) - int(cfg.is_hop(i+1, i, t)))

#
# interpolating operator < mo1^\dagger m01 >
#
def interp_m01(cfg, source_site, sink_site, source_time, step):
    i = source_site
    j = sink_site
    t = source_time
    a = step
    create  = m01(cfg, i, t)
    destroy = m01_dagger(cfg, j, t+a)
    corr = create * destroy
    return corr

#
# Given
#   - a configuration,
#   - a two-point correlation function accumulator
#   - a source vev accumulator, and
#   - a sink vev accumulator,
#
# These functions update the accumulators for a corresponding set of
# interpolating operators
#

### CHI_0
def update_chi0(cfg, src_site, src_times, tsteps,
        tp_corr_acc, vev_ini_acc, vev_fin_acc):
    # compute connected two-point correlations
    # two-point correlation functions averaged over
    # all source times, all separations between source and sink,
    # and all sink sites,
    # and the corresponding vacuum expectation
    i = src_site
    create  = np.zeros((len(src_times)))
    destroy = np.zeros((len(src_times)))
    for t in range(len(src_times)):
        # build arrays create and destroy
        create[t] += vac_expect(cfg, i, src_times[t])
        destroy_acc = 0.
        for j in range(i % 2, cfg.nsites, 2):
            destroy_acc += vac_expect(cfg, j, src_times[t])
        destroy[t] = destroy_acc / float(cfg.nsites / 2.)
        vev_acc_ini[t] += create[t]
        vev_acc_fin[t] += destroy[t]
    # once the arrays create and destroy are built, calculate the two-point
    # function
    ntsteps = len(tsteps)
    for a in range(ntsteps):
        tp_corr_inc = create[t] * destroy[(t + a) % ntsteps]
        tp_corr_acc[t][a] += tp_corr_inc
    return

### M_01
def update_m01(cfg, src_site, src_times, tsteps,
        tp_corr_acc, vev_ini_acc, vev_fin_acc):
    # compute connected two-point correlations
    # two-point correlation functions averaged over
    # all source times, all separations between source and sink,
    # and all sink sites,
    # and the corresponding vacuum expectation
    i = src_site
    create  = np.zeros((len(src_times)))
    destroy = np.zeros((len(src_times)))
    for t in range(len(src_times)):
        # build arrays create and destroy
        create[t] += m01(cfg, i, src_times[t])
        destroy_acc = 0.
        for j in range(i % 2, cfg.nsites, 2):
            destroy_acc += m01_dagger(cfg, j, src_times[t])
        destroy[t] = destroy_acc / float(cfg.nsites / 2.)
        vev_acc_ini[t] += create[t]
        vev_acc_fin[t] += destroy[t]
    return
    # once the arrays create and destroy are built, calculate the two-point
    # function
    ntsteps = len(tsteps)
    for a in range(ntsteps):
        tp_corr_inc = create[t] * destroy[(t + a) % ntsteps]
        tp_corr_acc[t][a] += tp_corr_inc
    return


### SMART
def update_smart(cfg, src_site1, src_site2, src_times, tsteps,
        tp_corr_acc, vev_ini_acc, vev_fin_acc):
    # compute connected two-point correlations
    # two-point correlation functions averaged over
    # all source times, all separations between source and sink,
    # and all sink sites,
    # and the corresponding vacuum expectation
    i = src_site1
    j = src_site2
    nsrctimes = len(src_times)
    nops1 = len(weights_a)
    nops2 = len(weights_b)
    # precompute arrays for source and sink at each time
    dagger = True
    create_a  = np.zeros((nsrctimes, nops1))
    create_b  = np.zeros((nsrctimes, nops2))
    destroy_a = np.zeros((nsrctimes, nops1))
    destroy_b = np.zeros((nsrctimes, nops2))
    for t in range(nsrctimes):
        create_a[t] = smart_op_1(cfg, i, src_times[t])
        create_b[t] = smart_op_2(cfg, j, src_times[t])
        destroy_a[t] = smart_op_1(cfg, i, src_times[t], dagger)
        destroy_b[t] = smart_op_2(cfg, j, src_times[t], dagger)
        destroy_a_inc = np.zeros(nops1)
        destroy_b_inc = np.zeros(nops2)
        for k in range(i % 2, cfg.nsites, 2):
            destroy_a_inc += smart_op_1(cfg, k, src_times[t], dagger)
            destroy_b_inc += smart_op_2(cfg, k, src_times[t], dagger)
        destroy_a[t] += destroy_a_inc / float(cfg.nsites / 2.)
        destroy_b[t] += destroy_b_inc / float(cfg.nsites / 2.)
    # now calculate the one-point functions
    create  = np.zeros(nsrctimes)
    destroy = np.zeros(nsrctimes)
    for t in range(nsrctimes):
        if cfg.is_bv(src_times[t]):
            create[t]  = smart_op(create_a[t-1], create_b[t])
            destroy[t] = smart_op(destroy_a[t-1], destroy_b[t])
        else:
            create[t]  = 0.
            destroy[t] = 0.
    # finally, calculate the two-point function
    ntsteps = len(tsteps)
    for a in range(ntsteps):
        tp_corr_inc = create[t] * destroy[(t + a) % ntsteps]
        tp_corr_acc[t][a] += tp_corr_inc
    return

if __name__ == '__main__':
### Subtract vaccuum?
    subtract_vac = True
### Specify model parameters
    nsites = 8
    ntimes = 80
    jw = 1.667
    mw = 0.167
    tw = 0.100
### Number of cfg files
    ncfgs = 1000000
    assert(ncfgs > 1)
    cfg = Cfg(nsites, ntimes)
### Print status every ... bins
    nprint = 1
### Bootstrap and binning
    nbins = 1000
    cfg_per_bin = int(ncfgs / nbins)
### Loop to read cfg files ###
    # choose fixed fermionic site for the source in the two-point corr function
    src_site = 1
    i = src_site
    # due to the checkerboard splitting, it only makes sense to average
    # over an even number of time steps;
    tsteps    = np.array(range(0, ntimes, 2))
    src_times = np.array(range(0, ntimes, 2))
### Compute connected 2-point correlation functions in each bin, write them out
    if not(os.path.isdir(res_bin_dir())):
        os.mkdir(res_bin_dir())
    for new_bin in range(nbins):
        # time averaging done over source-sink separation as well as source time
        tp_corr_acc     = np.zeros((len(src_times), len(tsteps)))
        # subtracting vacuum expectation value at each time separately
        vev_acc_ini     = np.zeros((len(src_times)))
        vev_acc_fin     = np.zeros((len(src_times)))
        # determine the range of configurations for this bin
        first_cfg    = new_bin   * cfg_per_bin
        next_bin_cfg = first_cfg + cfg_per_bin
        for ncfg in range(first_cfg, next_bin_cfg):
            # load a config from file
            fname = cfg_fname(cfg_dir(), nsites, ntimes, jw, mw, tw, ncfg, job_id)
            infile = open(fname, 'rb')
            cfg.load(infile)
            # compute connected two-point correlations
            # two-point correlation functions averaged over
            # all source times, all separations between source and sink,
            # and all sink sites,
            # and the corresponding vacuum expectation
<<<<<<< HEAD
            update_smart(cfg, src_site, src_site, src_times, tsteps,
                tp_corr_acc, vev_acc_ini, vev_acc_fin)
            # close file
            infile.close()
        # calculate the averages per bin
        # two-point correlation functions
        # first, computed separately for each source time
        for t in range(len(src_times)):
            # here, accumulators at each time have one value for each
            # source-sink time separation
            tp_corr_acc[t]     = 1./float(cfg_per_bin) * tp_corr_acc[t]
            vev_acc_ini[t]     = 1./float(cfg_per_bin) * vev_acc_ini[t]
            vev_acc_fin[t]     = 1./float(cfg_per_bin) * vev_acc_fin[t]
        # then, averaged over all source times
        # so that now you have ntimes values, one for each time separation
        # between source and sink
        tp_corr_avg     = np.average(tp_corr_acc, axis=0) # first axis = source time
        vev_acc_ini_avg = np.average(vev_acc_ini, axis=0)
        vev_acc_fin_avg = np.average(vev_acc_fin, axis=0)
### Calculate the (binned) connected two-point correlation functions
### Write out the correlation function into a file
        fname = res_fname_binned((res_bin_dir()),
                nsites, ntimes, jw, mw, tw,
                ncfgs,
                new_bin, nbins,
                op_string,
                job_id)
        results = np.stack((tsteps,
                tp_corr_avg,
                vev_acc_ini_avg * np.ones(len(tsteps)),
                vev_acc_fin_avg * np.ones(len(tsteps))),
                axis = -1)
        if (new_bin % nprint == 0):
            print("Saving file %s..." % fname)
        np.savetxt(fname, results, header = "tsteps, tp_corr_avg, vev_ini_avg, vev_fin_avg")



