#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numba import jit
from schwinger_helper import cfg_dir, res_bin_dir, cfg_fname, res_fname_binned
from schwinger import Cfg
from schwinger import alpha, can_hop
import os

# type of interpolating operator used (for filenames)
op_string = "m01"
# job settings
job_id = 20

@jit(nopython=True)
def bincount(cfglinks, link_upd):
    linkcounts = np.bincount( np.sum(np.abs(cfglinks), axis=1 ))
    link_upd[:len(linkcounts)] += linkcounts
    return link_upd

def update_flux_hist(cfg, flux_data):
    cfglinks = np.array(cfg.links, dtype=np.int64)
    maxflux = cfg.nsites
    link_upd = np.zeros(1+maxflux, dtype = np.int64)
    link_upd = bincount(cfglinks, link_upd)
    flux_data += link_upd
    return

###############################################################################
#                             SMART OPERATOR                                  #
###############################################################################

#
#   |IN EVEN PARITY SECTOR
#
def overcnt(nsites, separation):
    L = nsites / 2
    factor = 1
    if (L % separation == 0):
        factor = 2
    return 1./float(np.sqrt(factor))


###############################################################################
#                           CORRELATION FUNCTIONS                             #
###############################################################################
#
# vacuum expectation value
#
def vac_expect(cfg, site, time):
    i = site
    j = int(2*time)
    # correct for 0-indexed lattice sites
    a = cfg.n(i, j)
    b = cfg.n(i, j+1)
    c = cfg.n(i, j-1)
    occup = ((a and b) or (a and c))
    return float((-1)**(i+1) * occup)

#
# mesonic operator \chi^\dagger_i 1/2 * (\chi_i+1 + \chi_i-1)
#
def m01(cfg, site, time):
    i = site
    j = int(2*time)
    if (i % 2 == 1):            # if jumping from odd site
        m01 = float(0.5 * (int(cfg.is_hop(i, i+1, j-1)) - int(cfg.is_hop(i, i+1, j+1)))
                - int(cfg.is_hop(i, i-1, j)))
    elif (i % 2 == 0):          # if jumping from even site
        m01 = float(int(cfg.is_hop(i, i+1, j))
                - 0.5 * (int(cfg.is_hop(i, i-1, j-1)) - int(cfg.is_hop(i, i-1, j+1))))
    else:
        print("ARGHHHHHHHHHH")
        m01 = 0.
    return m01

# mesonic operator \chi^\dagger_i 1/2 * (\chi_i+1 + \chi_i-1)
#
def m01_dagger(cfg, site, time):
    i = site
    j = int(2*time)
    if (i % 2 == 1):            # if jumping to odd site
        m01d = float(0.5 * (int(cfg.is_hop(i+1, i, j-1)) - int(cfg.is_hop(i+1, i, j+1)))
                - int(cfg.is_hop(i-1, i, j)))
    elif (i % 2 == 0):          # if jumping to even site
        m01d = float(int(cfg.is_hop(i+1, i, j))
                - 0.5 * (int(cfg.is_hop(i-1, i, j-1)) - int(cfg.is_hop(i-1, i, j+1))))
    else:
        print("ARGHHHHHHHHHH")
        m01d = 0.
    return m01d

# link operator
#
def l01(cfg, site, time):
    i = site
    j = int(2*time)
    return float(cfg.l(i-1, j) + cfg.l(i, j))

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
        tp_corr_acc, vev_acc_ini, vev_acc_fin):
    # compute connected two-point correlations
    # two-point correlation functions averaged over
    # all source times, all separations between source and sink,
    # and all sink sites,
    # and the corresponding vacuum expectation
    i = src_site
    nsrctimes = len(src_times)
    create  = np.zeros(nsrctimes)
    destroy = np.zeros(nsrctimes)
    for t in range(nsrctimes):
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
    for t in range(nsrctimes):
        for a in range(ntsteps):
            tp_corr_inc = create[t] * destroy[(t + a) % nsrctimes]
            tp_corr_acc[t][a] += tp_corr_inc
    return

### M_01
def update_m01(cfg, src_site, src_times, tsteps,
        tp_corr_acc, vev_acc_ini, vev_acc_fin):
    # compute connected two-point correlations
    # two-point correlation functions averaged over
    # all source times, all separations between source and sink,
    # and all sink sites,
    # and the corresponding vacuum expectation
    i = src_site
    nsrctimes = len(src_times)
    create  = np.zeros(nsrctimes)
    destroy = np.zeros(nsrctimes)
    for t in range(nsrctimes):
        # build arrays create and destroy
        create[t] += m01(cfg, i, src_times[t])
        destroy_acc = 0.
        for j in range(i % 2, cfg.nsites, 2):
            destroy_acc += m01_dagger(cfg, j, src_times[t])
        destroy[t] = destroy_acc
        vev_acc_ini[t] += create[t]
        vev_acc_fin[t] += destroy[t]
    # once the arrays create and destroy are built, calculate the two-point
    # function
    ntsteps = len(tsteps)
    for t in range(nsrctimes):
        for a in range(ntsteps):
            tp_corr_inc = create[t] * destroy[(t + a) % nsrctimes]
            tp_corr_acc[t][a] += tp_corr_inc
    return


### L_01
def update_l01(cfg, src_site, src_times, tsteps,
        tp_corr_acc, vev_acc_ini, vev_acc_fin):
    # compute connected two-point correlations
    # two-point correlation functions averaged over
    # all source times, all separations between source and sink,
    # and all sink sites,
    # and the corresponding vacuum expectation
    i = src_site
    nsrctimes = len(src_times)
    create  = np.zeros(nsrctimes)
    destroy = np.zeros(nsrctimes)
    for t in range(nsrctimes):
        # build arrays create and destroy
        create[t] += l01(cfg, i, src_times[t])
        destroy_acc = 0.
        for j in range(i % 2, cfg.nsites, 2):
            destroy_acc += l01(cfg, j, src_times[t])
        destroy[t] = destroy_acc
        vev_acc_ini[t] += create[t]
        vev_acc_fin[t] += destroy[t]
    # once the arrays create and destroy are built, calculate the two-point
    # function
    ntsteps = len(tsteps)
    for t in range(nsrctimes):
        for a in range(ntsteps):
            tp_corr_inc = create[t] * destroy[(t + a) % nsrctimes]
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
    tsteps    = np.array(range(0, int(ntimes/2), 1))
    src_times = np.array(range(0, int(ntimes/2), 1))
    # for making a histrogram of total flux
    # flux_data = np.zeros(nsites + 1, dtype=np.int64)
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
            #update_chi0(cfg, src_site, src_times, tsteps,
            #   tp_corr_acc, vev_acc_ini, vev_acc_fin)
            update_m01(cfg, src_site, src_times, tsteps,
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
            #print("%d/%d" % (new_bin, nbins))
            #print(flux_data)
        np.savetxt(fname, results, header = "tsteps, tp_corr_avg, vev_ini_avg, vev_fin_avg")
    #np.savetxt("flux_data.txt", flux_data)



