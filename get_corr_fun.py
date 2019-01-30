#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from schwinger_helper import cfg_dir, res_bin_dir, cfg_fname, res_fname_binned
from schwinger import Cfg
import os

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
    return 0.5 * (cfg.is_hop(i+1, i, t) + cfg.is_hop(i-1, i, t))

# mesonic operator \chi^\dagger_i 1/2 * (\chi_i+1 + \chi_i-1)
#
def m01_dagger(cfg, site, time):
    i = site
    t = time
    return 0.5 * (cfg.is_hop(i, i+1, t) + cfg.is_hop(i, i-1, t))

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

if __name__ == '__main__':
### Subtract vaccuum?
    subtract_vac = True
### Specify model parameters
    nsites = 8
    ntimes = 40
    jw = 1.667
    mw = 0.167
    tw = 0.100
### Number of cfg files
    ncfgs = 10000
    assert(ncfgs > 1)
    cfg = Cfg(nsites, ntimes)
### Print status every ... bins
    nprint = 10
### Bootstrap and binning
    nbins = 100
    cfg_per_bin = ncfgs / nbins
### Loop to read cfg files ###
    # choose fixed fermionic site for the source in the two-point corr function
    src_site = 1
    i = src_site
    # due to the checkerboard splitting, it only makes sense to average
    # over an even number of time steps; consequently,
    # (good) source times are only integer numbers.
    tsteps    = np.array(range(1, ntimes, 2))
    src_times = np.array(range(2, ntimes, 2))
### Compute connected 2-point correlation functions in each bin, write them out
    if not(os.path.isdir(res_bin_dir())):
        os.mkdir(res_bin_dir())
    for new_bin in xrange(80, nbins):
        # time averaging done over source-sink separation as well as source time
        tp_corr_acc     = np.zeros((len(src_times), len(tsteps)))
        # subtracting vacuum expectation value at each time separately
        vev_acc_ini     = np.zeros((len(src_times)))
        vev_acc_fin     = np.zeros((len(src_times)))
        # determine the range of configurations for this bin
        first_cfg    = new_bin   * cfg_per_bin
        next_bin_cfg = first_cfg + cfg_per_bin
        for ncfg in xrange(first_cfg, next_bin_cfg):
            # load a config from file
            fname = cfg_fname(cfg_dir(), nsites, ntimes, jw, mw, tw, ncfg)
            infile = open(fname, 'r')
            cfg.load(infile)
            # compute connected two-point correlations
            # two-point correlation functions averaged over
            # all source times, all separations between source and sink,
            # and all sink sites,
            # and the corresponding vacuum expectation
            for t in xrange(len(src_times)):
                i = src_site
                # calculate vacuum expectation values for source and sink
                # vev_acc_ini[t] += vac_expect(cfg, i, src_times[t])
                vev_fin_inc = 0.
                vev_ini_inc = 0.
                for j in xrange(i % 2, cfg.nsites, 2):
                    vev_ini_inc += m01(cfg, j, src_times[t])
                for j in xrange(1, cfg.nsites, 2):
                    vev_fin_inc += m01_dagger(cfg, j, src_times[t])
                vev_ini_inc = vev_ini_inc / float(cfg.nsites / 2.)
                vev_fin_inc = vev_fin_inc / float(cfg.nsites / 2.)
                vev_acc_ini[t] += vev_ini_inc
                vev_acc_fin[t] += vev_fin_inc
                for a in xrange(len(tsteps)):
                    tp_corr_inc = 0.
                    for j in xrange(1, cfg.nsites, 2):
                        tp_corr_inc += interp_m01(cfg, i, j,
                                        src_times[t], tsteps[a])
                    tp_corr_inc = tp_corr_inc / float(cfg.nsites / 2.)
                    tp_corr_acc[t][a] += tp_corr_inc
            # close file
            infile.close()
        # calculate the averages per bin
        # two-point correlation functions
        # first, computed separately for each source time
        for t in xrange(len(src_times)):
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
                new_bin, nbins)
        results = np.stack((tsteps,
                tp_corr_avg,
                vev_acc_ini_avg * np.ones(len(tsteps)),
                vev_acc_fin_avg * np.ones(len(tsteps))),
                axis = -1)
        if (new_bin % nprint == 0):
            print("Saving file %s..." % fname)
        np.savetxt(fname, results, header = "tsteps, tp_corr_avg, vev_ini_avg, vev_fin_avg")



