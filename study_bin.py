#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from schwinger_helper import res_bin_dir, res_fname_binned

if __name__ == '__main__':
###### INPUT ######
    bins = [1000, 500, 100]
    nensembles = 100
    tau0 = 6
###################
    ncfgs = 10000
### Specify model parameters
    nsites = 8
    ntimes = 80
    jw = 1.667
    mw = 0.167
    tw = 0.100
    for nbins in bins:
        tp_corr = np.zeros(nbins)
        vev_ini = np.zeros(nbins)
        vev_fin = np.zeros(nbins)
        for new_bin in range(nbins):
            fname = res_fname_binned(res_bin_dir(), nsites, ntimes,
                                        jw, mw, tw,
                                        ncfgs, new_bin, nbins)
            res = np.transpose(np.loadtxt(fname))
            tp_corr[new_bin] = res[1][tau0]      # array with nsteps entries
            vev_ini[new_bin] = res[2][tau0]
            vev_fin[new_bin] = res[3][tau0]
        # find averages and standard deviations
        tp_corr_bin_avg = np.average(tp_corr)
        vev_ini_bin_avg = np.average(vev_ini)
        vev_fin_bin_avg = np.average(vev_fin)
        tp_corr_bin_std = np.std(tp_corr, ddof=1)
        vev_ini_bin_std = np.std(vev_ini, ddof=1)
        vev_fin_bin_std = np.std(vev_fin, ddof=1)
        # print
        print("points per bin: %d" % (int(ncfgs/nbins),))
        print("CORRF: %f, %f (%f)" % (tp_corr_bin_avg, tp_corr_bin_std, tp_corr_bin_std/tp_corr_bin_avg,))
        print("VEV 1: %f, %f (%f)" % (vev_ini_bin_avg, vev_ini_bin_std, vev_ini_bin_std/vev_ini_bin_avg,))
        print("VEV 2: %f, %f (%f)" % (vev_fin_bin_avg, vev_fin_bin_std, vev_fin_bin_std/vev_fin_bin_avg,))
