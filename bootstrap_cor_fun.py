#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from schwinger_helper import res_dir, res_bin_dir, res_fname_binned, res_fname_bootstrapped
import os

# Given:
#   - new_bin, the current bin number (datafile) to be added to ensembles
#   - bin_it, smallest unread bin index for a current ensemble
#   - ensemble, an array of bin numbers in a given ensemble
#   - nbins, a total number of bins
#   - print_updates, whether or not to print updates (for debugging, mostly)
# returns (factor, bin_ind), where:
#   - factor is how many times the current bin number is used in the ensemble
#   - bin_ind is the new smallest unread bin index in a given ensemble
def get_increment_factor(new_bin, bin_it, ensemble, nbins, print_updates = False):
    factor  = 0
    bin_ind = int(bin_it)
    if (print_updates):
        print ("Current leftmost unread index is %d" % (bin_it[ens]))
    # figure out how many time current bin needs to be counted
    # in this ensemble
    # if haven't finished calculating for this ensemble yet...
    if (int(bin_it) < nbins):
        # iterate from last unused bin index
        for bin_ind in range(int(bin_it), nbins):
            current_ind = ensemble[bin_ind]
            if (current_ind == new_bin):
                # count how many times this bin is included
                # in this ensemble
                factor += 1
            # stop if current bin index gives a bin larger than new_bin
            elif (current_ind > new_bin):
                break
            # if last bin index was used, the ensemble is finished
            if (bin_ind == nbins - 1):
                bin_ind += 1
            else:
                assert(current_ind >= new_bin)
        if (print_updates):
            print ("New leftmost index is %d" % (bin_it[ens]))
            print ("Ensemble %.3d includes the correlation function..." % (ens, ))
            print ("%.3d times" % (factor, ))
    elif (print_updates):
        print ("...already completed the ensemble!")
    return (factor, bin_ind)


if __name__ == '__main__':
### Number of bins => size of ensembles
    ncfgs = 10000
    nbins = 100
    nensembles = 100
    nprint_bin = nbins / 10
### Specify model parameters
    nsites = 8
    ntimes = 80
    jw = 1.667
    mw = 0.167
    tw = 0.100
    # due to the checkerboard splitting, it only makes sense to average
    # over an even number of time steps; consequently,
    # (good) source times are only integer numbers.
    tsteps    = np.array(range(0, ntimes, 2))
    nsteps    = len(tsteps)
### Generate random ensembles
    # each row is an ensemble of bin indices, sorted
    ens_tp_corr = np.sort(
                np.random.randint(nbins, size=(nensembles, nbins)),
                axis = 1)
    ens_vev_ini = np.sort(
            np.random.randint(nbins, size=(nensembles, nbins)),
            axis = 1)
    ens_vev_fin = np.sort(
            np.random.randint(nbins, size=(nensembles, nbins)),
            axis = 1)
### Loop over bins in order
    bin_it_tp_corr  = np.zeros(nensembles)          # to loop over bin indices in each ensemble
    bin_it_vev_ini  = np.zeros(nensembles)
    bin_it_vev_fin  = np.zeros(nensembles)
    tp_corr_acc     = np.zeros((nensembles, nsteps))# averaging for each time step
    vev_ini_acc     = np.zeros((nensembles, nsteps))
    vev_fin_acc     = np.zeros((nensembles, nsteps))
    for new_bin in xrange(nbins):
        if (new_bin % nprint_bin == 0):
            print ("Including bin %.3d..." % (new_bin, ))
        fname = res_fname_binned(res_bin_dir(), nsites, ntimes,
                                    jw, mw, tw,
                                    ncfgs, new_bin, nbins)
        # print ("Opening file %s" % (fname, ))
        res = np.transpose(np.loadtxt(fname))
        tp_corr = res[1]                     # array with nsteps entries
        vev_ini = res[2]
        vev_fin = res[3]
        # loop over ensembles
        print_updates = False
        for ens in xrange(nensembles):
            ####################################################################
            #                       TWO-POINT CORRELATION                      #
            ####################################################################
            factor, bin_ind = get_increment_factor(new_bin,
                                                    bin_it_tp_corr[ens],
                                                    ens_tp_corr[ens],
                                                    nbins, print_updates)
            # save new leftmost unused bin index for this ensemble
            bin_it_tp_corr[ens] = bin_ind
            tp_corr_acc[ens]    += float(factor) * tp_corr
            ####################################################################
            #                            VACUUM                                #
            ####################################################################
            # INITIAL
            factor, bin_ind = get_increment_factor(new_bin,
                                                    bin_it_vev_ini[ens],
                                                    ens_vev_ini[ens],
                                                    nbins, print_updates)
            # save new leftmost unused bin index for this ensemble
            bin_it_vev_ini[ens] = bin_ind
            vev_ini_acc[ens] += float(factor) * vev_ini
            # FINAL
            factor, bin_ind = get_increment_factor(new_bin,
                                                    bin_it_vev_fin[ens],
                                                    ens_vev_fin[ens],
                                                    nbins, print_updates)
            # save new leftmost unused bin index for this ensemble
            bin_it_vev_fin[ens] = bin_ind
            vev_fin_acc[ens] += float(factor) * vev_fin
    # average within each ensemble
    tp_corr_ens_avg = 1./float(nbins) * tp_corr_acc
    vev_ini_ens_avg = 1./float(nbins) * vev_ini_acc
    vev_fin_ens_avg = 1./float(nbins) * vev_fin_acc
    # finf connected correlations
    # TODO check with background subtraction removed
    cn_tp_corr_ens_avg = tp_corr_ens_avg - vev_ini_ens_avg * vev_fin_ens_avg
    print vev_ini_ens_avg[0]
    print vev_fin_ens_avg[0]
    print cn_tp_corr_ens_avg[0]
    # find the average of ensemble averages and their standard deviation
    cn_tp_corr_avg = np.average(cn_tp_corr_ens_avg, axis = 0)
    cn_tp_corr_std = np.std(cn_tp_corr_ens_avg, axis=0, ddof=1)
    print cn_tp_corr_avg
    # print out data file
    results = np.stack((tsteps,
            cn_tp_corr_avg, cn_tp_corr_std),
            axis = -1)
    writeout_dir = res_dir() + "/bootstrapped"
    fname = res_fname_bootstrapped(writeout_dir,
            nsites, ntimes, jw, mw, tw,
            ncfgs, nbins, nensembles)
    if not(os.path.isdir(writeout_dir)):
        os.mkdir(writeout_dir)
    print("writing binned data to file %s" % fname)
    np.savetxt(fname, results, header = "tsteps, cn_corr_avg, cn_corr_std")
