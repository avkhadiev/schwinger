# -*- coding: utf-8 -*-

import numpy as np
from schwinger_helper import res_dir, res_bin_dir, res_fname_binned, res_fname_bootstrapped
import os

# type of interpolating operator used (for filenames)
op_string = "m01"

# job ids to combine (must have same parameters)
job_ids = [11]

### fraction of bins from each job --- for collectings statistics while the jobs are still running :p
nbins_per_job_ready = 1000

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
        print ("Current leftmost unread index is %d" % (bin_it))
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
            print ("New leftmost index is %d" % (bin_it))
            # print ("Ensemble %.3d includes the correlation function..." % (ens, ))
            print ("%.3d times" % (factor, ))
    elif (print_updates):
        print ("...already completed the ensemble!")
    return (factor, bin_ind)

################################################################################
#                                   BOOTSTRAP                                  #
################################################################################
# TODO add loop over job ids
def bootstrap(ncfgs_per_job,
        nbins_per_job, nensembles,
        nsites, ntimes, jw, mw, tw, tsteps,
        job_ids):
    print_updates = False
    binfrac = (float(nbins_per_job_ready) / float(nbins_per_job))
    njobs = len(job_ids)
    ncfgs = int(ncfgs_per_job * njobs * binfrac)
    nbins = int(nbins_per_job * njobs * binfrac)
    nprint_bin = nbins / 10000
    nsteps    = len(tsteps)
### Generate random ensembles
    # each row is an ensemble of bin indices, sorted
    ens_tp_corr = np.sort(
                np.random.randint(nbins, size=(nensembles, nbins)),
                axis = 1)
    # vaccuum e-value sampled from the same indices
    ens_vev_ini = ens_tp_corr
    ens_vev_fin = ens_tp_corr
    #ens_vev_ini = np.sort(
    #        np.random.randint(nbins, size=(nensembles, nbins)),
    #        axis = 1)
    #ens_vev_fin = np.sort(
    #        np.random.randint(nbins, size=(nensembles, nbins)),
    #        axis = 1)
### Loop over bins in order
    bin_it_tp_corr  = np.zeros(nensembles)          # to loop over bin indices in each ensemble
    bin_it_vev_ini  = np.zeros(nensembles)
    bin_it_vev_fin  = np.zeros(nensembles)
    tp_corr_acc     = np.zeros((nensembles, nsteps))# averaging for each time step
    vev_ini_acc     = np.zeros((nensembles, nsteps))
    vev_fin_acc     = np.zeros((nensembles, nsteps))
    for ijob in range(njobs):
        job_id = job_ids[ijob]
        print("Working on job %02d..." % (job_id, ))
        for new_bin in range(nbins_per_job_ready):
            # bins are numbered within each job, but are combined for bootstrap
            glob_bin = ijob * nbins_per_job_ready + new_bin
            if (print_updates and (glob_bin % nprint_bin == 0)):
                print ("Including bin %.3d..." % (glob_bin, ))
            fname = res_fname_binned(res_bin_dir(),
                                        nsites, ntimes,
                                        jw, mw, tw,
                                        # here the the number of configs is within job_id
                                        ncfgs_per_job,
                                        # here the the bin is numbered within job_id
                                        new_bin, nbins_per_job,
                                        op_string,
                                        job_id)
            # print ("Opening file %s" % (fname, ))
            res = np.transpose(np.loadtxt(fname))
            tp_corr = res[1]                     # array with nsteps entries
            vev_ini = res[2]
            vev_fin = res[3]
            # loop over ensembles
            for ens in range(nensembles):
                ####################################################################
                #                       TWO-POINT CORRELATION                      #
                ####################################################################
                # here the bin number needs to refer to that of combined jobs
                factor, bin_ind = get_increment_factor(glob_bin,
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
                factor, bin_ind = get_increment_factor(glob_bin,
                                                        bin_it_vev_ini[ens],
                                                        ens_vev_ini[ens],
                                                        nbins, print_updates)
                # save new leftmost unused bin index for this ensemble
                bin_it_vev_ini[ens] = bin_ind
                vev_ini_acc[ens] += float(factor) * vev_ini
                # FINAL
                factor, bin_ind = get_increment_factor(glob_bin,
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
    #print(np.average(tp_corr_ens_avg, axis = 0))
    #print(np.average(vev_ini_ens_avg * vev_fin_ens_avg, axis = 0))
    # find connected correlations
    cn_tp_corr_ens_avg = tp_corr_ens_avg - vev_ini_ens_avg * vev_fin_ens_avg
    # cn_tp_corr_ens_avg[cn_tp_corr_ens_avg < 0] = np.nan
    #print(np.average(cn_tp_corr_ens_avg, axis=0))
    # calculate effective mass
    ratio = np.zeros(int(ntimes/2) - eff_mass_step)
    for pt in range(len(cn_tp_corr_ens_avg) - eff_mass_step):
        ratio[pt] = cn_tp_corr_ens_avg[pt]/cn_tp_corr_ens_avg[pt+eff_mass_step]
    ratio[ratio < 0] = np.nan
    eff_mass_ens_avg = np.log(ratio)
    # find the average of ensemble averages and their standard deviation
    eff_mass_avg = np.nanmean(eff_mass_ens_avg, axis=0)
    eff_mass_std = np.nanstd(eff_mass_ens_avg, axis=0, ddof=1)
    # divide out by the lattice spacing (time step)
    eff_mass_avg = eff_mass_avg / (eff_mass_step * 2. * tw)
    eff_mass_std = eff_mass_std / (eff_mass_step * 2. * tw)
    print(eff_mass_avg)
    print(eff_mass_std)
    return eff_mass_avg, eff_mass_std

if __name__ == '__main__':
### job ids
    njobs = len(job_ids)
### Number of bins => size of ensembles
    ncfgs_per_job   = 1000000
    nbins_per_job   = 1000
    ncfgs           = int(ncfgs_per_job * njobs)
    nbins           = int(nbins_per_job * njobs)
    nensembles = int(nbins_per_job * njobs)
    # nprint_bin = nbins / 10
### Specify model parameters
    nsites = 8
    ntimes = 80
    jw = 1.667
    mw = 0.167
    tw = 0.100
    eff_mass_step = 1
    # due to the checkerboard splitting, it only makes sense to average
    # over an even number of time steps; consequently,
    # (good) source times are only integer numbers.
    tsteps    = np.array(range(0, ntimes, 2))
    # nsteps    = len(tsteps)
    eff_mass_avg, eff_mass_std = bootstrap(ncfgs_per_job,
            nbins_per_job, nensembles,
            nsites, ntimes, jw, mw, tw, tsteps,
            job_ids)
    # for effective mass M(tau), skip tau = 0
    tsteps    = tsteps[1:]
    print(tsteps)
    # print out data file
    # FIXME
    binfrac = (float(nbins_per_job_ready) / float(nbins_per_job))
    ncfgs = int(ncfgs_per_job * njobs * binfrac)
    nbins = int(nbins_per_job * njobs * binfrac)
    results = np.stack((tsteps,
            eff_mass_avg, eff_mass_std),
            axis = -1)
    writeout_dir = res_dir() + "/bootstrapped"
    fname = res_fname_bootstrapped(writeout_dir,
            nsites, ntimes, jw, mw, tw,
            ncfgs, nbins, nensembles,
            op_string)
    if not(os.path.isdir(writeout_dir)):
        os.mkdir(writeout_dir)
    print("writing binned data to file %s" % fname)
    np.savetxt(fname, results, header = "tsteps, cn_corr_avg, cn_corr_std")
