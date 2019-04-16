#!/usr/bin/env python
# -*- coding: utf-8 -*-

output_dir = "/Users/Arthur/Documents/research/mc_sim/"

# Returns name of the folder where config files reside (from current directory)
def cfg_dir():
    return (output_dir + "/" + "cfg")

# Returns name of the folder where data with corr functions resides
# (from current directory)
def res_dir():
    return (output_dir + "/" + "data")

# Returns name of the folder where binned data with corr functions resides
# (from current directory) --- this data can then be bootstrapped
def res_bin_dir():
    bin_dir = res_dir() + "/" + "binned"
    return bin_dir

# Returns name of the folder where observables (tracked during sampling) are
# located
def obs_dir():
    return (output_dir + "/" + "obs")

#   dir = folder name, from current directory
#   namebase = base name in the naming convention
#   nsites = number of fermionic sites
#   ntimes = number of sites along the imaginary time axis
#   jw = dimensionless parameter J/w
#   mw = dimensionless parameter m/w
#   tw = dimensionless parameter Delta_tau * w
#   ncfg = number of the sampled configuration


# Returns the name of the observables file (which tracked during sampling)
def obs_fname(obs_dir, nsites, ntimes, jw, mw, tw, ncorr, job_id):
    fname = ("%s/obs_%03d_%03dx%03d_%.3f_%.3f_%.3f_%07d.npy"
                % (obs_dir,
                    job_id,
                    nsites, ntimes,
                    jw, mw, tw,
                    ncorr)
                # "." separating integer from decimal in model paramaters
                #   is replaced w "p"
            ).replace(".", "p", 3)
    return fname


# gets the file name for a configuration file,
# according to the naming convention
def cfg_fname(cfg_dir, nsites, ntimes, jw, mw, tw, ncfg, job_id):
# dir/cfg_nsitesxntimes_JW_MW_TW_NSWEEP.npy
    fname = ("%s/cfg_%03d_%03dx%03d_%.3f_%.3f_%.3f_%07d.npy"
                % (cfg_dir,
                    job_id,
                    nsites, ntimes,
                    jw, mw, tw,
                    ncfg)
                # "." separating integer from decimal in model paramaters
                #   is replaced w "p"
            ).replace(".", "p", 3)
    return fname

# gets the file name for a binned result (data) file,
def res_fname_binned(res_dir, nsites, ntimes, jw, mw, tw, ncfg, nbin, nbins, op_string, job_id):
    fname = ("%s/data_%03d_%03dx%03d_%.3f_%.3f_%.3f_%07d_%06d-%06d_%s.npy"
                % (res_dir,
                    job_id,
                    nsites, ntimes,
                    jw, mw, tw,
                    ncfg,
                    nbin, nbins,
                    op_string)
                # "." separating integer from decimal in model paramaters
                #   is replaced w "p"
            ).replace(".", "p", 3)
    return fname

# gets the file name for a bootrstrapped result (data) file,
def res_fname_bootstrapped(res_dir, nsites, ntimes, jw, mw, tw,
        ncfg, nbins, nensembles, op_string):
    fname = ("%s/data_%03dx%03d_%.3f_%.3f_%.3f_%07d_%06d_%06d_%s.npy"
                % (res_dir,
                    nsites, ntimes,
                    jw, mw, tw,
                    ncfg,
                    nbins,
                    nensembles, op_string)
                # "." separating integer from decimal in model paramaters
                #   is replaced w "p"
            ).replace(".", "p", 3)
    return fname

# gets the file name for a plot of effective mass
def eff_mass_fig_name(nsites, ntimes, jw, mw, tw):
    fname = ("eff_mass_%03dx%03d_%.3f_%.3f_%.3f.png"
                % (nsites, ntimes,
                    jw, mw, tw)
                # "." separating integer from decimal in model paramaters
                #   is replaced w "p"
            ).replace(".", "p", 3)
    return fname
