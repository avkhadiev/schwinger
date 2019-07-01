#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from schwinger_helper import cfg_dir, res_bin_dir, cfg_fname, res_fname_binned
from schwinger import Cfg
import os

job_id = 1

def print_links(cfg, src_times):
    nsrctimes = len(src_times)
    for t in range(nsrctimes):
        links = cfg.get_links(src_times[t])
        lambdamax = np.max(np.abs(links))
        lambdasq = np.dot(links, links)
        if ((lambdamax > 1) or (lambdasq > 6)):
            print((lambdamax, lambdasq))
            print(src_times[t])
            print(cfg)

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
    # due to the checkerboard splitting, it only makes sense to average
    # over an even number of time steps;
    src_times = np.array(range(0, ntimes, 2))
### Compute connected 2-point correlation functions in each bin, write them out
    if not(os.path.isdir(res_bin_dir())):
        os.mkdir(res_bin_dir())
    for ncfg in range(ncfgs):
        # load a config from file
        fname = cfg_fname(cfg_dir(), nsites, ntimes, jw, mw, tw, ncfg, job_id)
        infile = open(fname, 'rb')
        cfg.load(infile)
        print_links(cfg, src_times)
        # close file
        infile.close()
