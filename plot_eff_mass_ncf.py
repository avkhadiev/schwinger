#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from schwinger_helper import res_dir, res_fname_bootstrapped

if __name__ == '__main__':
### Specify model parameters
    nsites = 20
    ntimes = 40
    ncfs   = [1000, 10000]
    ncor  = 10
    neql  = 100
    nbins = 100
    nens  = 100
    jw = 1.00
    mw = 1.00
    tw = 0.10
### Interpolating operator
    src_site = 1
### How many points to plot
    nsteps  = int(ntimes/2)
    start   = 0                 # omit first # pts
    #start  += 1                # always omit the dt = 0 point
    finish  = int(2 * nsteps/3) # omit last # points
    inter = 1                   # plot points with interval
    npoints = int(nsteps) - (start + finish)
    trunc = start + inter * npoints
    assert(trunc <= nsteps)
### Set up the figure
    ax = plt.figure().gca()
    ax.set_title((r'$\sigma\left(\Gamma(\tau)\right)$'
                  r' decreases with $'
                  r'N_{\mathrm{cf}}$'), fontsize = 12)
    ttl = ax.title
    ttl.set_position([.5, 1.02])
    ax.set_xlabel(r'$w\Delta\tau$',                 fontsize = 14)
    ax.set_ylabel(r'$\Gamma(\tau)/(2 w\Delta\tau)$', fontsize = 14)
    eff_mass_str = (r'$\Gamma(\tau)=$'
                  r'$\log\left(\frac{G(\tau)}{G(\tau+\Delta\tau)}\right)$'
                  r' for $G(\tau)='
                  r'\sum_{t\;\mathrm{even}}\,\langle\,'
                  r'\left(\sum_{n}\chi^\dagger_n(t+\tau)\chi_n(t+\tau)\right)'
                  r'^\dagger'
                  r'\chi^\dagger_{s}(t)\chi_{s}(t)\rangle_\mathrm{conn}$')
    textstr = '\n'.join((
        r'$%3d \times %3d$  lat' % (nsites, ntimes),
        r'$\frac{J}{w}=%.3f$' % (jw, ),
        r'$\frac{m}{w}=%.3f$' % (mw, ),
        r'$w\Delta\tau=%.2f$' % (tw, ),
        "\n" # skip extra line for readability
        r'$N_{\mathrm{cor}}=%4d$' % (ncor,),
        r'$N_{\mathrm{eql}}=%4d$' % (neql,),
        r'$N_{\mathrm{bin}}=%4d$' % (nbins,),
        r'$N_{\mathrm{ens}}=%4d$' % (nens,)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.0)
    # place text with parameter details in upper right corner
    ax.text(1.04, 0.50, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='center', bbox=props)
    # place text with effective mass formula on the botton
    ax.text(0.50, -0.25, eff_mass_str, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment = 'center', bbox=props)
    for ncf in ncfs:
### Read the data file in
        fname          = res_fname_bootstrapped(res_dir() + "/bootstrapped", nsites, ntimes, jw, mw, tw, ncf, nbins, nens)
        res            = np.transpose(np.loadtxt(fname))
        tsteps         = res[0]
        tp_cn_corr     = res[1]
        tp_cn_corr_err = res[2]
        # truncate
        tsteps         = tsteps[start:trunc:inter]
        tp_cn_corr     = tp_cn_corr[start:trunc:inter]
        tp_cn_corr_err = tp_cn_corr_err[start:trunc:inter]
        # log[ Gn / Gm ], m = n+1
        eff_mass = np.log(tp_cn_corr[:-1]/tp_cn_corr[1:])
        # Sqrt[ dx^2 / x^2 + dy^2 / y^2 ]
        eff_mass_err = np.sqrt(
                    (tp_cn_corr_err[:-1]/tp_cn_corr[:-1])**2
                    + (tp_cn_corr_err[1:]/tp_cn_corr[1:])**2)
        # size of the step is tw * 2 = t/a
        eff_mass = eff_mass / (2. * tw)
        eff_mass_err = eff_mass_err / (2. * tw)
### Plot
        # ax.plot(tsteps[:-1], eff_mass, 'o',
        ax.errorbar(tsteps[:-1], eff_mass, yerr = eff_mass_err, fmt='o',
                label = ((r'$N_{\mathrm{cf}}=%4d$' % (ncf,))))
    ax.legend(frameon = False, loc = 'upper left', fontsize = 12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # sets integer ticks
    plt.subplots_adjust(right=0.8, bottom=0.25)
    # plt.tight_layout(pad = 4.5)
    plt.savefig("eff_mass_ncf.png", dpi=300)
    plt.show()


