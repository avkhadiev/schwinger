#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from schwinger_helper import res_dir, res_fname_bootstrapped, eff_mass_fig_name

if __name__ == '__main__':
### Gap prediction
    same_parity_gap = 1.82902
    odd_parity_gap  = 1.25957
### Specify model parameters
    nsites = 8
    ntimes = 40
    ncf   = 100000
    ncor  = 10
    neql  = 500
    nbins = 100
    nens  = 100
    jw = 1.667
    mw = 0.167
    tw = 0.100
### Interpolating operator
    src_site = 1
### How many points to plot
    nsteps  = int(ntimes/2)
    start   = 1                     # omit first # pts
    finish  = 1                     # omit last # points
    inter = 1                       # plot points with interval
    npoints = int(nsteps) - (start + finish)
    trunc = start + inter * npoints
    assert(trunc <= nsteps)
### Set up the figure
    ax = plt.figure().gca()
    ax.set_title((r'$\Gamma(\tau)$ gets the same-parity gap '), fontsize = 12)
    ttl = ax.title
    ttl.set_position([.5, 1.02])
    ax.set_xlabel(r'$w\Delta\tau$',              fontsize = 14)
    ax.set_ylabel(r'$\Gamma(\tau)/(2 w\Delta\tau)$', fontsize = 14)
    eff_mass_str = '\n'.join((
                  r'$\Gamma(\tau)=$'
                  r'$\log\left(\frac{G(\tau)}{G(\tau+\Delta\tau)}\right)$'
                  r' for $G(\tau)='
                  r'\sum_{t\;\mathrm{even}}\,\langle\,' # sum over even sites
                  r'\left(\sum_{n\;\mathrm{even}}'       # sum over odd  sites
                  r'\chi^\dagger_n(t+\tau)\chi_n(t+\tau)\right)'
                  r'^\dagger'
                  r'\chi^\dagger_{s}(t)\chi_{s}(t)\rangle_\mathrm{conn}$',
                  r'where s = %d' % (src_site + 1, ),   # correct for 0-indexed
                  ))
    textstr = '\n'.join((
        r'$%3d \times %3d$  lat' % (nsites, ntimes),
         r'$\frac{J}{w}=%.3f$' % (jw, ),
        r'$\frac{m}{w}=%.3f$' % (mw, ),
        r'$w\Delta\tau=%.2f$' % (tw, ),
        "\n" # skip extra line for readability
        r'$N_{\mathrm{cor}}=%4d$' % (ncor,),
        r'$N_{\mathrm{eql}}=%4d$' % (neql,),
        r'$N_{\mathrm{cfg}}=%4d$' % (ncf,),
        r'$N_{\mathrm{bin}}=%4d$' % (nbins,),
        r'$N_{\mathrm{ens}}=%4d$' % (nens,)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.0)
    # place text with parameter details in upper right corner
    ax.text(1.04, 0.50, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='center', bbox=props)
    # place text with effective mass formula on the botton
    ax.text(0.50, -0.25, eff_mass_str, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment = 'center', bbox=props)
### Read the data file in
    fname       = res_fname_bootstrapped(res_dir() + "/bootstrapped", nsites, ntimes, jw, mw, tw, ncf, nbins, nens)
    res         = np.transpose(np.loadtxt(fname))
    tsteps      = res[0]
    cn_tp_corr     = res[1]
    cn_tp_corr_err = res[2]
    tsteps      = tsteps[start:trunc:inter]
    cn_tp_corr     = cn_tp_corr[start:trunc:inter]
    cn_tp_corr_err = cn_tp_corr_err[start:trunc:inter]
    # remove spurious values (for large timestep artefacts)
    # print cn_tp_corr
    # cn_tp_corr[cn_tp_corr < 0] = np.nan
    # log[ Gn / Gm ], m = n+1
    eff_mass = np.log(cn_tp_corr[:-1]/cn_tp_corr[1:])
    # Sqrt[ dx^2 / x^2 + dy^2 / y^2 ]
    eff_mass_err = np.sqrt(
                (cn_tp_corr_err[:-1]/cn_tp_corr[:-1])**2
                + (cn_tp_corr_err[1:]/cn_tp_corr[1:])**2)
    # size of the step is tw * 2 = t/a
    eff_mass = eff_mass / (2. * tw)
    eff_mass_err = eff_mass_err / (2. * tw)
    print eff_mass
    print eff_mass_err
### Plot
    # ax.plot(tsteps[:-1], eff_mass, 'o',
    # gaps from exact diagonalization
    ax.plot(tsteps[:-1], same_parity_gap * np.ones(len(tsteps[:-1])),
            color = 'r', linewidth = 2, alpha = 0.5,
            label = r'gap to $\pi=+1$ state, $%.5f$' % (same_parity_gap))
    #ax.plot(tsteps[:-1], odd_parity_gap * np.ones(len(tsteps[:-1])),
    #        color = 'b', linewidth = 2, alpha = 0.5,
    #        label = r'gap to $\pi=-1$ state, $%.5f$' % (odd_parity_gap))
    # MC
    ax.errorbar(tsteps[:-1], eff_mass, yerr = eff_mass_err, fmt='o',
              label = r'MC')
            # label = ((r'$w\Delta\tau=%.3f$' % (tw,))))
            # label = ((r'$\frac{m}{w}=%.3f$' % (mw, ))))
            # label = ((r'$\frac{J}{w}=%.3f$' % (jw, ))))
            # label = ((r'$%3d \times %3d$  lat' % (nsites, ntimes))))
    ax.legend(frameon = False, loc = 'lower left', fontsize = 12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # sets integer ticks
    plt.subplots_adjust(right=0.77, bottom=0.30)
    # plt.tight_layout(pad = 4.5)
    plt.savefig(eff_mass_fig_name(nsites, ntimes, jw, mw, tw), dpi=300)
    plt.show()


