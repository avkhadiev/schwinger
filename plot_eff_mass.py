#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
from schwinger_helper import res_dir, res_fname_bootstrapped, eff_mass_fig_name

op_string = 's1'

# for nice scientific notation
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
fmt = mticker.FuncFormatter(g)

if __name__ == '__main__':
### Gap prediction
    same_parity_gap = 1.84074
    odd_parity_gap  = 1.26463
    Lambda          = 4
###
    exact_E1  = "mathematica/E1_gap_exact_naive.txt"
    exact_E2  = "mathematica/E2_gap_exact_naive.txt"
    is_exact = False
### Specify model parameters
    nsites = 8
    ntimes = 80
    ncf  = 10000000
    ncor  = 10
    neql  = 500
    nbins = 10000
    nens  = 10000
    jw = 1.667
    mw = 0.167
    tw = 0.100
### Interpolating operator
    src_site = 1
### How many points to plot
    nsteps  = int(ntimes/2)
    start   = 0                     # omit first # pts
    finish  = int(0.66 * nsteps)    # omit last # points
    inter = 1                       # plot points with interval
    npoints = int(nsteps) - (start + finish)
    trunc = start + inter * npoints
    assert(trunc <= nsteps)
### Set up the figure
    ax = plt.figure().gca()
    if (op_string == 'chi0'):
        op = '\chi_{01}'
    elif (op_string == 'm01'):
        op = 'M_{01}'
    elif (op_string == 'smart'):
        op = '\hat{O}_{\mathrm{S}}'
    ax.set_title("$\Gamma(\\tau)$ with $%s$ as interpolating operator" % (op, ), fontsize = 12)
    ttl = ax.title
    ttl.set_position([.5, 1.02])
    ax.set_xlabel(r'$w\Delta\tau$',                  fontsize = 14)
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
        ''.join((r'$N_{\mathrm{eql}}=$', "{}".format(fmt(neql  )))),
        ''.join((r'$N_{\mathrm{cfg}}=$', "{}".format(fmt(ncf )))),
        ''.join((r'$N_{\mathrm{bin}}=$', "{}".format(fmt(nbins )))),
        ''.join((r'$N_{\mathrm{ens}}=$', "{}".format(fmt(nens  ))))))
    props = dict(boxstyle='round', facecolor='white', alpha=0.0)
    # place text with parameter details in upper right corner
    ax.text(1.04, 0.50, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='center', bbox=props)
    # place text with effective mass formula on the botton
    # ax.text(0.50, -0.25, eff_mass_str, transform=ax.transAxes, fontsize=10,
    #     verticalalignment='top', horizontalalignment = 'center', bbox=props)
### Read the data file in
    fname       = res_fname_bootstrapped(res_dir() + "/bootstrapped", nsites, ntimes, jw, mw, tw, ncf, nbins, nens, op_string)
    res         = np.transpose(np.loadtxt(fname))
    tsteps         = res[0]
    eff_mass     = res[1]
    eff_mass_err = res[2]
    tsteps       = tsteps[start:trunc:inter]
    eff_mass     = eff_mass[start:trunc:inter]
    eff_mass_err = eff_mass_err[start:trunc:inter]
    # remove spurious values (for large timestep artefacts)
    # cn_tp_corr[cn_tp_corr < 0] = np.nan
    # cn_tp_corr_err[cn_tp_corr < 0] = np.nan
    # log[ Gn / Gm ], m = n+1
    # tsteps          = tsteps[1::]
    # eff_mass        = np.log(cn_tp_corr[:-1]/cn_tp_corr[1:])
    # Sqrt[ dx^2 / x^2 + dy^2 / y^2 ]
    # eff_mass_err = np.sqrt(
    #             (cn_tp_corr_err[:-1]/cn_tp_corr[:-1])**2
    #             + (cn_tp_corr_err[1:]/cn_tp_corr[1:])**2)
    # # size of the step is tw * 2 = t/a
    # eff_mass = eff_mass / (2. * tw)
    # eff_mass_err = eff_mass_err / (2. * tw)
    # download exact data
    if (op_string == 'chi0'):
        exact = np.transpose(np.loadtxt(exact_E2))
        is_exact = True
    elif (op_string == 'm01'):
        exact = np.transpose(np.loadtxt(exact_E1))
        is_exact = True
    else:
        is_exact = False
    if(is_exact):
        exact_eff_mass  = exact[0]
        exact_tsteps    = exact[1]
        exact_tsteps = exact_tsteps[:len(eff_mass)]
        exact_eff_mass = exact_eff_mass[:len(eff_mass)]
### Plot
    # Exact
        ax.plot(exact_tsteps, exact_eff_mass,
                    color = 'orange', linewidth = 2, alpha = 0.5,
                    label = r'exact')
    # MC
    ax.errorbar(tsteps, eff_mass, yerr = eff_mass_err, fmt='o',
            label = r'MC')
            # label = ((r'$N_{\mathrm{cf}}=%4d$' % (ncf,))))
            # label = ((r'$w\Delta\tau=%.3f$' % (tw,))))
            # label = ((r'$\frac{m}{w}=%.3f$' % (mw, ))))
            # label = ((r'$\frac{J}{w}=%.3f$' % (jw, ))))
            # label = ((r'$%3d \times %3d$  lat' % (nsites, ntimes))))
    # gaps from exact diagonalization
    if (op_string == 'chi0'):
        ax.plot(tsteps, same_parity_gap * np.ones(len(tsteps)),
                color = 'r', linewidth = 2, alpha = 0.5,
                label = r'$m_{\mathrm{S}^{+}}=%.3f, \tilde{\Lambda}=%d$' % (same_parity_gap, Lambda, ))
    elif ((op_string == 'm01') or (op_string == 'smart')):
        ax.plot(tsteps, odd_parity_gap * np.ones(len(tsteps)),
                color = 'b', linewidth = 2, alpha = 0.5,
                label = r'$m_{\mathrm{V}^{-}}=%.3f, \tilde{\Lambda}=%d$'  % (odd_parity_gap, Lambda, ))
    ax.legend(frameon = False, loc = 'lower left', fontsize = 12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # sets integer ticks
    # plt.ylim((-1.0, 2.1))
    plt.subplots_adjust(right=0.77, bottom=0.18)
    #plt.tight_layout(pad = 4.5)
    plt.savefig(eff_mass_fig_name(nsites, ntimes, jw, mw, tw), dpi=300)
    plt.show()


