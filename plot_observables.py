#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from schwinger_helper import obs_dir, obs_fname
import numpy as np

# computes a simple moving average
def sma(arr, window):
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

# computes a cumulative moving average
def cma(arr):
    cumsum = np.cumsum(arr)
    return cumsum / np.arange(1.0, float(len(arr)) + 1., 1.0)

if __name__ == '__main__':
### Specify MC settings
    ncor = 10            # how often to save configuration
    neql = 50    * ncor  # length of equilibration
    ncf  = 100   * ncor  # length of sampling
### Expected condensate density in ground state from exact diagonalization
    expected_condensate = -0.324714
### Specify model parameters
    nsites = 8
    ntimes = 20
    jw = 1.667
    mw = 0.167
    tw = 0.100
### Binning data to calculate average
    nbins = 100
    binsize = int(ncf / 200)
### How much to plot
    start = 0
    trunc = ncf
    inter = 1
    sma_window = binsize                         # moving average window
### Read file
    obsfname  = obs_fname(obs_dir(), nsites, ntimes, jw, mw, tw, ncor)
    data_file = open(obsfname, 'r')
    data      = np.loadtxt(data_file)
    sweeps    = data[:, 0]
    vev       = data[:, 1]
    acc_rate  = abs(data[:, 2])
### Calculate binned values
#    vev_cut = vev[neql:]
#    vev_binned = [vev[binsize * k:binsize * (k+1)].mean()
#                    for k in range(0, len(vev_cut)/binsize)]
#    vev_binned_avg = np.average(vev_binned)
#    print vev_binned_avg
### Calculate averages and errors
    vev_avg = np.average(vev[neql:])
    vev_std = np.std(vev[neql:], ddof=1)
    acc_rate_avg = np.average(acc_rate[neql:])
    acc_rate_std = np.std(acc_rate[neql:], ddof=1)
### Truncate
    sweeps  = sweeps[start:trunc:inter]
    vev     = vev[start:trunc:inter]
    acc_rate= acc_rate[start:trunc:inter]
    textstr = '\n'.join((
        r'$%3d \times %3d$  lat' % (nsites, ntimes),
         r'$\frac{J}{w}=%.3f$' % (jw, ),
        r'$\frac{m}{w}=%.3f$' % (mw, ),
        r'$w\Delta\tau=%.2f$' % (tw, ),
        "\n" # skip extra line for readability
        r'$N_{\mathrm{cor}}=%4d$' % (ncor,),
        r'$N_{\mathrm{eql}}=%4d$' % (neql / ncor,),
        r'$N_{\mathrm{cfg}}=%4d$' % (ncf / ncor,)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.0)

###############################################################################
#                               CONDENSATE                                    #
###############################################################################
    ax = plt.figure().gca()
    ax.set_title((r'chiral condensate MC sampling'), fontsize = 12)
    ttl = ax.title
    ttl.set_position([.5, 1.02])
    ax.set_xlabel(r'sweep', fontsize=12)
    ax.set_ylabel(r'$\sum_i (-1)^{i} \chi^\dagger_i \chi_i$', fontsize=12)
    ax.plot(sweeps, vev,
            color = 'gray', linewidth = 0.5,
            label = r'instantaneous',
            alpha = 0.5)
    ax.plot(sweeps[sma_window-1:], sma(vev, sma_window),
            color = 'b', linewidth = 1,
            label = (r'simple moving average, $w=%d$' % (sma_window,)),
            alpha = 0.5)
    # cumulative sum only after equilibration
    ax.plot(sweeps[neql:], cma(vev[neql:]),
            color = 'purple', linewidth = 1,
            label = r'cumulative moving average')
    # average + uncertainty
    ax.plot(sweeps, vev_avg * np.ones(len(sweeps)),
            color = 'r', linewidth = 2,
            label = r'total average',
            alpha = 0.3)
    ax.fill_between(sweeps,
            vev_avg - vev_std * np.ones(len(sweeps)),
            vev_avg + vev_std * np.ones(len(sweeps)),
            color = 'r', alpha = 0.3)
    # expected from exact diagonalization in the GS
    ax.plot(sweeps, expected_condensate * np.ones(len(sweeps)),
            color = 'k', linewidth = 2, linestyle = '--',
            label = r'$\langle \bar{\psi}\psi \rangle$ in ground state')
    # insert information on the average
    vev_avg_str = ( r'$\langle\sum_i(-1)^i \chi^\dagger_i \chi_i \rangle'
                    r'= %.5f \pm %.5f$' % (vev_avg, vev_std,))
    ax.text(0.50, -0.25, vev_avg_str, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment = 'center', bbox=props)
    # ax.set_ylim(None, 1.0)
    # add nequil marker
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    plt.axvline(x=neql, linewidth=3, color = 'g', alpha = 0.5)
    plt.text(neql + ncor, ymin + 0.95 * y_range,
            "equilibration",
            rotation=90, verticalalignment='top')
    # place text with parameter details in upper right corner
    ax.text(1.04, 0.50, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='center', bbox=props)
    plt.legend(frameon = False, loc='upper right', fontsize = 10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # sets integer ticks
    plt.subplots_adjust(right=0.8, left=0.15, bottom=0.25)
    plt.savefig("mc_run_condensate.png", dpi=300)

###############################################################################
#                              ACCEPTANCE RATE                                #
###############################################################################
    ax = plt.figure().gca()
    ax.set_title((r'acceptance rate MC sampling'), fontsize = 12)
    ttl = ax.title
    ttl.set_position([.5, 1.02])
    ax.set_xlabel(r'sweep', fontsize=12)
    ax.set_ylabel(r'acceptance rate', fontsize=12)
    plt.plot(sweeps, acc_rate,
            color = 'gray', linewidth = 0.5,
            label = r'instantaneous',
            alpha = 0.5)
    plt.plot(sweeps[sma_window-1:], sma(acc_rate, sma_window),
            color = 'b', linewidth = 1,
            label = (r'simple moving average, $w=%d$' % (sma_window,)),
            alpha = 0.5)
    # cumulative sum only after equilibration
    plt.plot(sweeps[neql:], cma(acc_rate[neql:]),
            color = 'purple', linewidth = 1,
            label = r'cumulative moving average')
    # average + uncertainty
    ax.plot(sweeps, acc_rate_avg * np.ones(len(sweeps)),
            color = 'r', linewidth = 2,
            label = r'total average',
            alpha = 0.3)
    ax.fill_between(sweeps,
            acc_rate_avg - acc_rate_std * np.ones(len(sweeps)),
            acc_rate_avg + acc_rate_std * np.ones(len(sweeps)),
            color = 'r', alpha = 0.3)
    # insert information on the average
    acc_rate_avg_str = (r'avg acc rate'
                        r'$= %.5f \pm %.5f$' % (acc_rate_avg, acc_rate_std, ))
    ax.text(0.50, -0.25, acc_rate_avg_str, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment = 'center', bbox=props)
    # add nequil marker
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    plt.axvline(x=neql, linewidth=3, color = 'g', alpha = 0.5)
    plt.text(neql + ncor, ymin + 0.95 * y_range,
            "equilibration",
            rotation=90, verticalalignment='top')
    # place text with parameter details in upper right corner
    ax.text(1.04, 0.50, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='center', bbox=props)
    plt.legend(frameon = False, loc='upper right', fontsize = 10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # sets integer ticks
    plt.subplots_adjust(right=0.8, left=0.15, bottom=0.25)
    plt.savefig("mc_run_accrate.png", dpi=300)
