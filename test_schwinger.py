#!/usr/bin/env python
# -*- coding: utf-8 -*-

from schwinger import Cfg, LocalUpdates, can_hop, hop, are_equal, alpha
import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

###############################################################################
#                             MODEL PARAMETERS                                #
###############################################################################
def define_model_params():
    global m
    global J
    global w
    global Delta_tau
    a = 1.
    g = 1.
    m = 0.5
    w = 1./(2. * a)
    J = (a * g ** 2)/2.
    Delta_tau = 0.1 * a
def print_info(cfg, local, n_upd, t_upd):
    sw = "%35s"              # string width
    fp = "%20.8f"            # float precision
    print("\n")
    print((sw + " (%d, %d)")
        % ("Updating the 3x3 patch centered at",
            n_upd, t_upd))
#    print((sw + fp)
#        % ("s = ",
#            local.patch_s(cfg, n_upd, t_upd)))
    print((sw + fp)
        % ("Transition probability is p = ",
            local.p_acc(cfg, n_upd, t_upd)))
#    print((sw + fp)
#        % ("Ratio of configuration weights R = ",
#            local.patch_R(cfg, n_upd, t_upd)))
    print("\n")
def test_even_odd():
    n_upd = 1
    t_upd = 2
    cfg = Cfg(4, 4)
    local = LocalUpdates(m, J, w, Delta_tau)
    # update for given config
    print("Current config\n")
    print(cfg)
    print_info(cfg, local, n_upd, t_upd)
    local.update(cfg, n_upd, t_upd)
    print_info(cfg, local, n_upd, t_upd)
    local.update(cfg, n_upd, t_upd)     # change back
    print("Change site parity\n")
    n_upd += 1
    t_upd -= 1
    print_info(cfg, local, n_upd, t_upd)
    local.update(cfg, n_upd, t_upd)
    print_info(cfg, local, n_upd, t_upd)
    local.update(cfg, n_upd, t_upd)     # change back
def test_local_updates():
    n_upd = 3
    t_upd = 2
    cfg = Cfg(4, 4)
    local = LocalUpdates(m, J, w, Delta_tau)
    print(cfg)
    print_info(cfg, local, n_upd, t_upd)
    local.update(cfg, n_upd, t_upd)
    print(cfg)
    print_info(cfg, local, n_upd, t_upd)
    local.update(cfg, n_upd, t_upd)
    print(cfg)
def test_choose_patch():
    nsites = 8
    ntimes = 16
    nsweeps = 500 * 10
    cfg = Cfg(nsites, ntimes)
    local = LocalUpdates(m, J, w, Delta_tau)
    white_squares = local.white_squares(nsites, ntimes)
    print(cfg)
    gen = []
    for i in range(int(nsweeps * (nsites * ntimes / 2))):
        (site, time) = (local.choose_patch(cfg))
        gen.append((site, time))
    counter = collections.Counter(gen)
    occurences = [counter[pair] for pair in white_squares]
    #### make a bar chart ####
    coord_labels = ["(%s, %s)" % coord_pair for coord_pair in white_squares]
    indices = range(len(coord_labels))
    ax = plt.figure().gca()
    ax.bar(indices, occurences, align='center', alpha=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(indices, coord_labels)
    plt.xticks(rotation=90)
    ax.xaxis.set_tick_params(labelsize=3)
    ax.set_title("Testing RNG for sweeping")
    ax.set_xlabel('white square coordinate')
    ax.set_ylabel('frequency')
    nsweeps_str = r'$N_{\mathrm{swps}}=%4d$' % (nsweeps, )
    props = dict(boxstyle='round', facecolor='white', alpha=0.0)
    # place text with parameter details on the bottom
    ax.text(0.50, -0.25, nsweeps_str, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment = 'center', bbox=props)
    plt.subplots_adjust(bottom=0.23)
    plt.show()
# tests hopping methods (the ones outside the cfg class)
def test_hopping():
    print("testing hopping...")
    n_upd_1 = 0
    t_upd_1 = 1
    cfg = Cfg(8, 8)
    local = LocalUpdates(m, J, w, Delta_tau)
    local.update(cfg, n_upd_1, t_upd_1)
    sites0 = cfg.get_sites(0)
    sites1 = cfg.get_sites(2)
    links0 = cfg.get_links(0)
    links1 = cfg.get_links(2)
    print(cfg)
    print("T=0")
    print(sites0)
    print(links0)
    print("T=1")
    print(sites1)
    print(links1)
    print("jumping from 0 to 1 at 1...")
    #print("Can hop from 1 to 0: %d" % (int(can_hop(sites0, 0, 1))))
    #sites0, links0 =  hop(sites0, links0, 0, 1)
    #print("after hopping:")
    #print(sites0)
    #print(links0)
    #print(are_equal(sites0, links0, sites1, links1))
    print(alpha(cfg, 1, 0, 0))
    print(cfg)
#
def test_is_bv():
    print("testing is_bv()...")
    n_upd_1 = 0
    t_upd_1 = 1
    cfg = Cfg(8, 8)
    local = LocalUpdates(m, J, w, Delta_tau)
    print(cfg)
    print("Testing bare vacuum at t=%d: %d" % (t_upd_1, cfg.is_bv(t_upd_1), ))
    local.update(cfg, n_upd_1, t_upd_1)
    print(cfg)
    print("Testing bare vacuum at t=%d: %d" %(t_upd_1, cfg.is_bv(t_upd_1), ))
    print("Testing bare vacuum at t=%d: %d" %(t_upd_1 + 1, cfg.is_bv(t_upd_1 + 1), ))
    print("Testing bare vacuum at t=%d: %d" %(t_upd_1 + 2, cfg.is_bv(t_upd_1 + 2), ))
def test_is_hop():
# mesonic operator \chi^\dagger_i 1/2 * (\chi_i+1 + \chi_i-1)
    def m01(cfg, site, time):
        i = site
        t = time
        # return 0.5 * (cfg.is_hop(i, i-1, t) + cfg.is_hop(i, i+1, t))
        return 0.5 * (alpha(cfg, i, i-1, t) - alpha(cfg, i, i+1, t))
# mesonic operator \chi^\dagger_i 1/2 * (\chi_i+1 + \chi_i-1)
    def m01_dagger(cfg, site, time):
        i = site
        t = time
        # return 0.5 * (cfg.is_hop(i-1, i, t) + cfg.is_hop(i+1, i, t))
        return 0.5 * (alpha(cfg, i-1, i, t) - alpha(cfg, i+1, i, t))
# interpolating operator < mo1^\dagger m01 >
    def interp_m01(cfg, source_site, sink_site, source_time, step):
        i = source_site
        j = sink_site
        t = source_time
        a = step
        create  = m01(cfg, i, t)
        destroy = m01_dagger(cfg, j, t+a)
        corr = create * destroy
        return corr
    print("testing is_hop()...")
    n_upd_1 = 0
    t_upd_1 = 1
    n_upd_2 = 1
    t_upd_2 = 2
    cfg = Cfg(8, 8)
    local = LocalUpdates(m, J, w, Delta_tau)
    # hops from
    local.update(cfg, n_upd_1, t_upd_1)
    print(cfg)
    print("Hopping from 1 to 0 at t=0 (1): %s (%s)"  % (cfg.is_hop(1, 0, 0), cfg.is_hop(1, 0, 1), ))
    print("Hopping from 1 to 0 at t=2:     %s"       % (cfg.is_hop(1, 0, 2), ))
    print("hopping from 0 to 1 at t=2 (3): %s (%s)"  % (cfg.is_hop(0, 1, 2), cfg.is_hop(0, 1, 3), ))
    print("M01(1, 0)    = %.3f, M01(1, 1)    = %.3f" % (m01(cfg, 1, 0), m01(cfg, 1, 1), ))
    print("M01^dn(1, 2) = %.3f, M01^dn(1, 3) = %.3f" % (m01_dagger(cfg, 1, 2), m01_dagger(cfg, 1, 3), ))
    print("G(1, 1, 0, 2) = %.3f, G(1, 1, 1, 2) = %.3f"      % (interp_m01(cfg, 1, 1, 0, 2), interp_m01(cfg, 1, 1, 0, 2), ))
    print("G(1, 1, 0, 3) = %.3f, G(1, 1, 1, 4) = %.3f"      % (interp_m01(cfg, 1, 1, 0, 4), interp_m01(cfg, 1, 1, 0, 4), ))
    # hops back
    local.update(cfg, n_upd_1, t_upd_1)
    local.update(cfg, n_upd_2, t_upd_2)
    print(cfg)
    print("Hopping from 1 to 2 at t=0 (1): %s (%s)"  % (cfg.is_hop(1, 2, 0), cfg.is_hop(1, 2, 1), ))
    print("Hopping from 1 to 0 at t=2:     %s"       % (cfg.is_hop(1, 2, 2), ))
    print("Hopping from 2 to 1 at t=2 (3): %s (%s)"  % (cfg.is_hop(2, 1, 2), cfg.is_hop(2, 1, 3), ))
    print("M01(1, 0)    = %.3f, M01(1, 1)    = %.3f" % (m01(cfg, 1, 0), m01(cfg, 1, 1), ))
    print("M01^dn(1, 2) = %.3f, M01^dn(1, 3) = %.3f" % (m01_dagger(cfg, 1, 2), m01_dagger(cfg, 1, 3), ))
    print("G(1, 1, 0, 2) = %.3f, G(1, 1, 1, 2) = %.3f"      % (interp_m01(cfg, 1, 1, 0, 2), interp_m01(cfg, 1, 1, 0, 2), ))
    print("G(1, 1, 0, 4) = %.3f, G(1, 1, 1, 4) = %.3f"      % (interp_m01(cfg, 1, 1, 0, 4), interp_m01(cfg, 1, 1, 0, 4), ))
###############################################################################

if __name__ == '__main__':
    define_model_params()
    test_hopping()


