#!/usr/bin/env python
# -*- coding: utf-8 -*-

from schwinger import Cfg, LocalUpdates

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
    print "\n"
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
    print "\n"
def test_even_odd():
    n_upd = 1
    t_upd = 2
    cfg = Cfg(4, 4)
    local = LocalUpdates(m, J, w, Delta_tau)
    # update for given config
    print "Current config\n"
    print(cfg)
    print_info(cfg, local, n_upd, t_upd)
    local.update(cfg, n_upd, t_upd)
    print_info(cfg, local, n_upd, t_upd)
    local.update(cfg, n_upd, t_upd)     # change back
    print "Change site parity\n"
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
def test_is_hop():
    print("testing is_hop()...")
    n_upd_1 = 0
    t_upd_1 = 3
    n_upd_2 = 2
    t_upd_2 = 3
    cfg = Cfg(4, 4)
    local = LocalUpdates(m, J, w, Delta_tau)
    # hops from
    local.update(cfg, n_upd_1, t_upd_1)
    local.update(cfg, n_upd_2, t_upd_2)
    print(cfg)
    print("Hopping from 1 to 0 at t=0 (1): %s (%s)"  % (cfg.is_hop(1, 0, 0), cfg.is_hop(1, 0, 1), ))
    print("Hopping from 0 to 1 at t=0 (1): %s (%s)"  % (cfg.is_hop(0, 1, 0), cfg.is_hop(0, 1, 1), ))
    print("Hopping from 0 to 1 at t=2 (3): %s (%s)"  % (cfg.is_hop(0, 1, 2), cfg.is_hop(0, 1, 3), ))
    print("Hopping from 3 to 2 at t=2 (3): %s (%s)"  % (cfg.is_hop(3, 2, 2), cfg.is_hop(3, 2, 3), ))
    print("Hopping from 2 to 3 at t=2 (3): %s (%s)"  % (cfg.is_hop(2, 3, 2), cfg.is_hop(2, 3, 3), ))
    print("Hopping from 2 to 3 at t=0 (1): %s (%s)"  % (cfg.is_hop(2, 3, 0), cfg.is_hop(2, 3, 1), ))
###############################################################################

if __name__ == '__main__':
    define_model_params()
    # print J
    # test_local_updates()
    # test_even_odd()
    test_is_hop()

