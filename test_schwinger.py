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

###############################################################################

if __name__ == '__main__':
    define_model_params()
    print J
    test_local_updates()
    # test_even_odd()

