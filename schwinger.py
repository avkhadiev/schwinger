 #!/usr/bin/env python
 # -*- coding: utf-8 -*-

import numpy as np
# import numba
# from numba import jitclass
from schwinger_helper import cfg_dir, cfg_fname, obs_dir, obs_fname
from random import randrange
import os

###############################################################################
#                                CONFIGURATION                                #
###############################################################################
# cfg_spec = [
#         ('nsites', int64),
#         ('ntimes', int64),
#         ('lmax', int64),
#         ('beta', float64),
#         ('size', )
#         ('sites')
#         ('links')
#     ]
class Cfg():
    """Configuration of the 1+1D lattice"""
    ndim = 2
    def __init__(self, nsites, ntimes, lmax = 1):
        self.nsites = abs(int(round(nsites)))
        self.ntimes = abs(int(round(ntimes)))
        self.lmax = abs(int(round(lmax)))
        assert(self.nsites >= 2)                     # using staggered fermions
        assert(self.nsites % 2 == 0)
        assert(self.ntimes > 0)
        self.beta = (self.ntimes - 1) / 2.      # due to checkerboard splitting
        self.size = (self.ntimes, self.nsites)
        self.sites = np.array([[(site % 2) for site in range(nsites)]
                        for time in range(ntimes)],
                     dtype = np.int)
        self.links = np.zeros(self.size, dtype = np.int)
    def __str__(self):
        s = ''
        # variables to make sure the string is nicely aligned
        col_zfill = 3   # number of zeros in numbering columns
        row_zfill = 3   # number of zeros in numbering rows
        lspaces = 2     # number of spaces between row number and the rest
        link_spaces = 2 # number of spaces between a site and a link number
        n_paddings  = 1 # number of rows to pad between time steps
        # given number of sites per row,
        # returns string to pad rows to improve readability
        row_padding = (lspaces + row_zfill) * ' '
        for site in range(self.nsites):
            row_padding += '|' + 3 * link_spaces * ' '
        # 0th column printed on either end to show periodicity
        row_padding += '|'
        col_numbering = (row_zfill + lspaces - 1) * ' '
        for site in range(self.nsites):
            col_numbering += str(site).zfill(col_zfill) + 2 * link_spaces * ' '
        # 0th column printed on either end to show periodicity
        col_numbering += str(0).zfill(col_zfill)
        # build the lattice string from the top row down;
        # include 0th row at the top
        for step in reversed(range(self.ntimes + 1)):
            s += str(step % self.ntimes).zfill(row_zfill) + lspaces * ' '
            for site in range(self.nsites):
                s += '{0:d}'.format(self.n(site, step))
                s += link_spaces * ' '
                s += '{0:+d}'.format(self.l(site, step))
                s += link_spaces * ' '
            # include 0th column on either end to show periodicity
            s += str(self.n(0, step))
            if (step != 0):
                s += '\n' + n_paddings * (row_padding + '\n')
            else:
                s += '\n' + col_numbering
        return s
    # returns link between site and site + 1 at time time
    def l(self, site, time):
        return self.links[time % self.ntimes][site % self.nsites]
    # returns the occupation number at site site at time step
    def n(self, site, time):
        return self.sites[time % self.ntimes][site % self.nsites]
    # calculates a dot product between chain at time t1 and t2
    #   = 1 if all occupation numbers and links coincide
    #   = 0 otherwise
    def dot(self, t1, t2):
        s1 = self.sites[t1 % self.ntimes]
        s2 = self.sites[t2 % self.ntimes]
        l1 = self.links[t1 % self.ntimes]
        l2 = self.links[t2 % self.ntimes]
        s_eq = int(np.array_equal(s1, s2))
        l_eq = int(np.array_equal(l1, l2))
        inner_prod = s_eq * l_eq
        return inner_prod

    # takes from_site, i, to_site, j and at_time, t.
    # returns:
    #   1, if there was a hopping from site i to site j at time t
    #   0, if there wasn't hopping.
    #
    # asserts that j = (i \pm 1) mod nsites
    #
    # !!! assumes the square whose bottom left corner is at (0, 0) is black !!!
    #
    # if i (j) == 0, let i (j) = nsites
    #
    # Due to checkerboard splitting, black squares where hopping can happen
    # are at even times for some sites and odd times for others =>
    # shifting time may be required:
    def is_hop(self, from_site, to_site, at_time):
        i = from_site
        j = to_site
        t = at_time
        site_dist = (abs((i % self.nsites) - (j % self.nsites)))
        assert((site_dist == 1) or (site_dist == self.nsites - 1))
        #
        # due to checkerboard splitting, shift time if required
        #
        # first determine if specified coordinates give hopping on a shaded
        # square or a white square
        #
        # to check if a square is white, first
        # choose bottom-left corner of the square
        if (((i % self.nsites == 0)) or ((j % self.nsites) == 0)):
            max_index = max(i % self.nsites, j % self.nsites)
            if(max_index == 1):
                bleft = 0
            elif(max_index == (self.nsites - 1)):
                bleft = self.nsites-1
            else:
                print("checking if square is white in is_hop(): unexpected case")
        else:
            bleft = min(i, j)
        # assuming the square with bleft 0, 0 is black,
        # a square is white if bleft + t = odd number
        is_white = ((bleft + t) % 2 == 1)
        if (is_white):
            # if the square is white and the time coordinate is odd,
            # shift the time back
            if (t % 2 == 1):
                t = t - 1
            # if the square is white and the time coordinate is even,
            # shift the time forward
            elif (t % 2 == 0):
                t = t + 1
        # to have a hop from i to j at time t, must have
        # n(i, t) == n(j, t+1) == 1 and n(j, t) == n(i, t+1) == 0
        occpd = (self.n(i ,t) == self.n(j, t+1) == 1)
        empty = (self.n(j, t) == self.n(i, t+1) == 0)
        is_hop = occpd & empty
        return is_hop

    # fermion hops from (site, time) to (site + 1, time);
    # connecting link value decreases by 1
    # assumes (site, time) is occupied and (site + 1, time) is unoccupied!
    def hop_forward(self, site, time):
        assert(self.n(site,     time) == 1)
        assert(self.n(site + 1, time) == 0)
        self.sites[time % self.ntimes][ site      % self.nsites] =  0
        self.sites[time % self.ntimes][(site + 1) % self.nsites] =  1
        self.links[time % self.ntimes][ site      % self.nsites] -= 1
    # fermion hops from (site, time) to (site - 1, time);
    # connecting link value increases by 1
    # assumes (site, time) is occupied and (site + 1, time) is unoccupied!
    def hop_back(self, site, time):
        assert(self.n(site,     time) == 1)
        assert(self.n(site - 1, time) == 0)
        self.sites[time % self.ntimes][ site      % self.nsites] =  0
        self.sites[time % self.ntimes][(site - 1) % self.nsites] =  1
        self.links[time % self.ntimes][(site - 1) % self.nsites] += 1
    # saves a numpy array of [sites, links] in outfile
    # TODO much more efficient way to store
    def save(self, outfile):
        np.save(outfile, np.array([self.sites, self.links]))
    # loads a numpy array of [sites, links] from infile
    # updates the instance to reflect the file contents
    def load(self, infile):
        loaded_arr = np.load(infile)
        self.sites = loaded_arr[0]
        self.links = loaded_arr[1]
        assert(self.sites.shape == self.links.shape)
        nsites, ntimes = self.sites.shape
        assert(self.nsites >= 2)                     # using staggered fermions
        assert(self.nsites % 2 == 0)
        assert(self.ntimes > 0)
        self.size = (self.ntimes, self.nsites)
        self.beta = (self.ntimes - 1) / 2.      # due to checkerboard splitting

###############################################################################
#                               LOCAL UPDATES                                 #
###############################################################################

class LocalUpdates():
    """Local updates of 1+1D lattice"""
    def __init__(self, m, J, w, Delta_tau):
        self.m = m
        self.J = J
        self.w = abs(w)
        self.Delta_tau = abs(Delta_tau)
    def params(self):
        return (self.m, self.J, self.w, self.Delta_tau)
    def __str__(self):
        'm = %3.2f\nJ= %3.2f\nw=%3.2f,Delta_tau=%3.2f' % self.params()
# returns a random (site, time) of a white square,
# a center of a 3x3 patch for the local update.
    def choose_patch(self, cfg):
        # first, generate random time;
        # due to bc, t >= 1 and t <= ntimes - 1 (0-indexed)
        # if the bottom left square is black,
        # the sum of coordinates of a bottom left corner
        # for any white square is an odd number
        # otherwise it is an even number
        if (self.is_square_white(1, 0)):
            min_coord_sum = 1
        else:
            min_coord_sum = 0
        patch_time = np.random.randint(0, cfg.ntimes)
        #   patch_time              coord_sum                   patch_site
        #   even                    even                        even
        #   even                    odd                         odd
        #   odd                     even                        odd
        #   odd                     odd                         even
        patch_time_even = (patch_time    % 2 == 0)
        coord_sum_even  = (min_coord_sum % 2 == 0)
        patch_site_even = (patch_time_even == coord_sum_even)
        if (patch_site_even):
            min_patch_site = 0
        else:
            min_patch_site = 1
        patch_site = randrange(min_patch_site, cfg.nsites, 2)
        return (patch_site, patch_time)

# given config and (site, time) for a *white* square, returns:
#
#   - True,  if a local update can be performed; or
#   - False, if a local update cannot be performed.
#
# Local updates are done on a 3x3 patch centered at the given white square.
    def can_update(self, cfg, site, time):
        return (abs(self.patch_s(cfg, site, time)) == 2)

# given config and (site, time) for a *white* square,
# returns the acceptance probability for the transition
# from the current configuration to
# the configuration after the local update
# on the 3x3 patch centered at the given white square
    def p_acc(self, cfg, site, time):
        assert(self.is_square_white(site, time))
        R = self.patch_R(cfg, site, time)
        p = R / (1 + R)
        return p

    def update(self, cfg, site, time):
        s = self.patch_s(cfg, site, time)
        # straight line is on the left side of the white square =>
        # hopping will decrease link value
        if (s == 2):
            cfg.hop_forward(site, time)
            cfg.hop_forward(site, time + 1)
        # straight line is on the right side of the white square =>
        # hopping will increase link value
        elif (s == -2):
            cfg.hop_back(site + 1, time)
            cfg.hop_back(site + 1, time + 1)
        # else update is not allowed


# given the (site, time) coordinates of a square on the checkerboard,
# asserts whether they correspond to the bottom left corner of a white square
# assumes the square whose bottom left corner is at (0, 0) is black
    def is_square_white(self, site, time):
        return ((site + time) % 2 == 1)

# generates a list of all white squares as coordinates of their
# lower-left corner --- (site, time).
# required for updates
    def white_squares(self, nsites, ntimes):
        # if the bottom left square is black,
        # the sum of coordinates of a bottom left corner
        # for any white square is an odd number
        # otherwise it is an even number
        if (self.is_square_white(1, 0)):
            min_coord_sum = 1
        else:
            min_coord_sum = 0
        white_squares = [(site, time)
            for site in range(nsites)
            for time in range(ntimes)
            if ((site + time) % 2 == min_coord_sum)]
        return white_squares

# given config and (site, time) for a *white* square,
# returns the ratio R = W_new / W_old,
# where W_new is the W_old is the weight of the current configuration, and
# W_new is the weight of the configuration after the local update on the
# 3x3 patch centered at the given white square
    def patch_R(self, cfg, site, time):
        assert(self.is_square_white(site, time))
        kronecker_delta = int(self.can_update(cfg, site, time))
        m, J, w, Delta_tau = self.params()
        k = cfg.l(site, time - 1)
        s = self.patch_s(cfg, site, time)
        t = self.patch_t(cfg, site, time)
        u = self.patch_u(cfg, site, time)
        v = self.patch_v(cfg, site, time)
        R = ( kronecker_delta
                * np.tanh(w * Delta_tau) ** (2 * u)
                * np.cosh(w * Delta_tau) ** (s * v)
                * np.exp(-J * Delta_tau * (t - s * k))
                * np.exp( m * Delta_tau * s * ((-1) ** (site))) )
        return R

# given config and (site, time) for a *white* square,
# uses occupation numbers on the corners of the square to
# compute and return a number indicating whether a local update
# can be performed on the patch centerd at the white square:
#
#   straight line on left side of the square  -> 2
#   straight line on right side of the square -> -2
#   any other configuration -> impossible to perform a local update
#
# is also be used to determine the sign of the
# mass-proportional part of the exponent.
    def patch_s(self, cfg, site, time):
        assert(self.is_square_white(site, time))
        s = (cfg.n(site, time) + cfg.n(site, time + 1)
                - cfg.n(site + 1, time + 1) - cfg.n(site + 1, time))
        return s

# given config and (site, time) for a *white* square
# quantifies the number of hops below the central white square on the patch
#
#   no hop below ->  1
#   1 hop below  -> -1;
#
# helps identify the sign of the J-proportional part of the exponent.
    def patch_t(self, cfg, site, time):
        assert(self.is_square_white(site, time))
        t = 1 - 2 * abs(cfg.n(site + 1, time) - cfg.n(site + 1, time - 1))
        return t

# given config and (site, time) for a *white* square
# quantifies the number of hops a world line does on
# the patch centered at the square:
#
#   0 hops ->  1
#   1 hop  ->  0
#   2 hops -> -1;
#
# is used to determine the power of hyperbolic tangent
    def patch_u(self, cfg, site, time):
        assert(self.is_square_white(site, time))
        u = (1  - abs(cfg.n(site + 1, time)     - cfg.n(site + 1, time - 1))
                - abs(cfg.n(site + 1, time + 2) - cfg.n(site + 1, time + 1)))
        return u

# given config and (site, time) for a *white* square
# computes the difference of occupation numbers at site -1 and site + 2
# quantifies the number of hops a world line does on
# the patch centered at the square:
#
#   Both sites empty or both full ->  0
#   Left side full, right, empty  ->  1
#   Left side empty, right, full  -> -1;
#
# together withi patch_s, determines the power of hyperbolic cosine
    def patch_v(self, cfg, site, time):
        assert(self.is_square_white(site, time))
        v = cfg.n(site - 1, time) - cfg.n(site + 2, time)
        return v

# given the configuration, performs a sweep through the lattice
# selects a 3x3 patch of the lattice uniformly at random N times, where
# N is the number of legal (overlapping) 3x3 patches
# if the patch can be updated, updates it
# with a probability given by cfg.p_acc(...)
    def sweep(self, cfg):
        naccepts = 0
        npatches = int(cfg.nsites * cfg.ntimes / 2)
        for patch in range(npatches):
            p_site, p_time = self.choose_patch(cfg)
            # if update on given patch is allowed,
            # update with probability p_acc
            if (self.can_update(cfg, p_site, p_time)
                    and (np.random.random() < self.p_acc(cfg, p_site, p_time))):
                self.update(cfg, p_site, p_time)
                naccepts += 1
        return (naccepts, npatches)

###############################################################################
#                       LOCAL HAMILTONIAN MONTE-CARLO                         #
###############################################################################

class LocalMC():
    """A quantum Monte-Carlo simulation"""
    def __init__(self, cfg, upd_machine, measur_obs = False, cfg_dir = "cfg", obs_dir = "obs"):
        self.cfg = cfg
        self.upd = upd_machine
        self.measure_obs = measure_obs
        self.cfg_dir = cfg_dir
        self.obs_dir = obs_dir
        m, J, w, deltaTau = self.upd.params()
        self.obsfname = obs_fname(obs_dir, nsites, ntimes,
                    float(J/w), float(m/w), float(deltaTau * w),
                    n_corr)
        if not(os.path.isdir(cfg_dir)):
            os.mkdir(cfg_dir)
        if (self.measure_obs):
            if not(os.path.isdir(obs_dir)):
                os.mkdir(obs_dir)
    def __str__(self):
        m, J, w, Delta_tau = self.upd.params()
        s = ("QMC of %03dx%03d lattice with params (J/w = %.2f, m/w = %.2f, w * Delta_tau = %.2f)."
                    % (cfg.nsites, cfg.ntimes, J/w, m/w, w * Delta_tau))
        return s
# given:
#
#   current_sweep --- current sweep in the evolve loop
#   n_sweeps --- number of sweeps to perform
#   n_corr ---  practically, how often to save a configuration if sampling
#   n_print_sweeps --- how often to print sampling status
#
# performs n_sweeps * n_corr sweeps, updating cfg via upd_machine
# saves sampled configrations if sample = False
#
# returns (current_sweep, avg_rate), where avg_rate is the average acceptance
# ration
    def do_sweeps(self, current_sweep, n_sweeps, n_corr, n_print_sweeps, obsfile):
        rate_acc = 0.
        # get dimless model parameters for naming output files
        m, J, w, Delta_tau = self.upd.params()
        # in n_print == 0
        if n_print_sweeps == 0:
            n_print_sweeps = n_sweeps + 1
        # if n_corr == 0, do not perform sampling
        if n_corr == 0:
            n_corr = n_sweeps + 1
        for a_sweep in range(n_sweeps):
            current_sweep += 1
            naccepts, ntrials = self.upd.sweep(self.cfg)
            rate = (float(naccepts) / float(ntrials))
            rate_acc += rate
            if (self.measure_obs):
                vev = measure_vev(cfg)
                obsfile.write('%d \t %.8f \t %.8f \n' % (current_sweep, vev, rate))
            # save configuration to a binary file if sampling is required
            if ((a_sweep + 1) % n_corr == 0):
                fname = cfg_fname(self.cfg_dir,
                                 self.cfg.nsites, self.cfg.ntimes,
                                 float(J/w), float(m/w), float(Delta_tau * w),
                                 int(a_sweep/n_corr))
                outfile = open(fname, 'wb')         # will write bytes through numpy
                # print("saving %s..." % fname)
                self.cfg.save(outfile)
                outfile.close()
            if ((a_sweep) % n_print_sweeps == 0):
                print ("acceptance (%4d / %4d) * 100 prcnt = %4.2f prcnt"
                       % (naccepts, ntrials, rate * 100.))
                print ("%5d out of %5d sweeps completed (%.2f prcnt)" % (a_sweep, n_sweeps, float(a_sweep)/float(n_sweeps) * 100, ))
        avg_rate = rate_acc / n_sweeps
        # returns average acceptance ratio
        return (current_sweep, avg_rate)

    def evolve(self, n_corr, n_equil, n_sample, n_print):
        #if (self.measure_obs):
        obsfile = open(self.obsfname, 'w')      # will write strings, not bytes
        sweep = 0
### Equilibrate the lattice #
        print("Equilibrating the lattice")
        should_print = False
        sweep, avg_rate = sim.do_sweeps(sweep,
                            n_equil_sweeps,
                            int(should_print),
                            n_print_sweeps,
                            obsfile)
        print("Average acceptance rate is %.3f prcnt" % (avg_rate * 100.))
### Sample configurations   #
        print("Sampling configurations")
        sweep, avg_rate = sim.do_sweeps(sweep,
                                    n_sampl_sweeps,
                                    n_corr,
                                    n_print_sweeps,
                                    obsfile)
        print("Average acceptance rate is %.3f prcnt" % (avg_rate * 100.))
        if (self.measure_obs):
            obsfile.close()

###############################################################################
#                               MEASUREMENTS                                  #
###############################################################################
def measure_tp_corr(cfg, src_site, tstep):
    i = src_site
    a = tstep
    src_times = np.array(range(0,   cfg.ntimes, 1))
    snk_sites = np.array(range(0,   cfg.nsites, 1))
    tp_corr = 0.0
    # compute two-point correlation functions averaged over
    # all source times and all sink sites
    for t in src_times:
        tp_corr_inc = 0.0
        for j in snk_sites:
            corr = (float(((-1)**(j) * cfg.n(j, t + a))
                        * ((-1)**(i) * cfg.n(i, t))))
            tp_corr_inc += corr
        tp_corr_inc = tp_corr_inc / float(len(snk_sites))
        tp_corr += tp_corr_inc
    tp_corr = tp_corr / len(src_times)
    return tp_corr

def measure_vev(cfg):
    times = np.array(range(0,   cfg.ntimes, 1))
    sites = np.array(range(0,   cfg.nsites, 1))
    vev = 0.0
    # compute two-point correlation functions averaged over
    # all source times and all sink sites
    for t in times:
        vev_inc = 0.0
        for j in sites:
            vev_inc += float((-1)**(j) * cfg.n(j, t))
        vev_inc = vev_inc / float(len(sites))
        vev += vev_inc
    vev = vev / len(times)
    return vev

# given three dimensionless quanitities that specify the model:
#
#   J / w,          electric field value
#   m / w,          particle mass
#   Delta_tau * w,  time step ^ **
#
# specifies all the internal variables --- m, J, w, and Delta_tau, ---
# used in computing acceptance probabilities for local updates
#
# ** N.B. wt = pi in *real* space is roughly the period
#   of particle density oscillations
def define_model_params(jw, mw, tw):
    global m
    global J
    global w
    global Delta_tau
    a = 1.
    w = 1./(2. * a)
    J = jw * w
    m = mw * w
    Delta_tau = tw / w

# given a configuration and a time index t_test,
# tests dot products state vectors at all times
# with a state vector at time t_test
# prints out the config and the dot product
def test_dot(cfg, t_test):
    print("Testing dot products on the following configuration:")
    print(cfg)
    print("Testing dot products of sites @ t = %d with..." % (int(t_test), ))
    for step in range(cfg.ntimes):
        print("t = %.3d: %.1d" % (step, cfg.dot(int(t_test), step), ))

###############################################################################
#                               SIMULATION                                    #
###############################################################################

if __name__ == '__main__':
### Specify MC settings
    n_corr = 10                     # how often to save configuration
    n_equil_sweeps = 50  * n_corr   # length of equilibration
    n_sampl_sweeps = 100 * n_corr   # length of sampling
    n_print_sweeps = 10  * n_corr   # how often to print sim status
    measure_obs    = True
### Specify model parameters
    nsites = 8
    ntimes = 20
    # for Savage's paper:   1.60, 0.16, 0.50
    # for Muschik's papers: 1.00, 1.00, 0.05
    # for roughly 20% acc:  0.30, 0.02, 0.50
    jw = 1.667
    mw = 0.167
    tw = 0.100
    define_model_params(jw, mw, tw)
    local = LocalUpdates(m, J, w, Delta_tau)
### Initialize the simulation ###
    cfg = Cfg(nsites, ntimes)
    upd = LocalUpdates(m, J, w, Delta_tau)
#    if (measure_obs):
    if not(os.path.isdir(obs_dir())):
        os.mkdir(obs_dir())
    obsfname = obs_fname(obs_dir(), nsites, ntimes, jw, mw, tw, n_corr)
    # obsfile = open(obsfname, 'w')   # will writes string, not bytes
    sim = LocalMC(cfg, upd, measure_obs, cfg_dir(), obs_dir())
### Evolve ###
    sim.evolve(n_corr, n_equil_sweeps, n_sampl_sweeps, n_print_sweeps)


