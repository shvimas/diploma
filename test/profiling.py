import cProfile
from vg_pricing import price_vg
from heston_pricing import price_heston
from ls_pricing import price_ls
import numpy as np

num_strikes = 1000


def profile_heston():
    kappa = 5
    theta = 1
    sigma = .5
    rho = .5
    v0 = .2
    p = (kappa, theta, sigma, rho, v0)

    spot = 100
    strike = np.array([100 + i * 10 for i in range(num_strikes)])
    mat = 1.2
    rate = .01
    q = rate
    is_call = True
    a = (spot, strike, mat, rate, q, is_call)

    cProfile.runctx(statement='price_heston(pars, args)',
                    globals={},
                    locals={"pars": p, "args": a, "price_heston": price_heston},
                    sort=1)


def profile_vg():
    nu = .1
    theta = .1
    sigma = .05
    p = (nu, theta, sigma)

    spot = 100
    strike = np.array([100 + i * 10 for i in range(num_strikes)])
    mat = 1.2
    rate = .01
    q = rate
    is_call = True
    a = (spot, strike, mat, rate, q, is_call)

    cProfile.runctx(statement='price_vg(pars, args)',
                    globals={},
                    locals={"pars": p, "args": a, "price_vg": price_vg},
                    sort=1)


def profile_ls():
    sigma = .5
    alpha = 1.6
    p = (sigma, alpha)

    spot = 100
    strike = np.array([100 + i * 10 for i in range(num_strikes)])
    mat = 1.2
    rate = .01
    q = rate
    is_call = True
    a = (spot, strike, mat, rate, q, is_call)

    cProfile.runctx(statement='price_ls(pars, args)',
                    globals={},
                    locals={"pars": p, "args": a, "price_ls": price_ls},
                    sort=1)


if __name__ == "__main__":
    profile_heston()
    profile_vg()
    profile_ls()