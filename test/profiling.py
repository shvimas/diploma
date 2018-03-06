# import cProfile
import vg_pricing as vg
import heston_pricing as he
import ls_pricing as ls
import numpy as np
from fft import FFT
from time import time
from datetime import timedelta

num_strikes = 10000

spot = 1000
strike = np.array([100 + i for i in range(num_strikes)])
mat = 1.2
rate = .01
q = rate
is_call = True
a = (spot, strike, mat, rate, q, is_call)


def td_decorator(func, model: str):
    def wrapper(*args, **kwargs):
        print(f'Profiling {model}')
        t0 = time()
        val = func(*args, **kwargs)
        print(f'Time spent for {num_strikes} strikes: {timedelta(seconds=time() - t0)}\n')
        return val
    return wrapper


def profile_heston():
    kappa = 5
    theta = 1
    sigma = .5
    rho = .5
    v0 = .2
    p = (kappa, theta, sigma, rho, v0)

    td_decorator(he.price_heston, 'Heston')(pars=p, args=a)
    td_decorator(FFT(model='heston', args=a).price, 'FFT Heston')(p)


def profile_vg():
    nu = .1
    theta = .1
    sigma = .05
    p = (nu, theta, sigma)

    td_decorator(vg.price_vg, 'VG')(pars=p, args=a)
    td_decorator(FFT(model='vg', args=a).price, 'FFT VG')(p)


def profile_ls():
    sigma = .5
    alpha = 1.6
    p = (sigma, alpha)

    td_decorator(ls.price_ls, 'LS')(pars=p, args=a)
    td_decorator(FFT(model='ls', args=a).price, 'FFT LS')(p)


if __name__ == "__main__":
    profile_heston()
    profile_vg()
    profile_ls()
