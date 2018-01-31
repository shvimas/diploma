from scipy import *
import numpy as np
from integration import integrate_simpson_vectorized
import warnings as wr
from config import inf_price


def price_ls(pars: tuple, args: tuple) -> ndarray:
    if (len(args)) != 6:
        raise Exception("args should have 6 parameters: s, k, t, r, q, is_call")

    if (len(pars)) != 2:
        raise Exception("pars should have 2 parameters: sigma, alpha")

    alpha, sigma = pars
    beta = get_beta(sigma=sigma, alpha=alpha)
    s, k, t, r, q, is_call = args
    s = float(s)  # just to be sure

    if type(k) is not np.ndarray:
        if (type(k) is float) | (type(k) is int):
            k = np.array([float(k)])
        else:
            raise Exception("k(strikes) should be either np.array or numeric")

    if is_call:
        func = ls_call_price
    else:
        func = ls_put_price

    return s * func(strikes=(np.log(k / s)), beta=beta, r=r, d=q, t=t, alpha=alpha, sigma=sigma)


def get_beta(sigma: float, alpha: float) -> float:
    return 1.


def ls_log_return_fourier_transf(u: ndarray, r: float, d: float, t: float, sigma: float, alpha: float) -> ndarray:
    mu = sigma ** alpha / cos(pi * alpha / 2)
    return exp(1j * u * (r - d + mu) * t - (1j * u * sigma) ** alpha / cos(pi * alpha / 2) * t)


def ls_psi(v: ndarray, beta: float, r: float, d: float, t: float, sigma: float, alpha: float) -> ndarray:
    # see formula (6) in Carr-Madan
    return exp(-r * t) * (ls_log_return_fourier_transf(v - (beta + 1) * 1j, r, d, t, sigma, alpha) /
                          (beta * beta + beta - v * v + 1j * (2 * beta + 1) * v))


def ls_integrand(v: ndarray, k: ndarray,
                 beta: float, r: float, d: float, t: float, sigma: float, alpha: float) -> ndarray:
    return np.real(exp(-1j * v * k) * ls_psi(v, beta, r, d, t, sigma, alpha))


def ls_a(beta: float, r: float, d: float, t: float, sigma: float, alpha: float) -> float:
    # a is a truncation point in recovering the option's price integration
    result = 1
    while abs(ls_psi(result, beta, r, d, t, sigma, alpha)) > 1e-4:
        result *= 1.5
    return result


def ls_call_price(strikes: ndarray, beta: float, r: float, d: float, t: float, sigma: float, alpha: float) -> ndarray:
    # spot = 1
    a = 2 * ls_a(beta, r, d, t, sigma, alpha)
    result = np.array([])

    wr.filterwarnings('error')
    try:
        for strike in strikes:
            def integrand(v: ndarray): return ls_integrand(v, strike, beta, r, d, t, sigma, alpha)

            res_val = 1 / pi * exp(-beta * strike) * integrate_simpson_vectorized(integrand, 0, a)
            result = np.append(result, res_val)
    except Warning:
        result = np.array([inf_price] * len(strikes))

    return result


def ls_put_price(strikes: ndarray, beta: float, r: float, d: float, t: float, sigma: float, alpha: float) -> ndarray:
    call_prices = ls_call_price(strikes, beta, r, d, t, sigma, alpha)
    return call_prices + exp(-r * t) * strikes - exp(-d * t)  # spot = 1
