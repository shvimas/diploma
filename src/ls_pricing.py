from scipy import *
import numpy as np
import integration
from scipy.integrate import quad
import warnings as wr
from config import inf_price
from data_helpers import not_less_than_zero


def price_ls(pars: tuple, args: tuple) -> ndarray:
    if (len(args)) != 6:
        raise Exception("args should have 6 parameters: s, k, t, r, q, is_call")

    if (len(pars)) != 2:
        raise Exception("pars should have 2 parameters: sigma, alpha")

    alpha, sigma = pars
    beta = get_beta(sigma=sigma, alpha=alpha)
    s, k, t, r, q, is_call = args

    if type(s) is not float and type(s) is not int:
        raise ValueError(f's must be float or int; passed {type(s)}')

    if type(k) is not np.ndarray and type(k) is not np.float64:
        if (type(k) is float) | (type(k) is int):
            k = np.array([float(k)])
        else:
            raise ValueError(f"k(strikes) should be either np.array or numeric; passed {type(k)}")

    wr.filterwarnings('error')
    try:
        call_prices = ls_call_price(strikes=(np.log(k / s)), beta=beta, r=r, d=q, t=t, alpha=alpha, sigma=sigma)
    except Warning:
        call_prices = np.array([inf_price] * len(k))

    if is_call:
        return not_less_than_zero(call_prices).flatten()
    else:
        return not_less_than_zero(call_prices.flatten() + exp(-r * t) * k - exp(-q * t) * s)


# noinspection PyUnusedLocal
def get_beta(sigma: float, alpha: float) -> float:
    return 1.25


def ls_log_return_fourier_transf(u: ndarray, r: float, d: float, t: float,
                                 sigma: float, alpha: float) -> ndarray:
    mu = sigma ** alpha / cos(pi * alpha / 2)
    return exp(1j * u * (r - d + mu) * t - (1j * u * sigma) ** alpha / cos(pi * alpha / 2) * t)


def ls_psi(v: ndarray, beta: float, r: float, d: float, t: float,
           sigma: float, alpha: float) -> ndarray:
    # see formula (6) in Carr-Madan
    return exp(-r * t) * (ls_log_return_fourier_transf(v - (beta + 1) * 1j, r, d, t, sigma, alpha) /
                          (beta * beta + beta - v * v + 1j * (2 * beta + 1) * v))


def ls_integrand_vectorized(v: ndarray, k: ndarray,
                            beta: float, r: float, d: float, t: float, sigma: float, alpha: float) -> ndarray:
    return np.real(np.multiply(exp(-1j * np.mat(k).transpose() @ np.mat(v)), ls_psi(v, beta, r, d, t, sigma, alpha)))


def ls_integrand(v: ndarray, k: float, beta: float, r: float, d: float, t: float,
                 sigma: float, alpha: float) -> ndarray:
    return np.real(exp(-1j * v * k) * ls_psi(v, beta, r, d, t, sigma, alpha))


def ls_a(beta: float, r: float, d: float, t: float, sigma: float, alpha: float) -> float:
    # a is a truncation point in recovering the option's price integration
    result = 1
    while abs(ls_psi(result, beta, r, d, t, sigma, alpha)) > 1e-4:
        result *= 1.5
    return result


def ls_call_price(strikes: ndarray, beta: float, r: float, d: float, t: float,
                  sigma: float, alpha: float) -> ndarray:
    # spot = 1
    a = 2 * ls_a(beta, r, d, t, sigma, alpha)

    wr.filterwarnings('error')
    try:
        '''
        def integrand(v: ndarray):
            return ls_integrand_vectorized(v, strikes, beta, r, d, t, sigma, alpha)

        result = 1 / pi * exp(-beta * strikes) * \
            integration.integrate_simpson_vectorized(f=integrand, lower=0, upper=a).transpose()
        '''
        result = []
        for strike in strikes:
            def integrand(v): return ls_integrand(v, strike, beta, r, d, t, sigma, alpha)

            res_val = 1 / pi * exp(-beta * strike) * quad(integrand, 0, a)[0]
            result = np.append(result, res_val)

    except Warning:
        result = np.array([inf_price] * len(strikes))

    return result
