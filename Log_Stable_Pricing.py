from scipy import *
import numpy as np
from integration import integrate_simpson_vectorized


def price_ls(pars: tuple, args: tuple) -> ndarray:
    if (len(args)) != 6:
        raise Exception("args should have 6 parameters: s, k, t, r, q, is_call")

    if (len(pars)) != 3:
        raise Exception("pars should have 3 parameters: sigma, alpha, beta")

    sigma, alpha, beta = pars
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

    return s * func(strikes=k / s, beta=beta, r=r, d=q, t=t, alpha=alpha, sigma=sigma)
    # return np.array(list(map(
    #    lambda i: s * func(k=k[i] / s, r=r, d=q, t=t), range(len(k)))))


def ls_log_return_fourier_transf(u, r, d, t, sigma, alpha):

    mu = sigma ** alpha / cos(pi * alpha / 2)
    return exp(1j * u * (r - d) * t + (1j * u * mu - (1j * u * sigma) ** alpha / cos(pi * alpha / 2)))


def ls_psi(v, beta, r, d, t, sigma, alpha):
    # see formula (6) in Carr-Madan
    return exp(-r * t) * \
           ls_log_return_fourier_transf(v - (beta + 1) * 1j, r, d, t, sigma, alpha) / \
           (beta * beta + beta - v * v + 1j * (2 * beta + 1) * v)


def ls_integrand5(v, k, beta, r, d, t, sigma, alpha):
    return np.real(exp(-1j * v * k) * ls_psi(v, beta, r, d, t, sigma, alpha))


def ls_a(beta, r, d, t, sigma, alpha):
    # a is a truncation point in recovering the option's price integration
    result = 1
    while abs(ls_psi(result, beta, r, d, t, sigma, alpha)) > 1e-4:
        result *= 1.5
    return result


def ls_call_price(strikes, beta, r, d, t, sigma, alpha):
    a = 2 * ls_a(beta, r, d, t, sigma, alpha)
    h1 = 0.05
    h2 = 0.05
    # works with 0.02 USD precision with at t=0.025
    n1 = np.ceil(3 / h1)
    n2 = np.ceil((a - 3) / h2)
    v1 = np.arange(1, (n1 + 1) - 1) * h1
    v2 = 3 + np.arange(1, (n2 + 1) - 1) * h2
    
    psi1 = ls_psi(v1, beta, r, d, t, sigma, alpha)
    psi2 = ls_psi(v2, beta, r, d, t, sigma, alpha)

    result = np.array([0] * len(strikes))

    for j in range(1, len(strikes)):
        integrand5_1 = np.real(exp(-1j * v1 * strikes[j]) * psi1)
        integrand5_2 = np.real(exp(-1j * v2 * strikes[j]) * psi2)
        result[j] = 1 / pi * exp(-beta * strikes[j]) * \
            (integrate_simpson_vectorized(integrand5_1, 0, n1 * h1)
                + integrate_simpson_vectorized(integrand5_2, 3, 3 + n2 * h2))

        if result[j] < 0:
            result[j] = 0

    return result


def ls_put_price(strikes, beta, r, d, t, sigma, alpha):
    call_prices = ls_call_price(strikes, beta, r, d, t, sigma, alpha)
    return call_prices + exp(-r * t) * strikes - exp(-d * t)  # spot = 1
