from scipy import *
import numpy as np
import integration
import warnings as wr
import config
import helper_funcs as hf

'''
    2 * kappa * theta > sigma ** 2
'''


def price_heston(pars: tuple, args: tuple, strict=False, check=True, bounds_only=True) -> ndarray:
    if (len(args)) != 6:
        raise Exception("args should have 6 parameters: s, k, t, r, q, is_call")

    if (len(pars)) != 5:
        raise Exception("pars should have 5 parameters: kappa, theta, sigma, rho, v0")

    kappa, theta, sigma, rho, v0 = pars
    if check and bad_pars(*pars, bounds_only=bounds_only):
        raise ValueError(f'bad parameters for heston: {pars}')

    s, k, t, r, q, is_call = args

    if type(s) is not float and type(s) is not int:
        raise ValueError(f's must be float or int; passed {type(s)}')

    if type(k) is not ndarray and type(k) is not np.float64:
        if (type(k) is float) | (type(k) is int):
            k = np.array([float(k)])
        else:
            raise ValueError(f"k(strikes) should be either np.array or numeric; passed {type(k)}")

    if is_call:
        func = heston_call_value_int
    else:
        func = heston_put_value_int

    wr.filterwarnings('error')

    try:
        return np.array(list(map(
            lambda strike: hf.not_less_than_zero(func(kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0,
                                                                r=r, q=q, t=t, s0=s, k=strike)),
            k))).flatten()
    except Warning:
        if strict:
            raise ValueError(f"failed to model prices with {pars}")
        return np.array([config.inf_price] * len(k))


def bad_pars(kappa, theta, sigma, rho, v0, bounds_only: bool) -> bool:
    result = theta <= 0
    result |= sigma <= 0
    result |= (rho < 0) | (rho > 1)
    result |= v0 <= 0
    if not bounds_only:
        result |= 2 * kappa * theta <= sigma ** 2
    return result


def heston_put_value_int(kappa, theta, sigma, rho, v0, r, q, t, s0, k):
    c0 = heston_call_value_int(kappa, theta, sigma, rho, v0, r, q, t, s0, k)
    return c0 + k * exp(-r * t) - s0 * exp(-q * t)


def heston_call_value_int(kappa, theta, sigma, rho, v0, r, q, t, s0, k):
    a = s0 * exp(-q * t) * heston_pvalue(kappa, theta, sigma, rho, v0, r, q, t, s0, k, 1)
    b = k * exp(-r * t) * heston_pvalue(kappa, theta, sigma, rho, v0, r, q, t, s0, k, 2)
    return a - b


def heston_pvalue(kappa, theta, sigma, rho, v0, r, q, t, s0, k, typ):
    return 0.5 + (1 / pi) * integration.integrate_simpson_vectorized(
        lambda phi: int_function_1(phi, kappa, theta, sigma, rho, v0, r, q, t, s0, k, typ))


# COPY FROM R
def int_function_1(phi, kappa, theta, sigma, rho, v0, r, q, t, s0, k, typ):
    ans = np.real(np.exp(-1j * np.outer(np.log(k), phi)) *
                  int_function_2(phi, kappa, theta, sigma, rho, v0, r, q, t, s0, typ) / (1j * phi))

    return ans


def int_function_2(phi, kappa, theta, sigma, rho, v0, r, q, t, s0, typ):
    if typ == 1:
        cf = cf_heston(phi - 1j, kappa, theta, sigma, rho, v0, r, q, t, s0) / (s0 * exp((r - q) * t))
    else:
        cf = cf_heston(phi, kappa, theta, sigma, rho, v0, r, q, t, s0)
    return cf


def cf_heston(phi, kappa, theta, sigma, rho, v0, r, q, t, s0):
    sigma2 = sigma ** 2
    rsip = rho * sigma * 1j * phi
    d = sqrt((rsip - kappa) ** 2 + sigma2 * (1j * phi + phi ** 2))
    g = (kappa - rsip - d) / (kappa - rsip + d)
    exp_dt = exp(-d * t)
    
    cf1 = 1j * phi * (log(s0) + (r - q) * t)
    cf2 = theta * kappa / sigma2 * ((kappa - rsip - d) * t - 2 * log((1 - g * exp_dt) / (1 - g)))
    cf3 = v0 / sigma2 * (kappa - rsip - d) * (1 - exp_dt) / (1 - g * exp_dt)
    
    return exp(cf1 + cf2 + cf3)
