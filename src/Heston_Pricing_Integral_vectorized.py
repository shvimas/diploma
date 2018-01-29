from scipy import *
import numpy as np
from integration import integrate_simpson_vectorized

'''
    2 * kappa * theta > sigma ** 2
'''


def price_heston(pars: tuple, args: tuple) -> ndarray:
    if (len(args)) != 6:
        raise Exception("args should have 6 parameters: s, k, t, r, q, is_call")

    if (len(pars)) != 5:
        raise Exception("pars should have 5 parameters: kappa, theta, sigma, rho, v0")

    kappa, theta, sigma, rho, v0 = pars
    s, k, t, r, q, is_call = args

    if type(k) is not ndarray:
        if (type(k) is float) | (type(k) is int):
            k = np.array([float(k)])
        else:
            raise Exception("k(strikes) should be either np.array or numeric")

    # natural constraint
    #if 2 * kappa * theta <= sigma ** 2:
    #    return np.array([Inf] * len(k))

    if is_call:
        func = heston_call_value_int
    else:
        func = heston_put_value_int

    return np.array(list(map(
        lambda i: func(kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0, r=r, q=q, t=t, s0=s, k=k[i]),
        range(len(k))))
    )


def heston_put_value_int(kappa, theta, sigma, rho, v0, r, q, t, s0, k):
    c0 = heston_call_value_int(kappa, theta, sigma, rho, v0, r, q, t, s0, k)
    return c0 + k * exp(-r * t) - s0


def heston_call_value_int(kappa, theta, sigma, rho, v0, r, q, t, s0, k):
    a = s0 * heston_pvalue(kappa, theta, sigma, rho, v0, r, q, t, s0, k, 1)
    b = k * exp(-r * t) * heston_pvalue(kappa, theta, sigma, rho, v0, r, q, t, s0, k, 2)
    return a - b


def heston_pvalue(kappa, theta, sigma, rho, v0, r, q, t, s0, k, typ):
    integral = integrate_simpson_vectorized(
        lambda phi: int_function_1(phi, kappa, theta, sigma, rho, v0, r, q, t, s0, k, typ),
        lower=1e-14,
        upper=50)
    return 0.5 + (1 / pi) * integral


# COPY FROM R
def int_function_1(phi, kappa, theta, sigma, rho, v0, r, q, t, s0, k, typ):
    ans = np.real(np.exp(-1j * np.outer(np.log(k), phi)) *
                  int_function_2(phi, kappa, theta, sigma, rho, v0, r, q, t, s0, typ) / (1j * phi))

    return ans[0]  # no need to store one number as np.array


def int_function_2(phi, kappa, theta, sigma, rho, v0, r, q, t, s0, typ):
    if typ == 1:
        cf = cf_heston(phi - 1j, kappa, theta, sigma, rho, v0, r, q, t, s0) / (s0 * exp(r * t))
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
