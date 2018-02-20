from scipy import *
import numpy as np
from integration import integrate, integrate_simpson_vectorized
import warnings as wr
from config import inf_price
from data_helpers import not_less_than_zero


def price_vg(pars: tuple, args: tuple, strict=False, check=False, bounds_only=True) -> ndarray:

    if len(pars) != 3:
        raise Exception("pars should have 3 parameters: nu, theta, sigma")
    nu, theta, sigma = pars
    if check and bad_pars(*pars, bounds_only=bounds_only):
        raise ValueError(f"bad parameters for vg: {pars}")

    if len(args) != 6:
        raise Exception("args should have 6 parameters: s, k, tau, r, q, is_call")
    s, k, tau, r, q, is_call = args

    if type(s) is not float and type(s) is not int:
        raise ValueError(f's must be float or int; passed {type(s)}')

    if type(k) is not np.ndarray and type(k) is not np.float64:
        if (type(k) is float) | (type(k) is int):
            k = np.array([float(k)])
        else:
            raise ValueError(f"k(strikes) should be either np.array or numeric; passed {type(k)}")

    wr.filterwarnings('error')

    try:
        call_prices = np.array(list(map(
            lambda i: call_price_vg(s=s, k=k[i], tau=tau, r=r, q=q, nu=nu, theta=theta, sigma=sigma),
            range(len(k)))))
    except Warning:
        if strict:
            raise ValueError(f"failed to model prices with {pars}")
        return np.array([inf_price] * len(k))

    if is_call:
        return not_less_than_zero(call_prices).flatten()
    else:
        return not_less_than_zero(
            call_prices.flatten() + np.array(k) * exp(-r * tau) - np.array([s] * len(k)) * exp(-q * tau))


def bad_pars(nu, theta, sigma, bounds_only: bool) -> bool:
    result = sigma <= 0
    result |= nu <= 0
    if bounds_only:
        result |= theta ** 2 + (2 * sigma ** 2) / nu <= 0
    return result


def call_price_vg(nu: float, theta: float, sigma: float, s, k: float, tau, r, q) -> float:
    v_p1 = 0.5 + 1 / pi * integrate_simpson_vectorized(
        f=lambda om: p1_value_vg(om=om, s=s, k=k, tau=tau, r=r, q=q, nu=nu, theta=theta, sigma=sigma))

    v_p2 = 0.5 + 1 / pi * integrate_simpson_vectorized(
        f=lambda om: p2_value_vg(om=om, s=s, k=k, tau=tau, r=r, q=q, nu=nu, theta=theta, sigma=sigma))

    return exp(-q * tau) * s * v_p1 - exp(-r * tau) * k * v_p2


def p1_value_vg(om, s, k, tau, r, q, nu, theta, sigma):
    return np.real(exp(-(0 + 1j) * log(k) * om) *
                   cf_vg(om - (0 + 1j), s, tau, r, q, nu, theta, sigma) /
                   ((0 + 1j) * om * s * exp((r - q) * tau)))


def p2_value_vg(om, s, k, tau, r, q, nu, theta, sigma):
    return np.real(exp(-(0 + 1j) * log(k) * om) *
                   cf_vg(om, s, tau, r, q, nu, theta, sigma) / ((0 + 1j) * om))


def cf_vg(om, s, tau, r, q, nu, theta, sigma):
    om1i = om * (0 + 1j)
    sigma2 = sigma ** 2
    w = log(1 - theta * nu - 0.5 * nu * sigma2) / nu
    temp = om1i * log(s) + om1i * (r - q + w) * tau
    temp = exp(temp)
    return temp / ((1 - om1i * theta * nu + 0.5 * sigma2 * nu * om ** 2) ** (tau / nu))
