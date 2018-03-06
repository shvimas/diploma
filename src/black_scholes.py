from scipy.stats import norm
import numpy as np
import helper_funcs


def price_bs(pars: tuple, args: tuple) -> np.ndarray:
    if (len(args)) != 6:
        raise Exception("args should have 6 parameters: s, k, t, r, q, is_call")

    if (len(pars)) != 1:
        raise Exception("pars should have 1 parameter: sigma")

    sigma = pars[0]
    spot, strikes, t, r, q, is_call = args
    if is_call:
        return helper_funcs.not_less_than_zero(bs_call_price(spot, strikes, r, q, t, sigma))
    else:
        return helper_funcs.not_less_than_zero(bs_put_price(spot, strikes, r, q, t, sigma))


def bs_call_price(spot: np.ndarray, strikes: np.ndarray,
                  r: float, q: float, t: float, sigma: float) -> np.ndarray:
    a = (np.log(spot / strikes) + (r - q + sigma * sigma / 2) * t) / (sigma * np.sqrt(t))
    b = a - sigma * np.sqrt(t)
    return spot * norm.cdf(a) * np.exp(-q * t) - strikes * np.exp(-r * t) * norm.cdf(b)


def bs_put_price(spot: np.ndarray, strikes: np.ndarray,
                 r: float, q: float, t: float, sigma: float) -> np.ndarray:
    call_prices = bs_call_price(spot, strikes, r, q, t, sigma)
    return call_prices + strikes * np.exp(-r * t) - spot * np.exp(-q * t)


def bs_delta(spot: float, strikes: np.ndarray,
             r: float, q: float, t: float, bs_sigma: float, is_call: bool) -> np.ndarray:
    d1 = (np.log(spot / strikes) + (r - q + bs_sigma * bs_sigma / 2) * t) / bs_sigma / t ** .5
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1
