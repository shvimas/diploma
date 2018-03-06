import numpy as np
from numpy import ndarray, pi, exp, cos, log, sqrt
import integration
import config
import warnings as wr
import helper_funcs as hf


def ls_log_return(u: ndarray, r: float, q: float, t: float, alpha: float, sigma: float) -> ndarray:
    mu = sigma ** alpha / cos(pi * alpha / 2)
    return exp(1j * u * (r - q + mu) * t - (1j * u * sigma) ** alpha / cos(pi * alpha / 2) * t)


def vg_log_return(u: ndarray, r: float, q: float, t: float, nu: float, theta: float, sigma: float) -> ndarray:
    # beware of typo in Carr-Madan, see Carr-Madan-Chang
    w = 1 / nu * log(1 - theta * nu - .5 * nu * sigma ** 2)
    return exp(1j * u * (r - q) * t) * exp(1j * u * w * t) * \
        (1 - 1j * theta * nu * u + .5 * nu * (sigma * u) ** 2) ** (-t / nu)


def heston_log_return(u: ndarray, r: float, q: float, t: float,
                      kappa: float, theta: float, sigma: float, rho: float, v0: float) -> ndarray:
    """
        http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/ng.pdf
    """
    sigma2 = sigma ** 2
    rsip = rho * sigma * 1j * u
    d = sqrt((rsip - kappa) ** 2 + sigma2 * (1j * u + u ** 2))
    g = (kappa - rsip - d) / (kappa - rsip + d)
    exp_dt = exp(-d * t)

    cf1 = 1j * u * (r - q) * t  # log(S0) assumed to be 0
    cf2 = theta * kappa / sigma2 * ((kappa - rsip - d) * t - 2 * log((1 - g * exp_dt) / (1 - g)))
    cf3 = v0 / sigma2 * (kappa - rsip - d) * (1 - exp_dt) / (1 - g * exp_dt)

    return exp(cf1 + cf2 + cf3)


supported_models = dict(ls=ls_log_return, vg=vg_log_return, heston=heston_log_return)
num_of_pars = dict(ls=2, vg=3, heston=5)


class FFT:
    def __init__(self, model: str, args: tuple):
        try:
            self.log_return = supported_models[model]
        except KeyError:
            raise ValueError('unsupported model: ' + model)
        if len(args) != 6:
            raise ValueError('args must consist of: spot, strikes, maturity, rate, dividend rate, is_call')
        self.model = model
        self.spot, self.strikes, self.t, self.r, self.d, self.is_call = args
        self.beta = 1

    def price(self, pars: tuple):
        if len(pars) != num_of_pars[self.model]:
            raise ValueError(f'expected {num_of_pars[self.model]} parameters, got {len(pars)}: {pars}')
        prices = self.call_price(pars) if self.is_call else self.put_price(pars)
        return hf.not_less_than_zero(prices)

    def psi(self, v: ndarray, pars: tuple) -> ndarray:
        # see formula (6) in Carr-Madan
        return exp(-self.r * self.t) * \
               (self.log_return(v - (self.beta + 1) * 1j, self.r, self.d, self.t, *pars) /
                (self.beta * self.beta + self.beta - v * v + 1j * (2 * self.beta + 1) * v))

    def integrand_vectorized(self, v: ndarray, pars: tuple) -> ndarray:
        return np.real(np.multiply(
            exp(-1j * np.mat(log(self. strikes / self.spot)).transpose() @ np.mat(v)),
            self.psi(v, pars)))

    def truncation_point(self, pars: tuple) -> float:
        # a is a truncation point in recovering the option's price integration
        result = 1
        while abs(self.psi(result, pars)) > 1e-4:
            result *= 1.5
        return result

    def call_price(self, pars: tuple) -> ndarray:

        wr.filterwarnings('error')

        try:
            a = 2 * self.truncation_point(pars)

            def integrand(v: ndarray): return self.integrand_vectorized(v, pars)

            result = 1 / pi * exp(-self.beta * log(self.strikes / self.spot)) * \
                integration.integrate_simpson_vectorized(f=integrand, lower=0, upper=a).transpose()

        except Warning:
            result = np.array([config.inf_price] * len(self.strikes))

        return result * self.spot

    def put_price(self, pars: tuple) -> ndarray:
        return self.call_price(pars) + self.strikes * exp(-self.r * self.t) - self.spot * exp(-self.d * self.t)
