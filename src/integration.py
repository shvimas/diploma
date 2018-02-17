import numpy as np
from scipy.integrate import simps, quad
import config as cfg


class Integrator:
    def __init__(self):
        self.n = cfg.n
        self.lower = cfg.lower
        self.upper = cfg.upper
        self.integrate_impl = Integrator.integrate_impl_simps
        self.integrate_impl_1d = lambda f, lower, upper: quad(func=f, a=lower, b=upper)[0]

    def integrate_1d(self, f, lower=None, upper=None):
        if lower is None:
            lower = 0
        if upper is None:
            upper = self.upper
        return self.integrate_impl_1d(f=f, lower=lower, upper=upper)

    def integrate(self, f, lower=None, upper=None, n=None) -> np.matrix:
        if n is None:
            n = self.n
        if lower is None:
            lower = self.lower
        if upper is None:
            upper = self.upper
        return self.integrate_impl(f=f, lower=lower, upper=upper, n=n)

    @staticmethod
    def integrate_impl_1(f, lower, upper, n) -> np.matrix:
        h = (upper - lower) / n
        q = np.arange(0, n + 1)
        ff = f(lower + q * h)
        w = np.array(3 + (-1) ** (q + 1))
        w[0] = np.array([1])
        w[n] = np.array([1])
        result = h / 3 * np.dot(ff, w)
        return result

    @staticmethod
    def integrate_impl_simps(f, lower, upper, n) -> np.matrix:
        h = (upper - lower) / n
        q = np.arange(0, n + 1)
        x = lower + q * h
        return simps(y=f(x), x=x)


integrator = Integrator()


def integrate(f, lower=None, upper=None) -> float:
    return integrator.integrate_1d(f=f, lower=lower, upper=upper)


def integrate_simpson_vectorized(f, lower=None, upper=None, n=None) -> np.matrix:
    return integrator.integrate(f=f, lower=lower, upper=upper, n=n)
