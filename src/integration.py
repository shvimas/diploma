import numpy as np
from scipy.integrate import simps, quad


lower_ = 1e-30
upper_ = 100
n_ = 2000


class Integrator:
    def __init__(self):
        self.n = n_
        self.lower = lower_
        self.upper = upper_
        self.integrate_vectorized_impl = Integrator.integrate_simpson_vectorized_impl
        self.integrate_impl = lambda f, lower, upper: quad(func=f, a=lower, b=upper)[0]

    def integrate(self, f, lower=None, upper=None):
        if lower is None:
            lower = 0
        if upper is None:
            upper = self.upper
        return self.integrate_impl(f=f, lower=lower, upper=upper)

    def integrate_vectorized(self, f, lower=None, upper=None, n=None) -> np.matrix:
        if n is None:
            n = self.n
        if lower is None:
            lower = self.lower
        if upper is None:
            upper = self.upper
        return self.integrate_vectorized_impl(f=f, lower=lower, upper=upper, n=n)

    @staticmethod
    def integrate_simpson_vectorized_impl(f, lower, upper, n) -> np.matrix:
        h = (upper - lower) / n
        q = np.arange(0, n + 1)
        x = lower + q * h
        return simps(y=f(x), x=x)


integrator = Integrator()


def integrate(f, lower=None, upper=None) -> float:
    return integrator.integrate(f=f, lower=lower, upper=upper)


def integrate_simpson_vectorized(f, lower=None, upper=None, n=None) -> np.matrix:
    return integrator.integrate_vectorized(f=f, lower=lower, upper=upper, n=n)
