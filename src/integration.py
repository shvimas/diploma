import numpy as np
from scipy.integrate import simps
# import tensorflow as tf
import config as cfg


class Integrator:
    def __init__(self):
        self.n = cfg.n
        self.upper = cfg.upper
        self.integrate_impl = Integrator.integrate_impl_simps

    def integrate(self, f, lower, upper=None, n=None) -> np.matrix:
        if n is None:
            n = self.n
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


def integrate_simpson(f, lower, upper=cfg.upper, n=cfg.n) -> float:
    h = (upper - lower) / n
    result = h / 3 * (f(lower) + f(upper))
    for i in range(1, n):
        result = result + h / 3 * (3 + (-1) ** (i + 1)) * f(lower + i * h)
    return result


def integrate_simpson_vectorized(f, lower, upper=None, n=None) -> np.matrix:
    return integrator.integrate(f=f, lower=lower, upper=upper, n=n)


'''
def integrate_simpson_tensorflow(f, lower, upper, n=5000):
    h = (upper - lower) / n
    q = tf.range(start=0, limit=n+1, dtype=tf.float32)
    f_values = f(lower + q * h)
    w = 3 + (-1) ** (q + 1)
    w[0] = 0
    w[-1] = 0
    return h / 3 * tf.matmul(f_values, w, transpose_b=True)
'''
