import numpy as np


def integrate_simpson(f, a, b):
    # n should be even (2*k)
    n = 1000
    h = (b - a) / n
    result = h / 3 * (f(a) + f(b))
    for i in range(1, n):
        result = result + h / 3 * (3 + (-1) ** (i + 1)) * f(a + i * h)
    return result


def integrate_simpson_vectorized(f, lower, upper) -> float:
    n = 5000
    h = (upper - lower) / n
    q = np.arange(0, n + 1)
    ff = f(lower + q * h)
    w = np.array(3 + (-1) ** (q + 1))
    w[0] = np.array([1])
    w[n] = np.array([1])
    result = h / 3 * np.dot(ff, w)
    return result


def pair_max(seq1: np.ndarray, seq2: np.ndarray) -> np.ndarray:
    if len(seq1) != len(seq2):
        raise Exception("sequences must have the same length")

    return np.array(list(map(
        lambda i: max(seq1[i], seq2[i]),
        range(len(seq1))
    )))