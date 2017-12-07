import numpy as np
import scipy.optimize as opt
from sklearn.metrics import mean_absolute_error
from modeling import models


def mean(seq):
    return sum(seq) / len(seq)


def mean_ratio(predicted, actual):
    tmp = list(map(
        lambda x: (x >= 1) * x + (x < 1) / x,
        predicted / actual
    ))

    return mean(tmp)


metrics = {
        "MAE": mean_absolute_error,
        "mean ratio": mean_ratio
    }


def apply_metric(predicted, actual, metric="MAE"):
    return metrics[metric](predicted, actual)


def is_good_enough(quality: float, metric: str) -> bool:
    values = {
        "MAE": 2,
        "mean ratio": 1.05
    }

    return quality < values[metric]


def optimize_func(func, args: tuple, bounds: tuple, pars_num):
    """

    :param func: function to optimize
    :param args: market parameters + quality metric as args[-2] + actual prices as args[-1]
    :param bounds: model parameters natural bounds
    :param pars_num: number of parameters; depends on model used

    :return: optimal parameters
    """

    if len(bounds) != pars_num:
        raise Exception("bounds' length should match pars_num")

    res = opt.differential_evolution(
        func=func,
        bounds=bounds,
        args=args
    )

    return res.x


def opt_helper(pars: tuple, args: tuple):
    """

    :param pars: model parameters
    :param args: market parameters with strikes as args[1] +
                    price modeling function as args[-3] +
                    quality metric as args[-2] +
                    actual prices as args[-1]

    :return: value of quality metric on passed parameters
    """

    k = args[1]
    model = args[-3]
    metric = args[-2]
    prices = args[-1]

    if model not in models.keys():
        raise Exception("Cannot use model " + model)

    if metric not in metrics.keys():
        raise Exception("Cannot use metric " + metric)

    if (type(prices) is not np.ndarray) | (type(k) is not np.ndarray) | (len(prices) != len(k)):
        raise Exception("strikes and prices should be np.arrays with same length")

    return apply_metric(models[model](pars=pars, args=args[:-3]), prices, metric)
